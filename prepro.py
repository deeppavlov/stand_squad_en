import tensorflow as tf
import random
from tqdm import tqdm
import ujson as json
from collections import Counter
import numpy as np
import pickle
import os
import unicodedata
from pymystem3 import Mystem
from nltk import word_tokenize

from subword_nmt.apply_bpe import BPE
from util import sber2squad


# def word_tokenize(sent):
#    doc = nlp(sent)
#    return [token.text for token in doc]


def get_pos(token, pos_model):
    res = pos_model.analyze(token)[0]
    if 'analysis' not in res or len(res['analysis']) == 0:
        return 'NONE'
    pos = res['analysis'][0]['gr'].split(',')[0]
    return pos

def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def preprocess_string(s, unicode_mapping=False, remove_unicode=True):
    s = s.replace("''", '" ').replace("``", '" ')
    if not unicode_mapping:
        return ''.join(c for c in s if not unicodedata.combining(c))
    else:
        raw2prepro_idxs = [len(s)] * (len(s) + 1)
        prepro2raw_idxs = [len(s)] * (len(s) + 1)
        preprocessed_s = ''
        for i, c in enumerate(s):
            if unicodedata.combining(c) and remove_unicode:
                raw2prepro_idxs[i] = -1
            else:
                preprocessed_s += c
                raw2prepro_idxs[i] = len(preprocessed_s) - 1
                prepro2raw_idxs[len(preprocessed_s) - 1] = i
        return preprocessed_s, raw2prepro_idxs, prepro2raw_idxs


def process_file(config, filename, data_type, word_counter=None, char_counter=None, bpe_counter=None, pos_counter=None,
                 remove_unicode=True, bpe_model=None, pos_model=None, is_test=False):
    print("Generating {} examples...".format(data_type))
    para_limit = config.test_para_limit if is_test else config.para_limit
    ques_limit = config.test_ques_limit if is_test else config.ques_limit
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in source["data"]:
            for para in tqdm(article["paragraphs"]):
                context_raw = para['context']
                context, r2p, p2r = preprocess_string(para["context"], unicode_mapping=True, remove_unicode=remove_unicode)
                context_tokens = [token.replace("''", '"').replace("``", '"') for token in word_tokenize(context)[:para_limit]]
                context_chars = [list(token) for token in context_tokens]
                context_bpe = []
                context_pos = []
                if bpe_model is not None:
                    context_bpe = [bpe_model.segment(token).split(' ') for token in context_tokens]

                if pos_model is not None:
                    context_pos = [get_pos(token, pos_model) for token in context_tokens]

                spans = convert_idx(context, context_tokens)
                if word_counter is not None:
                    for token in context_tokens:
                        word_counter[token] += len(para["qas"])
                        if char_counter is not None:
                            for char in token:
                                char_counter[char] += len(para["qas"])

                if bpe_counter is not None:
                    for token in context_bpe:
                        for bpe in token:
                            bpe_counter[bpe] += len(para["qas"])

                if pos_counter is not None:
                    for pos in context_pos:
                        pos_counter[pos] += len(para["qas"])

                for qa in para["qas"]:
                    total += 1
                    ques = preprocess_string(qa["question"], remove_unicode=remove_unicode)
                    ques_tokens = word_tokenize(ques)[:ques_limit]
                    ques_chars = [list(token) for token in ques_tokens]
                    ques_bpe = []
                    ques_pos = []
                    if bpe_model is not None:
                        ques_bpe = [bpe_model.segment(token).split(' ') for token in ques_tokens]

                    if pos_model is not None:
                        ques_pos = [get_pos(token, pos_model) for token in ques_tokens]

                    if word_counter is not None:
                        for token in ques_tokens:
                            word_counter[token] += 1
                            if char_counter is not None:
                                for char in token:
                                    char_counter[char] += 1

                    if bpe_counter is not None:
                        for token in context_bpe:
                            for bpe in token:
                                bpe_counter[bpe] += 1

                    if pos_counter is not None:
                        for pos in ques_pos:
                            pos_counter[pos] += 1

                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = preprocess_string(answer["text"], remove_unicode=remove_unicode)
                        # convert answer start index to index in preprocessed context
                        answer_start = r2p[answer['answer_start']]
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        if len(answer_span) == 0:
                            # there is no answer in context_tokens (mb because of para_limit)
                            continue
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                    if len(answer_texts) == 0 and len(qa["answers"]) != 0:
                        # all answers are in the end of long context
                        # skipping such QAs
                        continue
                    example = {"context_tokens": context_tokens, "context_chars": context_chars,
                               "context_bpe": context_bpe, "context_pos": context_pos,
                               "ques_tokens": ques_tokens, "ques_chars": ques_chars,
                               "ques_bpe": ques_bpe, "ques_pos": ques_pos,
                               "y1s": y1s, "y2s": y2s, "id": total}
                    examples.append(example)
                    eval_examples[str(total)] = {
                        "context": context, "spans": spans,
                        "answers": answer_texts, "uuid": qa["id"],
                        "context_raw": context_raw,
                        "raw2prepro": r2p,
                        "prepro2raw": p2r,
                    }
        random.shuffle(examples)
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples


def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            print('Using pretrained embdgs: {}'.format(emb_file))
            for line in tqdm(fh, total=size):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx,
                      token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def build_features(config, examples, data_type, out_file, word2idx_dict, char2idx_dict,
                   bpe2idx_dict=None, pos2idx_dict=None, is_test=False):

    para_limit = config.test_para_limit if is_test else config.para_limit
    ques_limit = config.test_ques_limit if is_test else config.ques_limit
    char_limit = config.char_limit
    bpe_limit = config.bpe_limit

    def filter_func(example):
        return len(example["context_tokens"]) > para_limit or len(example["ques_tokens"]) > ques_limit

    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    total_ = 0
    meta = {}
    for example in tqdm(examples):
        total_ += 1

        if filter_func(example):
            continue

        total += 1
        context_idxs = np.zeros([para_limit], dtype=np.int32)
        context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
        context_bpe_idxs = np.zeros([para_limit, bpe_limit], dtype=np.int32)
        context_pos_idxs = np.zeros([para_limit], dtype=np.int32)
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
        ques_bpe_idxs = np.zeros([ques_limit, bpe_limit], dtype=np.int32)
        ques_pos_idxs = np.zeros([ques_limit], dtype=np.int32)
        y1 = np.zeros([para_limit], dtype=np.float32)
        y2 = np.zeros([para_limit], dtype=np.float32)

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_emb(k, k2idx_dict):
            if k in k2idx_dict:
                return k2idx_dict[k]
            return 1

        for i, token in enumerate(example["context_tokens"]):
            context_idxs[i] = _get_word(token)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idxs[i] = _get_word(token)

        if config.use_char:
            for i, token in enumerate(example["context_chars"]):
                for j, char in enumerate(token):
                    if j == char_limit:
                        break
                    context_char_idxs[i, j] = _get_emb(char, char2idx_dict)

            for i, token in enumerate(example["ques_chars"]):
                for j, char in enumerate(token):
                    if j == char_limit:
                        break
                    ques_char_idxs[i, j] = _get_emb(char, char2idx_dict)

        if config.use_bpe:
            for i, token in enumerate(example["context_bpe"]):
                for j, bpe in enumerate(token):
                    if j == bpe_limit:
                        break
                    context_bpe_idxs[i, j] = _get_emb(bpe, bpe2idx_dict)

            for i, token in enumerate(example["ques_bpe"]):
                for j, bpe in enumerate(token):
                    if j == bpe_limit:
                        break
                    ques_bpe_idxs[i, j] = _get_emb(bpe, bpe2idx_dict)

        if config.use_pos:
            for i, token in enumerate(example["context_pos"]):
                context_pos_idxs[i] = _get_emb(token, pos2idx_dict)

            for i, token in enumerate(example["ques_pos"]):
                ques_pos_idxs[i] = _get_emb(token, pos2idx_dict)

        # if we have no answers in file (it means we are in predict mode)
        # then add dummy answers
        start, end = 0, 0
        if len(example["y1s"]) > 0:
            start, end = example["y1s"][-1], example["y2s"][-1]

        y1[start], y2[end] = 1.0, 1.0

        record = tf.train.Example(features=tf.train.Features(feature={
                                  "context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
                                  "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
                                  "context_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_char_idxs.tostring()])),
                                  "ques_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_idxs.tostring()])),
                                  "context_bpe_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_bpe_idxs.tostring()])),
                                  "ques_bpe_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_bpe_idxs.tostring()])),
                                  "context_pos_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_pos_idxs.tostring()])),
                                  "ques_pos_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_pos_idxs.tostring()])),
                                  "y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
                                  "y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
                                  "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))
                                  }))
        writer.write(record.SerializeToString())
    print("Build {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    writer.close()
    return meta


def build_features_notfdata(config, examples, data_type, word2idx_dict, char2idx_dict,
                   bpe2idx_dict=None, pos2idx_dict=None, is_test=False):

    para_limit = config.test_para_limit if is_test else config.para_limit
    ques_limit = config.test_ques_limit if is_test else config.ques_limit
    char_limit = config.char_limit
    bpe_limit = config.bpe_limit

    def filter_func(example):
        return len(example["context_tokens"]) > para_limit or len(example["ques_tokens"]) > ques_limit

    print("Processing {} examples...".format(data_type))
    total = 0
    total_ = 0
    meta = {}
    records = []
    for example in tqdm(examples):
        total_ += 1

        if filter_func(example):
            continue

        total += 1
        context_idxs = np.zeros([para_limit], dtype=np.int32)
        context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
        context_bpe_idxs = np.zeros([para_limit, bpe_limit], dtype=np.int32)
        context_pos_idxs = np.zeros([para_limit], dtype=np.int32)
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
        ques_bpe_idxs = np.zeros([ques_limit, bpe_limit], dtype=np.int32)
        ques_pos_idxs = np.zeros([ques_limit], dtype=np.int32)
        y1 = np.zeros([para_limit], dtype=np.float32)
        y2 = np.zeros([para_limit], dtype=np.float32)

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_emb(k, k2idx_dict):
            if k in k2idx_dict:
                return k2idx_dict[k]
            return 1

        for i, token in enumerate(example["context_tokens"]):
            context_idxs[i] = _get_word(token)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idxs[i] = _get_word(token)

        if config.use_char:
            for i, token in enumerate(example["context_chars"]):
                for j, char in enumerate(token):
                    if j == char_limit:
                        break
                    context_char_idxs[i, j] = _get_emb(char, char2idx_dict)

            for i, token in enumerate(example["ques_chars"]):
                for j, char in enumerate(token):
                    if j == char_limit:
                        break
                    ques_char_idxs[i, j] = _get_emb(char, char2idx_dict)

        if config.use_bpe:
            for i, token in enumerate(example["context_bpe"]):
                for j, bpe in enumerate(token):
                    if j == bpe_limit:
                        break
                    context_bpe_idxs[i, j] = _get_emb(bpe, bpe2idx_dict)

            for i, token in enumerate(example["ques_bpe"]):
                for j, bpe in enumerate(token):
                    if j == bpe_limit:
                        break
                    ques_bpe_idxs[i, j] = _get_emb(bpe, bpe2idx_dict)

        if config.use_pos:
            for i, token in enumerate(example["context_pos"]):
                context_pos_idxs[i] = _get_emb(token, pos2idx_dict)

            for i, token in enumerate(example["ques_pos"]):
                ques_pos_idxs[i] = _get_emb(token, pos2idx_dict)

        # if we have no answers in file (it means we are in predict mode)
        # then add dummy answers
        start, end = 0, 0
        if len(example["y1s"]) > 0:
            start, end = example["y1s"][-1], example["y2s"][-1]

        y1[start], y2[end] = 1.0, 1.0

        record = {
                  "context_idxs": context_idxs,
                  "ques_idxs": ques_idxs,
                  "context_char_idxs": context_char_idxs,
                  "ques_char_idxs": ques_char_idxs,
                  "context_bpe_idxs": context_bpe_idxs,
                  "ques_bpe_idxs": ques_bpe_idxs,
                  "context_pos_idxs": context_pos_idxs,
                  "ques_pos_idxs": ques_pos_idxs,
                  "y1": y1,
                  "y2": y2,
                  "id": example["id"],
                  }
        records.append(record)
    print("Build {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    return records, meta


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def train_bpe(config):
    print('Start BPE training...')
    from subword_nmt.learn_bpe import main as learn_bpe
    train = json.load(open(config.train_file, 'r'))
    train_texts = []
    for p in train['data'][0]['paragraphs']:
        train_texts.append(preprocess_string(' '.join(word_tokenize(p['context'])),
                                             remove_unicode=config.remove_unicode))
        for qas in p['qas']:
            train_texts.append(preprocess_string(' '.join(word_tokenize(qas['question'])),
                                                 remove_unicode=config.remove_unicode))

    learn_bpe(train_texts, outfile=open(config.bpe_codes_file, 'w'), num_symbols=config.bpe_merges_count)
    print('BPE trained. BPE codes saved to {}'.format(config.bpe_codes_file))


def prepro(config):
    word_counter, char_counter, bpe_counter, pos_counter = Counter(), None, None, None
    bpe_model = None
    pos_model = None
    if config.use_bpe:
        if not config.use_bpe_pretrained_codes:
            # train bpe on train set
            train_bpe(config)
            bpe_model = BPE(open(config.bpe_codes_file, 'r'))
        else:
            print('Loading BPE codes: {}'.format(config.bpe_pretrained_codes_file))
            bpe_model = BPE(open(config.bpe_pretrained_codes_file, 'r'))
        bpe_counter = Counter()

    if config.use_char:
        char_counter = Counter()

    if config.use_pos:
        pos_model = Mystem()
        pos_counter = Counter()

    train_examples, train_eval = process_file(
        config, config.train_file, "train", word_counter, char_counter, bpe_counter, pos_counter,
        config.remove_unicode, bpe_model, pos_model, is_test=False)
    dev_examples, dev_eval = process_file(
        config, config.dev_file, "dev", word_counter, char_counter, bpe_counter, pos_counter,
        config.remove_unicode, bpe_model, pos_model, is_test=False)
    test_examples, test_eval = process_file(
        config, config.test_file, "test",
        remove_unicode=config.remove_unicode, bpe_model=bpe_model, pos_model=pos_model, is_test=True)

    word_emb_file = config.fasttext_file if config.fasttext else config.glove_word_file
    char_emb_file = config.glove_char_file if config.pretrained_char else None
    char_emb_size = config.glove_char_size if config.pretrained_char else None
    char_emb_dim = config.glove_dim if config.pretrained_char else config.char_dim
    bpe_emb_file = config.glove_bpe_file if config.pretrained_bpe_emb else None
    bpe_emb_size = config.glove_bpe_size if config.pretrained_bpe_emb else None
    bpe_emb_dim = config.bpe_glove_dim if config.pretrained_bpe_emb else config.bpe_dim

    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, "word", emb_file=word_emb_file, size=config.glove_word_size, vec_size=config.glove_dim)
    char_emb_mat, char2idx_dict = None, None
    if config.use_char:
        char_emb_mat, char2idx_dict = get_embedding(
            char_counter, "char", emb_file=char_emb_file, size=char_emb_size, vec_size=char_emb_dim)
    bpe_emb_mat, bpe2idx_dict = None, None
    if config.use_bpe:
        bpe_emb_mat, bpe2idx_dict = get_embedding(
            bpe_counter, "bpe", emb_file=bpe_emb_file, size=bpe_emb_size, vec_size=bpe_emb_dim)

    pos_emb_mat, pos2idx_dict = None, None
    if config.use_pos:
        pos_emb_mat, pos2idx_dict = get_embedding(
            pos_counter, "pos", emb_file=None, size=None, vec_size=config.pos_dim)

    pickle.dump(word2idx_dict, open(config.word2idx_dict_file, 'wb'))
    pickle.dump(char2idx_dict, open(config.char2idx_dict_file, 'wb'))
    pickle.dump(bpe2idx_dict, open(config.bpe2idx_dict_file, 'wb'))
    pickle.dump(pos2idx_dict, open(config.pos2idx_dict_file, 'wb'))

    build_features(config, train_examples, "train",
                   config.train_record_file, word2idx_dict, char2idx_dict, bpe2idx_dict, pos2idx_dict)
    dev_meta = build_features(config, dev_examples, "dev",
                              config.dev_record_file, word2idx_dict, char2idx_dict, bpe2idx_dict, pos2idx_dict)
    test_meta = build_features(config, test_examples, "test",
                               config.test_record_file, word2idx_dict, char2idx_dict, bpe2idx_dict, pos2idx_dict, is_test=True)

    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.char_emb_file, char_emb_mat, message="char embedding")
    save(config.bpe_emb_file, bpe_emb_mat, message="bpe embedding")
    save(config.pos_emb_file, pos_emb_mat, message="pos embedding")
    save(config.train_eval_file, train_eval, message="train eval")
    save(config.dev_eval_file, dev_eval, message="dev eval")
    save(config.test_eval_file, test_eval, message="test eval")
    save(config.dev_meta, dev_meta, message="dev meta")
    save(config.test_meta, test_meta, message="test meta")


def prepro_predict(config):

    if config.use_bpe and config.use_bpe_pretrained_codes:
        bpe_model = BPE(open(config.bpe_pretrained_codes_file, 'r'))
    elif config.use_bpe and not config.use_bpe_pretrained_codes:
        bpe_model = BPE(open(config.bpe_codes_file, 'r'))
    else:
        bpe_model = None

    predict_examples, predict_eval = process_file(
        config, config.predict_file, "predict",
        remove_unicode=config.remove_unicode, bpe_model=bpe_model, is_test=True)

    word2idx_dict = pickle.load(open(config.word2idx_dict_file, 'rb'))
    char2idx_dict = pickle.load(open(config.char2idx_dict_file, 'rb'))
    bpe2idx_dict = pickle.load(open(config.bpe2idx_dict_file, 'rb'))
    pos2idx_dict = pickle.load(open(config.pos2idx_dict_file, 'rb'))

    predict_meta = build_features(config, predict_examples, "predict",
                               config.predict_record_file, word2idx_dict, char2idx_dict,
                               bpe2idx_dict, pos2idx_dict, is_test=True)

    save(config.predict_eval_file, predict_eval, message="predict eval")
    save(config.predict_meta, predict_meta, message="predict meta")


def prepro_test_sber(config):
    if config.use_bpe and config.use_bpe_pretrained_codes:
        bpe_model = BPE(open(config.bpe_pretrained_codes_file, 'r'))
    elif config.use_bpe and not config.use_bpe_pretrained_codes:
        bpe_model = BPE(open(config.bpe_codes_file, 'r'))
    else:
        bpe_model = None

    word2idx_dict = pickle.load(open(config.word2idx_dict_file, 'rb'))
    char2idx_dict = pickle.load(open(config.char2idx_dict_file, 'rb'))
    bpe2idx_dict = pickle.load(open(config.bpe2idx_dict_file, 'rb'))
    pos2idx_dict = pickle.load(open(config.pos2idx_dict_file, 'rb'))

    for datafile, datatype in zip([config.sber_public_file, config.sber_private_file], ['public', 'private']):
        datafile_squad = os.path.join(config.target_dir, "{}.json_squad".format(datatype))
        sber2squad(datafile, outfile=datafile_squad)
        data_examples, data_eval = process_file(
            config, datafile_squad, datatype,
            remove_unicode=config.remove_unicode, bpe_model=bpe_model, is_test=True)

        data_meta = build_features(config, data_examples, datatype,
                                   os.path.join(config.target_dir, "{}.tfrecords".format(datatype)),
                                   word2idx_dict, char2idx_dict, bpe2idx_dict, pos2idx_dict, is_test=True)

        save(os.path.join(config.target_dir, "{}_eval.json".format(datatype)), data_eval, message="{} eval".format(datatype))
        save(os.path.join(config.target_dir, "{}_meta.json".format(datatype)), data_meta, message="{} meta".format(datatype))

