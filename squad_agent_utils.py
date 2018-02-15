import random

from tqdm import tqdm
from nltk import word_tokenize

from prepro import preprocess_string, get_pos, convert_idx


MIN_CONTEXT_LENGTH = 70


def sber2squad(sber_tasks):
    squad_tasks = {'version': 1.1, 'data': [{'title': 'SberChallenge', 'paragraphs': []}]}
    for paragraph in sber_tasks['paragraphs']:
        squad_tasks['data'][0]['paragraphs'].append(paragraph)

        # In case of to short context - fill it with dummy symbols
        # context = paragraph['context']
        # if len(context) < MIN_CONTEXT_LENGTH:
        #     paragraph['context'] = '{} . {}'.format(context, '@' * (MIN_CONTEXT_LENGTH - len(context)))

    return squad_tasks


def squad_answer2sber(sber_tasks, squad_predict):
    sber_answers = sber_tasks
    sber_answers['answers'] = squad_predict
    return sber_answers


def process_file(config, squad_data, data_type, word_counter=None, char_counter=None, bpe_counter=None, pos_counter=None,
                 remove_unicode=True, bpe_model=None, pos_model=None, is_test=False):
    print("Generating {} examples...".format(data_type))
    para_limit = config.test_para_limit if is_test else config.para_limit
    ques_limit = config.test_ques_limit if is_test else config.ques_limit
    examples = []
    eval_examples = {}
    total = 0

    source = squad_data
    for article in source["data"]:
        for para in tqdm(article["paragraphs"]):
            context_raw = para['context']
            context, r2p, p2r = preprocess_string(para["context"], unicode_mapping=True, remove_unicode=remove_unicode)
            context_tokens = [token.replace("''", '"').replace("``", '"') for token in
                              word_tokenize(context)[:para_limit]]
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
