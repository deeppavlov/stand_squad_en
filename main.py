import tensorflow as tf
import ujson as json
import numpy as np
from tqdm import tqdm
import os
import requests
import json
import pickle

from model import Model
from util import get_record_parser, convert_tokens, evaluate, get_batch_dataset, get_dataset, squad_answer2sber, sber2squad
from prepro import prepro_predict, prepro_test_sber, process_file, build_features_notfdata
from subword_nmt.apply_bpe import BPE


def train(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.bpe_emb_file, "r") as fh:
        bpe_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.pos_emb_file, "r") as fh:
        pos_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)

    dev_total = meta["total"]

    print("Building model...")
    parser = get_record_parser(config)
    train_dataset = get_batch_dataset(config.train_record_file, parser, config)
    dev_dataset = get_dataset(config.dev_record_file, parser, config)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    train_iterator = train_dataset.make_one_shot_iterator()
    dev_iterator = dev_dataset.make_one_shot_iterator()

    model = Model(config, iterator, word_mat, char_mat, bpe_mat, pos_mat)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    loss_save = 100.0
    patience = 0
    lr = config.init_lr
    min_lr = config.min_lr

    with tf.Session(config=sess_config) as sess:
        writer = tf.summary.FileWriter(config.log_dir)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=None)
        train_handle = sess.run(train_iterator.string_handle())
        dev_handle = sess.run(dev_iterator.string_handle())
        sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
        sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))

        for _ in tqdm(range(1, config.num_steps + 1)):
            global_step = sess.run(model.global_step) + 1
            if global_step < config.freeze_steps:
                loss, train_op = sess.run([model.loss, model.train_op_f], feed_dict={
                    handle: train_handle})
            else:
                if global_step == config.freeze_steps:
                    print('Unfreezing embedding matrices')
                loss, train_op = sess.run([model.loss, model.train_op], feed_dict={
                                      handle: train_handle})

            if global_step % config.period == 0:
                loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/loss", simple_value=loss), ])
                lr_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/lr", simple_value=lr), ])
                writer.add_summary(loss_sum, global_step)
                writer.add_summary(lr_sum, global_step)
            if global_step % config.checkpoint == 0:
                sess.run(tf.assign(model.is_train,
                                   tf.constant(False, dtype=tf.bool)))
                _, summ = evaluate_batch(
                    model, config.val_num_batches, train_eval_file, sess, "train", handle, train_handle)
                for s in summ:
                    writer.add_summary(s, global_step)

                metrics, summ = evaluate_batch(
                    model, dev_total // config.batch_size + 1, dev_eval_file, sess, "dev", handle, dev_handle)
                sess.run(tf.assign(model.is_train,
                                   tf.constant(True, dtype=tf.bool)))

                dev_loss = metrics["loss"]
                if dev_loss < loss_save:
                    loss_save = dev_loss
                    patience = 0
                else:
                    patience += 1
                if patience >= config.patience and lr > min_lr:
                    lr /= 2.0
                    loss_save = dev_loss
                    patience = 0
                sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))
                for s in summ:
                    writer.add_summary(s, global_step)
                writer.flush()
                filename = os.path.join(
                    config.save_dir, "model_{}.ckpt".format(global_step))
                saver.save(sess, filename)


def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle):
    answer_dict = {}
    losses = []
    for _ in tqdm(range(1, num_batches + 1)):
        qa_id, loss, yp1, yp2, = sess.run(
            [model.qa_id, model.loss, model.yp1, model.yp2], feed_dict={handle: str_handle})
        answer_dict_, _ = convert_tokens(
            eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
        answer_dict.update(answer_dict_)
        losses.append(loss)
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    metrics["loss"] = loss
    loss_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    f1_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ])
    em_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/em".format(data_type), simple_value=metrics["exact_match"]), ])
    return metrics, [loss_sum, f1_sum, em_sum]


def test(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.bpe_emb_file, "r") as fh:
        bpe_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.pos_emb_file, "r") as fh:
        pos_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.test_eval_file, "r") as fh:
        eval_file = json.load(fh)
    with open(config.test_meta, "r") as fh:
        meta = json.load(fh)

    total = meta["total"]

    print("Loading model...")
    test_batch = get_dataset(config.test_record_file, get_record_parser(
        config, is_test=True), config).make_one_shot_iterator()

    model = Model(config, test_batch, word_mat, char_mat, bpe_mat, pos_mat, trainable=False)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # TODO: add restoring from best model or from model name
        saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
        sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
        losses = []
        answer_dict = {}
        remapped_dict = {}
        for step in tqdm(range(total // config.batch_size + 1)):
            qa_id, loss, yp1, yp2 = sess.run(
                [model.qa_id, model.loss, model.yp1, model.yp2])
            answer_dict_, remapped_dict_ = convert_tokens(
                eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
            answer_dict.update(answer_dict_)
            remapped_dict.update(remapped_dict_)
            losses.append(loss)
        loss = np.mean(losses)
        metrics = evaluate(eval_file, answer_dict)
        with open(config.answer_file, "w") as fh:
            json.dump(remapped_dict, fh)
        print("Exact Match: {}, F1: {}".format(
            metrics['exact_match'], metrics['f1']))


def predict(config):

    prepro_predict(config)

    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.bpe_emb_file, "r") as fh:
        bpe_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.pos_emb_file, "r") as fh:
        pos_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.predict_eval_file, "r") as fh:
        predict_eval_file = json.load(fh)
    with open(config.predict_meta, "r") as fh:
        meta = json.load(fh)

    total = meta["total"]

    print("Loading model...")
    test_batch = get_dataset(config.predict_record_file, get_record_parser(
        config, is_test=True), config).make_one_shot_iterator()

    model = Model(config, test_batch, word_mat, char_mat,
                  bpe_mat, pos_mat, trainable=False)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # TODO: add restoring from best model or from model name
        saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
        print('Restoring from: {}'.format(tf.train.latest_checkpoint(config.save_dir)))
        sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
        answer_dict = {}
        remapped_dict = {}
        for step in tqdm(range(total // config.batch_size + 1)):
            qa_id, loss, yp1, yp2 = sess.run(
                [model.qa_id, model.loss, model.yp1, model.yp2])
            answer_dict_, remapped_dict_ = convert_tokens(
                predict_eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
            answer_dict.update(answer_dict_)
            remapped_dict.update(remapped_dict_)

        path_to_save_answer = config.predict_file + '_ans'
        with open(path_to_save_answer, "w") as fh:
            json.dump(remapped_dict, fh)

        print("Answer dumped: {}".format(path_to_save_answer))


def test_sber(config):

    prepro_test_sber(config)

    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.bpe_emb_file, "r") as fh:
        bpe_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.pos_emb_file, "r") as fh:
        pos_mat = np.array(json.load(fh), dtype=np.float32)

    for datafile, datatype in zip([config.sber_public_file, config.sber_private_file], ['public', 'private']):

        with open(os.path.join(config.target_dir, "{}_eval.json".format(datatype)), "r") as fh:
            data_eval_file = json.load(fh)
        with open(os.path.join(config.target_dir, "{}_meta.json".format(datatype)), "r") as fh:
            meta = json.load(fh)

        total = meta["total"]

        print("Loading model...")
        test_batch = get_dataset(os.path.join(config.target_dir, "{}.tfrecords".format(datatype)), get_record_parser(
            config, is_test=True), config).make_one_shot_iterator()

        model = Model(config, test_batch, word_mat, char_mat,
                      bpe_mat, pos_mat, trainable=False)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            if config.model_name == 'latest':
                checkpoint = tf.train.latest_checkpoint(config.save_dir)
            else:
                checkpoint = os.path.join(config.save_dir, config.model_name)
            print('Restoring from: {}'.format(checkpoint))
            saver.restore(sess, checkpoint)
            sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
            answer_dict = {}
            remapped_dict = {}
            for step in tqdm(range(total // config.batch_size + 1)):
                qa_id, loss, yp1, yp2 = sess.run(
                    [model.qa_id, model.loss, model.yp1, model.yp2])
                answer_dict_, remapped_dict_ = convert_tokens(
                    data_eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
                answer_dict.update(answer_dict_)
                remapped_dict.update(remapped_dict_)

            path_to_save_answer = os.path.join(config.answer_dir, '{}.json_squad_ans'.format(datatype))
            with open(path_to_save_answer, "w") as fh:
                json.dump(remapped_dict, fh)

            sber_ans = '.'.join(path_to_save_answer.split('.')[0:-1]) + '.json_ans'
            squad_answer2sber(datafile, path_to_save_answer, outfile=sber_ans)

            print("Answer dumped: {}".format(path_to_save_answer))

        tf.reset_default_graph()

    # evaluating
    url = 'http://api.aibotbench.com/rusquad/qas'
    headers = {'Content-Type': 'application/json', 'Accept': 'text/plain'}
    metrics = dict()
    f1, em = 0.0, 0.0
    for datatype in ['public', 'private']:
        sber_ans = open(os.path.join(config.answer_dir, '{}.json_ans'.format(datatype)), 'r').readline()
        res = requests.post(url, data=sber_ans, headers=headers)
        metrics[datatype] = eval(json.loads(res.text))
        f1 += metrics[datatype]['f1']
        em += metrics[datatype]['exact_match']
        print('{}: EM: {:.5f} F-1: {:.5f}'.format(datatype, metrics[datatype]['exact_match'], metrics[datatype]['f1']))
    print('EM avg: {:.5f} F-1 avg: {:.5f}'.format(em/2, f1/2))


def test_sber_onfly(config):
    print('Loading emb matrices')
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.bpe_emb_file, "r") as fh:
        bpe_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.pos_emb_file, "r") as fh:
        pos_mat = np.array(json.load(fh), dtype=np.float32)

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

    print("Loading model...")
    model = Model(config, None, word_mat, char_mat,
                  bpe_mat, pos_mat, trainable=False, use_tfdata=False)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if config.model_name == 'latest':
        checkpoint = tf.train.latest_checkpoint(config.save_dir)
    else:
        checkpoint = os.path.join(config.save_dir, config.model_name)
    print('Restoring from: {}'.format(checkpoint))
    saver.restore(sess, checkpoint)
    sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))

    for datafile, datatype in zip([config.sber_public_file, config.sber_private_file], ['public', 'private']):

        datafile_squad = os.path.join(config.target_dir, "{}.json_squad".format(datatype))
        sber2squad(datafile, outfile=datafile_squad)
        data_examples, data_eval = process_file(
            config, datafile_squad, datatype,
            remove_unicode=config.remove_unicode, bpe_model=bpe_model, is_test=True)

        data_features, data_meta = build_features_notfdata(config, data_examples, datatype,
                                   word2idx_dict, char2idx_dict, bpe2idx_dict, pos2idx_dict, is_test=True)

        total = data_meta["total"]

        answer_dict = {}
        remapped_dict = {}

        print(len(data_features))
        # hotfix добить длину data_examples до делителя config.batch_size
        while len(data_features) % config.batch_size != 0:
            data_features.append(data_features[-1])

        print(len(data_features))

        for step in tqdm(range(total // config.batch_size + 1)):

            def get_batch():
                batch_items = data_features[step * config.batch_size:(step + 1) * config.batch_size]
                batch = dict()
                for key in batch_items[0].keys():
                    batch[key] = np.stack([el[key] for el in batch_items])
                return batch

            batch = get_batch()

            qa_id, loss, yp1, yp2 = sess.run(
                [model.qa_id, model.loss, model.yp1, model.yp2], feed_dict={
                    model.c_ph: batch['context_idxs'],
                    model.q_ph: batch['ques_idxs'],
                    model.ch_ph: batch['context_char_idxs'],
                    model.qh_ph: batch['ques_char_idxs'],
                    model.cb_ph: batch['context_bpe_idxs'],
                    model.qb_ph: batch['ques_bpe_idxs'],
                    model.cp_ph: batch['context_pos_idxs'],
                    model.qp_ph: batch['ques_pos_idxs'],
                    model.y1_ph: batch['y1'],
                    model.y2_ph: batch['y2'],
                    model.qa_id: batch['id'],
                })

            answer_dict_, remapped_dict_ = convert_tokens(
                data_eval, qa_id.tolist(), yp1.tolist(), yp2.tolist())
            answer_dict.update(answer_dict_)
            remapped_dict.update(remapped_dict_)

        path_to_save_answer = os.path.join(config.answer_dir, '{}.json_squad_ans'.format(datatype))
        with open(path_to_save_answer, "w") as fh:
            json.dump(remapped_dict, fh)

        sber_ans = '.'.join(path_to_save_answer.split('.')[0:-1]) + '.json_ans'
        squad_answer2sber(datafile, path_to_save_answer, outfile=sber_ans)

        print("Answer dumped: {}".format(path_to_save_answer))

    # evaluating
    # TODO: CHANGE TO ENG URL
    url = 'http://api.aibotbench.com/rusquad/qas'
    headers = {'Content-Type': 'application/json', 'Accept': 'text/plain'}
    metrics = dict()
    f1, em = 0.0, 0.0
    for datatype in ['public', 'private']:
        sber_ans = open(os.path.join(config.answer_dir, '{}.json_ans'.format(datatype)), 'r').readline()
        res = requests.post(url, data=sber_ans, headers=headers)
        metrics[datatype] = eval(json.loads(res.text))
        f1 += metrics[datatype]['f1']
        em += metrics[datatype]['exact_match']
        print('{}: EM: {:.5f} F-1: {:.5f}'.format(datatype, metrics[datatype]['exact_match'], metrics[datatype]['f1']))
    print('EM avg: {:.5f} F-1 avg: {:.5f}'.format(em/2, f1/2))

