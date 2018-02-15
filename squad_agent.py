import numpy as np
import copy
import traceback
import json
import os
import pickle

import requests
from tqdm import tqdm
import tensorflow as tf

from model import Model
from util import convert_tokens
from subword_nmt.apply_bpe import BPE
from prepro import build_features_notfdata
from squad_agent_utils import sber2squad, squad_answer2sber, process_file
from config_dgx1_char_pretrained_train_full_eng import flags


class SquadAgent:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.model_config = flags.FLAGS
        self.kpi_name = self.config['kpi_name']
        self.model = None
        self.bpe_model = None
        self.tf_session = None
        self.model_dicts = None
        self.session_id = None
        self.numtasks = None
        self.tasks = None
        self.observations = None
        self.agent_params = None
        self.predictions = None
        self.answers = None
        self.score = None
        self.response_code = None

        sber_tasks_template_file = self.config['kpis'][self.kpi_name]['settings_agent']['sber_tasks_template_file']
        with open(sber_tasks_template_file, 'r') as template_json:
            template = json.load(template_json)

        self.tasks_template = template

    def init_agent(self):
        config = flags.FLAGS

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

        self.model = model
        self.bpe_model = bpe_model
        self.tf_session = sess
        self.model_dicts = {'word2idx_dict': word2idx_dict,
                                        'char2idx_dict': char2idx_dict,
                                        'bpe2idx_dict': bpe2idx_dict,
                                        'pos2idx_dict': pos2idx_dict}

    def _set_numtasks(self, numtasks):
        self.numtasks = numtasks

    def _get_tasks(self):
        get_url = self.config['kpis'][self.kpi_name]['settings_kpi']['rest_url']
        if self.numtasks in [None, 0]:
            test_tasks_number = self.config['kpis'][self.kpi_name]['settings_kpi']['test_tasks_number']
        else:
            test_tasks_number = self.numtasks
        get_params = {'stage': 'task_public', 'quantity': test_tasks_number}
        get_response = requests.get(get_url, params=get_params)
        tasks = json.loads(get_response.text)
        return tasks

    def _make_observations(self, tasks, human_input=False):
        if human_input:
            task = self.tasks_template
            task['paragraphs'][0]['context'] = tasks[0]
            task['paragraphs'][0]['qas'][0]['question'] = tasks[1]
            observations = sber2squad(task)
        else:
            observations = sber2squad(tasks)
        return observations

    def _get_predictions(self, observations):
        datatype = 'public'

        word2idx_dict = self.model_dicts['word2idx_dict']
        char2idx_dict = self.model_dicts['char2idx_dict']
        bpe2idx_dict = self.model_dicts['bpe2idx_dict']
        pos2idx_dict = self.model_dicts['pos2idx_dict']

        data_examples, data_eval = process_file(
            self.model_config, observations, datatype,
            remove_unicode=self.model_config.remove_unicode, bpe_model=self.bpe_model, is_test=True)

        data_features, data_meta = build_features_notfdata(self.model_config, data_examples, datatype,
                                                           word2idx_dict, char2idx_dict, bpe2idx_dict, pos2idx_dict,
                                                           is_test=True)

        total = data_meta["total"]

        answer_dict = {}
        remapped_dict = {}

        print(len(data_features))
        # hotfix добить длину data_examples до делителя config.batch_size
        while len(data_features) % self.model_config.batch_size != 0:
            data_features.append(data_features[-1])

        print(len(data_features))

        for step in tqdm(range(total // self.model_config.batch_size + 1)):

            def get_batch():
                batch_items = data_features[step * self.model_config.batch_size:(step + 1) * self.model_config.batch_size]
                batch = dict()
                for key in batch_items[0].keys():
                    batch[key] = np.stack([el[key] for el in batch_items])
                return batch

            batch = get_batch()

            qa_id, loss, yp1, yp2 = self.tf_session.run(
                [self.model.qa_id, self.model.loss, self.model.yp1, self.model.yp2], feed_dict={
                    self.model.c_ph: batch['context_idxs'],
                    self.model.q_ph: batch['ques_idxs'],
                    self.model.ch_ph: batch['context_char_idxs'],
                    self.model.qh_ph: batch['ques_char_idxs'],
                    self.model.cb_ph: batch['context_bpe_idxs'],
                    self.model.qb_ph: batch['ques_bpe_idxs'],
                    self.model.cp_ph: batch['context_pos_idxs'],
                    self.model.qp_ph: batch['ques_pos_idxs'],
                    self.model.y1_ph: batch['y1'],
                    self.model.y2_ph: batch['y2'],
                    self.model.qa_id: batch['id'],
                })

            answer_dict_, remapped_dict_ = convert_tokens(
                data_eval, qa_id.tolist(), yp1.tolist(), yp2.tolist())
            answer_dict.update(answer_dict_)
            remapped_dict.update(remapped_dict_)

        return remapped_dict

    def _make_answers(self, observations, predictions, human_input=False):
        if human_input:
            return predictions['dummy']
        else:
            tasks = copy.deepcopy(self.tasks)
            return squad_answer2sber(tasks, predictions)

    def _get_score(self, answers):
        post_headers = {'Accept': '*/*'}
        rest_response = requests.post(self.config['kpis'][self.kpi_name]['settings_kpi']['rest_url'],
                                      json=answers,
                                      headers=post_headers)
        return {'text': rest_response.text, 'status_code': rest_response.status_code}

    def _run_test(self):
        tasks = self._get_tasks()
        session_id = tasks['id']
        numtasks = tasks['total']
        self.tasks = tasks
        self.session_id = session_id
        self.numtasks = numtasks

        observations = self._make_observations(tasks)
        self.observations = observations

        predictions = self._get_predictions(observations)
        self.predictions = predictions

        answers = self._make_answers(observations, predictions)
        self.answers = answers

        score_response = self._get_score(answers)
        self.score = score_response['text']
        self.response_code = score_response['status_code']

    def _run_score(self, observation):
        observations = self._make_observations(observation, human_input=True)
        self.observations = observations
        predictions = self._get_predictions(observations)
        self.predictions = predictions
        answers = self._make_answers(observations, predictions, human_input=True)
        self.answers = answers

    def answer(self, input_task):
        # try:
            if isinstance(input_task, list):
                print("%s human input mode..." % self.kpi_name)
                self._run_score(input_task)
                result = copy.deepcopy(self.answers)
                print("%s action result:  %s" % (self.kpi_name, result))
                return result
            elif isinstance(input_task, int):
                print("%s API mode..." % self.kpi_name)
                self._set_numtasks(input_task)
                self._run_test()
                print("%s score: %s" % (self.kpi_name, self.score))
                result = copy.deepcopy(self.tasks)
                result.update(copy.deepcopy(self.answers))
                return result
            else:
                return {"ERROR": "{}".format(traceback.extract_stack())}
