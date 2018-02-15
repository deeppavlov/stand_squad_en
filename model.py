import tensorflow as tf
from func import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net


class Model(object):
    def __init__(self, config, batch=None, word_mat=None,
                 char_mat=None, bpe_mat=None, pos_mat=None, trainable=True, use_tfdata=True):
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)

        N, PL, QL, CL, BL = config.batch_size, config.test_para_limit, config.test_ques_limit, config.char_limit, config.bpe_limit

        if not use_tfdata:
            self.c_ph = tf.placeholder(shape=(N, PL), dtype=tf.int32, name='c_ph')
            self.q_ph = tf.placeholder(shape=(N, QL), dtype=tf.int32, name='q_ph')
            self.ch_ph = tf.placeholder(shape=(N, PL, CL), dtype=tf.int32, name='c_ch_ph')
            self.qh_ph = tf.placeholder(shape=(N, QL, CL), dtype=tf.int32, name='q_ch_ph')
            self.cb_ph = tf.placeholder(shape=(N, PL, BL), dtype=tf.int32, name='c_bpe_ph')
            self.qb_ph = tf.placeholder(shape=(N, QL, BL), dtype=tf.int32, name='q_bpe_ph')
            self.cp_ph = tf.placeholder(shape=(N, PL), dtype=tf.int32, name='c_pos_ph')
            self.qp_ph = tf.placeholder(shape=(N, QL), dtype=tf.int32, name='q_pos_ph')
            self.y1_ph = tf.placeholder(shape=(N, PL), dtype=tf.int32, name='y1_ph')
            self.y2_ph = tf.placeholder(shape=(N, PL), dtype=tf.int32, name='y2_ph')
            self.qa_id = tf.placeholder(shape=(N,), dtype=tf.int64, name='qa_id_ph')
        else:
            self.c_ph, self.q_ph, self.ch_ph, self.qh_ph, self.cb_ph, self.qb_ph, \
                self.cp_ph, self.qp_ph, self.y1_ph, self.y2_ph, self.qa_id = batch.get_next()

        self.is_train = tf.get_variable(
            "is_train", shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
            word_mat, dtype=tf.float32), trainable=False)
        if config.use_char:
            self.char_mat = tf.get_variable(
                "char_mat", initializer=tf.constant(char_mat, dtype=tf.float32),
                trainable=config.train_pretrained_char)
        if config.use_bpe:
            self.bpe_mat = tf.get_variable(
                "bpe_mat", initializer=tf.constant(bpe_mat, dtype=tf.float32),
                trainable=config.train_pretrained_bpe_emb)
        if config.use_pos:
            self.pos_mat = tf.get_variable("pos_mat", initializer=tf.constant(
                pos_mat, dtype=tf.float32), trainable=True)

        self.c_mask = tf.cast(self.c_ph, tf.bool)
        self.q_mask = tf.cast(self.q_ph, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

        self.c_maxlen = tf.reduce_max(self.c_len)
        self.q_maxlen = tf.reduce_max(self.q_len)
        self.c = tf.slice(self.c_ph, [0, 0], [N, self.c_maxlen])
        self.q = tf.slice(self.q_ph, [0, 0], [N, self.q_maxlen])
        self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
        self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
        if config.use_char:
            self.ch = tf.slice(self.ch_ph, [0, 0, 0], [N, self.c_maxlen, CL])
            self.qh = tf.slice(self.qh_ph, [0, 0, 0], [N, self.q_maxlen, CL])
        if config.use_bpe:
            self.cb = tf.slice(self.cb_ph, [0, 0, 0], [N, self.c_maxlen, BL])
            self.qb = tf.slice(self.qb_ph, [0, 0, 0], [N, self.q_maxlen, BL])
        if config.use_pos:
            self.cp = tf.slice(self.cp_ph, [0, 0], [N, self.c_maxlen])
            self.qp = tf.slice(self.qp_ph, [0, 0], [N, self.q_maxlen])
        self.y1 = tf.slice(self.y1_ph, [0, 0], [N, self.c_maxlen])
        self.y2 = tf.slice(self.y2_ph, [0, 0], [N, self.c_maxlen])

        if config.use_char:
            self.ch_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
            self.qh_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])
        if config.use_bpe:
            self.cb_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.cb, tf.bool), tf.int32), axis=2), [-1])
            self.qb_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.qb, tf.bool), tf.int32), axis=2), [-1])

        self.ready()

        if trainable:
            self.lr = tf.get_variable(
                "lr", shape=[], dtype=tf.float32, trainable=False)
            self.opt = tf.train.AdadeltaOptimizer(
                learning_rate=self.lr, epsilon=1e-6)
            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)

            capped_grads, _ = tf.clip_by_global_norm(gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)
            
            grads_without_freezed = [(grad, var) for (grad, var) in grads if 'char_mat' not in var.name and 'bpe_mat' not in var.name]
            gradients_f, variables_f = zip(*grads_without_freezed)
            capped_grads_f, _ = tf.clip_by_global_norm(gradients_f, config.grad_clip)
            self.train_op_f = self.opt.apply_gradients(
                zip(capped_grads_f, variables_f), global_step=self.global_step)

    def ready(self):
        config = self.config
        N, PL, QL, CL, BL, d, dc, dg, dbpe, dbpeh = config.batch_size, self.c_maxlen, self.q_maxlen, \
                                                   config.char_limit, config.bpe_limit, config.hidden, \
                                                   config.glove_dim if config.pretrained_char else config.char_dim, config.char_hidden, \
                                                   config.bpe_glove_dim if config.pretrained_bpe_emb else config.bpe_dim, config.bpe_hidden
        gru = cudnn_gru if config.use_cudnn else native_gru

        with tf.variable_scope("emb"):
            if config.use_char:
                with tf.variable_scope("char"):
                    ch_emb = tf.reshape(tf.nn.embedding_lookup(
                        self.char_mat, self.ch), [N * PL, CL, dc])
                    qh_emb = tf.reshape(tf.nn.embedding_lookup(
                        self.char_mat, self.qh), [N * QL, CL, dc])
                    ch_emb = dropout(
                        ch_emb, keep_prob=config.keep_prob, is_train=self.is_train)
                    qh_emb = dropout(
                        qh_emb, keep_prob=config.keep_prob, is_train=self.is_train)
                    cell_fw = tf.contrib.rnn.GRUCell(dg)
                    cell_bw = tf.contrib.rnn.GRUCell(dg)
                    _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, ch_emb, self.ch_len, dtype=tf.float32)
                    ch_emb = tf.concat([state_fw, state_bw], axis=1)
                    _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, qh_emb, self.qh_len, dtype=tf.float32)
                    qh_emb = tf.concat([state_fw, state_bw], axis=1)
                    qh_emb = tf.reshape(qh_emb, [N, QL, 2 * dg])
                    ch_emb = tf.reshape(ch_emb, [N, PL, 2 * dg])

            if config.use_bpe:
                with tf.variable_scope("bpe"):
                    cb_emb = tf.reshape(tf.nn.embedding_lookup(
                        self.bpe_mat, self.cb), [N * PL, BL, dbpe])
                    qb_emb = tf.reshape(tf.nn.embedding_lookup(
                        self.bpe_mat, self.qb), [N * QL, BL, dbpe])
                    cb_emb = dropout(
                        cb_emb, keep_prob=config.keep_prob, is_train=self.is_train)
                    qb_emb = dropout(
                        qb_emb, keep_prob=config.keep_prob, is_train=self.is_train)
                    cell_fw = tf.contrib.rnn.GRUCell(dbpeh)
                    cell_bw = tf.contrib.rnn.GRUCell(dbpeh)
                    _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, cb_emb, self.cb_len, dtype=tf.float32)
                    cb_emb = tf.concat([state_fw, state_bw], axis=1)
                    _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, qb_emb, self.qb_len, dtype=tf.float32)
                    qb_emb = tf.concat([state_fw, state_bw], axis=1)
                    qb_emb = tf.reshape(qb_emb, [N, QL, 2 * dbpeh])
                    cb_emb = tf.reshape(cb_emb, [N, PL, 2 * dbpeh])

            with tf.name_scope("word"):
                c_emb = tf.nn.embedding_lookup(self.word_mat, self.c)
                q_emb = tf.nn.embedding_lookup(self.word_mat, self.q)

            if config.use_char:
                c_emb = tf.concat([c_emb, ch_emb], axis=2)
                q_emb = tf.concat([q_emb, qh_emb], axis=2)

            if config.use_bpe:
                c_emb = tf.concat([c_emb, cb_emb], axis=2)
                q_emb = tf.concat([q_emb, qb_emb], axis=2)

            if config.use_pos:
                cp_emb = tf.nn.embedding_lookup(self.pos_mat, self.cp)
                qp_emb = tf.nn.embedding_lookup(self.pos_mat, self.qp)
                c_emb = tf.concat([c_emb, cp_emb], axis=2)
                q_emb = tf.concat([q_emb, qp_emb], axis=2)

        with tf.variable_scope("encoding"):
            rnn = gru(num_layers=3, num_units=d, batch_size=N, input_size=c_emb.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            c = rnn(c_emb, seq_len=self.c_len)
            q = rnn(q_emb, seq_len=self.q_len)

        with tf.variable_scope("attention"):
            qc_att = dot_attention(c, q, mask=self.q_mask, hidden=d,
                                   keep_prob=config.keep_prob, is_train=self.is_train)
            rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=qc_att.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            att = rnn(qc_att, seq_len=self.c_len)

        with tf.variable_scope("match"):
            self_att = dot_attention(
                att, att, mask=self.c_mask, hidden=d, keep_prob=config.keep_prob, is_train=self.is_train)
            rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=self_att.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            match = rnn(self_att, seq_len=self.c_len)

        with tf.variable_scope("pointer"):
            init = summ(q[:, :, -2 * d:], d, mask=self.q_mask,
                        keep_prob=config.ptr_keep_prob, is_train=self.is_train)
            pointer = ptr_net(batch=N, hidden=init.get_shape().as_list(
            )[-1], keep_prob=config.ptr_keep_prob, is_train=self.is_train)
            logits1, logits2 = pointer(init, match, d, self.c_mask)

        with tf.variable_scope("predict"):
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            outer = tf.matrix_band_part(outer, 0, 15)
            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits1, labels=self.y1)
            losses2 = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits2, labels=self.y2)
            self.loss = tf.reduce_mean(losses + losses2)

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step
