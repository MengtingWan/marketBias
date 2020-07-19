import numpy as np
import tensorflow as tf
import pandas as pd
import dataset
import os
import sys
from utils import Model


class MF(Model):

    def __init__(self, DATA_NAME, config):
        super().__init__(DATA_NAME, 'MF', config)

    def mf_block(self):

        n_user, n_item = self.n_user, self.n_item
        HIDDEN_DIM, LAMBDA = self.config['hidden_dim'], self.config['lbda']

        u = tf.placeholder(tf.int32, [None])
        i = tf.placeholder(tf.int32, [None])
        y = tf.placeholder(tf.float32, [None])

        user_emb = tf.get_variable("user_emb", [n_user, HIDDEN_DIM],
                                   initializer=tf.random_uniform_initializer(-0.01, 0.01))
        item_emb = tf.get_variable("item_emb", [n_item, HIDDEN_DIM],
                                   initializer=tf.random_uniform_initializer(-0.01, 0.01))
        u_emb = tf.nn.embedding_lookup(user_emb, u)
        i_emb = tf.nn.embedding_lookup(item_emb, i)
        s = tf.reduce_sum(tf.multiply(u_emb, i_emb), 1, keep_dims=False)
        score = tf.tensordot(u_emb, item_emb, axes=[[1], [1]])

        l2_norm = tf.add_n([tf.reduce_mean(tf.multiply(u_emb, u_emb)),
                            tf.reduce_mean(tf.multiply(i_emb, i_emb))])

        user_bias = tf.get_variable("user_bias", [n_user],
                                    initializer=tf.constant_initializer(0))
        item_bias = tf.get_variable("item_bias", [n_item],
                                    initializer=tf.constant_initializer(0))

        i_b = tf.nn.embedding_lookup(item_bias, i)
        u_b = tf.nn.embedding_lookup(user_bias, u)
        b = tf.get_variable("global_bias", [],
                            initializer=tf.constant_initializer(0))

        s += i_b + u_b + b
        score += tf.reshape(item_bias, [1, n_item])
        l2_norm += tf.add_n([tf.reduce_mean(tf.multiply(u_b, u_b)),
                             tf.reduce_mean(tf.multiply(i_b, i_b))])

        diff = s - y

        loss = tf.reduce_mean(tf.multiply(diff, diff)) + LAMBDA*l2_norm

        return [u, i, y, s], score, [loss], diff

    def next_train_batch(self, BATCH_SIZE):

        N_MAX = self.N_MAX
        N_BATCH = N_MAX//BATCH_SIZE
        index_selected = np.random.permutation(
            self.data_train.shape[0])[:N_MAX]

        for i in range(0, N_BATCH*BATCH_SIZE, BATCH_SIZE):
            current_index = index_selected[i:(i+BATCH_SIZE)]
            xu1 = self.data_train[current_index, 0].astype(np.int32)
            xi1 = self.data_train[current_index, 1].astype(np.int32)
            y1 = self.data_train[current_index, 2].astype(np.float32)
            au1 = self.user_attr[xu1].astype(np.int32)
            ai1 = self.item_attr[xi1].astype(np.int32)
            aui1 = - np.ones(au1.shape[0])
            _ind = (au1 >= 0) & (aui1 >= 0)
            aui1[_ind] = self.item_user_attr_map[au1[_ind], ai1[_ind]]
            aui1 = aui1.astype(np.int32)
            yield xu1, xi1, y1, au1, ai1, aui1

    def get_validation_batches(self, dataInput, BATCH_SIZE):

        rtn = []
        for i in range(0, dataInput.shape[0], BATCH_SIZE):
            xu1 = dataInput[i:(i+BATCH_SIZE), 0].astype(np.int32)
            xi1 = dataInput[i:(i+BATCH_SIZE), 1].astype(np.int32)
            y1 = dataInput[i:(i+BATCH_SIZE), 2].astype(np.float32)
            au1 = self.user_attr[xu1].astype(np.int32)
            ai1 = self.item_attr[xi1].astype(np.int32)
            aui1 = - np.ones(au1.shape[0])
            _ind = (au1 >= 0) & (aui1 >= 0)
            aui1[_ind] = self.item_user_attr_map[au1[_ind], ai1[_ind]]
            aui1 = aui1.astype(np.int32)
            rtn.append([xu1, xi1, y1, au1, ai1, aui1])
        return rtn

    def model_constructor(self):

        n_item_group, n_user_group = len(
            self.item_attr_ids), len(self.user_attr_ids)
        [u, i, y, s], score, [loss], diff = self.mf_block()
        ai = tf.placeholder(tf.int32, [None])
        au = tf.placeholder(tf.int32, [None])
        aui = tf.placeholder(tf.int32, [None])
        x_ai = tf.one_hot(ai, n_item_group)
        x_au = tf.one_hot(au, n_user_group)
        x_aui = tf.one_hot(aui, n_user_group)

        PROTECT_ITEM, PROTECT_USER, PROTECT_USER_ITEM = self.config['protect_item_group'], self.config[
            'protect_user_group'], self.config['protect_user_item_group']

        eps = tf.constant(1e-15)
        fstats = tf.constant(0.0)

        if PROTECT_ITEM:
            zi = tf.multiply(tf.expand_dims(diff, -1), x_ai)

            sum_i = tf.reduce_sum(zi, axis=0)
            count_i = tf.reduce_sum(x_ai, axis=0)
            mean_group_i = sum_i/(count_i + eps)

            mean_all_i = tf.reduce_sum(sum_i)/(tf.reduce_sum(count_i) + eps)

            mean_diff_i = mean_group_i - mean_all_i
            var_between_i = tf.reduce_sum(tf.multiply(
                count_i, mean_diff_i*mean_diff_i))/(n_item_group - 1)

            diff_group_i = zi - tf.expand_dims(mean_group_i, 0)
            var_within_i = tf.reduce_sum(
                tf.multiply(x_ai, diff_group_i*diff_group_i))

            f_i = var_between_i * \
                tf.nn.relu(tf.reduce_sum(count_i) - n_item_group) / \
                (var_within_i + eps)

            fstats += f_i

        if PROTECT_USER:
            zu = tf.multiply(tf.expand_dims(diff, -1), x_au)

            sum_u = tf.reduce_sum(zu, axis=0)
            count_u = tf.reduce_sum(x_au, axis=0)
            mean_group_u = sum_u/(count_u + eps)

            mean_all_u = tf.reduce_sum(sum_u)/(tf.reduce_sum(count_u) + eps)

            mean_diff_u = mean_group_u - mean_all_u
            var_between_u = tf.reduce_sum(tf.multiply(
                count_u, mean_diff_u*mean_diff_u))/(n_user_group - 1)

            diff_group_u = zu - tf.expand_dims(mean_group_u, 0)
            var_within_u = tf.reduce_sum(
                tf.multiply(x_au, diff_group_u*diff_group_u))

            f_u = var_between_u * \
                tf.nn.relu(tf.reduce_sum(count_u) - n_user_group) / \
                (var_within_u + eps)
            fstats += f_u

        if PROTECT_USER_ITEM:
            zui = tf.multiply(tf.expand_dims(diff, -1), x_aui)

            sum_ui = tf.reduce_sum(zui, axis=0)
            count_ui = tf.reduce_sum(x_aui, axis=0)
            mean_group_ui = sum_ui/(count_ui + eps)

            mean_all_ui = tf.reduce_sum(sum_ui)/(tf.reduce_sum(count_ui) + eps)

            mean_diff_ui = mean_group_ui - mean_all_ui
            var_between_ui = tf.reduce_sum(tf.multiply(
                count_ui, mean_diff_ui*mean_diff_ui))/(n_user_group*n_item_group - 1)

            diff_group_ui = zui - tf.expand_dims(mean_group_ui, 0)
            var_within_ui = tf.reduce_sum(
                tf.multiply(x_aui, diff_group_ui*diff_group_ui))

            f_ui = var_between_ui * \
                tf.nn.relu(tf.reduce_sum(count_ui) - n_user_group *
                           n_item_group)/(var_within_ui + eps)
            fstats += f_ui

        C = self.config['C']
        if C is not None:
            loss += C*fstats

        LEARNING_RATE = self.config['learning_rate']
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        return [u, i, y, au, ai, aui], score, [loss], diff, [optimizer]
