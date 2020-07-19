import numpy as np
import tensorflow as tf
import pandas as pd
import sys
import dataset
import os
import heapq
import statsmodels.api as sm
from statsmodels.formula.api import ols

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_ROOT)

DATA_DIR = os.path.join(BASE_DIR, "data/")
MODEL_DIR = os.path.join(BASE_DIR, "model/")
OUTPUT_DIR = os.path.join(BASE_DIR, "result/")

for DIR in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    if not os.path.exists(DIR):
        os.makedirs(DIR)


class Model(object):

    def __init__(self, DATA_NAME, METHOD_NAME, config):
        self.DATA_NAME = DATA_NAME
        self.config = config

        config_string = "_".join([str(config[k]) for k in sorted(
            list(config.keys())) if config[k] is not None])

        self.MODEL_NAME = "_".join(
            [DATA_NAME, METHOD_NAME]) + "_" + config_string

    def assign_data(self, n_user, n_item,
                    user_attr, item_attr,
                    user_attr_ids, item_attr_ids,
                    data_train, data_validation,
                    N_MAX=500000):
        self.n_user = n_user
        self.n_item = n_item
        self.item_attr = item_attr
        self.user_attr = user_attr
        self.item_attr_ids = item_attr_ids
        self.user_attr_ids = user_attr_ids
        self.item_user_attr_map = np.arange(len(item_attr_ids)*len(user_attr_ids)).reshape(
            len(item_attr_ids), len(user_attr_ids))
        self.data_train = data_train
        self.data_validation = data_validation
        self.N_MAX = min(data_train.shape[0], N_MAX)

    def assign_neg_samples(self, neg_samples):
        self.neg_samples = neg_samples
        self.N_NEG = neg_samples.shape[1]

    def assign_user_item_train_map(self, user_item_train_map):
        self.user_item_train_map = user_item_train_map

    def next_train_batch(self, BATCH_SIZE):
        pass

    def get_validation_batches(self, dataInput, BATCH_SIZE):
        pass

    def model_constructor(self):
        pass

    def train(self):
        BATCH_SIZE = self.config['batch_size']
        MODEL_NAME = self.MODEL_NAME

        EPOCHS = 1000
        max_noprogress = 10

        batches_validation = self.get_validation_batches(
            self.data_validation, BATCH_SIZE)

        print("start training "+MODEL_NAME+" ...")
        sys.stdout.flush()
        config = tf.ConfigProto()
        with tf.Graph().as_default(), tf.Session(config=config) as session:

            variables, scores, losses, errors, optimizers = self.model_constructor()
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            _loss_train_min = 1e10
            _loss_vali_min = 1e10
            _loss_vali_old = 1e10
            n_noprogress = 0

            for epoch in range(1, EPOCHS):
                _count, _count_sample = 0, 0
                _loss_train = [0 for _ in range(len(losses))]

                print("epoch: ", epoch)
                print("=== current batch: ", end="")
                for _vars in self.next_train_batch(BATCH_SIZE):
                    feed = dict(zip(variables, _vars))

                    _loss_batch, _ = session.run([losses, optimizers],
                                                 feed_dict=feed)
                    for _i, _l in enumerate(_loss_batch):
                        _loss_train[_i] += _l*_vars[0].shape[0]

                    _count += 1.0
                    _count_sample += _vars[0].shape[0]
                    if _count % 500 == 0:
                        print(int(_count), end=", ")
                        sys.stdout.flush()
                print("complete!")

                for _i in range(len(_loss_train)):
                    _loss_train[_i] /= _count_sample

                if _loss_train[0] < _loss_train_min:
                    _loss_train_min = _loss_train[0]

                print("=== training: primary loss: {:.4f}, min loss: {:.4f}".format(
                    _loss_train[0], _loss_train_min), end=";  ")
                for _i, _l in enumerate(_loss_train[1:]):
                    print(" aux_loss"+str(_i)+": {:.4f}".format(_l), end="")
                print("")

                _count, _count_sample = 0, 0
                _loss_vali = [0 for _ in range(len(losses))]
                for _vars in batches_validation:
                    feed = dict(zip(variables, _vars))

                    _loss_batch = session.run(losses, feed_dict=feed)
                    for _i, _l in enumerate(_loss_batch):
                        _loss_vali[_i] += _l*_vars[0].shape[0]
                    _count += 1
                    _count_sample += _vars[0].shape[0]

                for _i in range(len(_loss_vali)):
                    _loss_vali[_i] /= _count_sample

                if _loss_vali[0] <= _loss_vali_min:
                    _loss_vali_min = _loss_vali[0]
                    n_noprogress = 0
                    saver.save(session, os.path.join(
                        MODEL_DIR, self.MODEL_NAME + ".model.ckpt"))

                if (_loss_vali[0] > _loss_vali_old) or (_loss_train[_i] > _loss_train_min):
                    n_noprogress += 1
                _loss_vali_old = _loss_vali[0]

                print("=== validation: primary loss: {:.4f}, min loss: {:.4f}".format(
                    _loss_vali[0], _loss_vali_min), end=";  ")
                for _i, _l in enumerate(_loss_vali[1:]):
                    print(" aux_loss"+str(_i)+": {:.4f}".format(_l), end="")
                print("")

                print("=== #no progress: ", n_noprogress)
                sys.stdout.flush()

                if n_noprogress > max_noprogress:
                    break
            saver.restore(session, os.path.join(
                MODEL_DIR, self.MODEL_NAME + ".model.ckpt"))
        print("done!")
        sys.stdout.flush()

    def batch(self, dataInput, BATCH_SIZE):
        res = []
        for i in range(0, dataInput.shape[0], BATCH_SIZE):
            res.append(dataInput[i:(i+BATCH_SIZE), :])
        return res

    # make sure the first 3 vars are user, item, rating
    def evaluate_rating(self, df_validation, df_test):
        BATCH_SIZE = self.config['batch_size']

        res = []
        df_mse, df_mae, df_count = None, None, None

        with tf.Graph().as_default(), tf.Session() as session:
            variables, scores, losses, errors, optimizers = self.model_constructor()
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(session, os.path.join(
                MODEL_DIR, self.MODEL_NAME + ".model.ckpt"))

            print('evaluating rating ...')

            print('evaluating validation set ...')
            sys.stdout.flush()
            batches_vali = self.batch(
                df_validation[['user_id', 'item_id', 'rating']].values, BATCH_SIZE)

            _errors = np.array([])
            for _vars in batches_vali:
                feed = {variables[0]: _vars[:, 0].astype(np.int32),
                        variables[1]: _vars[:, 1].astype(np.int32),
                        variables[2]: _vars[:, 2].astype(np.float32)}
                _errors_batch = session.run(errors, feed_dict=feed)
                _errors = np.append(_errors, _errors_batch)

            df_validation['error'] = _errors
            f, p = sm.stats.anova_lm(
                ols('error ~ model_attr*user_attr - model_attr - user_attr', data=df_validation).fit()).values[0, -2:]

            res.append([np.mean(_errors*_errors),
                        np.mean(np.absolute(_errors)), f, p])

            print('evaluating test set ...')
            sys.stdout.flush()
            batches_test = self.batch(
                df_test[['user_id', 'item_id', 'rating']].values, BATCH_SIZE)

            _errors = np.array([])
            for _vars in batches_test:
                feed = {variables[0]: _vars[:, 0].astype(np.int32),
                        variables[1]: _vars[:, 1].astype(np.int32),
                        variables[2]: _vars[:, 2].astype(np.float32)}
                _errors_batch = session.run(errors, feed_dict=feed)
                _errors = np.append(_errors, _errors_batch)

            df_test['error'] = _errors
            f, p = sm.stats.anova_lm(
                ols('error ~ model_attr*user_attr - model_attr - user_attr', data=df_test).fit()).values[0, -2:]

            res.append([np.mean(_errors*_errors),
                        np.mean(np.absolute(_errors)), f, p])

        res = np.array(res)

        print("Results on validation data: ", end='')
        print(", ".join([str(np.round(_m, 3)) for _m in res[0, :]]))
        print("Results on test data: ", end='')
        print(", ".join([str(np.round(_m, 3)) for _m in res[1, :]]))

        sys.stdout.flush()

        return res

    def evaluate_ranking(self, df_validation, df_test, topK=10):
        BATCH_SIZE = self.config['batch_size']
        n_item = self.n_item
        item_attr = self.item_attr
        user_attr = self.user_attr
        item_attr_ids = self.item_attr_ids
        user_attr_ids = self.user_attr_ids
        user_item_train_map = self.user_item_train_map

        res = []
        df_ndcg, df_count = None, None

        with tf.Graph().as_default(), tf.Session() as session:
            variables, scores, losses, errors, optimizers = self.model_constructor()
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(session, os.path.join(
                MODEL_DIR, self.MODEL_NAME + ".model.ckpt"))

            print('evaluating ranking ...')

            print('evaluating validation set ...')
            sys.stdout.flush()
            batches_vali = self.batch(
                df_validation[['user_id', 'item_id']].values, BATCH_SIZE)

            _metric = []
            _recommended_distribution = []
            all_index = set(np.arange(n_item))
            for _vars in batches_vali:
                feed = {variables[0]: _vars[:, 0].astype(np.int32),
                        variables[1]: _vars[:, 1].astype(np.int32)}
                _scores_batch = session.run(scores, feed_dict=feed)
                for _k, _u in enumerate(_vars[:, 0]):
                    _u = int(_u)
                    _i = int(_vars[_k, 1])
                    if _u in user_item_train_map:
                        valid_items = np.array(
                            list((all_index - set(user_item_train_map[_u]))), dtype=int)
                    else:
                        valid_items = np.array(list(all_index), dtype=int)
                    _s = _scores_batch[_k, :]
                    tmp = heapq.nlargest(
                        topK, zip(valid_items, _s), key=lambda e: e[1])
                    _topK_items, _topK_scores = list(zip(*tmp))
                    _dist = np.zeros(len(item_attr_ids))
                    for _ti in _topK_items:
                        _dist[item_attr[_ti]] += 1
                    _recommended_distribution.append(_dist)
                    _pos_i = np.sum(_s[_i] <= _s[valid_items])
                    _metric.append(
                        [1.0/np.log2(_pos_i+1), 1 - _pos_i/len(valid_items)])
            _metric = np.array(_metric, dtype=float)

            _recommended_distribution = np.array(_recommended_distribution)
            for _ai in range(len(item_attr_ids)):
                df_validation[item_attr_ids[_ai]
                              ] = _recommended_distribution[:, _ai]
            df_recommend = df_validation.groupby(
                'user_attr')[item_attr_ids].sum()
            df_count = df_validation.groupby(['user_attr', 'model_attr'])[
                'rating'].size().unstack()
            df_recommend = df_recommend.loc[df_count.index, df_count.columns]

            p, q = df_count.values.flatten(), df_recommend.values.flatten()
            p1 = p/p.sum()
            q1 = q/q.sum()
            KL = (p1*np.log(p1/q1)).sum()
            res.append(np.append(np.mean(_metric, axis=0), KL))

            print('evaluating test set ...')
            sys.stdout.flush()
            batches_test = self.batch(
                df_test[['user_id', 'item_id']].values, BATCH_SIZE)

            _metric = []
            _recommended_distribution = []
            all_index = set(np.arange(n_item))
            for _vars in batches_test:
                feed = {variables[0]: _vars[:, 0].astype(np.int32),
                        variables[1]: _vars[:, 1].astype(np.int32)}
                _scores_batch = session.run(scores, feed_dict=feed)
                for _k, _u in enumerate(_vars[:, 0]):
                    _u = int(_u)
                    _i = int(_vars[_k, 1])
                    if _u in user_item_train_map:
                        valid_items = np.array(
                            list((all_index - set(user_item_train_map[_u]))), dtype=int)
                    else:
                        valid_items = np.array(list(all_index), dtype=int)
                    _s = _scores_batch[_k, :]
                    tmp = heapq.nlargest(
                        topK, zip(valid_items, _s), key=lambda e: e[1])
                    _topK_items, _topK_scores = list(zip(*tmp))
                    _dist = np.zeros(len(item_attr_ids))
                    for _ti in _topK_items:
                        _dist[item_attr[_ti]] += 1
                    _recommended_distribution.append(_dist)
                    _pos_i = np.sum(_s[_i] <= _s[valid_items])
                    _metric.append(
                        [1.0/np.log2(_pos_i+1), 1 - _pos_i/len(valid_items)])
            _metric = np.array(_metric, dtype=float)

            _recommended_distribution = np.array(_recommended_distribution)
            for _ai in range(len(item_attr_ids)):
                df_test[item_attr_ids[_ai]] = _recommended_distribution[:, _ai]
            df_recommend = df_test.groupby(
                'user_attr')[item_attr_ids].sum().transpose()
            df_count = df_test.groupby(['model_attr', 'user_attr'])[
                'rating'].size().unstack()
            df_recommend = df_recommend.loc[df_count.index, df_count.columns]

            p, q = df_count.values.flatten(), df_recommend.values.flatten()
            p1 = p/p.sum()
            q1 = q/q.sum()
            KL = (p1*np.log(p1/q1)).sum()
            res.append(np.append(np.mean(_metric, axis=0), KL))

        res = np.array(res)

        print("Results on validation data: ", end='')
        print(", ".join([str(np.round(_m, 3)) for _m in res[0, :]]))
        print("Results on test data: ", end='')
        print(", ".join([str(np.round(_m, 3)) for _m in res[1, :]]))

        sys.stdout.flush()

        return res
