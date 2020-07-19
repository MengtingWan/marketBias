import numpy as np
import tensorflow as tf
import pandas as pd
import os
import sys

import argparse
import dataset
from utils import OUTPUT_DIR
from model import MF


def run_mf(DATA_NAME, METHOD_NAME, dim_set, lbda_set, lr_set, C_set,
           protect_item_group=None, protect_user_group=None,  protect_user_item_group=None, pos_thr=4):

    myData = dataset.Dataset(DATA_NAME)
    batch_size = 512
    param_set = [(_d, _l, _lr, _C)
                 for _d in dim_set for _l in lbda_set for _lr in lr_set for _C in C_set]

    if METHOD_NAME in ['MF']:

        user_item_train_map = myData.get_user_item_train_map()

        for (hidden_dim, lbda, learning_rate, C) in param_set:
            config = {'hidden_dim': hidden_dim, 'lbda': lbda,
                      'learning_rate': learning_rate, 'batch_size': batch_size,
                      'C': C,
                      'protect_item_group': protect_item_group,
                      'protect_user_group': protect_user_group,
                      'protect_user_item_group': protect_user_item_group}
            configStr = "_".join([str(config[k]) for k in sorted(
                list(config.keys())) if config[k] is not None])
            outputStr = os.path.join(
                OUTPUT_DIR, DATA_NAME+"_"+METHOD_NAME+"_"+configStr)

            myModel = MF(DATA_NAME, config)

            columns = ['user_id', 'item_id', 'rating']
            myModel.assign_data(myData.n_user, myData.n_item,
                                myData.user_attr, myData.item_attr,
                                myData.user_attr_ids, myData.item_attr_ids,
                                myData.data[columns].loc[myData.data['split'] == 0].values.astype(
                                    int),
                                myData.data[columns].loc[myData.data['split'] == 1].values.astype(int))

            myModel.train()

            columns = ['user_id', 'item_id',
                       'rating', 'model_attr', 'user_attr']
            _res = myModel.evaluate_rating(myData.data[columns].loc[myData.data['split'] == 1],
                                           myData.data[columns].loc[myData.data['split'] == 2])

            pd.DataFrame(_res,
                         index=['validation', 'test'],
                         columns=['MSE', 'MAE', 'F-stat', 'p-value']).to_csv(outputStr+"_rating_results.csv")

            myModel.assign_user_item_train_map(user_item_train_map)
            _res = myModel.evaluate_ranking(myData.data[columns].loc[(myData.data['split'] == 1) & (myData.data['rating'] >= pos_thr)],
                                            myData.data[columns].loc[(myData.data['split'] == 2) & (myData.data['rating'] >= pos_thr)])
            pd.DataFrame(_res,
                         index=['validation', 'test'],
                         columns=['NDCG', 'AUC', 'KL']).to_csv(outputStr+"_ranking_results.csv")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        help="specify a dataset from [modcloth, electronics]")
    parser.add_argument('--method',
                        help="specify a training method from [MF]")
    parser.add_argument('--hidden_dim', default=10, type=int)
    #parser.add_argument('--lbda', default=0.1, type=float)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    #parser.add_argument('--C', default=0, type=float)
    parser.add_argument('--protect_item_group', default=1, type=int)
    parser.add_argument('--protect_user_group', default=1, type=int)
    parser.add_argument('--protect_user_item_group', default=1, type=int)

    args = parser.parse_args()

    kappa = [args.protect_item_group, args.protect_user_group, args.protect_user_item_group]
    lbda_set = [0.01, 0.1, 1.0, 10]
    C_set = [0.5, 1.0, 5.0, 10.0]

    # vanilla MF
    if sum(kappa) == 0:
        run_mf(DATA_NAME=args.dataset, METHOD_NAME=args.method,
            dim_set=[args.hidden_dim],
            lbda_set=lbda_set,
            lr_set=[args.learning_rate],
            C_set=[0],
            protect_item_group=kappa[0],
            protect_user_group=kappa[1],
            protect_user_item_group=kappa[2])
    # MF with correlation losses
    else:
        run_mf(DATA_NAME=args.dataset, METHOD_NAME=args.method,
            dim_set=[args.hidden_dim],
            lbda_set=lbda_set,
            lr_set=[args.learning_rate],
            C_set=C_set,
            protect_item_group=kappa[0],
            protect_user_group=kappa[1],
            protect_user_item_group=kappa[2])