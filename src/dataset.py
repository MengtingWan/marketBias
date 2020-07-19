import numpy as np
import pandas as pd
import sys
import os
from utils import DATA_DIR

class Dataset(object):

    def __init__(self, DATA_NAME):
        self.DATA_NAME = DATA_NAME
        
        print("Initializing dataset:", DATA_NAME)
        sys.stdout.flush()

        data = pd.read_csv(os.path.join(DATA_DIR, "df_"+DATA_NAME+".csv"))
        data['item_id'].loc[data['item_id'].isna()] = ''
        data['user_id'].loc[data['user_id'].isna()] = ''
        
        item_id_vals, item_ids = pd.factorize(data['item_id'].values)
        user_id_vals, user_ids = pd.factorize(data['user_id'].values)
        item_attr_vals, item_attr_ids = pd.factorize(data['model_attr'].values)
        user_attr_vals, user_attr_ids = pd.factorize(data['user_attr'].values)
        
        tmp = dict(zip(data['item_id'].values, item_attr_vals))
        self.item_attr = np.array([tmp[_i] for _i in item_ids], dtype=int)
        tmp = dict(zip(data['user_id'].values, user_attr_vals))
        self.user_attr = np.array([tmp[_i] for _i in user_ids], dtype=int)
        
        data['item_id'] = item_id_vals
        data['user_id'] = user_id_vals
        
        self.item_ids = item_ids
        self.user_ids = user_ids
        self.item_attr_ids = item_attr_ids
        self.user_attr_ids = user_attr_ids

        self.n_item = data['item_id'].max()+1
        self.n_user = data['user_id'].max()+1
        
        self.data = data[['user_id','item_id','rating','split','model_attr','user_attr']]
        
        print("Successfully initialized!")
        print(self.data.shape[0], "training records")
        print("about", self.n_user, "users and", self.n_item, "items are loaded!")
        sys.stdout.flush()
        
        
    def get_user_item_train_map(self):
        
        data = self.data
        
        user_item_train_map = (self.data.loc[(self.data['rating']>=4) & (self.data['split'] == 0)]).groupby(
            ['user_id'])['item_id'].apply(list).to_dict()
        
        return user_item_train_map

    def get_neg_samples(self, N_NEG=10):

        user_item_map = (self.data.loc[self.data['rating']>=4]).groupby(['user_id'])['item_id'].apply(list).to_dict()
        
        print("Start sampling negative examples ...")
        neg_samples = []
        count = 0
        print("current progress for", self.n_user, "users: ", end="")
        sys.stdout.flush()       
        for u in range(self.n_user):
            if count % 5000 == 0:
                print(count, end=", ")
                sys.stdout.flush()
            count += 1
            p = np.ones(self.n_item)
            if u in user_item_map:
                pos_items = np.array(user_item_map[u], dtype=int)
                p[pos_items] = 0
            p /= np.sum(p)
            neg_items = np.random.choice(self.n_item, size=N_NEG, p=p)
            neg_samples.append(neg_items)
        print("done!")
        sys.stdout.flush()
        
        return np.array(neg_samples, dtype=int)
        