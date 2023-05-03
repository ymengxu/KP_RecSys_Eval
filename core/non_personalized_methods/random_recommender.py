import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

from core import MAIN_DIRECTORY
from utils.data_importer import DataImporter
from utils.evaluation import Evaluation

class RandomRecommender:
    def __init__(self, dataset_name, 
                 data_folder_path=os.path.join(MAIN_DIRECTORY, 'data_clean')):
        """a random recommender"""
        DI = DataImporter(dataset_name, data_folder_path)
        train, _, _, test_user, test_ytrue = DI.get_data('ncf')
        self.n_items = DI.info['n']
        self.train = train
        self.test_user = test_user
        self.test_ytrue = test_ytrue
    
    
    def train_model(self):
        # nothing to train
        return
    
    
    def get_recommendation(self, user_list:list, k:int):
        """recommend random items to a list of users, excluding interacted items"""
        users, items = [], []
        item = list(range(self.n_items))
        for user in user_list:
            user = [user] * len(item) 
            users.extend(user)
            items.extend(item)

        all_predictions = pd.DataFrame(data={"user_num": users, "item_num":items})
        merged = pd.merge(self.train, all_predictions, on=["user_num", "item_num"], how="outer")
        all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

        all_predictions = shuffle(all_predictions)
        all_predictions = all_predictions.groupby('user_num').head(k)[['user_num', 'item_num']]\
                .groupby('user_num')['item_num'].apply(lambda x: list(x)).reset_index()\
                .sort_values('user_num', ascending=True)
        pred_ls = all_predictions['item_num'].tolist()

        return pred_ls
    

    def get_test_metrics(self, K):
        pred_ls = self.get_recommendation(self.test_user, max(K))
        test_evaluator = Evaluation(batch_size = len(self.test_user), K=K, how='test')
        mean_precision, mean_recall, HR, mean_ndcg, MRR, MAP = test_evaluator.evaluate(pred_ls, self.test_ytrue)
        return mean_precision, mean_recall, HR, mean_ndcg, MRR, MAP