import os
import cornac
import numpy as np

from core import MAIN_DIRECTORY
from utils.data_importer import DataImporter
from utils.evaluation import Evaluation


class PopularityRecommender:
    def __init__(self, dataset_name, 
                 data_folder_path=os.path.join(MAIN_DIRECTORY, 'data_clean')):
        """popularity recommender using cornac.models.MostPop()"""
        DI = DataImporter(dataset_name, data_folder_path)
        train, _, _, test_user, test_ytrue = DI.get_data('popularity')
        self.train = train
        self.test_user = test_user
        self.test_ytrue = test_ytrue
        self.model = None
    
    
    def train_model(self):
        most_pop = cornac.models.MostPop()
        most_pop.fit(self.train)
        self.model = most_pop
    
    
    def get_recommendation(self, user_list:list, k:int):
        """recommend popular items to a list of users, excluding interacted items"""
        if self.model is None:
            print('PopularityRecommender: model not trained yet. Call train_model() first.')
            return
        
        pred_ls = []
        train_mat = self.train.csr_matrix
        for u in user_list:
            interacted_items = list(np.nonzero(train_mat[u])[1])
            res = self.model.score(u)
            res = np.argsort(res)[::-1]
            y_pred = [i for i in res[:k+len(interacted_items)] if i not in interacted_items][:k]
            pred_ls.append(list(y_pred))
        return pred_ls
    

    def get_test_metrics(self, K):
        if self.model is None:
            print('PopularityRecommender: model not trained yet. Call train_model() first.')
            return
        
        pred_ls = self.get_recommendation(self.test_user, max(K))
        test_evaluator = Evaluation(batch_size = len(self.test_user), K=K, how='test')
        mean_precision, mean_recall, HR, mean_ndcg, MRR, MAP = test_evaluator.evaluate(pred_ls, self.test_ytrue)
        return mean_precision, mean_recall, HR, mean_ndcg, MRR, MAP