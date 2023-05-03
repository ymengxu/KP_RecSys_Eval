import os
import cornac
import numpy as np

from core import MAIN_DIRECTORY
from utils.data_importer import DataImporter
from utils.evaluation import Evaluation


class KNNRecommender:
    def __init__(self, dataset_name, 
                 data_folder_path=os.path.join(MAIN_DIRECTORY, 'data_clean')):
        """nearest neighbor based recommender using cornac.models.UserKNN() and cornac.models.ItemKNN()"""
        DI = DataImporter(dataset_name, data_folder_path)
        train, val_user, val_ytrue, test_user, test_ytrue = DI.get_data('knn')
        self.train = train
        self.val_user = val_user
        self.val_ytrue = val_ytrue
        self.test_user = test_user
        self.test_ytrue = test_ytrue
        self.model = None


    def train_model(self, base, metric, n_neighbors):
        assert base in ['user', 'item'], 'KNNRecommender: need to specify user or item-based'
        assert metric in ['cosine', 'pearson'], 'KNNRecommender: only support cosine or pearson similarity metric'
        # TODO: cornac.models.UserKNN also allows additional parameters:
        # 1. amplify: Amplifying the influence on similarity weights.
        # 2. weighting: The option for re-weighting the rating matrix. Supported types: ['idf', 'bm25'].
        
        if base=='user':
            if metric=='cosine':
                model = cornac.models.UserKNN(k=n_neighbors, similarity="cosine", name="UserKNN-Cosine", verbose=False)
            if metric=='pearson':
                model = cornac.models.UserKNN(k=n_neighbors, similarity="pearson", name="UserKNN-Pearson", verbose=False)
        if base=='item':
            if metric=='cosine':
                model = cornac.models.ItemKNN(k=n_neighbors, similarity="cosine", name="ItemKNN-Cosine", verbose=False)
            if metric=='pearson':
                model = cornac.models.ItemKNN(k=n_neighbors, similarity="pearson", name="ItemKNN-Pearson", verbose=False)
            
        model.fit(self.train)
        self.model = model
        self.params = {'base': base, 'metric': metric, 'n_neighbors': n_neighbors}


    def get_recommendation(self, user_list:list, k:int):
        if self.model is None:
            print('KNNRecommender: model not trained yet. Call train_model() first.')
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
    

    def get_validation_ndcg(self, k=10):
        """compute ndcg@10 for validation users"""
        if self.model is None:
            print('KNNRecommender: model not trained yet. Call train_model() first.')
            return

        pred_ls = self.get_recommendation(self.val_user, k=k)
        val_evaluator = Evaluation(batch_size = len(self.val_user), K=[k], how='val')
        mean_ndcg = val_evaluator.evaluate(pred_ls, self.val_ytrue)
        return mean_ndcg[0]
    

    def get_test_metrics(self, K):
        if self.model is None:
            print('KNNRecommender: model not trained yet. Call train_model() first.')
            return
        
        pred_ls = self.get_recommendation(self.test_user, max(K))
        test_evaluator = Evaluation(batch_size = len(self.test_user), K=K, how='test')
        mean_precision, mean_recall, HR, mean_ndcg, MRR, MAP = test_evaluator.evaluate(pred_ls, self.test_ytrue)
        return mean_precision, mean_recall, HR, mean_ndcg, MRR, MAP
