import os

import numpy as np
import pandas as pd
import scipy.sparse as sps

from lightfm import LightFM

from core import MAIN_DIRECTORY, SEED
from utils.data_importer import DataImporter
from utils.evaluation import Evaluation
from utils.early_stopping import EarlyStopping


# os.environ['OPENBLAS_NUM_THREADS'] = '1'


class LightFMRecommender:
    def __init__(self, dataset_name, data_folder_path=os.path.join(MAIN_DIRECTORY, 'data_clean'), use_text_feature=True, use_no_feature=False,
                        use_only_text=False):
        DI = DataImporter(dataset_name, data_folder_path)
        train, val_user, val_ytrue, test_user, test_ytrue, user_feature, item_feature = DI.get_data('lightfm', 
                                                        use_text_feature, use_no_feature, use_only_text)
        self.train = train
        self.val_user = val_user
        self.val_ytrue = val_ytrue
        self.test_user = test_user
        self.test_ytrue = test_ytrue
        self.user_feature = user_feature
        self.item_feature = item_feature
        self.model = None
        self.params = None


    def train_model(self, loss, no_components, learning_rate, item_alpha, user_alpha, 
                   epochs=600, early_stopping=True, eval_step=5, consecutive_eval_threshold=5,
                    seed=SEED, verbose=False):
        self.params = {
            'loss':loss,
            'no_components':int(no_components),
            'learning_rate':learning_rate,
            'item_alpha':item_alpha,
            'user_alpha':user_alpha
        }
        model = LightFM(
                **self.params,
                random_state=seed
            )
        if not early_stopping:
            model.fit(
                self.train,
                epochs=epochs,
                user_features=self.user_feature, 
                item_features=self.item_feature,
                num_threads=8,
                verbose=verbose
            )
            self.model = model
            self.params['iterations'] = epochs
        else:
            steps = epochs//eval_step  # maximum evaluation times
            ES = EarlyStopping(consecutive_eval_threshold=consecutive_eval_threshold)
            for i in range(1,steps+1):
                model.fit_partial(
                    self.train,
                    epochs=eval_step,
                    user_features=self.user_feature, 
                    item_features=self.item_feature,
                    num_threads=8,
                    verbose=False
                )
                self.model = model
                epoch_id = eval_step*i
                # evaluate
                eval_score = self.get_validation_ndcg()
                if verbose:
                    print('epoch {}: evaluation score = {}'.format(epoch_id, eval_score))
                
                stop_flag = ES.log(epoch_id, eval_score)
                if stop_flag:
                    break
            self.params['best_epoch'] = ES.best_epoch
            self.params['iterations'] = epoch_id


    def get_recommendation(self, users:list, k:int):
        if self.model is None:
            print('LightFMRecommender: Model not trained yet. Call train_model() first.')
            return

        # prediction score for each user-item pair
        b_u, q_u = self.model.get_user_representations(features = self.user_feature)
        b_i, p_i = self.model.get_item_representations(features = self.item_feature)
        b_u = b_u[users]
        q_u = q_u[users]
        scores = np.matmul(q_u, p_i.transpose()) + b_u.reshape((len(b_u), 1)) + b_i.reshape((1,len(b_i)))

        # top-k rec list for each user in users
        pred_ls = []
        for i in range(len(users)):
            score = pd.Series(scores[i])
            score = score.sort_values(ascending=False)

            # exclude interacted items
            interacted_items = self.train[users[i]].todense().flatten()
            interacted_items = np.nonzero(interacted_items)[1]
            if len(interacted_items) == 0:
                y_pred = score.index.tolist()[:k]
            else:
                y_pred = [item_num for item_num in score.index.tolist()[:k+len(interacted_items)]\
                          if item_num not in interacted_items][:k]
            pred_ls.append(y_pred)

        return pred_ls


    def get_validation_ndcg(self, k=10):
        """compute ndcg@10 for validation users"""
        pred_ls = self.get_recommendation(self.val_user, k=k)
        val_evaluator = Evaluation(batch_size = len(self.val_user), K=[k], how='val')
        mean_ndcg = val_evaluator.evaluate(pred_ls, self.val_ytrue)
        return mean_ndcg[0]
    
    
    def get_test_metrics(self, K):
        pred_ls = self.get_recommendation(self.test_user, max(K))
        test_evaluator = Evaluation(batch_size = len(self.test_user), K=K, how='test')
        mean_precision, mean_recall, HR, mean_ndcg, MRR, MAP = test_evaluator.evaluate(pred_ls, self.test_ytrue)
        return mean_precision, mean_recall, HR, mean_ndcg, MRR, MAP
