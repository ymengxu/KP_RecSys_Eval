import os

import implicit
import numpy as np

from core import MAIN_DIRECTORY, SEED
from utils.data_importer import DataImporter
from utils.evaluation import Evaluation
from utils.early_stopping import EarlyStopping


# os.environ['OPENBLAS_NUM_THREADS'] = '1'


class iALSRecommender:
    def __init__(self, dataset_name, data_folder_path=os.path.join(MAIN_DIRECTORY, 'data_clean')):
        DI = DataImporter(dataset_name, data_folder_path)
        train, val_user, val_ytrue, test_user, test_ytrue = DI.get_data('ials')
        self.train = train
        self.val_user = val_user
        self.val_ytrue = val_ytrue
        self.test_user = test_user
        self.test_ytrue = test_ytrue
        self.model = None
        self.params = None


    def train_model(self, factors, alpha, regularization, 
                    iterations=200, early_stopping=True, eval_step=5, consecutive_eval_threshold=5,
                    seed=SEED, verbose=False):
        """
        train a implicit.als.AlternatingLeastSquares model
        Args: 
            factors, alpha, regularization, iterations (default 200): model parameters in implicit.als.AlternatingLeastSquares
            ealy_stopping (bool): whether to apply early stopping 
            eval_step (int): used when `early_stopping` is True. 
                            compute the evaluation metric every x epochs.
                            default 5
            consecutive_eval_threshold (int): used when `early_stopping` is True. 
                                              how many consecutive validation steps with no improvement before stopping the training process
                                              default 5    
        """
        self.params = {'factors': factors,
                       'alpha': alpha,
                       'regularization': regularization, 
                       'iterations': iterations}
        
        if not early_stopping:
            # train the model at once
            model = implicit.als.AlternatingLeastSquares(
                factors=factors,
                regularization=regularization,
                iterations=iterations,
                num_threads=0,
                random_state=seed
            )
            model.fit(self.train*alpha, show_progress=verbose)
        else:
            model = implicit.als.AlternatingLeastSquares(
                factors=factors,
                regularization=regularization,
                iterations=eval_step,
                num_threads=0,
                random_state=seed
            )
            train = self.train*alpha
            steps = iterations//eval_step  # maximum evaluation times
            ES = EarlyStopping(consecutive_eval_threshold=consecutive_eval_threshold)
            for i in range(1, steps+1):
                model.fit(train, show_progress=False)
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

        self.model = model
        

    def get_recommendation(self, user_list:list, k:int):
        if self.model is None:
            print('iALSRecommender: Model not trained yet. Call train_model() first.')
            return
        
        train = self.train*self.params['alpha']
        pred_ls, _ = self.model.recommend(
            user_list,
            train[user_list],
            N=k,
            filter_already_liked_items=True
        )
        return pred_ls
    

    def get_validation_ndcg(self, k=10):
        """compute ndcg@10 for validation users"""
        if self.model is None:
            print('iALSRecommender: model not trained yet. Call train_model() first.')
            return
        
        pred_ls = self.get_recommendation(self.val_user, k=k)
        val_evaluator = Evaluation(batch_size = len(self.val_user), K=[k], how='val')
        mean_ndcg = val_evaluator.evaluate(pred_ls, self.val_ytrue)
        return mean_ndcg[0]
    
    
    def get_test_metrics(self, K):
        if self.model is None:
            print('iALSRecommender: model not trained yet. Call train_model() first.')
            return
        
        pred_ls = self.get_recommendation(self.test_user, max(K))
        test_evaluator = Evaluation(batch_size = len(self.test_user), K=K, how='test')
        mean_precision, mean_recall, HR, mean_ndcg, MRR, MAP = test_evaluator.evaluate(pred_ls, self.test_ytrue)
        return mean_precision, mean_recall, HR, mean_ndcg, MRR, MAP

