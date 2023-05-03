import os

import numpy as np
import cornac

from core import MAIN_DIRECTORY, SEED
# from core.neural_based_methods.bivae.cornac_bivaecf import BiVAECF
from utils.data_importer import DataImporter
from utils.evaluation import Evaluation
from utils.early_stopping import EarlyStopping


class BiVAERecommender:
    def __init__(self, dataset_name, data_folder_path=os.path.join(MAIN_DIRECTORY, 'data_clean')):
        DI = DataImporter(dataset_name, data_folder_path)
        train, val_user, val_ytrue, test_user, test_ytrue = DI.get_data('bivae')
        self.train = train
        self.val_user = val_user
        self.val_ytrue = val_ytrue
        self.test_user = test_user
        self.test_ytrue = test_ytrue
        self.model = None
        self.params = None
    
    
    def train_model(self, k:int, 
                    batch_size:int, learning_rate,
                    act_fn="tanh", likelihood="pois", n_epochs=600, 
                    early_stopping=False, eval_step=5, consecutive_eval_threshold=5, initial_training_epochs=100,
                    seed=SEED, use_gpu=True, verbose=False):
        """
        train a cornac.models.BiVAECF model
        Args: 
            k, batch_size, learning_rate, act_fn, likelihood, n_epochs (default 400): model params in cornac.models.BiVAECF
            ealy_stopping (bool): whether to apply early stopping 
            eval_step (int): used when `early_stopping` is True. 
                            compute the evaluation metric every x epochs.
                            default 5
            consecutive_eval_threshold (int): used when `early_stopping` is True. 
                                              how many consecutive validation steps with no improvement before stopping the training process
                                              default 5    
        """
        self.params = {'k': int(k), 
                       'encoder_structure': [2*int(k)],   # encoder structure set as 2*k
                       'batch_size': batch_size, 
                       'learning_rate': learning_rate, 
                       'act_fn': act_fn, 
                       'likelihood': likelihood, 
                       'n_epochs': int(n_epochs),
                       'seed': seed, 
                       'use_gpu': use_gpu,
                       'verbose': verbose
                       }
        if not early_stopping:
            # train the model at once
            model = cornac.models.BiVAECF(**self.params)
            model.fit(self.train)
            self.model = model

        else:
            print('BiVAERecommender: BiVAE not suitable for early stopping')
        #     self.params['n_epochs'] = eval_step
        #     model = BiVAECF(**self.params)
        #     model.fit(self.train)
            
        #     ES = EarlyStopping(consecutive_eval_threshold=consecutive_eval_threshold)
        #     # let the model train some epochs with no early stopping
        #     epoch_id = 0
        #     for i in range(initial_training_epochs//eval_step):
        #         model.train()
        #         self.model = model
        #         epoch_id += eval_step
        #         eval_score = self.get_validation_ndcg()
        #         ES.log(epoch_id, eval_score)
        #         if verbose:
        #             print('epoch {}: evaluation score = {}'.format(epoch_id, eval_score))


        #     ES.consecutive_evaluation_index = 0
        #     # apply early stopping for the following epochs
        #     for i in range((n_epochs-initial_training_epochs)//eval_step):
        #         model.train()
        #         self.model = model
        #         epoch_id += eval_step
        #         # evaluate
        #         eval_score = self.get_validation_ndcg()
        #         if verbose:
        #             print('epoch {}: evaluation score = {}'.format(epoch_id, eval_score))
                
        #         stop_flag = ES.log(epoch_id, eval_score)
        #         if stop_flag:
        #             break
        #     self.params['best_epoch'] = ES.best_epoch
        #     self.params['iterations'] = epoch_id
        #     self.params['best_score'] = ES.best_evaluation_score
     
    
    def get_recommendation(self, user_list:list, k:int):
        if self.model is None:
            print('BiVAERecommender: model not trained yet. Call train_model() first.')
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
            print('BiVAERecommender: model not trained yet. Call train_model() first.')
            return

        pred_ls = self.get_recommendation(self.val_user, k=k)
        val_evaluator = Evaluation(batch_size = len(self.val_user), K=[k], how='val')
        mean_ndcg = val_evaluator.evaluate(pred_ls, self.val_ytrue)
        return mean_ndcg[0]
    

    def get_test_metrics(self, K):
        if self.model is None:
            print('BiVAERecommender: model not trained yet. Call train_model() first.')
            return
        
        pred_ls = self.get_recommendation(self.test_user, max(K))
        test_evaluator = Evaluation(batch_size = len(self.test_user), K=K, how='test')
        mean_precision, mean_recall, HR, mean_ndcg, MRR, MAP = test_evaluator.evaluate(pred_ls, self.test_ytrue)
        return mean_precision, mean_recall, HR, mean_ndcg, MRR, MAP