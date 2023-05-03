from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.models.deeprec.deeprec_utils import prepare_hparams
from recommenders.utils.timer import Timer

import os

import pandas as pd

from core import MAIN_DIRECTORY, SEED
from utils.data_importer import DataImporter
from utils.evaluation import Evaluation
from utils.early_stopping import EarlyStopping


class LightGCNRecommender:
    def __init__(self, dataset_name, data_folder_path=os.path.join(MAIN_DIRECTORY, 'data_clean')):
        DI = DataImporter(dataset_name, data_folder_path)
        train, val_user, val_ytrue, test_user, test_ytrue = DI.get_data('lightgcn')
        self.train = train
        self.val_user = val_user
        self.val_ytrue = val_ytrue
        self.test_user = test_user
        self.test_ytrue = test_ytrue
        self.model = None
        self.params = None
        
    
    def train_model(self, embed_size, n_layers, batch_size, decay, learning_rate, epochs=300, 
                   early_stopping=False, eval_step=5, consecutive_eval_threshold=5, 
                   seed=SEED, verbose=True, 
                    savemodel_dir=os.path.join(MAIN_DIRECTORY, 'savemodel', 'lightgcn')):
        """
        train a recommenders.models.deeprec.models.graphrec.lightgcn.LightGCN model
        
        Args: 
            embed_size, n_layers, batch_size, decay, learning_rate, epochs (default 300): model params
            early_stopping (bool): whether to apply early stopping 
            eval_step (int): used when `early_stopping` is True. 
                            compute the evaluation metric every x epochs.
                            default 5
            consecutive_eval_threshold (int): used when `early_stopping` is True. 
                                              how many consecutive validation steps with no improvement before stopping the training process
                                              default 5    
        """
        self.params = {'embed_size': int(embed_size), 
                        'n_layers': n_layers,
                        # train
                        'batch_size': batch_size, 
                        'decay': decay, 
                        'learning_rate': learning_rate, 
                        'epochs': epochs,
                        'eval_epoch': eval_step
                      }
        if not early_stopping:
            hparams = prepare_hparams(**self.params, 
                                      model_type='lightgcn', 
                                        top_k=10, metrics=['ndcg'],
                                        save_model=False, save_epoch=100,
                                      MODEL_DIR=savemodel_dir
                                    )
            model = LightGCN(hparams, self.train, seed=seed)
            model.fit()
            self.model = model
            
        else:
            self.params['epochs'] = eval_step
            self.params['eval_epoch'] = -1
            hparams = prepare_hparams(**self.params, 
                                      model_type='lightgcn', 
                                        top_k=10, metrics=['ndcg'],
                                        save_model=False, save_epoch=100,
                                      MODEL_DIR=savemodel_dir
                                    )
            model = LightGCN(hparams, self.train, seed=seed)
            steps = epochs//eval_step   # maximum evaluation times
            ES = EarlyStopping(consecutive_eval_threshold=consecutive_eval_threshold)
            for i in range(1, steps+1):
                model.fit()
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
                
                
    def get_recommendation(self, user_list, k):
        if self.model is None:
            print('LightGCNRecommender: Model not trained yet. Call train_model() first.')
            return
        
        test = pd.DataFrame(user_list, columns=['userID'])
        topk_scores = self.model.recommend_k_items(test, top_k=k, remove_seen=True)
        topk_rec = topk_scores.groupby('userID')['itemID'].apply(lambda x: list(x)).reset_index()
        pred_ls = topk_rec['itemID'].tolist()
        
        return pred_ls
    
    
    def get_validation_ndcg(self, k=10):
        pred_ls = self.get_recommendation(self.val_user, k=k)
        val_evaluator = Evaluation(batch_size = len(self.val_user), K=[k], how='val')
        mean_ndcg = val_evaluator.evaluate(pred_ls, self.val_ytrue)
        return mean_ndcg[0]
        
        
    def get_test_metrics(self, K):
        pred_ls = self.get_recommendation(self.test_user, max(K))
        test_evaluator = Evaluation(batch_size = len(self.test_user), K=K, how='test')
        mean_precision, mean_recall, HR, mean_ndcg, MRR, MAP = test_evaluator.evaluate(pred_ls, self.test_ytrue)
        return mean_precision, mean_recall, HR, mean_ndcg, MRR, MAP
