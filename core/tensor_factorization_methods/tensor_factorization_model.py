import os
import sys
from time import time

import pandas as pd
import numpy as np
from numpy import mat
import random
import math
from core import MAIN_DIRECTORY
from core.tensor_factorization_methods.data_utils import d
from utils.data_importer import DataImporter
from utils.early_stopping import EarlyStopping
from utils.evaluation import Evaluation


class TensorRecommender:
    def __init__(self, dataset_name, 
                 data_folder_path=os.path.join(MAIN_DIRECTORY, 'data_clean'), 
                 use_text_feature=True, use_no_feature=False, use_only_text=False):
        DI = DataImporter(dataset_name, data_folder_path)
        train_dict, val_user, val_ytrue, test_user, test_ytrue, item_feature = DI.get_data('tensor', 
                                                                                           use_text_feature=use_text_feature,
                                                                                           use_no_feature=use_no_feature, 
                                                                                           use_only_text=use_only_text)
        self.train_data = train_dict['train_data']
        self.train_aux = train_dict['train_aux']
        self.train_record_aux = train_dict['train_record_aux']
        self.train_time_aux = train_dict['train_time_aux']
        self.val_user = val_user
        self.val_ytrue = val_ytrue
        self.test_user = test_user
        self.test_ytrue = test_ytrue
        self.item_feature = item_feature
        self.n_user = DI.info['m']
        self.n_item = DI.info['n']
        self.n_time = len(self.train_time_aux)
        self.n_item_feature = item_feature.shape[1]
        self.param_size_thre = 10**10
        self.model = None
        self.params = None


    def _init_params(self):
        """initialize model parameters"""
        user_item_k = self.params['user_item_k']
        item_time_k = self.params['item_time_k']

        # U: user-item: user latent feature matrix (n_user x user_item_k)
        U = np.array([np.array([(random.random() / math.sqrt(user_item_k)) for j in range(user_item_k)]) for i in range(self.n_user)])
        # Vu: user-item: item latent feature matrix (n_item x user_item_k)
        Vu = np.array([np.array([(random.random() / math.sqrt(user_item_k)) for j in range(user_item_k)]) for i in range(self.n_item)])
        # Vt: time-item: item latent feature matrix (n_item x (item_time_k)
        Vt = np.array([np.array([(random.random() / math.sqrt(item_time_k)) for j in range(item_time_k)]) for i in range(self.n_item)])
        # T: time-item: item latent feature matrix (n_time x (item_time_k)
        T = np.array([np.array([(random.random() / math.sqrt(item_time_k)) for j in range(item_time_k)]) for i in range(self.n_time)])
        # M: item feature coefficient matrix for user-item (n_user x n_item_feature)
        M = np.array([np.array([(random.random() / math.sqrt(self.n_item_feature)) for j in range(self.n_item_feature)]) for i in range(self.n_user)])
        # N: item feature coefficient matrix for time-item (n_time x n_item_feature)
        N = np.array([np.array([(random.random() / math.sqrt(self.n_item_feature)) for j in range(self.n_item_feature)]) for i in range(self.n_time)])

        self.model = {
            'M': M, 
            'N': N,
            'U': U,
            'Vu': Vu,
            'Vt': Vt,
            'T': T
        }


    def _train_epoch(self, batches_split):
        """train one epoch, update learning rate"""
        M = self.model['M']
        N = self.model['N']
        U = self.model['U']
        Vu = self.model['Vu']
        Vt = self.model['Vt']
        T = self.model['T']
        F = self.item_feature

        user_item_k = self.params['user_item_k'] 
        item_time_k = self.params['item_time_k']
        lambda_c = self.params['lambda_c']
        lambda_r = self.params['lambda_r'] 
        n_neg = self.params['n_neg'] 
        lr = self.params['current_lr']
        
        # iterate all train samples in one epoch
        for i in range(0, len(batches_split) - 1): 
            if abs(U.sum()) < self.param_size_thre:
                # initialize dU and dC to record the gradient
                dU = np.zeros((self.n_user, user_item_k))
                dVu = np.zeros((self.n_item, user_item_k))
                dVt = np.zeros((self.n_item, item_time_k))
                dT = np.zeros((self.n_time, item_time_k))
                dM = np.zeros((self.n_user, self.n_item_feature))
                dN = np.zeros((self.n_time, self.n_item_feature))

                # for each training sample in the batch
                for re in range(batches_split[i], batches_split[i + 1]):   
                    # train sample: [u, i, r]
                    p = self.train_data[re][0]
                    qi = self.train_data[re][1]
                    r = self.train_data[re][2]

                    UV = np.dot(U[p], Vu[qi])
                    VT = np.dot(Vt[qi], T[r])
                    MDF = np.dot(M[p], F[qi])
                    NEF = np.dot(N[r], F[qi])

                    Bi = UV + MDF
                    Ci = VT + NEF
                    Ai = Bi * Ci

                    num = 0
                    # choose sample_rate negative items, and calculate the gradient
                    while num < n_neg:
                        qj = int(random.uniform(0, self.n_item))
                        if (not qj in self.train_aux[p][0]) and (not qj in self.train_time_aux[r][1]):
                            num += 1   # counter
                            UV = np.dot(U[p], Vu[qj])
                            VT = np.dot(Vt[qj], T[r])
                            MDF = np.dot(M[p], F[qj])
                            NEF = np.dot(N[r], F[qj])

                            Bj = UV + MDF
                            Cj = VT + NEF
                            Aj = Bj * Cj

                            Bij = Bi - Bj
                            Cij = Ci - Cj
                            Aij = Ai - Aj

                            dU[p] += d(Aij) * (Ci * Vu[qi] - Cj * Vu[qj]) + lambda_c * d(Bij) * (Vu[qi] - Vu[qj])
                            dVu[qi] += d(Aij) * Ci * U[p] + lambda_c * d(Bij) * U[p]
                            dVu[qj] -= d(Aij) * Cj * U[p] + lambda_c * d(Bij) * U[p]
                            dM[p] += d(Aij) * (Ci * F[qi] - Cj * F[qj]) + lambda_c * d(Bij) * (F[qi] - F[qj])
                            dVt[qi] += d(Aij) * Bi * T[r] + lambda_c * d(Cij) * T[r]
                            dVt[qj] -= d(Aij) * Bj * T[r] + lambda_c * d(Cij) * T[r]
                            dT[r] += d(Aij) * (Bi * Vt[qi] - Bj * Vt[qj]) + lambda_c * d(Cij) * (Vt[qi] - Vt[qj])
                            dN[r] += d(Aij) * (Bi * F[qi] - Bj * F[qj]) + lambda_c * d(Cij) * (F[qi] - F[qj])

                # update the matrices
                U += lr * (dU - lambda_r * U)   # lambda_r*U: gradient of the regularization term
                Vu += lr * (dVu - lambda_r * Vu)
                Vt += lr * (dVt - lambda_r * Vt)
                T += lr * (dT - lambda_r * T)
                M += lr * (dM - lambda_r * M)
                N += lr * (dN - lambda_r * N)
            else:
                sys.exit('TensorRecommender.train: model parameter too large, diverge')
        
        # update learning rate
        lr = lr * 0.99
        self.params['current_lr'] = lr

        # update model
        self.model = {
            'M': M, 
            'N': N,
            'U': U,
            'Vu': Vu,
            'Vt': Vt,
            'T': T
        }

    def train_model(self, 
                    user_item_k, item_time_k,  
                    batch_size, lambda_c, lambda_r, n_neg, lr, 
                    n_epochs = 200, 
                    early_stopping=True, eval_step=5, consecutive_eval_threshold=5,
                    verbose=False):
        """
        train the tensor decompostion model
        Args:
            user_item_k (int): user-item latent feature dimension
            item_time_k (int): item-time latent feature dimension
            batch_size (int): batch size for training
            lambda_c: weighting parameter for coupled matrices
            lambda_r: regularization coefficient
            n_neg: number of negative items to sample for each positive training sample
            lr: learning rate
            n_epochs (int): default 200, training epochs
            early_stopping (bool): default True, whether to apply early stopping
            eval_step (int): used when `early_stopping` is True. 
                            compute the evaluation metric every x epochs.
                            default 5
            consecutive_eval_threshold (int): used when `early_stopping` is True. 
                                              how many consecutive validation steps with no improvement before stopping the training process
                                              default 5    
            verbose (bool): default True, whether to print training log
        """
        self.params = {
            'user_item_k': int(user_item_k),
            'item_time_k': int(item_time_k),
            'batch_size': int(batch_size),
            'lambda_c': lambda_c, 
            'lambda_r': lambda_r,
            'n_neg': int(n_neg),
            'lr': lr,
            'current_lr': lr,  # decrease lr during training
            'n_epochs': n_epochs,
            'verbose': verbose
        }
        self._init_params()
        # the number of train samples
        n_train = len(self.train_data)
        # split the train samples with a step of batch_size
        batches_split = range(0, n_train, batch_size)

        if not early_stopping:
            # train all epochs at once
            for epoch_id in range(1, n_epochs+1):
                t0 = time()
                self._train_epoch(batches_split)
                t1 = time()
                if verbose:
                    print(f'epoch {epoch_id} [{round(t1-t0,2)}s] finished.')
        else:
            ES = EarlyStopping(consecutive_eval_threshold=consecutive_eval_threshold)
            for epoch_id in range(1, n_epochs+1):
                t0 = time()
                self._train_epoch(batches_split)
                t1 = time()
                if epoch_id%eval_step==0:
                    # evaluate
                    eval_score = self.get_validation_ndcg()
                    t2 = time()
                    if verbose:
                        print(f'epoch {epoch_id} [{round(t1-t0,2)}s]: evaluation score = {eval_score} [{round(t2-t1,2)}s]')
                    
                    stop_flag = ES.log(epoch_id, eval_score)
                    if stop_flag:
                        break
                elif verbose:
                    print(f'epoch {epoch_id} [{round(t1-t0,2)}s] finished.')
            self.params['best_epoch'] = ES.best_epoch
            self.params['iterations'] = epoch_id
            self.params['best_eval_score'] = ES.best_evaluation_score

    

    def get_recommendation(self, user_list, k):
        if self.model is None:
            print('TensorRecommender: model not trained yet. Call train_model() first.')
            return
        
        # test the effectiveness
        U = mat(self.model['U'])
        Vu = mat(self.model['Vu'])
        Vt = mat(self.model['Vt'])
        T = mat(self.model['T'])
        M = mat(self.model['M'])
        N = mat(self.model['N'])
        F = mat(self.item_feature)

        # get the item recommendation at the last time slice
        r = self.n_time-1
        UV = np.array(U[user_list] * Vu.T + M[user_list] * F.T)
        VT = np.array(T[r] * Vt.T + N[r] * F.T)
        score = (UV * VT).tolist()

        all_items = list(range(self.n_item))
        users = []
        items = []
        preds = []
        for i in range(len(user_list)):
            users.extend([user_list[i]]*self.n_item)
            items.extend(all_items)
            preds.extend(score[i])
        all_predictions = pd.DataFrame(data={
            "user_num": users, "item_num":items, "prediction":preds
        })    
        
        # remove interacted_items
        train = pd.DataFrame(self.train_data, 
                            columns = ['user_num', 'item_num', 'timestamp'])
        train['rating'] = 1
        merged = pd.merge(train, all_predictions, on=["user_num", "item_num"], how="outer")
        all_predictions = merged[merged.rating.isnull()].drop(['rating','timestamp'], axis=1)
        all_predictions = all_predictions.sort_values('prediction', ascending=False)\
                .groupby('user_num').head(k)[['user_num', 'item_num']]\
                .groupby('user_num')['item_num'].apply(lambda x: list(x)).reset_index()\
                .sort_values('user_num', ascending=True)
        pred_ls = all_predictions['item_num'].tolist()
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
