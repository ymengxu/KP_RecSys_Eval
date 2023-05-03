import os

import numpy as np
import pandas as pd
import openpyxl


def write_results_to_excel(res:pd.DataFrame, path, sheet_name):
    if not os.path.isfile(path):
        res.to_excel(path, sheet_name=sheet_name)
    else:
        writer = pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='replace')
        writer.book = openpyxl.load_workbook(path)
        writer.sheets = {ws.title:ws for ws in writer.book.worksheets}
        res.to_excel(writer, sheet_name=sheet_name)
        writer.save()


def get_test_results(model, K)->pd.DataFrame:
    """for a given model trained on a given dataset, compute:
    precision@k, recall@k, HR@k, NDCG@k, MRR@k, MAP@k for each k in the K list"""

    mean_precision, mean_recall, HR, mean_ndcg, MRR, MAP = model.get_test_metrics(K=K)
    res = {
        'precision': mean_precision,
        'recall': mean_recall,
        'hit ratio': HR,
        'NDCG': mean_ndcg,
        'MRR': MRR,
        'MAP': MAP
    }
    res = pd.DataFrame.from_dict(res, orient='index').set_axis(K, axis=1)
    return res


class Evaluation:
    def __init__(self, batch_size, K:list, how:str):
        """
        compute all evaluation metrics for test users;
        compute ndcg@10 for validation users (hyperparameter tuning)

        Args:
            batch_size: number of val/test users in one batch
            K: a list of top-k requiremenst, ex: [5,10,20]
            how: ['test', 'val']
        """
        assert how in ['test', 'val'], "Parameter 'how' need to be either 'test' or 'val'"
        self.K = K
        self.batch_size = batch_size
        self.how = how
        self._init_metrics()


    def evaluate(self, pred_ls, true_ls):
        """evaluate the prediction results
        pred_ls: a list of prediction lists for each val/test user
        true_ls: a list of interacted items lists for each val/test user"""
        for i in range(len(pred_ls)):
            self._update(true_ls[i], pred_ls[i], i)
        return self._get_final_metrics()


    def clear(self):
        """clear all attributes for a new batch of users"""
        self._init_metrics()


    def _init_metrics(self):
        """initialize empty metric matrix (matrices)"""
        if self.how == 'val':
            self.ndcg = np.zeros((self.batch_size, len(self.K)))
        if self.how == 'test':
            self.precision = np.zeros((self.batch_size, len(self.K)))
            self.recall = np.zeros((self.batch_size, len(self.K)))
            self.hit = np.zeros((self.batch_size, len(self.K)))
            self.ndcg = np.zeros((self.batch_size, len(self.K)))
            self.rr = np.zeros((self.batch_size, len(self.K)))
            self.ap = np.zeros((self.batch_size, len(self.K)))


    def _update(self, y_true:list, y_pred:list, idx:int):
        """update metrics for a new val/test user"""
        if self.how == 'val':
            for j in range(len(self.K)):
                k = self.K[j]
                self.ndcg[idx,j] = self.get_ndcg(y_true, y_pred, k)
        if self.how == 'test':
            for j in range(len(self.K)):
                k = self.K[j]
                self.precision[idx,j] = self.get_precision(y_true, y_pred, k)
                self.recall[idx,j] = self.get_recall(y_true, y_pred, k)
                self.hit[idx,j] = self.get_hit(y_true, y_pred, k)
                self.ndcg[idx,j] = self.get_ndcg(y_true, y_pred, k)
                self.rr[idx,j] = self.get_rr(y_true, y_pred, k)
                self.ap[idx,j] = self.get_ap(y_true, y_pred, k)


    def _get_final_metrics(self):
        """get final average metrics for val/test users"""
        if self.how == 'val':
            mean_ndcg  = np.mean(self.ndcg, axis=0)
            return mean_ndcg
        if self.how == 'test':
            mean_precision = np.mean(self.precision, axis=0)
            mean_recall = np.mean(self.recall, axis=0)
            HR = np.mean(self.hit, axis=0)
            mean_ndcg  = np.mean(self.ndcg, axis=0)
            MRR = np.mean(self.rr, axis=0)
            MAP = np.mean(self.ap, axis=0)
            return mean_precision, mean_recall, HR, mean_ndcg, MRR, MAP


    def get_precision(self, y_true, y_pred, k):
        y_pred_k = y_pred[:k]
        common_items_k = [i for i in y_pred_k if i in y_true]
        return len(common_items_k)/k


    def get_recall(self, y_true, y_pred, k):
        y_pred_k = y_pred[:k]
        common_items_k = [i for i in y_pred_k if i in y_true]
        return len(common_items_k)/min([k,len(y_true)])


    def get_hit(self, y_true, y_pred, k):
        y_pred_k = y_pred[:k]
        common_items_k = [i for i in y_pred_k if i in y_true]
        return len(common_items_k)>0


    def get_ndcg(self, y_true, y_pred, k):
        e = 0.0000000001
        m = len(y_pred)
        Z_u = 0
        temp = 0
        for i in range(0, k):
            if i < m:
                Z_u += 1 / np.log2(i + 2)
            if y_pred[i] in y_true:
                temp += 1 / np.log2(i + 2)

        return temp/(Z_u+e)


    def get_rr(self, y_true, y_pred, k):
        y_pred_k = y_pred[:k]
        common_items_k = [i for i in y_pred_k if i in y_true]
        if len(common_items_k) == 0:
            return 0
        else:
            for i in range(k):
                if y_pred[i] in y_true:
                    break
            return 1/(i+1)


    def get_ap(self, y_true, y_pred, k):
        y_pred_k = y_pred[:k]
        common_items_k = [i for i in y_pred_k if i in y_true]
        if len(common_items_k) == 0:
            return 0
        else:
            ap = 0
            for i in range(k):
                ap += self.get_precision(y_true, y_pred, k)*(y_pred[i] in y_true)
            return ap/min([len(y_pred), k])


    def get_coverage(self, y_pred_all, n):
        return self(y_pred_all, n)
