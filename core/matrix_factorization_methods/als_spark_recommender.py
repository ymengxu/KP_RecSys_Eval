"""This class is not tested. It has an abnormally long evaluation time. """

import os

from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.functions import col, row_number

from core import MAIN_DIRECTORY
from utils.data_importer import DataImporter
from utils.evaluation import Evaluation


class ALSSparkRecommender:
    def __init__(self, dataset_name, data_folder_path=os.path.join(MAIN_DIRECTORY, 'data_clean')):
        DI = DataImporter(dataset_name, data_folder_path)
        train, val_user, val_ytrue, test_user, test_ytrue = DI.get_data('als_spark')
        self.train = train
        self.val_user = val_user
        self.val_ytrue = val_ytrue
        self.test_user = test_user
        self.test_ytrue = test_ytrue
        self.model = None
    

    def train_model(self, params):
        model = ALS(**params)
        model = model.fit(self.train)
        self.model = model

    
    def get_recommendation(self, user_list, k:int):
        """user_list: pyspark.sql.dataframe.DataFrame"""
        if self.model is None:
            print('ALSSparkRecommender: model not trained yet. Call train_model() first.')
            return
        
        # Get the cross join of all user-item pairs and score them.
        items = self.train.select('item_num').distinct()
        user_item = user_list.crossJoin(items)
        dfs_pred = self.model.transform(user_item)

        # Remove seen items
        dfs_pred_exclude_train = dfs_pred.alias("pred").join(
            self.train.alias("train"),
            (dfs_pred['user_num'] == self.train['user_num'])
            & (dfs_pred['item_num'] == self.train['item_num']),
            how="outer",
        )

        topk_scores = dfs_pred_exclude_train.filter(
            dfs_pred_exclude_train["train." + 'rating'].isNull()
        ).select(
            "pred." + 'user_num',
            "pred." + 'item_num',
            "pred." + 'prediction',
        )
        
        window_spec = Window.partitionBy('user_num').orderBy(col('prediction').desc())
        # generate top-k recommendation list
        items_for_user = (
            topk_scores.select(
                'user_num', 'item_num', 'prediction', row_number().over(window_spec).alias('rank')
            )
            .where(col('rank') <= k)
            .groupby('user_num')
            .agg(F.collect_list('item_num').alias('recommendation'))
        )
        
        pred_ls_df = items_for_user.toPandas(). # takes 1-2 minutes
        pred_ls_df = pred_ls_df.sort_values('user_num')
        pred_ls = pred_ls_df['recommendation'].map(lambda x: list(x)).tolist()
        return pred_ls
    
    
    def get_validation_ndcg(self, k=10):
        """compute ndcg@10 for validation users"""
        if self.model is None:
            print('ALSSparkRecommender: model not trained yet. Call train_model() first.')
            return

        pred_ls = self.get_recommendation(self.val_user, k=k)
        val_evaluator = Evaluation(batch_size = len(self.val_user), K=[k], how='val')
        mean_ndcg = val_evaluator.evaluate(pred_ls, self.val_ytrue)
        return mean_ndcg[0]
    

    def get_test_metrics(self, K):
        if self.model is None:
            print('ALSSparkRecommender: model not trained yet. Call train_model() first.')
            return
        
        pred_ls = self.get_recommendation(self.test_user, max(K))
        test_evaluator = Evaluation(batch_size = len(self.test_user), K=K, how='test')
        mean_precision, mean_recall, HR, mean_ndcg, MRR, MAP = test_evaluator.evaluate(pred_ls, self.test_ytrue)
        return mean_precision, mean_recall, HR, mean_ndcg, MRR, MAP
    

    def obj(self, params):
        """TODO"""
        return
