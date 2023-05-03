# Databricks notebook source
import os
import numpy as np
import pandas as pd

from core import TEST_K, DATA_CLEAN_PATH, RES_PATH
from core.non_personalized_methods.random_recommender import RandomRecommender
from utils.evaluation import write_results_to_excel

# if in databricks
IN_DATABRICKS = DATA_CLEAN_PATH.find('/Workspace/Repos/')>=0
if IN_DATABRICKS:
    LOCAL_ROOT_DIRECTORY = '/tmp/model_evaluation_data'
    DATA_CLEAN_PATH = os.path.join(LOCAL_ROOT_DIRECTORY, 'data_clean')
    RES_PATH = os.path.join(LOCAL_ROOT_DIRECTORY, 'res')

    if not os.path.exists(DATA_CLEAN_PATH):
        dbutils.fs.cp('abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/data_clean', 
                    'file:/tmp/model_evaluation_data/data_clean', True)
    dbutils.fs.mkdirs('file:'+RES_PATH)

# COMMAND ----------

def get_random_recommendation_results(rec, K, n_runs=10):
    """
    get the average test results for a random recommender over n runs
    Args:
        rec: RandomRecommender instance
        K: a list of top-k to compute
        n_runs: number of runs
    """
    metrics = ['precison', 'recall', 'hit ratio', 'NDCG', 'MRR', 'MAP']
    res = np.zeros((len(metrics), len(K)))
    for _ in range(n_runs):
        new_metrics = rec.get_test_metrics(K)
        for i in range(len(metrics)): 
            res[i] += new_metrics[i]
    res = res/n_runs
    res = {metrics[i]: res[i] for i in range(len(metrics))}
    res = pd.DataFrame.from_dict(res, orient='index').set_axis(K, axis=1)
    return res

# COMMAND ----------

# random recommend for 10 times
for data in ['movielens_100k', 'adobe_core5']:
    rec = RandomRecommender(data, DATA_CLEAN_PATH)
    res = get_random_recommendation_results(rec, TEST_K)
    write_results_to_excel(res, os.path.join(RES_PATH, 'test_results_RandomRecommender.xlsx'), data)

# COMMAND ----------

# if in databricks, save result to ADLS
if IN_DATABRICKS:
    dbutils.fs.cp(
        'file:/tmp/model_evaluation_data/res/test_results_RandomRecommender.xlsx', 
        'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/res/test_results_RandomRecommender.xlsx', 
        True
    )

# COMMAND ----------


