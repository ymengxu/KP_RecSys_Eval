# Databricks notebook source
import os

from core import TEST_K, DATA_CLEAN_PATH, RES_PATH
from utils.evaluation import write_results_to_excel, get_test_results
from core.non_personalized_methods.popularity_recommender import PopularityRecommender

# if in databricks
if DATA_CLEAN_PATH.find('/Workspace/Repos/')>=0:
    dbutils.fs.cp('abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/data_clean', 
                'file:/tmp/model_evaluation_data/data_clean', True)

    LOCAL_ROOT_DIRECTORY = '/tmp/model_evaluation_data'
    DATA_CLEAN_PATH = os.path.join(LOCAL_ROOT_DIRECTORY, 'data_clean')
    RES_PATH = os.path.join(LOCAL_ROOT_DIRECTORY, 'res')

    dbutils.fs.mkdirs('file:'+DATA_CLEAN_PATH)
    dbutils.fs.mkdirs('file:'+RES_PATH)

# COMMAND ----------

for data in ['adobe_core5', 'movielens_100k']: 
    pop_rec = PopularityRecommender(data, DATA_CLEAN_PATH)
    pop_rec.train_model()
    res = get_test_results(pop_rec, TEST_K)
    write_results_to_excel(res, os.path.join(RES_PATH, 'test_results_PopularityRecommender.xlsx'), data)

# COMMAND ----------

# if in databricks, save result to ADLS
if DATA_CLEAN_PATH.find('/tmp/model_evaluation_data')>=0:
    dbutils.fs.cp(
        'file:/tmp/model_evaluation_data/res/test_results_PopularityRecommender.xlsx', 
        'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/res/test_results_PopularityRecommender.xlsx', 
        True
    )
