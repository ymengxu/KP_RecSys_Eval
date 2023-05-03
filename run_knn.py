# Databricks notebook source
import os

from core import TEST_K, DATA_CLEAN_PATH, RES_PATH
from utils.evaluation import write_results_to_excel, get_test_results
from core.similarity_based_methods.knn_recommender import KNNRecommender

# if in databricks
if DATA_CLEAN_PATH.find('/Workspace/Repos/')>=0:
    # dbutils.fs.cp('abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/data_clean', 
    #               'file:/tmp/model_evaluation_data/data_clean', True)
        
    LOCAL_ROOT_DIRECTORY = '/tmp/model_evaluation_data'
    DATA_CLEAN_PATH = os.path.join(LOCAL_ROOT_DIRECTORY, 'data_clean')
    RES_PATH = os.path.join(LOCAL_ROOT_DIRECTORY, 'res')

    dbutils.fs.mkdirs('file:'+DATA_CLEAN_PATH)
    dbutils.fs.mkdirs('file:'+RES_PATH)

# COMMAND ----------

def optimize_knn(dataset, verbose=True):
    for base in ['user', 'item']:
        metric = 'cosine'
        print('Tuning KNNRecommender with base={} metric={} on dataset {}'.format(base, metric, dataset))
        if (dataset=='adobe') & (base=='user'):
            print('skip user-based KNNRecommender for dataset {} due to memory limit'.format(DATASET))
            continue

        knn_rec = KNNRecommender(dataset, DATA_CLEAN_PATH)

        # find the optimal number of neighbors
        # if 5 consecutive runs with no improvement, break
        optimal_param = {'x': [0], 'fun':0}
        last_run_ndcg = 0
        current_ndcg = 0
        consecutive_idx = 0
        print("Finding the optimal hyperparameters for the current model...")

        for n_neighbors in range(1, 100):   
            knn_rec.train_model(base, metric, n_neighbors)
            current_ndcg = knn_rec.get_validation_ndcg()
            if verbose:
                print('n_neighbors={} validation NDCG@10={}'.format(n_neighbors, current_ndcg))

            if current_ndcg > optimal_param['fun']:
                # new optimal param
                optimal_param = {'x':[n_neighbors], 'fun':current_ndcg}
            
            if current_ndcg > last_run_ndcg:
                # improvement
                consecutive_idx = 0
            else:
                consecutive_idx += 1
            
            if consecutive_idx >= 5:
                break

            last_run_ndcg = current_ndcg
        
        print('Tuning finished. ')

        result_txt_path = os.path.join(RES_PATH, 'knn_recommender_tuning_results.txt')
        if os.path.exists(result_txt_path):
            f = open(os.path.join(RES_PATH, 'knn_recommender_tuning_results.txt'), 'a')
        else: 
            f = open(os.path.join(RES_PATH, 'knn_recommender_tuning_results.txt'), 'w')

        print('Optimal hyperparamter for the current model={} with validation ndcg@10 = {}'.format(optimal_param['x'], optimal_param['fun']), 
              file=f)

        print('Computing test statistic for the best model...', file=f)
        knn_rec.train_model(base, metric, optimal_param['x'][0])
        res = get_test_results(knn_rec, TEST_K)
        print(res, file=f)
        f.close()
        
        # write_results_to_excel(res, os.path.join(RES_PATH, 'test_results_KNNRecommender.xlsx'),
        #                         dataset+'_'+base+'_'+metric)

# COMMAND ----------

for data in ['movielens_100k', 'adobe_core5']:
    optimize_knn(data)

# COMMAND ----------

# if in databricks, save result to ADLS
if DATA_CLEAN_PATH.find('/tmp/model_evaluation_data')>=0:
    # dbutils.fs.cp(
    #     'file:/tmp/model_evaluation_data/res/test_results_KNNRecommender.xlsx', 
    #     'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/res/test_results_KNNRecommender.xlsx', 
    #     True
    # )
    dbutils.fs.cp(
        'file:/tmp/model_evaluation_data/res/knn_recommender_tuning_results.txt', 
        'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/res/knn_recommender_tuning_results.txt', 
        True
    )
