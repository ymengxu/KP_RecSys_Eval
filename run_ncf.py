# Databricks notebook source
# MAGIC %%capture
# MAGIC !pip install tf_slim

# COMMAND ----------

import os
import pickle
import numpy as np

from core import TEST_K, SEED, DATA_CLEAN_PATH, RES_PATH, SAVEMODEL_PATH
from core.neural_based_methods.ncf.ncf_recommender import NCFRecommender

from utils.evaluation import get_test_results, write_results_to_excel

# if in databricks
IN_DATABRICKS = DATA_CLEAN_PATH.find('/Workspace/Repos/')>=0
if IN_DATABRICKS:
    MAIN_DIRECTORY = '/tmp/model_evaluation_data'
    DATA_CLEAN_PATH = os.path.join(MAIN_DIRECTORY, 'data_clean')
    RES_PATH = os.path.join(MAIN_DIRECTORY, 'res')

    if not os.path.exists(DATA_CLEAN_PATH):
        dbutils.fs.cp('abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/data_clean', 
        'file:/tmp/model_evaluation_data/data_clean', True)

    dbutils.fs.mkdirs('file:'+RES_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC # Fine tuning using hyperopt

# COMMAND ----------

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

space = {
    'n_factors': hp.choice('n_factors', [8,16,32,64]),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)), 
    'batch_size': hp.choice('batch_size', [32,64,128,256]),
    'n_neg': hp.choice('n_neg', [4,6,8])
}

# COMMAND ----------

def tune_ncf(data,
    tune_res_path=os.path.join(RES_PATH, 'ncf_recommender_tuning_results.txt'),
    test_res_path=os.path.join(RES_PATH, 'test_results_NCFRecommender.xlsx'),
    verbose=False
):
    rec = NCFRecommender(data, DATA_CLEAN_PATH)

    def objective(params):
        rec.train_model(**params, early_stopping=True, verbose=verbose, train_file_save_folder=DATA_CLEAN_PATH)
        mean_ndcg = rec.get_validation_ndcg()
        return {
            'loss':-mean_ndcg, 
            'status':STATUS_OK,
            'best_epoch':rec.params['best_epoch']
        }

    print('Tuning hyperparameters of NCF Recommender on dataset {}...'.format(data))
    trials = Trials()
    trials._random_state = np.random.RandomState(SEED)
    best = fmin(objective, space, algo=tpe.suggest, max_evals=50, trials=trials)

    # save result
    if os.path.exists(tune_res_path):
        f = open(tune_res_path, 'a')
    else:
        f = open(tune_res_path, 'w')

    print(f'NCF Recommender tuning results on dataset {data}...', 
        file=f)

    # get the best trial information
    best_trial_idx = np.argmin([trial_info['result']['loss'] for trial_info in trials.trials])
    optimal_params = {
        'n_factors': [8,16,32,64][trials.trials[best_trial_idx]['misc']['vals']['n_factors'][0]],
        'learning_rate': trials.trials[best_trial_idx]['misc']['vals']['learning_rate'][0],
        'batch_size': [32,64,128,256][trials.trials[best_trial_idx]['misc']['vals']['batch_size'][0]],
        'n_neg': [4,6,8][trials.trials[best_trial_idx]['misc']['vals']['n_neg'][0]],
        'epochs': trials.trials[best_trial_idx]['result']['best_epoch'],
    }
    print('The optimal hyperparamters are: \n_factors={}, learning_rate={}, batch_size={}, n_neg={}, iterations={}'.format(
        optimal_params['n_factors'], optimal_params['learning_rate'],
        optimal_params['batch_size'], optimal_params['n_neg'],
        optimal_params['epochs']
    ), file=f)
    print('Best validation NDCG@10 = {}'.format(-trials.trials[best_trial_idx]['result']['loss']), file=f)

    # retrain the model
    print('Retraining the model using optimal parameters...', file=f)
    rec.train_model(**optimal_params, early_stopping=False, train_file_save_folder=DATA_CLEAN_PATH, verbose=verbose)
    print(f'Validation NDCG@10 of the retrained model = {rec.get_validation_ndcg()}', file=f)

    # print test metrics
    print('Test results of the retrained model:', file=f)
    test_res = get_test_results(rec, TEST_K)
    print(test_res, file=f)
    write_results_to_excel(test_res, test_res_path, data)

    f.close()

# COMMAND ----------

tune_ncf('movielens_100k')
tune_ncf('adobe_core5')

# COMMAND ----------

# if in databricks, save result to ADLS
if IN_DATABRICKS:
    dbutils.fs.cp(
        'file:/tmp/model_evaluation_data/res/ncf_recommender_tuning_results.txt', 
        'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/res/ncf_recommender_tuning_results.txt', 
        True
    )
    dbutils.fs.cp(
        'file:/tmp/model_evaluation_data/res/test_results_NCFRecommender.xlsx', 
        'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/res/test_results_NCFRecommender.xlsx', 
        True
    )
