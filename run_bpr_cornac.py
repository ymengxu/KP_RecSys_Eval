# Databricks notebook source
import os

from core import TEST_K, SEED
from core.matrix_factorization_methods.bpr_cornac_recommender import BPRCornacRecommender
from utils.evaluation import get_test_results, write_results_to_excel

# if in databricks
if DATA_CLEAN_PATH.find('/Workspace/Repos/')>=0:
    # dbutils.fs.cp('abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/data_clean', 
    #               'file:/tmp/model_evaluation_data/data_clean', True)

    LOCAL_ROOT_DIRECTORY = '/tmp/model_evaluation_data'
    DATA_CLEAN_PATH = os.path.join(LOCAL_ROOT_DIRECTORY, 'data_clean')
    RES_PATH = os.path.join(LOCAL_ROOT_DIRECTORY, 'res')
    SAVEMODEL_PATH = os.path.join(LOCAL_ROOT_DIRECTORY, 'savemodel')

    dbutils.fs.mkdirs('file:'+DATA_CLEAN_PATH)
    dbutils.fs.mkdirs('file:'+RES_PATH)
    dbutils.fs.mkdirs('file:'+SAVEMODEL_PATH)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Fine tuning using hyperopt

# COMMAND ----------

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import numpy as np

space = {
    'k': hp.quniform('k', 10, 200, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)), 
    'lambda_reg': hp.loguniform('lambda_reg', np.log(0.00001), np.log(0.1))
}

# COMMAND ----------

# DBTITLE 1,adobe_core5
data = 'adobe_core5'
rec = BPRCornacRecommender(data, DATA_CLEAN_PATH)
def objective(params):
    rec.train_model(**params, early_stopping=True, verbose=False)
    mean_ndcg = rec.get_validation_ndcg()
    return {
        'loss':-mean_ndcg, 
        'status':STATUS_OK,
        'best_epoch':rec.params['best_epoch']
    }

print('Tuning hyperparameters of Cornac BPR Recommender on dataset {}'.format(data))
trials = Trials()
trials._random_state = np.random.RandomState(SEED)
best = fmin(objective, space, algo=tpe.suggest, max_evals=50, trials=trials)

# get the best trial information
best_trial_idx = np.argmin([trial_info['result']['loss'] for trial_info in trials.trials])
optimal_params = {
    'k': trials.trials[best_trial_idx]['misc']['vals']['k'][0], 
    'learning_rate': trials.trials[best_trial_idx]['misc']['vals']['learning_rate'][0],
    'lambda_reg': trials.trials[best_trial_idx]['misc']['vals']['lambda_reg'][0],
    'max_iter': trials.trials[best_trial_idx]['result']['best_epoch'],
}
print('The optimal hyperparamters are: \nfactors={}, learning_rate={}, regularization={}, iterations={}'.format(
    optimal_params['k'], optimal_params['learning_rate'], optimal_params['lambda_reg'], optimal_params['max_iter']
))
print('Best validation NDCG@10 = {}'.format(-trials.trials[best_trial_idx]['result']['loss']))

print('Retraining the model using optimal parameters...')
rec.train_model(**optimal_params, early_stopping=False)
print('Validation NDCG@10 of the retrained model =', rec.get_validation_ndcg())

# print test metrics
print('Test results of the retrained model:')
test_res = get_test_results(rec, TEST_K)
print(test_res)
write_results_to_excel(test_res, os.path.join(RES_PATH, 'test_results_BPRCornacRecommender.xlsx'), data)

# # save model
# model_info = {'model': rec.model, 'params': optimal_params}
# with open(os.path.join(SAVEMODEL_PATH, f'bpr_cornac_{data}.pkl'), 'wb') as flp:
#     pickle.dump(model_info, flp, protocol=pickle.HIGHEST_PROTOCOL)
# print('Model saved. ')

# COMMAND ----------

# DBTITLE 1,movielens_100k
data = 'movielens_100k'
rec = BPRCornacRecommender(data, DATA_CLEAN_PATH)

def objective(params):
    rec.train_model(**params, early_stopping=True, verbose=False)
    mean_ndcg = rec.get_validation_ndcg()
    return {
        'loss':-mean_ndcg, 
        'status':STATUS_OK,
        'best_epoch':rec.params['best_epoch']
    }

print('Tuning hyperparameters of Cornac BPR Recommender on dataset {}'.format(data))
trials = Trials()
trials._random_state = np.random.RandomState(SEED)
best = fmin(objective, space, algo=tpe.suggest, max_evals=50, trials=trials)

# get the best trial information
best_trial_idx = np.argmin([trial_info['result']['loss'] for trial_info in trials.trials])
optimal_params = {
    'k': trials.trials[best_trial_idx]['misc']['vals']['k'][0], 
    'learning_rate': trials.trials[best_trial_idx]['misc']['vals']['learning_rate'][0],
    'lambda_reg': trials.trials[best_trial_idx]['misc']['vals']['lambda_reg'][0],
    'max_iter': trials.trials[best_trial_idx]['result']['best_epoch'],
}
print('The optimal hyperparamters are: \nfactors={}, learning_rate={}, regularization={}, iterations={}'.format(
    optimal_params['k'], optimal_params['learning_rate'], optimal_params['lambda_reg'], optimal_params['max_iter']
))
print('Best validation NDCG@10 = {}'.format(-trials.trials[best_trial_idx]['result']['loss']))

print('Retraining the model using optimal parameters...')
rec.train_model(**optimal_params, early_stopping=False)
print('Validation NDCG@10 of the retrained model =', rec.get_validation_ndcg())

# print test metrics
print('Test results of the retrained model:')
test_res = get_test_results(rec, TEST_K)
print(test_res)
write_results_to_excel(test_res, os.path.join(RES_PATH, 'test_results_BPRCornacRecommender.xlsx'), data)

# # save model
# model_info = {'model': rec.model, 'params': optimal_params}
# with open(os.path.join(SAVEMODEL_PATH, f'bpr_cornac_{data}.pkl'), 'wb') as flp:
#     pickle.dump(model_info, flp, protocol=pickle.HIGHEST_PROTOCOL)
# print('Model saved. ')

# COMMAND ----------

# dbutils.fs.cp('file:/tmp/model_evaluation_data/res/test_results_BPRCornacRecommender.xlsx', 
# 'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/res/test_results_BPRCornacRecommender.xlsx', 
#               True)

# COMMAND ----------


