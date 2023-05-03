# Databricks notebook source
import os
import pickle
import numpy as np

from core import SEED, TEST_K, DATA_CLEAN_PATH, RES_PATH, SAVEMODEL_PATH
from core.matrix_factorization_methods.i_als_recommender import iALSRecommender
from utils.evaluation import write_results_to_excel, get_test_results

# if in databricks
if DATA_CLEAN_PATH.find('/Workspace/Repos/')>=0:
    # dbutils.fs.cp('abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/data_clean', 
    # 'file:/tmp/model_evaluation_data/data_clean', True)
    MAIN_DIRECTORY = '/tmp/model_evaluation_data'
    DATA_CLEAN_PATH = os.path.join(MAIN_DIRECTORY, 'data_clean')
    RES_PATH = os.path.join(MAIN_DIRECTORY, 'res')
    SAVEMODEL_PATH = os.path.join(MAIN_DIRECTORY, 'savemodel')
    
    dbutils.fs.mkdirs('file:'+DATA_CLEAN_PATH)
    dbutils.fs.mkdirs('file:'+RES_PATH)
    dbutils.fs.mkdirs('file:'+SAVEMODEL_PATH)

# COMMAND ----------

# DBTITLE 1,Fine Tuning using skopt.forest_minimize
from skopt import forest_minimize
from skopt.space import Integer, Real

SPACE_iALS = [
    Integer(10,200, name='factors', prior='uniform'),
    Real(0.001, 50, name='alpha', prior='log-uniform'),
    Real(0.00001, 0.01, name='regularization', prior='log-uniform')
]

for data in ['adobe', 'adobe_core5']:
    rec = iALSRecommender(data, DATA_CLEAN_PATH)
    print('Tuning hyperparameters of iALS Recommender on dataset {}...'.format(data))

    def objective(space):
        params = {'factors': space[0], 'alpha': space[1], 'regularization': space[2]}
        rec.train_model(**params, early_stopping=True, verbose=False)
        mean_ndcg = rec.get_validation_ndcg()
        return -mean_ndcg

    best_epochs = []
    def callback(res):
        # store the best epoch number in each trial
        best_epochs.append(rec.params['best_epoch'])

    result = forest_minimize(objective, SPACE_iALS, n_calls=50, 
        callback=callback, random_state=SEED, n_jobs=-1, verbose=False)
    
    print('Tuning finished.')

    # find the best epoch number in the trail with the best objective function value
    best_trial_idx = np.argmin(result['func_vals'])
    best_epoch = best_epochs[best_trial_idx]

    print('The optimal hyperparamters are: \nfactors={}, alpha={}, regularization={}, iterations={}'.format(
        result['x'][0], result['x'][1], result['x'][2], best_epoch
    ))
    print('Best validation NDCG@10 = {}'.format(result['fun']))

    print('Retraining model using the optimal hyperparamters found...')
    params = {'factors': result['x'][0], 'alpha': result['x'][1], 'regularization': result['x'][2], 
        'iterations': best_epoch}
    rec.train_model(**params, early_stopping=False)
    print('Validation NDCG@10 of the retrained model =', rec.get_validation_ndcg())

    # print test metrics
    print('Test results of the retrained model:')
    test_res = get_test_results(rec, TEST_K)
    print(test_res)
    write_results_to_excel(test_res, os.path.join(RES_PATH, 'test_results_iALSRecommender.xlsx'), data)

    # save the model
    print('Saving the retrained model...')
    model_info = {'model': rec.model, 'params': params}
    with open(os.path.join(SAVEMODEL_PATH, f'ials_{data}.pkl'), 'wb') as flp:
        pickle.dump(model_info, flp, protocol=pickle.HIGHEST_PROTOCOL)
    print('Model saved. ')

# COMMAND ----------

# DBTITLE 1,Fine tuning using hyperopt
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

space = {
    'factors': hp.quniform('factors', 10, 200, 1),
    'alpha': hp.loguniform('alpha', np.log(0.001), np.log(50)), 
    'regularization': hp.loguniform('regularization', np.log(0.00001), np.log(0.01))
}

for data in ['adobe', 'adobe_core5']:
    rec = iALSRecommender(data, DATA_CLEAN_PATH)
    def objective(params):
        rec.train_model(**params, early_stopping=True, verbose=False)
        mean_ndcg = rec.get_validation_ndcg()
        return {
            'loss':-mean_ndcg, 
            'status':STATUS_OK, 
            'best_epoch': rec.params['best_epoch']
        }

    print('Tuning hyperparameters of iALS Recommender on dataset {}...'.format(data))
    trials = Trials()
    trials._random_state = np.random.RandomState(SEED)
    best = fmin(objective, space, algo=tpe.suggest, max_evals=50, trials=trials)

    # get the best trial information
    best_trial_idx = np.argmin([trial_info['result']['loss'] for trial_info in trials.trials])
    optimal_params = {
        'alpha': trials.trials[best_trial_idx]['misc']['vals']['alpha'][0], 
        'factors': trials.trials[best_trial_idx]['misc']['vals']['factors'][0], 
        'regularization': trials.trials[best_trial_idx]['misc']['vals']['regularization'][0],
        'iterations': trials.trials[best_trial_idx]['result']['best_epoch']
    }
    print('The optimal hyperparamters are: \nfactors={}, alpha={}, regularization={}, iterations={}'.format(
        optimal_params['factors'], optimal_params['alpha'], optimal_params['regularization'], optimal_params['iterations']
    ))
    print('Best validation NDCG@10 = {}'.format(-trials.trials[best_trial_idx]['result']['loss']))

    # retrain the model
    rec.train_model(**optimal_params, early_stopping=False)
    print('Validation NDCG@10 of the retrained model =', rec.get_validation_ndcg())

    # print test metrics
    print('Test results of the retrained model:')
    test_res = get_test_results(rec, TEST_K)
    print(test_res)
    write_results_to_excel(test_res, os.path.join(RES_PATH, 'test_results_iALSRecommender.xlsx'), data)

    # save the model
    print('Saving the retrained model...')
    model_info = {'model': rec.model, 'params': optimal_params}
    with open(os.path.join(SAVEMODEL_PATH, f'ials_{data}.pkl'), 'wb') as flp:
        pickle.dump(model_info, flp, protocol=pickle.HIGHEST_PROTOCOL)
    print('Model saved. ')

# COMMAND ----------

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

space = {
    'factors': hp.quniform('factors', 10, 200, 1),
    'alpha': hp.loguniform('alpha', np.log(0.001), np.log(50)), 
    'regularization': hp.loguniform('regularization', np.log(0.00001), np.log(0.01))
}

data = 'movielens_100k'
rec = iALSRecommender(data, DATA_CLEAN_PATH)
def objective(params):
    rec.train_model(**params, early_stopping=True, verbose=False)
    mean_ndcg = rec.get_validation_ndcg()
    return {
        'loss':-mean_ndcg, 
        'status':STATUS_OK, 
        'best_epoch': rec.params['best_epoch']
    }

print('Tuning hyperparameters of iALS Recommender on dataset {}...'.format(data))
trials = Trials()
trials._random_state = np.random.RandomState(SEED)
best = fmin(objective, space, algo=tpe.suggest, max_evals=50, trials=trials)

# get the best trial information
best_trial_idx = np.argmin([trial_info['result']['loss'] for trial_info in trials.trials])
optimal_params = {
    'alpha': trials.trials[best_trial_idx]['misc']['vals']['alpha'][0], 
    'factors': trials.trials[best_trial_idx]['misc']['vals']['factors'][0], 
    'regularization': trials.trials[best_trial_idx]['misc']['vals']['regularization'][0],
    'iterations': trials.trials[best_trial_idx]['result']['best_epoch']
}
print('The optimal hyperparamters are: \nfactors={}, alpha={}, regularization={}, iterations={}'.format(
    optimal_params['factors'], optimal_params['alpha'], optimal_params['regularization'], optimal_params['iterations']
))
print('Best validation NDCG@10 = {}'.format(-trials.trials[best_trial_idx]['result']['loss']))

# retrain the model
rec.train_model(**optimal_params, early_stopping=False)
print('Validation NDCG@10 of the retrained model =', rec.get_validation_ndcg())

# print test metrics
print('Test results of the retrained model:')
test_res = get_test_results(rec, TEST_K)
print(test_res)
write_results_to_excel(test_res, os.path.join(RES_PATH, 'test_results_iALSRecommender.xlsx'), data)

# # save the model
# print('Saving the retrained model...')
# model_info = {'model': rec.model, 'params': optimal_params}
# with open(os.path.join(SAVEMODEL_PATH, f'ials_{data}.pkl'), 'wb') as flp:
#     pickle.dump(model_info, flp, protocol=pickle.HIGHEST_PROTOCOL)
# print('Model saved. ')

# COMMAND ----------

dbutils.fs.cp('file:/tmp/model_evaluation_data/res/test_results_iALSRecommender.xlsx', 
'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/res/test_results_iALSRecommender.xlsx', 
              True)

# dbutils.fs.cp('file:/tmp/model_evaluation_data/savemodel/', 
#               'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/savemodel/new', 
#                             True)

# COMMAND ----------


