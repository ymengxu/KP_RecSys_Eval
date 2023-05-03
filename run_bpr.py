# Databricks notebook source
# MAGIC %%capture
# MAGIC 
# MAGIC import os
# MAGIC import pickle
# MAGIC import numpy as np
# MAGIC 
# MAGIC from core import SEED, TEST_K, DATA_CLEAN_PATH, RES_PATH, SAVEMODEL_PATH
# MAGIC from core.matrix_factorization_methods.bpr_recommender import BPRRecommender
# MAGIC from utils.evaluation import write_results_to_excel, get_test_results
# MAGIC 
# MAGIC # dbutils.fs.cp('abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/data_clean', 
# MAGIC # 'file:/tmp/model_evaluation_data/data_clean', True)
# MAGIC MAIN_DIRECTORY = '/tmp/model_evaluation_data'
# MAGIC DATA_CLEAN_PATH = os.path.join(MAIN_DIRECTORY, 'data_clean')
# MAGIC RES_PATH = os.path.join(MAIN_DIRECTORY, 'res')
# MAGIC SAVEMODEL_PATH = os.path.join(MAIN_DIRECTORY, 'savemodel')
# MAGIC dbutils.fs.mkdirs('file:'+DATA_CLEAN_PATH)
# MAGIC dbutils.fs.mkdirs('file:'+RES_PATH)
# MAGIC dbutils.fs.mkdirs('file:'+SAVEMODEL_PATH)

# COMMAND ----------

# DBTITLE 1,Fine tuning using hyperopt
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

space = {
    'factors': hp.quniform('factors', 10, 200, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)), 
    'regularization': hp.loguniform('regularization', np.log(0.00001), np.log(0.1)),
    'iterations': hp.quniform('iterations', 100, 500, 1)
}

for data in ['adobe', 'adobe_core5']:
    rec = BPRRecommender(data, DATA_CLEAN_PATH)
    def objective(params):
        rec.train_model(**params, early_stopping=False)
        mean_ndcg = rec.get_validation_ndcg()
        return {
            'loss':-mean_ndcg, 
            'status':STATUS_OK
        }

    print('Tuning hyperparameters of BPR Recommender on dataset {}'.format(data))
    trials = Trials()
    trials._random_state = np.random.RandomState(SEED)
    best = fmin(objective, space, algo=tpe.suggest, max_evals=50, trials=trials)

    # get the best trial information
    best_trial_idx = np.argmin([trial_info['result']['loss'] for trial_info in trials.trials])
    optimal_params = {
        'factors': trials.trials[best_trial_idx]['misc']['vals']['factors'][0], 
        'learning_rate': trials.trials[best_trial_idx]['misc']['vals']['learning_rate'][0],
        'regularization': trials.trials[best_trial_idx]['misc']['vals']['regularization'][0],
        'iterations': trials.trials[best_trial_idx]['misc']['vals']['iterations'][0],
    }
    print('The optimal hyperparamters are: \nfactors={}, learning_rate={}, regularization={}, iterations={}'.format(
        optimal_params['factors'], optimal_params['learning_rate'], optimal_params['regularization'], optimal_params['iterations']
    ))
    print('Best validation NDCG@10 = {}'.format(-trials.trials[best_trial_idx]['result']['loss']))

    # retrain the model
    rec.train_model(**optimal_params, early_stopping=False)
    print('Validation NDCG@10 of the retrained model =', rec.get_validation_ndcg())

    # print test metrics
    print('Test results of the retrained model:')
    test_res = get_test_results(rec, TEST_K)
    print(test_res)
    write_results_to_excel(test_res, os.path.join(RES_PATH, 'test_results_BPRRecommender.xlsx'), data)

    # save the model
    print('Saving the retrained model...')
    model_info = {'model': rec.model, 'params': optimal_params}
    with open(os.path.join(SAVEMODEL_PATH, f'bpr_{data}.pkl'), 'wb') as flp:
        pickle.dump(model_info, flp, protocol=pickle.HIGHEST_PROTOCOL)
    print('Model saved. ')

# COMMAND ----------

# DBTITLE 1,Fine tuning using skopt.forest_minimize
from skopt import forest_minimize
from skopt.space import Integer, Real

SPACE_iALS = [
    Integer(10,200, name='factors', prior='uniform'),
    Real(0.00001, 0.1, name='learning_rate', prior='log-uniform'),
    Real(0.00001, 0.1, name='regularization', prior='log-uniform'),
    Integer(100, 500, name='iterations', prior='uniform')
]

for data in ['adobe_core5', 'adobe']:
    rec = BPRRecommender(data, DATA_CLEAN_PATH)
    print('Tuning hyperparameters of iALS Recommender on dataset {}...'.format(data))

    def objective(space):
        params = {'factors': space[0], 'learning_rate': space[1], 'regularization': space[2], 'iterations':space[3]}
        rec.train_model(**params, early_stopping=False)
        mean_ndcg = rec.get_validation_ndcg()
        return -mean_ndcg

    result = forest_minimize(objective, SPACE_iALS, n_calls=50, 
        random_state=SEED, n_jobs=-1, verbose=False)
    
    print('Tuning finished.')

    # find the optimal params
    print('The optimal hyperparamters are:', result['x'])
    print('Best validation NDCG@10 = {}'.format(result['fun']))

    print('Retraining model using the optimal hyperparamters found...')
    params = {
        'factors': result['x'][0], 
        'learning_rate': result['x'][1], 
        'regularization': result['x'][2], 
        'iterations': result['x'][3]
    }
    rec.train_model(**params, early_stopping=False)
    print('Validation NDCG@10 of the retrained model =', rec.get_validation_ndcg())

    # print test metrics
    print('Test results of the retrained model:')
    test_res = get_test_results(rec, TEST_K)
    print(test_res)
    # write_results_to_excel(test_res, os.path.join(RES_PATH, 'test_results_iALSRecommender.xlsx'), data)

    # # save the model
    # print('Saving the retrained model...')
    # model_info = {'model': rec.model, 'params': params}
    # with open(os.path.join(SAVEMODEL_PATH, f'ials_{data}.pkl'), 'wb') as flp:
    #     pickle.dump(model_info, flp, protocol=pickle.HIGHEST_PROTOCOL)
    # print('Model saved. ')

    break

# COMMAND ----------

dbutils.fs.cp('file:/tmp/model_evaluation_data/res/test_results_iALSRecommender.xlsx', 
'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/res/test_results_BPRRecommender.xlsx', 
              True)

dbutils.fs.cp('file:/tmp/model_evaluation_data/savemodel/', 
              'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/savemodel/new', 
                            True)

# COMMAND ----------


