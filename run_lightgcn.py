# Databricks notebook source
import os
import pickle

from core import TEST_K, SEED
from core.neural_based_methods.lightgcn_recommender import LightGCNRecommender
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
import numpy as np

space = {
    'embed_size': hp.quniform('embed_size', 10, 100, 1),
    'n_layers': hp.choice('n_layers', [1,2,3,4]),
    'batch_size': hp.choice('batch_size', [32, 64, 128, 256]),
    'decay': hp.loguniform('decay', np.log(0.000001), np.log(0.01)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1))
}

# COMMAND ----------

def tune_lightgcn(data,
    tune_res_path=os.path.join(RES_PATH, 'lightgcn_recommender_tuning_results.txt'),
    test_res_path=os.path.join(RES_PATH, 'test_results_LightGCNRecommender.xlsx'),
    verbose=False
):
    rec = LightGCNRecommender(data, DATA_CLEAN_PATH)
    def objective(params):
        rec.train_model(**params, early_stopping=True, verbose=verbose)
        mean_ndcg = rec.get_validation_ndcg()
        return {
            'loss':-mean_ndcg, 
            'status':STATUS_OK,
            'best_epoch': rec.params['best_epoch']
        }

    print('Tuning hyperparameters of LightGCN Recommender on dataset {}...'.format(data))
    trials = Trials()
    trials._random_state = np.random.RandomState(SEED)
    best = fmin(objective, space, algo=tpe.suggest, max_evals=50, trials=trials)

    # save result
    if os.path.exists(tune_res_path):
        f = open(tune_res_path, 'a')
    else:
        f = open(tune_res_path, 'w')

    print(f'LightGCN Recommender tuning results on dataset {data}...', 
        file=f)

    # get the best trial information
    best_trial_idx = np.argmin([trial_info['result']['loss'] for trial_info in trials.trials])
    optimal_params = {
        'embed_size': trials.trials[best_trial_idx]['misc']['vals']['embed_size'][0], 
        'n_layers': [1,2,3,4][trials.trials[best_trial_idx]['misc']['vals']['n_layers'][0]], 
        'batch_size': [32, 64, 128, 256][trials.trials[best_trial_idx]['misc']['vals']['batch_size'][0]],
        'decay': trials.trials[best_trial_idx]['misc']['vals']['decay'][0],
        'learning_rate': trials.trials[best_trial_idx]['misc']['vals']['learning_rate'][0],
        'epochs': trials.trials[best_trial_idx]['result']['best_epoch']
    }
    print('The optimal hyperparamters are: \nembed_sizes={}, n_layers={}, batch_size={}, decay={}, learning_rate={}, epochs={}'.format(
        optimal_params['embed_size'], optimal_params['n_layers'], 
        optimal_params['batch_size'], optimal_params['decay'], 
        optimal_params['learning_rate'], optimal_params['epochs']
    ), file=f)
    print('Best validation NDCG@10 = {}'.format(-trials.trials[best_trial_idx]['result']['loss']), file=f)

    # retrain the model
    print('Retraining model using the optimal params...', file=f)
    rec.train_model(**optimal_params, early_stopping=False, verbose=verbose)
    print(f'Validation NDCG@10 of the retrained model={rec.get_validation_ndcg()}', file=f)

    # print test metrics
    print('Test results of the retrained model:', file=f)
    test_res = get_test_results(rec, TEST_K)
    print(test_res, file=f)
    write_results_to_excel(test_res, test_res_path, data)

    # # save the model
    # print('Saving the retrained model...', file=f)
    # user_file = os.path.join(SAVEMODEL_PATH, 'lightgcn', data, 'user_embeddings.csv')
    # item_file = os.path.join(SAVEMODEL_PATH, 'lightgcn', data, 'item_embeddings.csv')
    # rec.model.infer_embedding(user_file, item_file)

    # model_info = {'params': optimal_params}
    # dbutils.fs.mkdirs('file:'+os.path.join(SAVEMODEL_PATH, 'lightgcn', data))
    # with open(os.path.join(SAVEMODEL_PATH, 'lightgcn', data, 'params.pkl'), 'wb') as flp:
    #     pickle.dump(model_info, flp, protocol=pickle.HIGHEST_PROTOCOL)
    # print('Model saved. ', file=f)

    f.close()

# COMMAND ----------

tune_lightgcn('movielens_100k')
tune_lightgcn('adobe_core5')

# COMMAND ----------

# if in databricks, save result to ADLS
if IN_DATABRICKS:
    dbutils.fs.cp(
        'file:/tmp/model_evaluation_data/res/lightgcn_recommender_tuning_results.txt', 
        'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/res/lightgcn_recommender_tuning_results.txt', 
        True
    )
    dbutils.fs.cp(
        'file:/tmp/model_evaluation_data/res/test_results_LightGCNRecommender.xlsx', 
        'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/res/test_results_LightGCNRecommender.xlsx', 
        True
    )

# COMMAND ----------


