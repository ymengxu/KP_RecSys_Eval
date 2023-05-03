# Databricks notebook source
import os
import warnings
warnings.filterwarnings("ignore") 

from core import DATA_CLEAN_PATH, TEST_K, SEED, RES_PATH
from core.neural_based_methods.wide_deep.wide_deep_recommender import WideDeepRecommender
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

    dbutils.fs.mkdirs('file:'+DATA_CLEAN_PATH)
    dbutils.fs.mkdirs('file:'+RES_PATH)

# COMMAND ----------

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import numpy as np

space = {
    'batch_size': hp.choice('batch_size', [32, 64, 128, 256]),
    'dnn_dropout': hp.quniform('dnn_dropout', 0.5, 0.8, 0.1),
    'embedding_dim': hp.quniform('embedding_dim', 20, 200, 1),
    'dnn_hidden_layer_1': hp.choice('dnn_hidden_layer_1', [16, 32, 64, 128]),   
    'dnn_hidden_layer_num': hp.quniform('dnn_hidden_layer_num', 2, 4, 1),
    'linear_optimizer': hp.choice('linear_optimizer', ['ftrl', 'adam', 'adagrad', 'adadelta']),
    'linear_optimizer_lr': hp.loguniform('linear_optimizer_lr', np.log(0.00001), np.log(0.1)), 
    'dnn_optimizer': hp.choice('dnn_optimizer', ['ftrl', 'adam', 'adagrad', 'adadelta']),
    'dnn_optimizer_lr': hp.loguniform('dnn_optimizer_lr', np.log(0.00001), np.log(0.1)),
}

# COMMAND ----------

def tune_wide_deep(
    data, 
    tune_res_path=os.path.join(RES_PATH, 'wide_deep_recommender_tuning_results.txt'),
    test_res_path=os.path.join(RES_PATH, 'test_results_WideDeepRecommender.xlsx'),
):
    rec = WideDeepRecommender(data, DATA_CLEAN_PATH)
    def objective(params):
        rec.train_model(**params, early_stopping=True, verbose=True, seed=SEED,
                        model_dir=os.path.join(RES_PATH, f'wide_deep_checkpoints/{data}'))
        return {
            'loss':-rec.params['best_eval_score'], 
            'status':STATUS_OK,
            'best_epoch': rec.params['best_epoch']
        }
    print('Tuning hyperparameters of Wide and Deep Recommender on dataset {}...'.format(data))
    trials = Trials()
    trials._random_state = np.random.RandomState(SEED)
    best = fmin(objective, space, algo=tpe.suggest, max_evals=50, trials=trials)

    # save result
    if os.path.exists(tune_res_path):
        f = open(tune_res_path, 'a')
    else:
        f = open(tune_res_path, 'w')

    print('Wide Deep Recommender tuning results on dataset {}...'.format(data), file=f)

    # get the best trial information
    best_trial_idx = np.argmin([trial_info['result']['loss'] for trial_info in trials.trials])

    optimal_params = {
        'batch_size': [32, 64, 128, 256][trials.trials[best_trial_idx]['misc']['vals']['batch_size'][0]],
        'dnn_dropout': trials.trials[best_trial_idx]['misc']['vals']['dnn_dropout'][0],
        'embedding_dim': trials.trials[best_trial_idx]['misc']['vals']['embedding_dim'][0],
        # hidden layers
        'dnn_hidden_layer_1': [16, 32, 64, 128][trials.trials[best_trial_idx]['misc']['vals']['dnn_hidden_layer_1'][0]],
        'dnn_hidden_layer_num': trials.trials[best_trial_idx]['misc']['vals']['dnn_hidden_layer_num'][0],
        # optimzier
        'linear_optimizer': ['ftrl', 'adam', 'adagrad', 'adadelta'][trials.trials[best_trial_idx]['misc']['vals']['dnn_hidden_layer_1'][0]],
        'linear_optimizer_lr': trials.trials[best_trial_idx]['misc']['vals']['linear_optimizer_lr'][0],
        'dnn_optimizer': ['ftrl', 'adam', 'adagrad', 'adadelta'][trials.trials[best_trial_idx]['misc']['vals']['dnn_optimizer'][0]],
        'dnn_optimizer_lr': trials.trials[best_trial_idx]['misc']['vals']['dnn_optimizer_lr'][0],
        # epochs
        'n_epochs': trials.trials[best_trial_idx]['result']['best_epoch']
    }

    print(
        'The optimal hyperparamters are: \nbatch_size={}, dnn_dropout={}, embedding_dim={},'.format(
            optimal_params['batch_size'], optimal_params['dnn_dropout'], optimal_params['embedding_dim'], 
        ),
    file=f
    )
    print(
        'dnn_hidden_layer_1={}, dnn_hidden_layer_num={}, '.format(
            optimal_params['dnn_hidden_layer_1'], optimal_params['dnn_hidden_layer_num'], 
        ),
    file=f
    )
    print(
        'linear_optimizer={}, linear_optimizer_lr={}, dnn_optimizer={}, dnn_optimizer_lr={}, n_epochs={}'.format(
            optimal_params['linear_optimizer'], optimal_params['linear_optimizer_lr'], 
            optimal_params['dnn_optimizer'], optimal_params['dnn_optimizer_lr'], 
            optimal_params['n_epochs'], 
        ),
    file=f
    )
    best_score = trials.trials[best_trial_idx]['result']['loss']
    print('Best validation NDCG@10 = {}'.format(best_score), file=f)

    # retrain the model
    print('Retraining model using the optimal params...', file=f)
    rec.train_model(**optimal_params, early_stopping=False, verbose=False, seed=SEED)
    print(f'Validation NDCG@10 of the retrained model = {rec.get_validation_ndcg()}', file=f)

    # print test metrics
    print('Test results of the retrained model:', file=f)
    test_res = get_test_results(rec, TEST_K)
    print(test_res, file=f)
    write_results_to_excel(test_res, test_res_path, data)

    f.close()

# COMMAND ----------

tune_wide_deep('movielens_100k')

# tune_wide_deep('adobe_core5')
# memory failure

# COMMAND ----------

# if in databricks, save result to ADLS
if IN_DATABRICKS:
    dbutils.fs.cp(
        'file:/tmp/model_evaluation_data/res/wide_deep_recommender_tuning_results.txt', 
        'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/res/wide_deep_recommender_tuning_results.txt', 
        True
    )
    dbutils.fs.cp(
        'file:/tmp/model_evaluation_data/res/test_results_WideDeepRecommender.xlsx', 
        'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/res/test_results_WideDeepRecommender.xlsx', 
        True
    )

# COMMAND ----------


