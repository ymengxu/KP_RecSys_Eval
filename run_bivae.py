# Databricks notebook source
import os
import numpy as np

from core import TEST_K, SEED
from utils.evaluation import write_results_to_excel, get_test_results
from core.neural_based_methods.bivae.bivae_recommender import BiVAERecommender

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

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL
import numpy as np

space = {
    'k': hp.quniform('k', 10, 100, 1),
    'batch_size': hp.choice('batch_size', [64, 128, 256, 512, 1024]),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)), 
    'act_fn': hp.choice('act_fn', ['sigmoid', 'tanh', 'elu', 'relu', 'relu6']),
    'likelihood': hp.choice('likelihood', ['bern', 'pois', 'gaus']),
    # 'n_epochs': hp.quniform('n_epochs', 100, 600, 100)
}

# COMMAND ----------

def tune_bivae(data,
    tune_res_path=os.path.join(RES_PATH, 'bivae_recommender_tuning_results.txt'),
    test_res_path=os.path.join(RES_PATH, 'test_results_BiVAERecommender.xlsx'),
    verbose=False,
):
    """tuning method unique for bivae: at each hyperparam combination, train bivae for 100, 200, ..., 600 epochs, use the epoch with best validation score"""
    rec = BiVAERecommender(data, DATA_CLEAN_PATH)
    epoch_list = [100,200,300,400,500,600]

    def objective(params):
        try:
            eval_score = np.zeros((len(epoch_list),1))
            for i in range(len(epoch_list)):
                rec.train_model(**params, n_epochs=epoch_list[i], verbose=verbose)
                eval_score[i] = rec.get_validation_ndcg()
            best_eval_score = np.max(eval_score)
            best_epoch = epoch_list[np.argmax(eval_score)]
            return {
                'loss':-best_eval_score, 
                'status':STATUS_OK,
                'best_epoch': best_epoch
            }
        except:
            return {
                'loss':0, 
                'status':STATUS_FAIL,
                'best_epoch': 0
            }

    print('Tuning hyperparameters of BiVAE Recommender on dataset {}...'.format(data))
    trials = Trials()
    trials._random_state = np.random.RandomState(SEED)
    best = fmin(objective, space, algo=tpe.suggest, max_evals=50, trials=trials)

    # save result
    if os.path.exists(tune_res_path):
        f = open(tune_res_path, 'a')
    else:
        f = open(tune_res_path, 'w')

    print(f'BiVAE Recommender tuning results on dataset {data}...', 
        file=f)

    # get the best trial information
    best_trial_idx = np.argmin([trial_info['result']['loss'] for trial_info in trials.trials])

    batch_size_ls = [64, 128, 256, 512, 1024]
    act_fn_ls = ['sigmoid', 'tanh', 'elu', 'relu', 'relu6']
    likelihhod_ls = ['bern', 'pois', 'gaus']

    optimal_params = {
        'k': trials.trials[best_trial_idx]['misc']['vals']['k'][0],
        'batch_size': batch_size_ls[trials.trials[best_trial_idx]['misc']['vals']['batch_size'][0]],
        'learning_rate': trials.trials[best_trial_idx]['misc']['vals']['learning_rate'][0],
        'act_fn': act_fn_ls[trials.trials[best_trial_idx]['misc']['vals']['act_fn'][0]],
        'likelihood': likelihhod_ls[trials.trials[best_trial_idx]['misc']['vals']['likelihood'][0]],
        'n_epochs': trials.trials[best_trial_idx]['result']['best_epoch']
    }

    print('The optimal hyperparamters are: \nk={}, batch_size={}, learning_rate={}, act_fn={}, likelihood={}, n_epochs={}'.format(
        optimal_params['k'], optimal_params['batch_size'], optimal_params['learning_rate'], 
        optimal_params['act_fn'], optimal_params['likelihood'],
        optimal_params['n_epochs']
    ), file=f)
    best_score = -np.min([trial_info['result']['loss'] for trial_info in trials.trials])
    print('Best validation NDCG@10 = {}'.format(best_score), file=f)

    # retrain the model
    print('Retraining model using the optimal params...', file=f)
    rec.train_model(**optimal_params, early_stopping=False, verbose=False)
    print(f'Validation NDCG@10 of the retrained model={rec.get_validation_ndcg()}', file=f)

    # print test metrics
    print('Test results of the retrained model:', file=f)
    test_res = get_test_results(rec, TEST_K)
    print(test_res, file=f)
    write_results_to_excel(test_res, test_res_path, data)

    # # save the model
    # print('Saving the retrained model...')
    # model_info = {'model': rec.model, 'params': optimal_params}
    # with open(os.path.join(SAVEMODEL_PATH, f'bivae_{data}.pkl'), 'wb') as flp:
    #     pickle.dump(model_info, flp, protocol=pickle.HIGHEST_PROTOCOL)
    # print('Model saved. ')

    f.close()

# COMMAND ----------

tune_bivae('movielens_100k')
tune_bivae('adobe_core5')

# COMMAND ----------

# if in databricks, save result to ADLS
if IN_DATABRICKS:
    dbutils.fs.cp(
        'file:/tmp/model_evaluation_data/res/bivae_recommender_tuning_results.txt', 
        'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/res/bivae_recommender_tuning_results.txt', 
        True
    )
    dbutils.fs.cp(
        'file:/tmp/model_evaluation_data/res/test_results_BiVAERecommender.xlsx', 
        'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/res/test_results_BiVAERecommender.xlsx', 
        True
    )
