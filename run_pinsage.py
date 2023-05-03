# Databricks notebook source
# MAGIC %%capture
# MAGIC !pip install dgl
# MAGIC !pip install openpyxl
# MAGIC !pip install dask

# COMMAND ----------

import os
import numpy as np

from core import SEED, TEST_K, DATA_CLEAN_PATH, RES_PATH
from core.neural_based_methods.pinsage.pinsage_recommender import PinSageRecommender
from utils.evaluation import write_results_to_excel, get_test_results

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

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

space = {
    'random_walk_length': hp.quniform('random_walk_length', 2, 5, 1),
    'random_walk_restart_prob': hp.quniform('random_walk_restart_prob', 0.3, 0.7, 0.1),
    'num_random_walks': hp.quniform('num_random_walks', 5, 15, 1),
    'num_neighbors': hp.quniform('num_neighbors', 3, 10, 1),
    'num_layers': hp.quniform('num_layers', 2, 4, 1),
    'hidden_dims': hp.quniform('hidden_dims', 16, 256, 1),
    'lr': hp.loguniform('lr', np.log(0.00001), np.log(0.1)), 
    'batch_size': hp.choice('batch_size', [64, 128, 256, 512])
}

# COMMAND ----------

def tune_pinsage(
    data,
    use_text_feature=True,
    use_no_feature=False,
    use_only_text=False, 
    tune_res_path=os.path.join(RES_PATH, 'pinsage_recommender_tuning_results.txt'),
    test_res_path=os.path.join(RES_PATH, 'test_results_PinSageRecommender.xlsx'),
    device='cpu',
):

    rec = PinSageRecommender(data, DATA_CLEAN_PATH, 
                             use_text_feature=use_text_feature, use_no_feature=use_no_feature,
                             use_only_text=use_only_text)
    
    def objective(params):
        rec.train_model(**params, device=device, early_stopping=True, verbose=True)
        return {
            'loss':-rec.params['best_eval_score'], 
            'status':STATUS_OK,
            'best_epoch':rec.params['best_epoch']
        }
    
    print('Tuning hyperparameters of PinSage Recommender on dataset {}...'.format(data))
    print(f'use_text_feature={use_text_feature}, use_no_feature={use_no_feature}, use_only_text={use_only_text}')
    trials = Trials()
    trials._random_state = np.random.RandomState(SEED)
    best = fmin(objective, space, algo=tpe.suggest, max_evals=50, trials=trials)

    # save result
    if os.path.exists(tune_res_path):
        f = open(tune_res_path, 'a')
    else:
        f = open(tune_res_path, 'w')

    print(f'PinSage Recommender tuning results on dataset {data} with use_text_feature={use_text_feature} and use_no_feature={use_no_feature} and use_only_text={use_only_text}...', 
        file=f)

    # get the best trial information
    best_trial_idx = np.argmin([trial_info['result']['loss'] for trial_info in trials.trials])
    optimal_params = {
        'random_walk_length': trials.trials[best_trial_idx]['misc']['vals']['random_walk_length'][0],
        'random_walk_restart_prob': trials.trials[best_trial_idx]['misc']['vals']['random_walk_restart_prob'][0], 
        'num_random_walks': trials.trials[best_trial_idx]['misc']['vals']['num_random_walks'][0],
        'num_neighbors': trials.trials[best_trial_idx]['misc']['vals']['num_neighbors'][0],
        'num_layers': trials.trials[best_trial_idx]['misc']['vals']['num_layers'][0],
        'hidden_dims': trials.trials[best_trial_idx]['misc']['vals']['hidden_dims'][0],
        'lr': trials.trials[best_trial_idx]['misc']['vals']['lr'][0], 
        'batch_size': [64, 128, 256, 512][trials.trials[best_trial_idx]['misc']['vals']['batch_size'][0]],
        'epochs': trials.trials[best_trial_idx]['result']['best_epoch']
    }

    print('The optimal hyperparamters are: \nrandom_walk_length={}, random_walk_restart_prob={}, num_random_walks={}, num_neighbors={}, num_layers={}, hidden_dims={}, lr={}, batch_size={}, epochs={}'.format(
            optimal_params['random_walk_length'], 
            optimal_params['random_walk_restart_prob'], 
            optimal_params['num_random_walks'], 
            optimal_params['num_neighbors'], 
            optimal_params['num_layers'], 
            optimal_params['hidden_dims'], 
            optimal_params['lr'],
            optimal_params['batch_size'],
            optimal_params['epochs']
        ), file=f
    )

    best_score = trials.trials[best_trial_idx]['result']['loss']
    print('Best validation NDCG@10 = {}'.format(best_score), file=f)

    # retrain the model
    print('Retraining model using the optimal params...', file=f)
    rec.train_model(**optimal_params, device='cpu', early_stopping=False, verbose=False)
    print(f'Validation NDCG@10 of the retrained model = {rec.get_validation_ndcg()}', file=f)

    # print test metrics
    print('Test results of the retrained model:', file=f)
    test_res = get_test_results(rec, TEST_K)
    print(test_res,file=f)
    
    sheet_name = data
    if use_no_feature:
        sheet_name = data+'_nofeature'
    elif use_text_feature:
        if use_only_text:
            sheet_name = data+'_onlytext'
        else:
            sheet_name = data
    else:
        sheet_name = data+'_notext'
    
    write_results_to_excel(test_res, test_res_path, sheet_name)

    f.close()

# COMMAND ----------

# tune_pinsage('movielens_100k')
# tune_pinsage('adobe_core5')
# # all feature


tune_pinsage('adobe_core5', use_text_feature=False, use_no_feature=False, use_only_text=False)
# no text
tune_pinsage('adobe_core5', use_text_feature=False, use_no_feature=True, use_only_text=False)
# no feature
tune_pinsage('adobe_core5', use_text_feature=True, use_no_feature=False, use_only_text=True)
# only text

# COMMAND ----------

# if in databricks, save result to ADLS
if IN_DATABRICKS:
    dbutils.fs.cp(
        'file:/tmp/model_evaluation_data/res/pinsage_recommender_tuning_results.txt', 
        'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/res/pinsage_recommender_tuning_results.txt', 
        True
    )
    dbutils.fs.cp(
        'file:/tmp/model_evaluation_data/res/test_results_PinSageRecommender.xlsx', 
        'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/res/test_results_PinSageRecommender.xlsx', 
        True
    )
