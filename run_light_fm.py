# Databricks notebook source
# MAGIC %%capture
# MAGIC !pip install openpyxl

# COMMAND ----------

import os
import pickle
import numpy as np

from core import SEED, TEST_K, DATA_CLEAN_PATH, RES_PATH, SAVEMODEL_PATH
from core.light_fm.light_fm_model import LightFMRecommender
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

    dbutils.fs.mkdirs('file:'+DATA_CLEAN_PATH)
    dbutils.fs.mkdirs('file:'+RES_PATH)

# COMMAND ----------

# DBTITLE 1,Fine tuning using skopt.forest_minimize
SPACE_lightFM = [
    Integer(10,200,name='no_components',prior='uniform'),
    Integer(100,300,name='epochs',prior='uniform'),
    Real(0.00001,0.1,name='learning_rate',prior='log-uniform'),
    Real(0.00001,0.1,name='item_alpha',prior='log-uniform'),
    Real(0.00001,0.1,name='user_alpha',prior='log-uniform')
]

for data in ['adobe', 'adobe_core5', 'mind_small']:
    print('Tuning hyperparameters of LightFM model on dataset', data)
    print('...')
    lightfm_rec = LightFMRecommender(data)
    res_warp = forest_minimize(lightfm_rec.obj_warp, SPACE_lightFM, n_calls=50,
                     random_state=2023, n_jobs=-1, verbose=True)
    res_bpr = forest_minimize(lightfm_rec.obj_bpr, SPACE_lightFM, n_calls=50,
                     random_state=2023, n_jobs=-1, verbose=True)
    if res_warp['fun'] < res_bpr['fun']:
        loss = 'warp'
        res = res_warp
    else:
        loss = 'bpr'
        res = res_bpr
    print('The optimal hyperparamters are:', loss, res['x'])
    print('Best validation NDCG@10 =', res['fun'])
    print('Retraining model using the optimal hyperparamters found...')
    print('...')
    lightfm_rec.train_model(loss, res['x'])
    print('Validation NDCG@10 of the retrained model =', lightfm_rec.evaluate())

    print('Saving the retrained model...')
    # save the model
    model_info = {'model': lightfm_rec.model, 'params': [loss]+res['x']}
    with open(os.path.join(MODEL_SAVE_PATH, 'lightfm_{data}.pkl'), 'wb') as flp:
        pickle.dump(model_info, flp, protocol=pickle.HIGHEST_PROTOCOL)
    print('Model saved.')

# COMMAND ----------

# MAGIC %md 
# MAGIC # Test     
# MAGIC compare between td-idf and transformer embedding

# COMMAND ----------

# DBTITLE 1,item tf-idf (0.0217)
# item tf-idf
rec = LightFMRecommender('adobe_core5', DATA_CLEAN_PATH)
print(rec.item_feature.shape)

params = {
    'loss':'warp', 
    'no_components':10, 
    'learning_rate':0.001, 
    'item_alpha':0.001, 
    'user_alpha':0.001 
}

rec.train_model(**params, early_stopping=True, verbose=True)
# 0.021711004497864037

# COMMAND ----------

# DBTITLE 1,item transformer embedding (0.0211)
# item transformer embedding
import scipy.sparse as sps

rec = LightFMRecommender('adobe_core5', DATA_CLEAN_PATH)
with open(os.path.join(DATA_CLEAN_PATH, 'adobe_core5', 'item_embedding.pkl'), 'rb') as f:
    item_embed = pickle.load(f)['item_embedding_matrix']
item_embed = sps.csr_matrix(item_embed)
item_identity = sps.identity(item_embed.shape[0])
item_topic = rec.item_feature[:, -82:]
rec.item_feature = sps.hstack([item_identity, item_embed, item_topic])
print(rec.item_feature.shape)

params = {
    'loss':'warp', 
    'no_components':10, 
    'learning_rate':0.001, 
    'item_alpha':0.001, 
    'user_alpha':0.001 
}

rec.train_model(**params, early_stopping=True, verbose=True)
# 0.021144924203205146

# COMMAND ----------

# DBTITLE 1,item tf-idf + user features (0.0257)
# item tf-idf + user features
rec = LightFMRecommender('adobe_core5', DATA_CLEAN_PATH)
print(rec.user_feature.shape)
print(rec.item_feature.shape)

params = {
    'loss':'warp', 
    'no_components':10, 
    'learning_rate':0.001, 
    'item_alpha':0.001, 
    'user_alpha':0.001 
}

rec.train_model(**params, early_stopping=True, verbose=True)
# 0.02565214710643036

# COMMAND ----------

# DBTITLE 1,item transformer embedding + user features (0.0197)
# item transformer embedding + user features
import scipy.sparse as sps

rec = LightFMRecommender('adobe_core5', DATA_CLEAN_PATH)
with open(os.path.join(DATA_CLEAN_PATH, 'adobe_core5', 'item_embedding.pkl'), 'rb') as f:
    item_embed = pickle.load(f)['item_embedding_matrix']
item_embed = sps.csr_matrix(item_embed)
item_identity = sps.identity(item_embed.shape[0])
item_topic = rec.item_feature[:, -82:]
rec.item_feature = sps.hstack([item_identity, item_embed, item_topic])
print(rec.item_feature.shape)
print(rec.user_feature.shape)

params = {
    'loss':'warp', 
    'no_components':10, 
    'learning_rate':0.001, 
    'item_alpha':0.001, 
    'user_alpha':0.001 
}

rec.train_model(**params, early_stopping=True, verbose=True)
# 0.019718542970094684

# COMMAND ----------

# MAGIC %md
# MAGIC # Fine tuning using hyperopt

# COMMAND ----------

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL

space = {
    'loss': hp.choice('loss', ['bpr', 'warp']),
    'no_components': hp.quniform('no_components', 10, 200, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)), 
    'user_alpha': hp.loguniform('user_alpha', np.log(0.00001), np.log(0.1)),
    'item_alpha': hp.loguniform('item_alpha', np.log(0.00001), np.log(0.1))
}

# COMMAND ----------

def tune_light_fm(data, 
    use_text_feature=True,
    use_no_feature=False,
    use_only_text=False,
    tune_res_path=os.path.join(RES_PATH, 'light_fm_recommender_tuning_results.txt'),
    test_res_path=os.path.join(RES_PATH, 'test_results_LightFMRecommender.xlsx'),
):

    rec = LightFMRecommender(data, DATA_CLEAN_PATH, use_text_feature=use_text_feature, use_no_feature=use_no_feature,
                                use_only_text=use_only_text)
    print(rec.user_feature.shape)
    print(rec.item_feature.shape)

    def objective(params):
        try:
            rec.train_model(**params, early_stopping=True, verbose=False)
            mean_ndcg = rec.get_validation_ndcg()
            return {
                'loss':-mean_ndcg, 
                'status':STATUS_OK,
                'best_epoch':rec.params['best_epoch']
            }
        except:
            return {
                'loss': 0,
                'status':STATUS_FAIL,
                'best_epoch':0
            }


    print('Tuning hyperparameters of LightFM Recommender on dataset {}'.format(data))
    print(f'use_text_feature={use_text_feature}, use_no_feature={use_no_feature}, use_only_text={use_only_text}')
    trials = Trials()
    trials._random_state = np.random.RandomState(SEED)
    best = fmin(objective, space, algo=tpe.suggest, max_evals=50, trials=trials)

    # save result
    if os.path.exists(tune_res_path):
        f = open(tune_res_path, 'a')
    else:
        f = open(tune_res_path, 'w')
    
    print(
        f'LightFM Recommender tuning results on dataset {data} with use_text_feature={use_text_feature} and use_no_feature={use_no_feature} and use_only_text={use_only_text}...', 
        file=f
    )

    # get the best trial information
    best_trial_idx = np.argmin([trial_info['result']['loss'] for trial_info in trials.trials])

    optimal_params = {
        'loss': ['bpr', 'warp'][trials.trials[best_trial_idx]['misc']['vals']['loss'][0]], 
        'no_components': trials.trials[best_trial_idx]['misc']['vals']['no_components'][0], 
        'learning_rate': trials.trials[best_trial_idx]['misc']['vals']['learning_rate'][0],
        'user_alpha': trials.trials[best_trial_idx]['misc']['vals']['user_alpha'][0],
        'item_alpha': trials.trials[best_trial_idx]['misc']['vals']['item_alpha'][0],
        'epochs': trials.trials[best_trial_idx]['result']['best_epoch']
    }

    print('The optimal hyperparamters are: \nloss={}, no_components={}, learning_rate={}, user_alpha={}, item_alpha={}, epochs={}'.format(
        optimal_params['loss'], optimal_params['no_components'], optimal_params['learning_rate'],
        optimal_params['user_alpha'], optimal_params['item_alpha'],
        optimal_params['epochs']
    ), file=f)
    print('Best validation NDCG@10 = {}'.format(-trials.trials[best_trial_idx]['result']['loss']), file=f)

    # retrain the model
    print('Retraining the model using optimal parameters...', file=f)
    rec.train_model(**optimal_params, early_stopping=False)
    print('Validation NDCG@10 of the retrained model =', rec.get_validation_ndcg(), file=f)

    # print test metrics
    print('Test results of the retrained model:', file=f)
    test_res = get_test_results(rec, TEST_K)
    print(test_res, file=f)
    
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

tune_light_fm('movielens_100k', use_text_feature=True, use_no_feature=False, use_only_text=False)
tune_light_fm('adobe_core5', use_text_feature=True, use_no_feature=False, use_only_text=False)
# all feature

tune_light_fm('adobe_core5', use_text_feature=False, use_no_feature=False, use_only_text=False)
# no text

tune_light_fm('adobe_core5', use_text_feature=False, use_no_feature=True, use_only_text=False)
# no feature

tune_light_fm('adobe_core5', use_text_feature=True, use_no_feature=False, use_only_text=True)
# only text

tune_light_fm('movielens_100k', use_text_feature=False, use_no_feature=True, use_only_text=False)
# no feature

# COMMAND ----------

if IN_DATABRICKS: 
    dbutils.fs.cp(
        'file:/tmp/model_evaluation_data/res/test_results_LightFMRecommender.xlsx', 
        'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/res/test_results_LightFMRecommender.xlsx', 
        True
    )

    dbutils.fs.cp(
        'file:/tmp/model_evaluation_data/res/light_fm_recommender_tuning_results.txt', 
        'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/res/light_fm_recommender_tuning_results.txt', 
        True
    )
