# Databricks notebook source
# MAGIC %%capture
# MAGIC !pip install tf_slim

# COMMAND ----------

import os
from time import time
import pandas as pd

from core import TEST_K, DATA_CLEAN_PATH, RES_PATH
from utils.evaluation import get_test_results, write_results_to_excel
from utils.utils_params_helper import get_train_eval_time

import warnings
warnings.filterwarnings("ignore")

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

# MAGIC %md
# MAGIC ## Evaluate the training and evaluation time on adobe_core5 for the following models: 
# MAGIC - Popularity
# MAGIC - UserKNN, ItemKNN
# MAGIC - iALS
# MAGIC - BPR (cornac)
# MAGIC - NCF
# MAGIC - BiVAE
# MAGIC - LightGCN
# MAGIC - LightFM
# MAGIC - PinSAGE
# MAGIC - Tensor
# MAGIC - Wide&Deep

# COMMAND ----------

MODEL_NAMES = [
    'Popularity', 
    'UserKNN', 'ItemKNN',
    'iALS', 'BPR', 
    'NCF', 'BiVAE', 'LightGCN',
    'LightFM', 'PinSage',
    # 'Tensor', 'Wide and Deep', 'Two tower'
]

train_time_ls = []
eval_time_ls = []
for model in MODEL_NAMES: 
    train_time, eval_time = get_train_eval_time(model, data_folder=DATA_CLEAN_PATH)
    train_time_ls.append(train_time)
    eval_time_ls.append(eval_time)

time_df = pd.DataFrame({
    'model': MODEL_NAMES,  
    'train time': train_time_ls, 
    'evaluation time': eval_time_ls 
})

write_results_to_excel(time_df, os.path.join(RES_PATH, 'train_val_time.xlsx'), 'time')

# COMMAND ----------


