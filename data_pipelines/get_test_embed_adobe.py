# Databricks notebook source
# MAGIC %%capture
# MAGIC !pip install sentence-transformers torch openpyxl

# COMMAND ----------

from sentence_transformers import SentenceTransformer, util
import torch

import os
import pickle
import requests
import pandas as pd

ADOBE_DATA_RAW_PATH = '/tmp/model_evaluation_data/data_raw/adobe'
DATA_CLEAN_PATH = '/tmp/model_evaluation_data/data_clean'
RES_PATH = '/tmp/model_evaluation_data/res'

dbutils.fs.mkdirs('file:'+RES_PATH)

# COMMAND ----------

# MAGIC %md 
# MAGIC Generate embeddings for adobe articles

# COMMAND ----------

model1_name = 'fine_tuned_Bio_Clinical_Bert_'
model2_name = 'Bio_ClinicalBERT'
model3_name = 'bert-base-uncased'
model4_name = 'all-mpnet-base-v2'
model5_name = 'all-MiniLM-L6-v2'

# load fine-tuned Bio-clinicalBERT 
dbutils.fs.cp('abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/QA/fine-tune-models/health/fine_tuned_Bio_Clinical_Bert_/', 'file:/tmp/fine_tuned_Bio_Clinical_Bert_', True)
#pretrained Bio-clinical
dbutils.fs.cp('abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/hf-models/emilyalsentzer/Bio_ClinicalBERT', 'file:/tmp/Bio_ClinicalBERT', True)
# bert
dbutils.fs.cp('abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/hf-models/bert-base-uncased', 'file:/tmp/bert-base-uncased', True)
#sentence-transformer for sentence similarity pre-trained
dbutils.fs.cp('abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/hf-models/sentence-transformers/all-mpnet-base-v2', 'file:/tmp/all-mpnet-base-v2', True)
dbutils.fs.cp('abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/hf-models/sentence-transformers/all-MiniLM-L6-v2', 'file:/tmp/all-MiniLM-L6-v2', True)

# load article text
dbutils.fs.cp('abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/data_raw/adobe', 
              'file:/tmp/model_evaluation_data/data_raw/adobe', True)

adobe_articles = pd.read_csv(os.path.join(ADOBE_DATA_RAW_PATH, 'page_meta.csv'))
adobe_articles['text'] = adobe_articles[['pagename_extracted', 'body_t']].fillna('').agg(' '.join, axis=1)

# COMMAND ----------

adobe_articles['text']

# COMMAND ----------

def generate_article_embedding(
    model_name, model_path, articles:pd.DataFrame, 
    text_col='text', id_col='pagename_extracted'
):
    """
    generate vector embeddings for each given article

    Args:
        model_name
        articles: a list of article strings
        model_path: where the model is stored
        text_col: column name in articles which to store the text info to embed
        id_col: id of each article, i.e. name
    
    Commands: 
        1. Starts pool to do multi_process_pool for faster encoding
        2. Encodes the articles fed into the function
        3. Stops the Pool 
        8. Returns vector embeddings
    """
    article_text = articles[text_col].tolist()

    # use gpu
    model = SentenceTransformer(model_path, device='cuda')

    #Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool()

    #Compute the embeddings using the multi-process pool
    emb = model.encode_multi_process(article_text, pool)
    #Optional: Stop the proccesses in the pool
    model.stop_multi_process_pool(pool)
    
    # map article title to embedding
    name2embed = dict(zip(articles[id_col].tolist(), emb))

    return {model_name: name2embed}

# COMMAND ----------

all_embed  = dict()
for model_name in [model1_name, model2_name, model3_name, model4_name, model5_name]:
    model_embed = generate_article_embedding(model_name, '/tmp/'+model_name, adobe_articles)
    all_embed.update(model_embed)

# COMMAND ----------

print(all_embed.keys())

# COMMAND ----------

dbutils.fs.rm('file:'+os.path.join(ADOBE_DATA_RAW_PATH, 'item_embedding.pkl'))

with open(os.path.join(ADOBE_DATA_RAW_PATH, 'item_embedding.pkl'), 'wb') as f:
    pickle.dump(all_embed, f, protocol=pickle.HIGHEST_PROTOCOL)

dbutils.fs.cp('file:'+os.path.join(ADOBE_DATA_RAW_PATH, 'item_embedding.pkl'), 
'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/data_raw/adobe/item_embedding.pkl', 
              True)

# COMMAND ----------

# MAGIC %md 
# MAGIC match article embeddings with adobe and adobe_core5 dataset

# COMMAND ----------

from core.content_based_model.content_based_recommender import ContentBasedRecommender
from utils.evaluation import get_test_results, write_results_to_excel
from core import TEST_K

import numpy as np
import scipy.sparse as sps

def create_item_embed_matrix(name_embed_dict:dict, item_id2num:dict):
    """create a matrix of item x embedding, sorted by item num"""
    item_num2id = {v:k for k,v in item_id2num.items()}
    embed_mat = np.array([name_embed_dict[item_num2id[i]] for i in range(len(item_id2num))])
    return embed_mat

# COMMAND ----------

# load article embeddings
dbutils.fs.cp('abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/data_raw/adobe/item_embedding.pkl', 
'file:'+os.path.join(ADOBE_DATA_RAW_PATH, 'item_embedding.pkl'), 
              True)

with open(os.path.join(ADOBE_DATA_RAW_PATH, 'item_embedding.pkl'), 'rb') as f:
    all_embed = pickle.load(f)

# load adobe and adobe_core5 item_id2num mapping
dbutils.fs.cp('abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/data_clean', 
              'file:/tmp/model_evaluation_data/data_clean', True)

# COMMAND ----------

# DBTITLE 1,test content based recommender using different embeddings
data = 'adobe_core5'
with open(os.path.join(DATA_CLEAN_PATH, data, 'mapping_id2num.pkl'), 'rb') as f:
    mapping = pickle.load(f)
item_id2num = mapping['item_id2num']


rec = ContentBasedRecommender(data, DATA_CLEAN_PATH)
for model_name in [
    'fine_tuned_Bio_Clinical_Bert_', 
    'Bio_ClinicalBERT', 
    'bert-base-uncased', 
    'all-mpnet-base-v2',
    'all-MiniLM-L6-v2'
]:
    item_embed_mat = create_item_embed_matrix(all_embed[model_name], item_id2num)
    rec.item_embed = item_embed_mat
    mean_ndcg = rec.get_validation_ndcg()
    res = get_test_results(rec, TEST_K)
    print(model_name, mean_ndcg)
    print(res)
    write_results_to_excel(res, os.path.join(RES_PATH, 'test_results_ContentBasedRecommender.xlsx'), model_name)

# COMMAND ----------

# test tdidf vectors 
item_feature = sps.load_npz(os.path.join(DATA_CLEAN_PATH, 'adobe_core5', 'item_feature.npz'))
item_feature = item_feature[:, item_feature.shape[0]:-82]
assert item_feature.shape[1] == 7595

rec = ContentBasedRecommender('adobe_core5', DATA_CLEAN_PATH)
rec.item_embed = item_feature.todense()
mean_ndcg = rec.get_validation_ndcg()
res = get_test_results(rec, TEST_K)
print('tdidf', mean_ndcg)
print(res)
write_results_to_excel(res, os.path.join(RES_PATH, 'test_results_ContentBasedRecommender.xlsx'), 'tfidf')

# COMMAND ----------

# tfidf + topic
item_feature = sps.load_npz(os.path.join(DATA_CLEAN_PATH, 'adobe_core5', 'item_feature.npz'))
item_feature = item_feature[:, item_feature.shape[0]:]
assert item_feature.shape[1] == 82+7595

rec = ContentBasedRecommender('adobe_core5', DATA_CLEAN_PATH)
rec.item_embed = item_feature.todense()
mean_ndcg = rec.get_validation_ndcg()
res = get_test_results(rec, TEST_K)
print('tfidf + topic', mean_ndcg)
print(res)
write_results_to_excel(res, os.path.join(RES_PATH, 'test_results_ContentBasedRecommender.xlsx'), 'tfidf_topic')

# COMMAND ----------

# test using none text feature
item_feature = sps.load_npz(os.path.join(DATA_CLEAN_PATH, 'adobe_core5', 'item_feature.npz'))
item_feature = item_feature[:, -82:]
assert item_feature.shape[1] == 82

rec = ContentBasedRecommender('adobe_core5', DATA_CLEAN_PATH)
rec.item_embed = item_feature.todense()
mean_ndcg = rec.get_validation_ndcg()
res = get_test_results(rec, TEST_K)
print('only topic', mean_ndcg)
print(res)
write_results_to_excel(res, os.path.join(RES_PATH, 'test_results_ContentBasedRecommender.xlsx'), 'topic')

# COMMAND ----------

# use embedding of all-mpnet-base-v2 in hybrid recommenders
item_embed_mat = create_item_embed_matrix(all_embed['all-mpnet-base-v2'], item_id2num)

save_info = {'item_embedding_matrix': item_embed_mat}
dbutils.fs.rm('file:'+os.path.join(DATA_CLEAN_PATH, 'adobe_core5', 'item_embedding.pkl'))
with open(os.path.join(DATA_CLEAN_PATH, 'adobe_core5', 'item_embedding.pkl'), 'wb') as f:
    pickle.dump(save_info, f, protocol=pickle.HIGHEST_PROTOCOL)

dbutils.fs.cp('file:'+os.path.join(DATA_CLEAN_PATH, 'adobe_core5', 'item_embedding.pkl'), 
'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/data_clean/adobe_core5/item_embedding.pkl', 
              True)

# COMMAND ----------

dbutils.fs.cp('file:'+os.path.join(RES_PATH, 'test_results_ContentBasedRecommender.xlsx'), 
'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/data_clean/res/test_results_ContentBasedRecommender.xlsx', 
              True)

# COMMAND ----------


