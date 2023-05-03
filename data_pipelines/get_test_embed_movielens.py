# Databricks notebook source
# MAGIC %%capture
# MAGIC !pip install sentence-transformers torch

# COMMAND ----------

from sentence_transformers import SentenceTransformer, util
import torch

import os
import pickle
import pandas as pd

ML_DATA_RAW_PATH = '/tmp/model_evaluation_data/data_raw/movielens_100k'
DATA_CLEAN_PATH = '/tmp/model_evaluation_data/data_clean'
RES_PATH = '/tmp/model_evaluation_data/res'

dbutils.fs.mkdirs('file:'+RES_PATH)

# COMMAND ----------

model1_name = 'fine_tuned_Bio_Clinical_Bert_'
model2_name = 'Bio_ClinicalBERT'
model3_name = 'bert-base-uncased'
model4_name = 'all-mpnet-base-v2'
model5_name = 'all-MiniLM-L6-v2'

# bert
dbutils.fs.cp('abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/hf-models/bert-base-uncased', 'file:/tmp/bert-base-uncased', True)
#sentence-transformer for sentence similarity pre-trained
dbutils.fs.cp('abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/hf-models/sentence-transformers/all-mpnet-base-v2', 'file:/tmp/all-mpnet-base-v2', True)
dbutils.fs.cp('abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/hf-models/sentence-transformers/all-MiniLM-L6-v2', 'file:/tmp/all-MiniLM-L6-v2', True)

# load article text
dbutils.fs.cp('abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/data_raw/movielens_100k', 
              'file:/tmp/model_evaluation_data/data_raw/movielens_100k', True)

# COMMAND ----------

ml_articles = pd.read_csv(os.path.join(ML_DATA_RAW_PATH, 'item_meta.csv'))
ml_articles['text'] = ml_articles['title'].str.split(' ', expand=True)[0]

ml_articles['text'].head()

# COMMAND ----------

def generate_article_embedding(
    model_name, model_path, articles:pd.DataFrame, 
    text_col='text', id_col='itemID'
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
for model_name in [model3_name, model4_name, model5_name]:
    model_embed = generate_article_embedding(model_name, '/tmp/'+model_name, ml_articles)
    all_embed.update(model_embed)

# COMMAND ----------

print(all_embed.keys())

# COMMAND ----------

# save
with open(os.path.join(ML_DATA_RAW_PATH, 'item_embedding.pkl'), 'wb') as f:
    pickle.dump(all_embed, f, protocol=pickle.HIGHEST_PROTOCOL)

dbutils.fs.cp('file:'+os.path.join(ML_DATA_RAW_PATH, 'item_embedding.pkl'), 
'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/data_raw/movielens_100k/item_embedding.pkl', 
              True)

# COMMAND ----------

# MAGIC %md
# MAGIC match embedding with movielens items

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

# load movielens_100k item_id2num mapping
dbutils.fs.cp('abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/data_clean/movielens_100k', 
              'file:/tmp/model_evaluation_data/data_clean/movielens_100k', True)

# COMMAND ----------

data = 'movielens_100k'
with open(os.path.join(DATA_CLEAN_PATH, data, 'mapping_id2num.pkl'), 'rb') as f:
    mapping = pickle.load(f)
item_id2num = mapping['item_id2num']


rec = ContentBasedRecommender(data, DATA_CLEAN_PATH)
for model_name in [
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
    write_results_to_excel(res, os.path.join(RES_PATH, 'test_results_ContentBasedRecommender_ml.xlsx'), model_name)

# COMMAND ----------

item_embed_mat = create_item_embed_matrix(all_embed['all-mpnet-base-v2'], item_id2num)
save_info = {'item_embedding_matrix': item_embed_mat}
with open(os.path.join(DATA_CLEAN_PATH, 'movielens_100k', 'item_embedding.pkl'), 'wb') as f:
    pickle.dump(save_info, f)

# COMMAND ----------

# use none text features
from sklearn.preprocessing import MultiLabelBinarizer

item_feature = pd.read_csv(os.path.join(DATA_CLEAN_PATH, 'movielens_100k', 'item_feature.csv'))[['item_num', 'genre', 'year']]

genre_ls = item_feature['genre'].str.split('|').map(lambda x: list(set(x)))
mlb = MultiLabelBinarizer(sparse_output=True)
feature_genre = pd.DataFrame.sparse.from_spmatrix(
    mlb.fit_transform(genre_ls),
    index=item_feature.index,
    columns=mlb.classes_
).reset_index(drop=True).iloc[:,1:]

item_feature = pd.concat([item_feature[['item_num', 'year']], feature_genre], axis=1)
item_feature = pd.get_dummies(item_feature, columns=['year']).sort_values('item_num', ascending=True)
item_feature = item_feature.drop('item_num', axis=1).values

rec = ContentBasedRecommender('movielens_100k', DATA_CLEAN_PATH)
rec.item_embed = item_feature.astype(np.float32)
mean_ndcg = rec.get_validation_ndcg()
res = get_test_results(rec, TEST_K)
print('genre + year', mean_ndcg)
print(res)
write_results_to_excel(res, os.path.join(RES_PATH, 'test_results_ContentBasedRecommender_ml.xlsx'), 'genre_year')

# COMMAND ----------

# embedding + genre + year
rec = ContentBasedRecommender('movielens_100k', DATA_CLEAN_PATH)
item_embed_mat = create_item_embed_matrix(all_embed['all-mpnet-base-v2'], item_id2num)
item_embed_feature = np.concatenate((item_embed_mat, item_feature), axis=-1)
assert item_embed_feature.shape[1] == 768+item_feature.shape[1]

rec.item_embed = item_feature.astype(np.float32)
mean_ndcg = rec.get_validation_ndcg()
res = get_test_results(rec, TEST_K)
print('embed + genre + year', mean_ndcg)
print(res)

# COMMAND ----------

dbutils.fs.cp('file:'+os.path.join(RES_PATH, 'test_results_ContentBasedRecommender_ml.xlsx'), 
'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/data_clean/res/test_results_ContentBasedRecommender_ml.xlsx', 
              True)

# COMMAND ----------


