from sentence_transformers import util

import os
import pandas as pd
import numpy as np
from utils.data_importer import DataImporter
from utils.evaluation import Evaluation
from core import MAIN_DIRECTORY

def get_similar_articles(query_num, embed_mat, k, article_info, item_num2id):
    query_embed = embed_mat[query_num]
    query_name = item_num2id[query_num]
    query_text = article_info.loc[article_info['pagename_extracted'] == query_name, 'body_t'].values
    sim_items = semantic_search(query_embed, embed_mat, k)
    sim_items_name = [item_num2id[i] for i in sim_items]
    sim_items_text = article_info.loc[article_info['pagename_extracted'].isin(sim_items_name), 'body_t'].tolist()

    return query_text, sim_items_text


def semantic_search(query_embed, all_embed, k):
    """
    Input: 
        * query_embed: embedding of the query text, assuming query is part of corpus
        * all_embed: embedding of all text in corpus
        * k: k recommendations returned
    
    Returns:
        dict of similar text corpus id: similarity score
    """
    # get the corpus index and the cosine similarity 
    hits = util.semantic_search(query_embed, all_embed, top_k = k+1, query_chunk_size = 100,
                        corpus_chunk_size = 100000, score_function = util.cos_sim)[0][1:] 
                        # exclude the query itself
    index_list = [d['corpus_id'] for d in hits]
    # score_list = [d['score'] for d in hits]
    
    return index_list

class ContentBasedRecommender:
    def __init__(self, dataset_name, 
                data_folder_path=os.path.join(MAIN_DIRECTORY, 'data_clean')):
        DI = DataImporter(dataset_name, data_folder_path)
        train, val_user, val_ytrue, test_user, test_ytrue, item_embed = DI.get_data('content')

        train['timestamp'] = pd.to_datetime(train['timestamp'])
        # get the latest interacted item for each user in the training set
        train_latest_interaction = train.sort_values(['user_num', 'timestamp'], ascending=False).groupby('user_num').head(1).reset_index()[['user_num', 'item_num']]
        latest_items = dict(zip(
            train_latest_interaction['user_num'].tolist(), 
            train_latest_interaction['item_num'].tolist(), 
        ))
        # get all interactions for each user in the training set
        interacted_items = train.groupby('user_num')['item_num'].agg(lambda x: list(set(x))).reset_index()
        interacted_items = dict(zip(
            interacted_items['user_num'].tolist(), 
            interacted_items['item_num'].tolist(), 
        ))

        self.latest_items = latest_items
        self.interacted_items = interacted_items
        self.val_user = val_user
        self.val_ytrue = val_ytrue
        self.test_user = test_user
        self.test_ytrue = test_ytrue
        self.item_embed = item_embed

    
    def get_recommendation(self, user_list, k):
        pred_ls = []
        for u in user_list:
            latest_item = self.latest_items[u]
            interacted_items = self.interacted_items[u]
            sim_items = semantic_search(
                self.item_embed[latest_item], 
                self.item_embed, 
                k+len(interacted_items)
            )
            y_pred = [i for i in sim_items if i not in interacted_items][:k]
            pred_ls.append(y_pred)
        return pred_ls
    
    
    def get_validation_ndcg(self, k=10):
        """compute ndcg@10 for validation users"""
        pred_ls = self.get_recommendation(self.val_user, k=k)
        val_evaluator = Evaluation(batch_size = len(self.val_user), K=[k], how='val')
        mean_ndcg = val_evaluator.evaluate(pred_ls, self.val_ytrue)
        return mean_ndcg[0]


    def get_test_metrics(self, K):
        pred_ls = self.get_recommendation(self.test_user, max(K))
        test_evaluator = Evaluation(batch_size = len(self.test_user), K=K, how='test')
        mean_precision, mean_recall, HR, mean_ndcg, MRR, MAP = test_evaluator.evaluate(pred_ls, self.test_ytrue)
        return mean_precision, mean_recall, HR, mean_ndcg, MRR, MAP