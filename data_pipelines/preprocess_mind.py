# Databricks notebook source

"""
1. Clean train and val interaction from orginal MIND dataset, with columns ['user_num', 'news_num', 'date_time']
Note: in train.csv, interactions with no "date_time" information are from week1-4, else from week5
2. save a mapping dictionary for user and item id2num
3. get news feature: ID one-hot encoding + title TF-IDF + category one-hot encoding
"""

import os
import pickle

import numpy as np
import pandas as pd
import scipy
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

from core import MAIN_DIRECTORY


# $ python -m spacy download en_core_web_md

# loading in spacy medium model that has 20k word2vec embeddings
nlp = spacy.load('en_core_web_md', disable=['parser', 'ner', 'textcat'])
# load in stop words
stop_words = nlp.Defaults.stop_words
stop_words = list(stop_words)
stop_words = stop_words + ['d', 'll', 'm', 'n', 's', 't', 've', '\n']
stop_words = set(stop_words)


def clean_mind_train_val(mind_train:pd.DataFrame, mind_val:pd.DataFrame):
    """clean the original MIND train and validation dataframe in the following format:
    each row is an interaction (click), including week1-5, columns: ['user_id', 'news_id', 'date-time'].
    The original mind train and validation datasets only have interaction data in week5.
    Interactions in week1-4 are stored as "previous_history" in the two data frames.


    Args:
        mind_train: original behaviors dataframe in "/MINDlarge_train/"
        mind_val: original behaviors dataframe in "/MINDlarge_dev/"

    Returns:
        train: pd dataframe, interactions in week1-4 + week5 Mon-Sat, (week1-4 interactions have no date_time info)
        val: pd dataframe, interactions in week5 Sun
    """
    mind_train = mind_train.copy()
    mind_val = mind_val.copy()

    # pick out clicked articles from impression logs
    for df in [mind_train, mind_val]:
        df.columns = ['user_id', 'date_time', 'previous_history', 'impression_log']
    for df in [mind_train, mind_val]:
        imps = df['impression_log'].tolist()
        imps = [i.split(' ') for i in imps]
        clicks = []
        for imp in imps:
            try:
                clicks.append([i.split('-')[0] for i in imp if i.split('-')[1] == '1'])
            except:
                clicks.append([])
        df['clicked_articles'] = pd.Series(clicks)
        df['n_clicks'] = df['clicked_articles'].map(lambda x: len(x))

    ## train
    # get interactions (clicks) in week1-4
    mind_train_val = pd.concat([mind_train, mind_val])
    mind_pastClicks = mind_train_val.groupby('user_id')['previous_history'].min().reset_index().fillna('')
    mind_pastClicks['previous_history'] = mind_pastClicks['previous_history'].str.split(' ')\
                                    .map(lambda x: [i for i in x if i != ''])
    mind_pastClicks['n_clicks'] = mind_pastClicks['previous_history'].map(lambda x: len(x))
    # remove users with no past clicks
    mind_pastClicks = mind_pastClicks.loc[mind_pastClicks['n_clicks']>0, ['user_id', 'previous_history']]
    # explode past clicks: one interaction per row
    mind_pastClicks = mind_pastClicks.explode('previous_history')\
                        .rename(columns = {'previous_history': 'news_id'})
    # all clicks in training file (week1-4 + week5 Mon-Sat)
    mind_train_explode = mind_train[['user_id', 'date_time', 'clicked_articles']].explode('clicked_articles')\
                        .rename(columns={'clicked_articles':'news_id'})
    mind_train_all = pd.concat([mind_pastClicks, mind_train_explode], axis=0)

    ## val
    mind_val_explode = mind_val[['user_id', 'date_time', 'clicked_articles']].explode('clicked_articles')\
                        .rename(columns={'clicked_articles':'news_id'})

    return mind_train_all, mind_val_explode


def generate_news_features(train:pd.DataFrame, val:pd.DataFrame, news_all:pd.DataFrame):
    """generate features for each relevant mind news article: news ID encoding + title TF-IDF + category one-hot encoding,
    also output a news id2num mapping

    Args:
        train: cleaned mind_train interaction data, each row is an interaction, columns ['user_id', 'news_id', 'date_time']
        val: cleaned mind_val interaction data, each row is an interaction, columns ['user_id', 'news_id', 'date_time']
        news_all: metadata of all related news articles, "news.tsv" combined from train/, dev/, test/ folders,
                  columns ['news_ID', 'category', 'title', 'title_lemma']

    Returns:
        news_feature: news feature sparse matrix,
        items_id2num: mapping, items are sorted as:
                      items interacted in week5 -> items only interacted in week1-4 -> uninteracted but relevant items
    """
    news_all = news_all.copy()

    ## all items
    # items interacted in week5
    items_week5 = set(list(train.loc[train['date_time'].notna(), 'news_id'].unique()) + list(val['news_id'].unique()))
    print('Number of items interacted in week5:', len(items_week5))  # 19206

    # items interacted ONLY in week1-4
    items_week1_4 = set(train.loc[train['date_time'].isna(), 'news_id'].unique())
    items_week1_4_unique = items_week1_4 - items_week5
    print('Number of items interacted ONLY in week1-4:', len(items_week1_4_unique))  # 77494

    # items uninteracted but relevant
    items_week5 = list(items_week5)
    items_week5.sort()
    items_week1_4_unique = list(items_week1_4_unique)
    items_week1_4_unique.sort()
    items_interacted = items_week5 + items_week1_4_unique
    items_relevant = news_all.loc[~news_all['news_ID'].isin(items_interacted), 'news_ID'].unique().tolist()
    items_relevant.sort()
    print('Number of items uninteracted but relevant:', len(items_relevant))  # 33679
    items_all = items_interacted + items_relevant  # 130379

    # create item id2num mapping
    items_id2num = dict(zip(items_all, range(len(items_all))))


    ## item features: news ID encoding + title TF-IDF + category one-hot encoding
    news_all = news_all.set_index('news_ID').loc[items_all]
    # ID
    news_feature_ID = scipy.sparse.eye(news_all.shape[0]).tocsr().astype(np.float32)   # 130379
    # title
    vectorizer = TfidfVectorizer(min_df = 2)  # exclude words that only appear in one article
    news_feature_title = vectorizer.fit_transform(news_all['title_lemma'])
    news_feature_title = news_feature_title.tocsr().astype(np.float32)    # 25112
    # category
    news_feature_category = scipy.sparse.csr_matrix(pd.get_dummies(news_all['category']).values).tocsr().astype(np.float32)  # 18
    news_feature = scipy.sparse.hstack([news_feature_ID, news_feature_title, news_feature_category])

    return news_feature, items_id2num


def get_mind_interactions(train:pd.DataFrame, val:pd.DataFrame, user_id2num:dict, item_id2num:dict):
    """output train and val interaction with columns ['user_num', 'item_num', 'datetime']

    Args:
        train: cleaned mind_train interaction data, each row is an interaction, columns ['user_id', 'news_id', 'date-time']
        val: cleaned mind_val interaction data, each row is an interaction, columns ['user_id', 'news_id', 'date-time']
        user_id2num, item_id2num: mapping

    Returns:
        train, val: pd.DataFrame
    """
    train = train.copy()
    val = val.copy()

    for df in [train, val]:
        df['user_num'] = df['user_id'].map(user_id2num)
        df['news_num'] = df['news_id'].map(item_id2num)

    return train[['user_num', 'news_num', 'date_time']], val[['user_num', 'news_num', 'date_time']]

# load data
mind_train = pd.read_csv(os.path.join(MAIN_DIRECTORY, 'data_raw/mind/MINDlarge_train/behaviors.tsv'), header=None, sep='\t').iloc[:,1:]
mind_val = pd.read_csv(os.path.join(MAIN_DIRECTORY, 'data_raw/mind/MINDlarge_dev/behaviors.tsv'), header=None, sep='\t').iloc[:,1:]
mind_test = pd.read_csv(os.path.join(MAIN_DIRECTORY, 'data_raw/mind/MINDlarge_test/behaviors.tsv'), header=None, sep='\t').iloc[:,1:]
news_train = pd.read_csv(os.path.join(MAIN_DIRECTORY, 'data_raw/mind/MINDlarge_train/news.tsv'), header=None, sep='\t')
news_val = pd.read_csv(os.path.join(MAIN_DIRECTORY, 'data_raw/mind/MINDlarge_dev/news.tsv'), header=None, sep='\t')
news_test = pd.read_csv(os.path.join(MAIN_DIRECTORY, 'data_raw/mind/MINDlarge_test/news.tsv'), header=None, sep='\t')
news_all = pd.concat([news_train, news_val, news_test]).drop_duplicates()[[0,1,3]]\
                            .rename(columns={0:'news_ID', 1:'category', 3:'title'})

# clean news title: lemmatize + remove stopwords
news_title_ls = []
for title in news_all['title']:
    doc = nlp(title)
    doc_list = []
    doc_list = [token.lemma_ for token in doc if token.lemma_ not in stop_words and not token.is_punct]
    news_title_ls.append(doc_list)
news_all['title_lemma'] = [' '.join(title) for title in news_title_ls]

# clean
train, val = clean_mind_train_val(mind_train, mind_val)
news_feature, items_id2num = generate_news_features(train, val, news_all)

# user id2num: include all users in train, val, test
all_users = set()
for df in [train, val, mind_test]:
    all_users.update(set(df['user_id'].unique()))  # 876956
print('Number of users:', len(all_users))
all_users = list(all_users)
all_users.sort()

user_id2num = dict(zip(all_users, range(len(all_users))))

# final train, val
train, val = get_mind_interactions(train, val, user_id2num, items_id2num)

# save
train.to_csv(os.path.join(MAIN_DIRECTORY, 'data_clean/mind/train.csv'), index=False)
val.to_csv(os.path.join(MAIN_DIRECTORY, 'data_clean/mind/val.csv'), index=False)
with open(os.path.join(MAIN_DIRECTORY, 'data_clean/mind/mapping.pickle'), 'wb') as handle:
    mapping = {'user_id2num': user_id2num,
              'item_id2num': items_id2num}
    pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
scipy.sparse.save_npz(os.path.join(MAIN_DIRECTORY, 'data_clean/mind/item_feature.npz'), news_feature)
