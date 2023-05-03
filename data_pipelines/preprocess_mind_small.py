# Databricks notebook source

"""
1. Clean train and val interaction from orginal MIND dataset, with columns ['user_id', 'news_id', 'date_time']
note: interactions from the previous 4 weeks has date_time as the last day of the 4 weeks
2. save a mapping dictionary for user and item id2num
3. get metadata (text, category) of news appearing in train, val, test, save in data_raw/mind
"""

import os
import pickle

import math
import datetime
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import MAIN_DIRECTORY


# $ python -m spacy download en_core_web_md

# # loading in spacy medium model that has 20k word2vec embeddings
# nlp = spacy.load('en_core_web_md', disable=['parser', 'ner', 'textcat'])
# # load in stop words
# stop_words = nlp.Defaults.stop_words
# stop_words = list(stop_words)
# stop_words = stop_words + ['d', 'll', 'm', 'n', 's', 't', 've', '\n']
# stop_words = set(stop_words)

def clean_mind_train_val(mind_train:pd.DataFrame, mind_val:pd.DataFrame, val_ratio=0.1, remove_cold_user=True):
    """clean the original MIND train and validation dataframe (serve as test here) in the following format:
    each row is an interaction (click), including week1-5, columns: ['user_id', 'news_id', 'date-time'].
    The original mind train and validation datasets only have interaction data in week5.
    Interactions in week1-4 are stored as "previous_history" in the two data frames.
    In the original mind_train dataset, take the last 10% interactions as validation set. 

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
                        .rename(columns = {'previous_history': 'item_id'})
    print('number of interactions in week1-4:', mind_pastClicks.shape[0])

    # select last 10% clicks in week5 Mon-Sat as validation set
    mind_train_explode = mind_train[['user_id', 'date_time', 'clicked_articles']].explode('clicked_articles')\
                        .rename(columns={'clicked_articles':'item_id'})
    mind_train_explode['date_time'] = pd.to_datetime(mind_train_explode['date_time'])
    mind_train_explode = mind_train_explode.sort_values('date_time', ascending=True)
    n_val = math.floor(mind_train_explode.shape[0]*val_ratio)
    val = mind_train_explode.iloc[-n_val:,].reset_index(drop=True)

    # training set: interactions in week1-4 + 90% in week5 Mon-Sat
    train_1 = mind_pastClicks.copy()   # no date_time info in train_1
    train_2 = mind_train_explode.iloc[:-n_val,]
    # impute date_time in train_1: one day before the earliest day in train_2
    previous_date = train_2['date_time'].min().date()-datetime.timedelta(days=1)
    previous_datetime = datetime.datetime.combine(previous_date, datetime.time())
    train_1['date_time'] = previous_datetime
    train_1['date_time'] = pd.to_datetime(train_1['date_time'])
    train = pd.concat([train_1, train_2]).reset_index(drop=True)
    
    # test
    test = mind_val[['user_id', 'date_time', 'clicked_articles']].explode('clicked_articles')\
                        .rename(columns={'clicked_articles':'item_id'}).reset_index(drop=True)
    test['date_time'] = pd.to_datetime(test['date_time'])
    test = test.sort_values('date_time', ascending=True)
    
    if remove_cold_user:
        # remove cold users in val and test
        train_user = train['user_id'].unique().tolist()
        val = val.loc[val['user_id'].isin(train_user)].reset_index(drop=True)
        test = test.loc[test['user_id'].isin(train_user)].reset_index(drop=True)
    
    print('number of interactions in train:', train.shape[0])
    print('number of interactions in val:', val.shape[0])
    print('number of interactions in test:', test.shape[0])
    
    return train, val, test


def get_id2num_mapping(train:pd.DataFrame, val:pd.DataFrame=None, test:pd.DataFrame=None):
    """
    map user_id and item_id to numbers, user and item number sorted as following:
    1. train user/item first, sorted by id 
    2. unique val user/item next, sorted by id
    3. unique test user/item last, sorted by id
    
    Args:
        train, val, test: columns ['user_id', 'item_id', 'timestamp']
    
    Return:
        user_id2num: mapping 
        item_id2num: mapping
    """
    users = sorted(train['user_id'].unique().tolist())
    items = sorted(train['item_id'].unique().tolist())
    print('number of users in train set: {}'.format(len(users)))
    print('number of items in train set: {}'.format(len(items)))
    if val is not None:
        print('number of validation users: {}'.format(len(val['user_id'].unique())))
    if test is not None:
        print('number of test users: {}'.format(len(test['user_id'].unique())))

    print('Adding cold users & items from validation & test set (if any):')
    for df in [val, test]:
        if df is not None:
            new_users = set(df['user_id'].unique())
            new_items = set(df['item_id'].unique())
            unique_new_users = sorted(list(new_users-set(users)))
            unique_new_items = sorted(list(new_items-set(items)))
            print('added {} cold users'.format(len(unique_new_users)))
            print('added {} cold items'.format(len(unique_new_items)))
            users.extend(unique_new_users)
            items.extend(unique_new_items)

    user_id2num = dict(zip(users, range(len(users))))
    item_id2num = dict(zip(items, range(len(items))))

    return user_id2num, item_id2num


def get_related_news(train, val, test):
    """filter out articles appeared in train, val, test from all news articles provided

    Args:
        train, val, test: cleaned mind_train interaction data, each row is an interaction, columns ['user_id', 'news_id', 'date_time']
        
    Returns:
        related_news: news article metadata df, columns ['item_id', 'text', 'category']
    """
    news_train = pd.read_csv(os.path.join(MAIN_DIRECTORY, 'data_raw/mind/MINDsmall_train/news.tsv'), header=None, sep='\t')
    news_val = pd.read_csv(os.path.join(MAIN_DIRECTORY, 'data_raw/mind/MINDsmall_dev/news.tsv'), header=None, sep='\t')
    news_all = pd.concat([news_train, news_val]).drop_duplicates()[[0,3,1]]\
                            .rename(columns={0:'item_id', 1:'category', 3:'title'})
    
    related_news = pd.concat([train, val, test])['item_id'].unique().tolist()
    related_news = news_all.loc[news_all['item_id'].isin(related_news)].reset_index(drop=True)

    print('number of related news in train+val+test:', related_news.shape[0])
    return related_news


if __name__ == '__main__':
    mind_train = pd.read_csv(os.path.join(MAIN_DIRECTORY, 'data_raw/mind/MINDsmall_train/behaviors.tsv'), header=None, sep='\t').iloc[:,1:]
    mind_val = pd.read_csv(os.path.join(MAIN_DIRECTORY, 'data_raw/mind/MINDsmall_dev/behaviors.tsv'), header=None, sep='\t').iloc[:,1:]
    train, val, test = clean_mind_train_val(mind_train, mind_val)
    user_id2num, item_id2num = get_id2num_mapping(train, val, test)
    for df in [train, val, test]:
        df['user_num'] = df['user_id'].map(user_id2num)
        df['item_num'] = df['item_id'].map(item_id2num)

    # news metadata
    news_df = get_related_news(train, val, test)

    # save
    train.to_csv(os.path.join(MAIN_DIRECTORY, 'data_clean', 'mind_small', 'train.csv'), index=False)
    val.to_csv(os.path.join(MAIN_DIRECTORY, 'data_clean', 'mind_small', 'val.csv'), index=False)
    test.to_csv(os.path.join(MAIN_DIRECTORY, 'data_clean', 'mind_small', 'test.csv'), index=False)
    mapping = {'user_id2num': user_id2num, 'item_id2num': item_id2num}
    with open(os.path.join(MAIN_DIRECTORY, 'data_clean', 'mind_small', 'mapping_id2num.pkl'), 'wb') as f:
        pickle.dump(mapping, f)
    news_df.to_csv(os.path.join(MAIN_DIRECTORY, 'data_raw', 'mind', 'related_news.csv'), index=False)