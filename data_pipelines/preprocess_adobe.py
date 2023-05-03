# Databricks notebook source

"""
1. Extract user and page metadata, store in os.path.join(MAIN_DIRECTORY, './data_raw/adobe/') folder
2. Generate user and page feature
3. (optional) Train-val-test split: 7:1:2 by time
4. Clean interaction data: user and page id to num, also generate a 5-core interaction dataset
5. create user feature df and item feature sparse df, sorted by user and item num 
6. user & page feature and interaction data stored in os.path.join(MAIN_DIRECTORY, 'data_clean/adobe/') and 
os.path.join(MAIN_DIRECTORY, 'data_clean/adobe_core5/') folder
"""

import os
import re
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import MAIN_DIRECTORY


def extract_user_metadata(adobe:pd.DataFrame):
    """given the adobe dataset, output one data frame: user_meta

    Args:
        adobe: adobe interaction dataset, columns ['Unnamed: 0.1', 'Unnamed: 0', 'guid', 'pagename', 'date_time', 'gndr',
                                               'age_in_yrs', 'rownumber', 'pagename_extracted', 'title_s',
                                               'category_ss', 'health_topics', 'health_topic_id_ss', 'keywords_s',
                                               'body_t', 'health_topics_converted', 'health_topic_id_ss_converted',
                                               'category_ss_converted', 'title_s_cleaned', 'body_t_cleaned',
                                               'title_s_lower']
    Returns:
        user_meta: pd dataframe, user metadata, columns ['guid', 'gender', 'age']
    """
    user_cols = ['guid', 'gndr', 'age_in_yrs']
    user_meta = adobe.loc[~adobe.duplicated(subset = user_cols), user_cols].set_axis(['guid', 'gender', 'age'], axis=1).reset_index(drop=True)
    assert user_meta.shape[0] == len(np.unique(user_meta['guid']))

    # for users lacking gender information: assume male
    user_meta['gender'] = user_meta['gender'].fillna('M')
    user_meta['gender'] = user_meta['gender'].map(lambda x: 0 if x=='F' else 1)
    # for users lacking age information: assume median age in the dataset
    user_meta['age'] = user_meta['age'].fillna(user_meta['age'].median())
    
    print('total users: {num_users}'.format(num_users=user_meta.shape[0]))
    # unique users: 645278
    
    return user_meta

def extract_page_metadata(adobe:pd.DataFrame):
    """given the adobe dataset, output one data frame: page_meta_clean

    Args:
        adobe: adobe interaction dataset, columns ['Unnamed: 0.1', 'Unnamed: 0', 'guid', 'pagename', 'date_time', 'gndr',
                                               'age_in_yrs', 'rownumber', 'pagename_extracted', 'title_s',
                                               'category_ss', 'health_topics', 'health_topic_id_ss', 'keywords_s',
                                               'body_t', 'health_topics_converted', 'health_topic_id_ss_converted',
                                               'category_ss_converted', 'title_s_cleaned', 'body_t_cleaned',
                                               'title_s_lower']

    Returns:
        page_meta_clean: pd dataframe, each row is a unique page, columns ['pagename_extracted',
                                                                       'title_s_cleaned',
                                                                       'body_t', 'body_t_cleaned',
                                                                       'health_topic_id_ss_converted',
                                                                       'health_topics_cleaned', 'keywords_s']
    """
    
    page_cols = ['pagename_extracted', 'title_s_cleaned', # title cleaned for stopwords
             'body_t', 'body_t_cleaned',
             'health_topics', 'health_topic_id_ss_converted', 'keywords_s']
    page_meta = adobe.loc[~adobe.duplicated(subset = page_cols), page_cols]

    ## clean page metadata
    # clean health topics
    page_meta = page_meta.fillna('')
    page_meta['health_topics_cleaned'] = page_meta['health_topics'].str.replace('"', "'")\
                                        .str.strip("[]").str.lower().str.split("', '")\
                                        .map(lambda x: [category.replace("'", '') for category in x])\
                                        .map(lambda x: [re.sub(r",? ", "-", category) for category in x])\
                                        .astype(str)\
                                        .str.replace('"', "'")\
                                        .str.strip('[]').str.replace("'", '')
    # health topic in this format: topic1, topic2-more-words, topic3-more-words
    page_meta['health_topic_id_ss_converted'] = page_meta['health_topic_id_ss_converted'].str.replace(' ', ', ')
    # health topic id in this format: id1, id2, id3
    page_meta_clean = page_meta.drop('health_topics', axis = 1)

    # remove page duplicate rows with no body or no topic info
    page_meta_clean = page_meta_clean.loc[~(   page_meta_clean.duplicated(subset = ['pagename_extracted']) &
                    ((page_meta_clean['body_t'] == '') | (page_meta_clean['health_topic_id_ss_converted'] == ''))   ), ]
    page_meta_clean = page_meta_clean.loc[~(   page_meta_clean.duplicated(subset = ['pagename_extracted'], keep='last') &
                    ((page_meta_clean['body_t'] == '') | (page_meta_clean['health_topic_id_ss_converted'] == ''))   ), ]
    # remaining duplicate pages: two pages with the same name: one main page, one subsequent page -> combine contents
    dup_page = page_meta_clean.loc[page_meta_clean.duplicated(subset = ['pagename_extracted'], keep=False),]
    dup_page = dup_page.groupby(['pagename_extracted', 'title_s_cleaned'])[['body_t',
                                             'body_t_cleaned',
                                             'health_topic_id_ss_converted',
                                             'keywords_s',
                                             'health_topics_cleaned']].agg(', '.join).reset_index()
    for k in dup_page.columns:
        dup_page[k] = dup_page[k].str.strip(', ')
    # finish cleaning duplicates
    page_meta_clean = page_meta_clean.loc[~page_meta_clean.duplicated(subset = ['pagename_extracted'], keep=False),]
    page_meta_clean = pd.concat([page_meta_clean, dup_page])

    # print("number of articles with no article body: {num_items}".format(num_items=page_meta_clean.loc[page_meta_clean['body_t_cleaned'] == '', ].shape[0]))
    # 326
    #### TODO: articles without article body, try add in healthwise data in CosmosDB
    ####       this step can also be done at an earlier stage
    # # join healthwise AzureDB data for pages with no article body
    # page_meta_clean = page_meta_clean.fillna('')
    # page_meta_nobody = page_meta_clean.loc[page_meta_clean['body_t'] == ''].copy()
    # healthwise = pd.read_csv(os.path.join(MAIN_DIRECTORY, 'data_raw/adobe/healthwise_metadata.csv'))
    # healthwise['title_t_lower'] = healthwise['title_t'].str.lower()
    # page_meta_nobody = page_meta_nobody.merge(healthwise[['title_t_lower', 'article_body']],
    #                        left_on = 'pagename_extracted', right_on = 'title_t_lower', how = 'left')
    # page_meta_nobody = page_meta_nobody.drop(['title_t_lower', 'body_t'], axis=1)\
    #                     .rename(columns={'article_body': 'body_t'})
    # page_meta_clean = page_meta_clean.loc[~(page_meta_clean['body_t'] == '')]
    # page_meta_clean = pd.concat([page_meta_clean, page_meta_nobody]).reset_index().iloc[:,1:]
    
    page_meta_clean = page_meta_clean.reset_index(drop=True)
    print('total items: {num_items}'.format(num_items=page_meta_clean.shape[0]))

    return page_meta_clean


def generate_user_features(user:pd.DataFrame):
    """Given the user metadata, create a user x features data frame

    Args:
        user: user meteadata, pd dataframe, each row is a unique user, columns ['guid', 'gender', 'age']

    Returns:
        user_feature: pd dataframe, features include age + gender one-hot encoding
    """
    # TODO: in the future, maybe add EHR data for users

    return user


def get_adobe_interaction(adobe:pd.DataFrame):
    """get the interaction data with user and page indicated by indices corresponding to user_feature and page_feature

    Args:
        adobe: adobe interaction dataset, columns ['Unnamed: 0.1', 'Unnamed: 0', 'guid', 'pagename', 'date_time', 'gndr',
                                               'age_in_yrs', 'rownumber', 'pagename_extracted', 'title_s',
                                               'category_ss', 'health_topics', 'health_topic_id_ss', 'keywords_s',
                                               'body_t', 'health_topics_converted', 'health_topic_id_ss_converted',
                                               'category_ss_converted', 'title_s_cleaned', 'body_t_cleaned',
                                               'title_s_lower']

    Returns:
        interaction: pd dataframe, each row is an interaction, columns ['user_id', 'item_id', 'timestamp']

    """
    interaction = adobe[['guid', 'pagename_extracted', 'date_time', 'rownumber']]\
                    .groupby(['guid', 'pagename_extracted', 'date_time'])['rownumber'].sum()\
                    .reset_index().sort_values('date_time')[['guid', 'pagename_extracted', 'date_time']]\
                    .set_axis(['user_id', 'item_id', 'date_time'], axis=1)
    interaction['timestamp'] = pd.to_datetime(interaction['date_time'])
    # interaction['timestamp'] = interaction['timestamp'].map(lambda x: int(x.strftime("%s")))
    interaction = interaction.reset_index(drop=True)
    
    return interaction[['user_id', 'item_id', 'timestamp']]

def get_adobe_interaction_5core(interaction:pd.DataFrame):
    """In the adobe dataset, only keep users with at least 5 interactions (5-core)

    Args:
        interaction: all adobe interactions, columns ['user_id', 'item_id', 'timestamp']

    Returns:
        adobe_core5: pd dataframe, 5-core adobe interaction data
    """
    # select only users with at least 5 interactions
    views_all_users = interaction.groupby('user_id')['item_id'].count()
    valid_users = views_all_users.loc[views_all_users >= 5].index.values
    adobe_core5 = interaction.loc[interaction['user_id'].isin(valid_users)]

    return adobe_core5.sort_values('timestamp').reset_index(drop=True)

def train_val_test_split_by_time(df:pd.DataFrame, 
                                 val_ratio=0.1, test_ratio=0.2, 
                                 filter_cold_user = False):
    """Given the cleaned adobe interaction dataframe,
    split train/val/test.
    Assume df is ready sorted by timestamp.

    Args:
        df: interaction dataframe with columns ['user_id', 'item_id', 'timestamp']
        val_ratio
        test_ratio
        filter_cold_user: if True, filter out the cold users in val and test set
    """

    n = df.shape[0]
    n_test = int(n*test_ratio)
    n_val = int(n*val_ratio)
    n_train = n-n_val-n_test
    assert n_train+n_val+n_test==n, 'Size of 3 sets does not add to the total size'

    train = df.iloc[:n_train, ].reset_index(drop=True)
    val = df.iloc[n_train:n_train+n_val, ].reset_index(drop=True)
    test = df.iloc[-n_test:, ].reset_index(drop=True)

    if filter_cold_user:
        train_users = set(train['user_id'].unique())
        val_users = set(val['user_id'].unique())
        test_users = set(test['user_id'].unique())
        common_train_val_users = train_users.intersection(val_users)
        common_train_test_users = train_users.intersection(test_users)

        val = val.loc[val['user_id'].isin(common_train_val_users),].reset_index(drop=True)
        test = test.loc[test['user_id'].isin(common_train_test_users),].reset_index(drop=True)
        
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


def generate_page_features(page:pd.DataFrame, train:pd.DataFrame):
    """Given the page metadata, create a page x features data frame

    Args:
        page: page metadata, pd dataframe, each row is a unique page, columns ['pagename_extracted',
                                                                            'title_s_cleaned',
                                                                            'body_t', 'body_t_cleaned',
                                                                            'health_topic_id_ss_converted', 'keywords_s',
                                                                            'health_topics_cleaned']
        train: columns ['user_id', 'item_id', 'timestamp']
    
    Commands:
        generate ID one-hot encoding + text (title+article body) tf-idf + topic one-hot encoding (tf-idf using only pages in the training set)

    Returns:
        page_feature: pd dataframe, features include ID one-hot encoding + title tf-idf + body tf-idf + topic one-hot encoding
    """
    train_items = train['item_id'].unique().tolist()
    ## generate features
    page_feature = page[['pagename_extracted']].copy()
    
    # ID one-hot encoding
    # page_feature_ID = pd.get_dummies(page_feature, columns = ['index'])

    # text ID-IDF
    page['text'] = page[['title_s_cleaned', 'body_t_cleaned']].fillna('').agg(' '.join, axis=1)
    vectorizer = TfidfVectorizer(min_df = 5)
    training_text = page.loc[page['pagename_extracted'].isin(train_items), 'text'].tolist()
    assert len(training_text) == len(train_items), 'TDIDF Vectorizer: training corpus length mismatch'
    vectorizer.fit(training_text)
    page_feature_text = vectorizer.transform(page['text'])
    text_names = vectorizer.get_feature_names_out()
    page_feature_text = page_feature_text.todense().tolist()
    page_feature_text = pd.DataFrame(page_feature_text, columns=text_names)
    print('text tdidf feature length: {}'.format(page_feature_text.shape[1]))

    # health topics one-hot encoding
    topic_ls = page['health_topic_id_ss_converted'].fillna('').str.split(', ')
    topic_ls = topic_ls.map(lambda x: list(set(x)))
    mlb = MultiLabelBinarizer(sparse_output=True)
    page_feature_topic = pd.DataFrame.sparse.from_spmatrix(
        mlb.fit_transform(topic_ls),
        index=page.index,
        columns=mlb.classes_
    ).reset_index(drop=True).iloc[:,1:]
    print('topic feature length: {}'.format(page_feature_topic.shape[1]))

    # combine
    page_feature = pd.concat([page_feature, page_feature_text, page_feature_topic], axis=1)

    return page_feature


def create_feature_matrix(user_feature:pd.DataFrame, item_feature:pd.DataFrame, 
                          user_id2num:dict, item_id2num:dict):
    """based on the user and item index from user_id2num, item_id2num, generate 
    1. user feature dataframe, columns ['user_id', 'user_num', 'gender', 'age']
    2. item feature sparse matrix, rows sorted by item index (item_num)
    """
    
    user_df = user_feature.copy()
    user_df['user_num'] = user_df['guid'].map(user_id2num)
    user_df = user_df.dropna().sort_values('user_num').reset_index(drop=True)
    user_df['user_num'] = user_df['user_num'].astype(int)
    assert user_df.shape[0] == len(user_id2num)

    item_df = item_feature.copy()
    item_df['item_num'] = item_df['pagename_extracted'].map(item_id2num)
    item_df = item_df.dropna().sort_values('item_num').reset_index(drop=True)
    # add item ID one-hot encoding
    item_df_ID = pd.get_dummies(item_df['item_num']).reset_index(drop=True)
    item_df = item_df.drop(['pagename_extracted', 'item_num'], axis=1)
    item_df = pd.concat([item_df_ID, item_df], axis=1)
    item_mat = sps.csr_matrix(item_df.values)
    assert item_mat.shape[0] == len(item_id2num)
    print('number of item features: {}'.format(item_mat.shape[1]))

    return user_df, item_mat


if __name__ == '__main__':
    adobe = pd.read_csv(os.path.join(MAIN_DIRECTORY, 'data_raw/adobe/healthwise_Adobe_joined.csv'))

    # extract user and page metadata
    user_meta = extract_user_metadata(adobe)
    page_meta = extract_page_metadata(adobe)

    # user and page feature
    user_feature = generate_user_features(user_meta)

    # interaction data: all, 5-core
    interaction = get_adobe_interaction(adobe)
    interaction_core5 = get_adobe_interaction_5core(interaction)

    # optional: train, val, test split
    train, val, test = train_val_test_split_by_time(interaction, filter_cold_user=True)
    train_core5, val_core5, test_core5 = train_val_test_split_by_time(interaction_core5, filter_cold_user=True)

    # final steps
    # map id to num + arrange user and item feature by num 
    print('\nadobe stats:')
    page_feature = generate_page_features(page_meta, train)
    print('number of interactions in train:', train.shape[0])
    print('number of interactions in val:', val.shape[0])
    print('number of interactions in test:', test.shape[0])
    user_id2num, item_id2num = get_id2num_mapping(train, val, test)
    mapping = {'user_id2num':user_id2num, 'item_id2num':item_id2num}
    with open(os.path.join(MAIN_DIRECTORY, 'data_clean', 'adobe', 'mapping_id2num.pkl'), 'wb') as f:
        pickle.dump(mapping, f)
    for df in [train, val, test]:
        df['user_num'] = df['user_id'].map(user_id2num)
        df['item_num'] = df['item_id'].map(item_id2num)
    user_df, item_mat = create_feature_matrix(user_feature, page_feature, user_id2num, item_id2num)

    print('\nadobe_core5 stats:')
    page_feature = generate_page_features(page_meta, train_core5)
    print('number of interactions in train:', train_core5.shape[0])
    print('number of interactions in val:', val_core5.shape[0])
    print('number of interactions in test:', test_core5.shape[0])
    user_id2num_core5, item_id2num_core5 = get_id2num_mapping(train_core5, val_core5, test_core5)
    mapping = {'user_id2num':user_id2num_core5, 'item_id2num':item_id2num_core5}
    with open(os.path.join(MAIN_DIRECTORY, 'data_clean', 'adobe_core5', 'mapping_id2num.pkl'), 'wb') as f:
        pickle.dump(mapping, f)
    for df in [train_core5, val_core5, test_core5]:
        df['user_num'] = df['user_id'].map(user_id2num_core5)
        df['item_num'] = df['item_id'].map(item_id2num_core5)
    user_df_core5, item_mat_core5 = create_feature_matrix(user_feature, page_feature, user_id2num_core5, item_id2num_core5)
    
    ## save
    # metadata: data_raw
    user_meta.to_csv(os.path.join(MAIN_DIRECTORY, 'data_raw', 'adobe', 'user_meta.csv'), index=False)
    page_meta.to_csv(os.path.join(MAIN_DIRECTORY, 'data_raw', 'adobe', 'page_meta.csv'), index=False)

    # interaction data
    train.to_csv(os.path.join(MAIN_DIRECTORY, 'data_clean', 'adobe', 'train.csv'), index=False)
    val.to_csv(os.path.join(MAIN_DIRECTORY, 'data_clean', 'adobe', 'val.csv'), index=False)
    test.to_csv(os.path.join(MAIN_DIRECTORY, 'data_clean', 'adobe', 'test.csv'), index=False)
    train_core5.to_csv(os.path.join(MAIN_DIRECTORY, 'data_clean', 'adobe_core5', 'train.csv'), index=False)
    val_core5.to_csv(os.path.join(MAIN_DIRECTORY, 'data_clean', 'adobe_core5', 'val.csv'), index=False)
    test_core5.to_csv(os.path.join(MAIN_DIRECTORY, 'data_clean', 'adobe_core5', 'test.csv'), index=False)

    # features
    user_df.to_csv(os.path.join(MAIN_DIRECTORY, 'data_clean', 'adobe', 'user_feature.csv'), index=False)
    sps.save_npz(os.path.join(MAIN_DIRECTORY, 'data_clean', 'adobe', 'item_feature.npz'), item_mat)
    user_df_core5.to_csv(os.path.join(MAIN_DIRECTORY, 'data_clean', 'adobe_core5', 'user_feature.csv'), index=False)
    sps.save_npz(os.path.join(MAIN_DIRECTORY, 'data_clean', 'adobe_core5', 'item_feature.npz'), item_mat_core5)