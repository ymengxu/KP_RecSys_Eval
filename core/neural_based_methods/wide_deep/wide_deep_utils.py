import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
import scipy.sparse as sps


def prepare_wide_deep_data(
    dataset_name, info,
    train_path, val_path, test_path, user_feature_path, item_feature_path
):
    # training data 
    train = pd.read_csv(train_path)[[info['user'], info['item']]]
    train['rating']=1
    train = train.set_axis(['userID', 'itemID', 'rating'], axis=1)

    # val/test data
    val = pd.read_csv(val_path)[[info['user'], info['item']]]\
        .sort_values(info['user'],ascending=True)\
        .groupby(info['user'])[info['item']].agg(lambda x: list(set(x))).reset_index()
    val_user = val[info['user']].tolist()
    val_ytrue = val[info['item']].tolist()

    test = pd.read_csv(test_path)[[info['user'], info['item']]]\
        .sort_values(info['user'],ascending=True)\
        .groupby(info['user'])[info['item']].agg(lambda x: list(set(x))).reset_index()
    test_user = test[info['user']].tolist()
    test_ytrue = test[info['item']].tolist()

    # use and item feature
    categorical_columns_vocab_list = {}
    numeric_columns_len = {}
    if dataset_name == 'movielens_100k':
        # user feature
        user_feature = pd.read_csv(user_feature_path)[[info['user'], 'age', 'gender', 'occupation']]\
            .sort_values(info['user'], ascending=True)\
            .rename({info['user']:'userID'}, axis=1)\
            .astype({'userID':int})
        categorical_columns_vocab_list['occupation'] = user_feature['occupation'].unique().tolist()
        numeric_columns_len['age'] = 1
        numeric_columns_len['gender'] = 1

        # item feature
        item_feature = pd.read_csv(item_feature_path)[[info['item'], 'genre', 'year']]\
            .sort_values(info['item'], ascending=True)\
            .rename({info['item']:'itemID'}, axis=1)\
            .astype({'itemID':int})
        genre_ls = item_feature['genre'].str.split('|').map(lambda x: list(set(x)))
        mlb = MultiLabelBinarizer(sparse_output=True)
        feature_genre = pd.DataFrame.sparse.from_spmatrix(
            mlb.fit_transform(genre_ls),
            index=item_feature.index,
            columns=mlb.classes_
        ).reset_index(drop=True).iloc[:,1:].values.tolist()
        item_feature['genre'] = feature_genre
        item_feature['year'].loc[item_feature['year'].isnull()] = item_feature['year'].mode()[0]
        item_feature['year'] = item_feature['year'].astype(int)
        categorical_columns_vocab_list['year'] = item_feature['year'].unique().tolist()
        numeric_columns_len['genre'] = len(item_feature['genre'].iloc[0])

        categorical_columns = list(categorical_columns_vocab_list.keys())
        numeric_columns = list(numeric_columns_len.keys())

    if dataset_name == 'adobe_core5': 
        # user feature
        user_feature = pd.read_csv(user_feature_path)[[info['user'], 'age', 'gender']]\
            .sort_values(info['user'], ascending=True)\
            .rename({info['user']:'userID'}, axis=1)\
            .astype({'userID':int})
        numeric_columns_len['age'] = 1
        numeric_columns_len['gender'] = 1
        # item features
        item_feature = sps.load_npz(item_feature_path)
        # text tf-idf feature
        item_feature = item_feature[:,item_feature.shape[0]:]
        item_feature = pd.DataFrame(item_feature.todense()).reset_index().rename(columns={'index':'itemID'})
        item_feature_tfidf = item_feature.iloc[:,1:-82].values.tolist()
        assert len(item_feature_tfidf[0]) == 7595
        item_feature_topic = item_feature.iloc[:,-82:].values.tolist()
        assert len(item_feature_topic[0]) == 82
        # item_num
        item_feature = item_feature[['itemID']]
        # item_feature['text'] = item_feature_tfidf
        item_feature['topic'] = item_feature_topic
        numeric_columns_len['topic'] = len(item_feature['topic'].iloc[0])
        # numeric_columns_len['text'] = len(item_feature['text'].iloc[0])

        categorical_columns = list(categorical_columns_vocab_list.keys())
        numeric_columns = list(numeric_columns_len.keys())

    # columns information
    columns_info = {
        'label_columns': 'rating',
        'id_columns': ['userID','itemID'],
        'id_columns_vocab_size': {
            'userID': user_feature.shape[0],
            'itemID': item_feature.shape[0],
        },
        'categorical_columns': categorical_columns,
        'categorical_columns_vocab_list': categorical_columns_vocab_list, 
        'numeric_columns': numeric_columns,
        'numeric_columns_len': numeric_columns_len
    }
        
    train = train.merge(user_feature, how='left').merge(item_feature, how='left')
    
    return train, val_user, val_ytrue, test_user, test_ytrue, columns_info, user_feature, item_feature


def get_wide_and_deep_columns(
    id_columns:list,
    id_columns_vocab_size:dict,
    embedding_dim:int, 
    categorical_columns:list=None, 
    categorical_columns_vocab_list:dict=None,
    numeric_columns:list=None,
    numeric_columns_len:dict=None,
    crossed_feat_dim=1000
):
    wide_columns, deep_columns = [], []
    
    # ID columns: one-hot encoding -> embedding
    for column_name in id_columns:
        id_col = tf.feature_column.categorical_column_with_vocabulary_list(
                                  column_name, vocabulary_list=list(range(id_columns_vocab_size[column_name])))
        wide_columns.append(id_col)
        wrapped_col = tf.feature_column.embedding_column(
                                  id_col,
                                  dimension=embedding_dim,
                                  combiner='mean')
        deep_columns.append(wrapped_col)

    # crossed column for user and item id column
    user_item_crossed = tf.feature_column.crossed_column([wide_columns[0], wide_columns[1]], crossed_feat_dim)
    wide_columns.append(user_item_crossed)

    # catgeorical columns: one-hot encoding -> embedding
    if categorical_columns is not None:
        for column_name in categorical_columns:
            cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
                                    column_name, vocabulary_list=categorical_columns_vocab_list[column_name])
            wide_columns.append(cat_col)
            wrapped_col = tf.feature_column.embedding_column(
                                    cat_col,
                                    dimension=embedding_dim,
                                    combiner='mean')
            deep_columns.append(wrapped_col)
        
    # numeric column
    if numeric_columns is not None:
        for column_name in numeric_columns:
            if column_name == 'age':
                user_age = tf.feature_column.numeric_column('age', shape=(1,), dtype=tf.float32)
                user_age_buckets = tf.feature_column.bucketized_column(user_age, boundaries=[18, 35])
                deep_columns.append(user_age)
                wide_columns.append(user_age_buckets)
            else:
                numeric_col = tf.feature_column.numeric_column(column_name, shape=(numeric_columns_len[column_name],), dtype=tf.float32)
                deep_columns.append(numeric_col)
    
    return wide_columns, deep_columns