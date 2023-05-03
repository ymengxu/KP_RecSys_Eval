import os
import pandas as pd
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import MultiLabelBinarizer
from datetime import datetime

from core import MAIN_DIRECTORY


def datetime_to_num(date): 
    year = date.year
    month = date.month
    day = date.day
    return year*365+month*12+day


def d(x):
    # sigmoid function for BPR, d(x) = sigmoid(-x)
    if x > 10:
        return 0
    if x < -10:
        return 1
    if x >= -10 and x <= 10:
        return 1.0 / (1.0 + np.exp(x))
    

def prepare_tensor_data(
    dataset_name, info, 
    train_path, val_path, test_path, item_feature_path,
    use_text_feature=True,
    use_no_feature=False,
    use_only_text=False,
):
    """
    get data for tensor decomposition recommender
    Args:   
        dataset_name (str): either adobe_core5 or movielens_100k
        info (dict): info to indicate user, item, and time column name in the stored interaction data frame
        train_path, val_path, test_path, item_feature_path (str): path to the respective stored data
    Returns:
        train_dict (dict): dict with keys 'train_data', 'train_aux', 'train_record_aux', 'train_time_aux'
        val_user, val_ytrue (list): val data
        test_user, test_ytrue (list): test data
        item_feature (np.array): item_feature matrix 
    """
    assert dataset_name in ['adobe_core5', 'movielens_100k'], 'prepare_tensor_data: dataset name must be adobe_core5 or movielens_100k'
    assert not (use_text_feature and use_no_feature), "get_data_tensor: argument conflict"
    assert not (use_only_text and use_no_feature), "get_data_tensor: argument conflict"
    assert not (not use_text_feature and use_only_text), "get_data_tensor: argument conflict"
    assert not (use_only_text and dataset_name=='movielens_100k'), "get_data_tensor: argument conflict"
    
    train = pd.read_csv(train_path)[[info['user'], info['item'], info['time']]]
    val = pd.read_csv(val_path)[[info['user'], info['item'], info['time']]]
    test = pd.read_csv(test_path)[[info['user'], info['item'], info['time']]]

    # time to num, discretized by day
    if dataset_name == 'movielens_100k':
        train[info['time']] = train[info['time']].map(lambda x: datetime.fromtimestamp(x))
    train[info['time']] = pd.to_datetime(train[info['time']])
    train[info['time']] = (train[info['time']] - train[info['time']].min()).dt.days
    train[info['time']] = train[info['time']].map(dict(zip(
        train[info['time']].unique(), range(len(train[info['time']].unique()))
    )))
    
    # train file format: [[u,i,r],...,[u,i,r]]
    train_data = train.values.tolist()

    # val/test user, ytrue
    val = val.groupby(info['user'])[info['item']].agg(lambda x: list(set(x))).reset_index()
    val_user = val[info['user']].tolist()
    val_ytrue = val[info['item']].tolist()

    test = test.groupby(info['user'])[info['item']].agg(lambda x: list(set(x))).reset_index()
    test_user = test[info['user']].tolist()
    test_ytrue = test[info['item']].tolist()
    

    # train_aux file format: [[[i,i],[r]],...,[[i,i,i],[r,r]]]
    # u-th elemet [[i,i],[r]] is the set of purchase items and time of user u
    train_aux_item = train.groupby(info['user'])[info['item']].apply(lambda x: list(set(x))).reset_index()
    train_aux_time = train.groupby(info['user'])[info['time']].apply(lambda x: list(set(x))).reset_index()
    train_aux = train_aux_item.merge(train_aux_time)
    train_aux = train_aux[[info['item'], info['time']]].values.tolist()

    # train_record_aux format: [[[i,r]],...,[[i,r],[i,r],[i,r]]]
    # u-th elemet [[i,r],[i,r],[i,r]] is the set of purchased item-time pairs of user u
    train['item_time'] = train[[info['item'], info['time']]].values.tolist()
    train_record_aux = train.groupby(info['user'])['item_time'].apply(list).values.tolist()
    
    # train_time_aux format: [[[u,u,u,u],[i,i,i]],...,[[u,u],[i,i,i]]]
    # r-th elemet [[u,u],[i,i,i]] is the set of users who purchased something in time r and the set of items purchsed in r
    train_time_aux = train.groupby(info['time']).agg({
        info['user']: lambda x: list(set(x)),
        info['item']: lambda x: list(set(x))
    }, axis=1)
    train_time_aux = train_time_aux.sort_values(info['time'], ascending=True)[[info['user'], info['item']]].values.tolist()

    if dataset_name == 'adobe_core5':
        # item feature: ID encoding + text tf-idf + health topic multi-hot encoding
        item_feature = sps.load_npz(item_feature_path)
        item_feature = np.array(item_feature[:,item_feature.shape[0]:].todense())
        assert item_feature.shape[1] == 7595+82, 'prepare_tensor_data: adobe_core5 item feature length error'
        if not use_text_feature: 
            item_feature = item_feature[:,-82:]
            assert item_feature.shape[1] == 82
        elif use_only_text:
            item_feature = item_feature[:,:-82]
            assert item_feature.shape[1] == 7595
            
    if dataset_name == 'movielens_100k':
        # item feature: ID encoding + genre + year
        item_feature = pd.read_csv(item_feature_path)[[info['item'], 'genre', 'year']].sort_values(info['item'], ascending=True)
        genre_ls = item_feature['genre'].str.split('|').map(lambda x: list(set(x)))
        mlb = MultiLabelBinarizer(sparse_output=True)
        feature_genre = pd.DataFrame.sparse.from_spmatrix(
            mlb.fit_transform(genre_ls),
            index=item_feature.index,
            columns=mlb.classes_
        ).reset_index(drop=True).iloc[:,1:]
        feature_year = pd.get_dummies(item_feature['year'], columns=['year'])
        item_feature = pd.concat([feature_genre, feature_year], axis=1).values

    train_dict = {
        'train_data': train_data,
        'train_aux': train_aux,
        'train_record_aux': train_record_aux, 
        'train_time_aux': train_time_aux
    }

    if use_no_feature: 
        item_feature = np.eye(item_feature.shape[0])
    else:
        item_feature = np.concatenate((np.eye(item_feature.shape[0]), item_feature), dtype=np.float32, axis=-1)
    
    return train_dict, val_user, val_ytrue, test_user, test_ytrue, item_feature