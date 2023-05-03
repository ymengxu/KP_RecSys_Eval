import pandas as pd
import os
import scipy.sparse as sps

import nvtabular as nvt
from nvtabular.ops import *
from merlin.schema.tags import Tags
from merlin.models.utils.dataset import unique_rows_by_features

import dask.dataframe as dd


def prepare_merlin_data(data:str, info:dict, 
                        train_path, val_path, test_path, user_feature_path, item_feature_path, 
                        use_text_feature=True, use_no_feature=False, use_only_text=False,
                        category_temp_directory=None, 
                        parquet_path='/tmp/merlin_data'):
    """prepare data from merlin models"""

    assert not (use_text_feature and use_no_feature), "get_data_two_tower: argument conflict"
    assert not (use_only_text and use_no_feature), "get_data_two_tower: argument conflict"
    assert not (not use_text_feature and use_only_text), "get_data_two_tower: argument conflict"
    assert not (use_only_text and data=='movielens_100k'), "get_data_two_tower: argument conflict"

    if category_temp_directory is not None: 
        os.makedirs(category_temp_directory, exist_ok=True)
        
    # load pandas df
    train = pd.read_csv(train_path)[[info['user'], info['item']]].astype(int)
    val = pd.read_csv(val_path)[[info['user'], info['item']]].astype(int)
    test = pd.read_csv(test_path)[[info['user'], info['item']]].astype(int)

    train['rating'] = 1
    val['rating'] = 1
    test['rating'] = 1

    if data == 'movielens_100k':
        user = pd.read_csv(user_feature_path)[[info['user'], 'age', 'gender', 'occupation']]
        item = pd.read_csv(item_feature_path)[[info['item'], 'genre', 'year']]
        item['genre'] = item['genre'].str.split('|')
        item['year'] = item['year'].fillna(item['year'].median()).astype(int)
        user[info['user']] = user[info['user']].astype(int)
        item[info['item']] = item[info['item']].astype(int)

    if data == 'adobe_core5':
        # item features
        item_feature = sps.load_npz(item_feature_path)
        # text tf-idf feature
        item_feature = item_feature[:,item_feature.shape[0]:]
        item_feature = pd.DataFrame(item_feature.todense()).reset_index().rename(columns={'index':info['item']})
        item_feature_tfidf = item_feature.iloc[:,1:-82].values.tolist()
        assert len(item_feature_tfidf[0]) == 7595
        item_feature_topic = item_feature.iloc[:,-82:].values.tolist()
        assert len(item_feature_topic[0]) == 82

        item = item_feature[[info['item']]].copy()
        item['text'] = item_feature_tfidf
        item['topic'] = item_feature_topic
        
        # user features
        user = pd.read_csv(user_feature_path)[[info['user'], 'age', 'gender']]\
                        .sort_values(info['user'],ascending=True)
        # normalize age
        user['age'] = (user['age']-user['age'].mean())/user['age'].std()
    
    train = train.merge(user, on=info['user']).merge(item, on=info['item'])
    val = val.merge(user, on=info['user']).merge(item, on=info['item'])
    test = test.merge(user, on=info['user']).merge(item, on=info['item'])

    train = dd.from_pandas(train, npartitions=1)
    val = dd.from_pandas(val, npartitions=1)
    test = dd.from_pandas(test, npartitions=1)

    # # to nvtabular dataset
    # os.makedirs(parquet_path, exist_ok=True)
    # train_parquet_path = os.path.join(parquet_path, "train.parquet")
    # val_parquet_path = os.path.join(parquet_path, "valid.parquet")
    # test_parquet_path = os.path.join(parquet_path, "test.parquet")

    # train.to_parquet(os.path.join(parquet_path, "train.parquet"))
    # val.to_parquet(os.path.join(parquet_path, "valid.parquet"))
    # test.to_parquet(os.path.join(parquet_path, "test.parquet"))

    # train_ds = nvt.Dataset(train_parquet_path, engine="parquet")
    # val_ds = nvt.Dataset(val_parquet_path, engine="parquet")
    # test_ds = nvt.Dataset(test_parquet_path, engine="parquet")

    train_ds = nvt.Dataset(train)
    val_ds = nvt.Dataset(val)
    test_ds = nvt.Dataset(test)


    # feature engineering
    if data == 'movielens_100k':
        item_feature = (
            ['genre', 'year'] 
            >> nvt.ops.Categorify(out_path=category_temp_directory)
            >> TagAsItemFeatures()
        )
        user_feature_cat = (
            ['gender', 'occupation'] 
            >> nvt.ops.Categorify(out_path=category_temp_directory) 
            >> TagAsUserFeatures()
        )
        user_feature_num = ['age'] >> nvt.ops.AddTags(tags=[Tags.CONTINUOUS, Tags.USER])
    if data == 'adobe_core5':
        item_feature = (
            ['topic'] 
            >> nvt.ops.Categorify(out_path=category_temp_directory)
            >> TagAsItemFeatures()
        )
        user_feature_cat = (
            ['gender'] 
            >> nvt.ops.Categorify(out_path=category_temp_directory) 
            >> TagAsUserFeatures()
        )
        user_feature_num = ['age'] >> nvt.ops.AddTags(tags=[Tags.CONTINUOUS, Tags.USER])
        item_feature_num = ['text'] >> TagAsItemFeatures()

    userId = (
        [info['user']] 
        >> nvt.ops.Categorify(out_path=category_temp_directory) 
        >> nvt.ops.AddTags(tags=[Tags.USER_ID, Tags.CATEGORICAL, Tags.USER])
    )
    itemId = (
        [info['item']] 
        >> nvt.ops.Categorify() 
        >> nvt.ops.AddTags(tags=[Tags.ITEM_ID, Tags.CATEGORICAL, Tags.ITEM])
    )
    
    if not use_no_feature:
        if data == 'movielens_100k':
            workflow = nvt.Workflow(userId + itemId + user_feature_cat + user_feature_num + item_feature)
        if data == 'adobe_core5':
            if use_text_feature:
                if use_only_text:
                    workflow = nvt.Workflow(userId + itemId + item_feature_num)
                else:
                   workflow = nvt.Workflow(userId + itemId + user_feature_cat + user_feature_num + item_feature + item_feature_num)
            else:
                workflow = nvt.Workflow(userId + itemId + user_feature_cat + user_feature_num + item_feature)
    else:
        workflow = nvt.Workflow(userId + itemId)
        
    train_transformed = workflow.fit_transform(train_ds)
    val_transformed = workflow.transform(val_ds)
    test_transformed = workflow.transform(test_ds)


    # prepare data for evaluation
    candidate_features = unique_rows_by_features(train_transformed, Tags.ITEM, Tags.ITEM_ID)
    val_user = unique_rows_by_features(val_transformed, Tags.USER, Tags.USER_ID)
    test_user = unique_rows_by_features(test_transformed, Tags.USER, Tags.USER_ID)

    train_interaction = train_transformed.to_ddf().compute()[[info['user'], info['item']]]
    train_interaction['rating'] = 1
    val_interaction = val_transformed.to_ddf().compute()[[info['user'], info['item']]]
    val_ytrue = val_interaction.groupby('user_num')['item_num'].apply(lambda x: list(set(x)))\
        .reset_index().sort_values('user_num')['item_num'].tolist()
    test_interaction = test_transformed.to_ddf().compute()[[info['user'], info['item']]]
    test_ytrue = test_interaction.groupby('user_num')['item_num'].apply(lambda x: list(set(x)))\
        .reset_index().sort_values('user_num')['item_num'].tolist()
    
    return train_transformed, val_transformed, train_interaction, val_user, val_ytrue, test_user, test_ytrue, candidate_features