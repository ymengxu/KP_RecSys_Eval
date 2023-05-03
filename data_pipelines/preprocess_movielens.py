# Databricks notebook source
# MAGIC %%capture
# MAGIC dbutils.fs.cp('abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/MovieLens_test', 
# MAGIC 'file:/tmp/model_evaluation_data/MovieLens_test', True)
# MAGIC # dbutils.fs.cp('abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/data_clean', 
# MAGIC # 'file:/tmp/model_evaluation_data/data_clean', True)
# MAGIC 
# MAGIC import os
# MAGIC import pickle
# MAGIC import pandas as pd
# MAGIC 
# MAGIC import shutil
# MAGIC from zipfile import ZipFile
# MAGIC from recommenders.datasets import movielens
# MAGIC 
# MAGIC from core import SEED
# MAGIC 
# MAGIC ML_RAW_PATH = '/tmp/model_evaluation_data/MovieLens_test'
# MAGIC DATA_CLEAN_PATH = '/tmp/model_evaluation_data/data_clean/'
# MAGIC dbutils.fs.mkdirs('file:'+os.path.join(DATA_CLEAN_PATH, 'ml-100k'))

# COMMAND ----------

def train_val_test_split_by_time(df:pd.DataFrame, 
                                 val_ratio=0.1, test_ratio=0.2, 
                                 filter_cold_user = False, user_col='userID'):
    """Given the cleaned adobe interaction dataframe,
    split train/val/test.
    Assume df is ready sorted by timestamp.

    Args:
        df: interaction dataframe
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
        train_users = set(train[user_col].unique())
        val_users = set(val[user_col].unique())
        test_users = set(test[user_col].unique())
        common_train_val_users = train_users.intersection(val_users)
        common_train_test_users = train_users.intersection(test_users)

        val = val.loc[val[user_col].isin(common_train_val_users),].reset_index(drop=True)
        test = test.loc[test[user_col].isin(common_train_test_users),].reset_index(drop=True)
    
    return train, val, test


def load_user_df(
    local_cache_path, 
    size='100k', user_col='userID', 
    age_col='age', gender_col='gender', occupation_col='occupation', zipcode_col='zipcode'):
    """load movielens user data: age, gender, occupation, zipcode"""

    zip_path = os.path.join(local_cache_path, "ml-{}.zip".format(size))
    filepath = f'ml-{size}/u.user'
    user_path = os.path.join(local_cache_path, "u.user".format(size))
    with ZipFile(zip_path, "r") as z:
        with z.open(filepath) as zf, open(user_path, "wb") as f:
            shutil.copyfileobj(zf, f)
    user_df = pd.read_csv(
        user_path,
        sep='|',
        engine="python",
        names=[user_col, age_col, gender_col, occupation_col, zipcode_col],
        encoding="ISO-8859-1"
    )
    return user_df


def get_id2num_mapping(train:pd.DataFrame, val:pd.DataFrame=None, test:pd.DataFrame=None, user_col='userID', item_col='itemID'):
    """
    map user_id and item_id to numbers, user and item number sorted as following:
    1. train user/item first, sorted by id 
    2. unique val user/item next, sorted by id
    3. unique test user/item last, sorted by id
    
    Args:
        train, val, test: columns [user_col, item_col, 'timestamp']
    
    Return:
        user_id2num: mapping 
        item_id2num: mapping
    """
    users = sorted(train[user_col].unique().tolist())
    items = sorted(train[item_col].unique().tolist())
    print('number of users in train set: {}'.format(len(users)))
    print('number of items in train set: {}'.format(len(items)))
    if val is not None:
        print('number of validation users: {}'.format(len(val[user_col].unique())))
    if test is not None:
        print('number of test users: {}'.format(len(test[user_col].unique())))

    print('Adding cold users & items from validation & test set (if any):')
    for df in [val, test]:
        if df is not None:
            new_users = set(df[user_col].unique())
            new_items = set(df[item_col].unique())
            unique_new_users = sorted(list(new_users-set(users)))
            unique_new_items = sorted(list(new_items-set(items)))
            print('added {} cold users'.format(len(unique_new_users)))
            print('added {} cold items'.format(len(unique_new_items)))
            users.extend(unique_new_users)
            items.extend(unique_new_items)

    user_id2num = dict(zip(users, range(len(users))))
    item_id2num = dict(zip(items, range(len(items))))

    return user_id2num, item_id2num

# COMMAND ----------

df = movielens.load_pandas_df(
    size='100k',
    local_cache_path='/tmp/model_evaluation_data/MovieLens_test',
    header=["userID", "itemID", "rating", "timestamp"]
).sort_values('timestamp', ascending=True)
df['rating'] = (df['rating'] > 3).astype(int)
df = df.loc[df['rating']>0]

print('number of total users:', len(df['userID'].unique()))
print('number of total items:', len(df['itemID'].unique()))

train, val, test = train_val_test_split_by_time(df, filter_cold_user=True)
user_id2num, item_id2num = get_id2num_mapping(train, val, test)

print('train: n_users={}, n_items={}, n_interactions={}'.format(
    len(train['userID'].unique()), 
    len(train['itemID'].unique()), 
    train.shape[0]
))
print('val: n_users={}, n_items={}, n_interactions={}'.format(
    len(val['userID'].unique()), 
    len(val['itemID'].unique()), 
    val.shape[0]
))
print('test: n_users={}, n_items={}, n_interactions={}'.format(
    len(test['userID'].unique()), 
    len(test['itemID'].unique()), 
    test.shape[0]
))

for df in [train, val, test]:
    df['user_num'] = df['userID'].map(user_id2num)
    df['item_num'] = df['itemID'].map(item_id2num)



# COMMAND ----------

# features
item_feature = movielens.load_item_df(
    size="100k",
    local_cache_path='/tmp/model_evaluation_data/MovieLens_test',
    movie_col='itemID',
    title_col='title',
    genres_col='genre',
    year_col='year'
)
user_feature = load_user_df(local_cache_path='/tmp/model_evaluation_data/MovieLens_test')

# COMMAND ----------

# save
train_file = os.path.join(DATA_CLEAN_PATH, 'ml-100k', 'train.csv')
val_file = os.path.join(DATA_CLEAN_PATH, 'ml-100k', 'val.csv')
test_file = os.path.join(DATA_CLEAN_PATH, 'ml-100k', 'test.csv')
train.to_csv(train_file, index=False)
val.to_csv(val_file, index=False)
test.to_csv(test_file, index=False)

mapping = {'user_id2num': user_id2num, 'item_id2num':item_id2num}
with open(os.path.join(DATA_CLEAN_PATH, 'ml-100k', 'mapping_id2num.pkl'), 'wb') as f:
    pickle.dump(mapping, f)
    
item_file = os.path.join(ML_RAW_PATH, 'item_meta.csv')
user_file = os.path.join(ML_RAW_PATH, 'user_meta.csv')
item_feature.to_csv(item_file, index=False)
user_feature.to_csv(user_file, index=False)

# COMMAND ----------

# clean user feature
user_feature['user_num'] = user_feature['userID'].map(user_id2num)
user_feature = user_feature.loc[user_feature['user_num'].notnull()].reset_index(drop=True)
user_feature['age'] = (user_feature['age']-user_feature['age'].mean())/user_feature['age'].std()
user_feature['gender'] = (user_feature['gender']=='M').astype(int)
# user_feature = pd.get_dummies(user_feature, columns=['occupation']).drop('zipcode', axis=1)

# clean item feature
item_feature['item_num'] = item_feature['itemID'].map(item_id2num)
item_feature = item_feature.loc[item_feature['item_num'].notnull()].sort_values('item_num').reset_index(drop=True)

user_feature.to_csv(os.path.join(DATA_CLEAN_PATH, 'ml-100k', 'user_feature.csv'), index=False)
item_feature.to_csv(os.path.join(DATA_CLEAN_PATH, 'ml-100k', 'item_feature.csv'), index=False)

# COMMAND ----------

dbutils.fs.cp(
    'file:'+os.path.join(DATA_CLEAN_PATH, 'ml-100k'),
    'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/data_clean/movielens_100k', True)

dbutils.fs.rm('file:'+os.path.join(ML_RAW_PATH, "ml-100k.zip"))
dbutils.fs.cp(
    'file:'+ML_RAW_PATH, 
    'abfss://databricks-data@kpadlsschdevuws2a1.dfs.core.windows.net/model_evaluation_data/data_raw/movielens_100k', True)

# COMMAND ----------

item_feature.head()

# COMMAND ----------


