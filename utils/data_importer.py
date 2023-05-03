import os
import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import scipy.sparse as sps

from core import MAIN_DIRECTORY, SEED
from core.tensor_factorization_methods.data_utils import prepare_tensor_data

# try:
#     from pyspark.sql.types import StructType, StructField
#     from pyspark.sql.types import FloatType, IntegerType, LongType
#     from recommenders.utils.spark_utils import start_or_get_spark
# except ImportError:
#     pass  # skip this import if we are not in a Spark environment


class DataImporter:
    def __init__(self, dataset_name:str, data_folder_path=os.path.join(MAIN_DIRECTORY, 'data_clean')):
        """initialze the data importer, prepare data for different models"""
        assert dataset_name in ['adobe', 'adobe_core5', 'mind', 'mind_small', 'movielens_100k'], 'DataImporter: dataset not available'

        self.data_folder_path = data_folder_path
        self.dataset_name = dataset_name
        self.train_path = os.path.join(data_folder_path, dataset_name, 'train.csv')
        self.val_path = os.path.join(data_folder_path, dataset_name, 'val.csv')
        self.test_path = os.path.join(data_folder_path, dataset_name, 'test.csv')
        self.user_feature_path = os.path.join(data_folder_path, dataset_name, 'user_feature.csv')
        if dataset_name == 'adobe_core5':
            self.item_feature_path = os.path.join(data_folder_path, dataset_name, 'item_feature.npz')
        if dataset_name == 'movielens_100k':
            self.item_feature_path = os.path.join(data_folder_path, dataset_name, 'item_feature.csv')
        self.item_embedding_path = os.path.join(data_folder_path, dataset_name, 'item_embedding.pkl')
        self.info = {'item': 'item_num', 'user': 'user_num', 'time':'timestamp'}

        # m: #users, n: #items, user: column name for user, item: column name for item
        if dataset_name == 'adobe':
            self.info.update({'m': 466967, 'n': 3175})
        if dataset_name == 'adobe_core5':
            self.info.update({'m': 10921, 'n': 2899})
        if dataset_name == 'mind':
            self.info.update({'m': 876956, 'n': 96700})
        if dataset_name == 'mind_small':
            self.info.update({'m': 92787, 'n': 52866})
            self.info['time'] = 'date_time'
        if dataset_name == 'movielens_100k':
            self.info.update({'m': 666, 'n': 1362})
            

    def _get_data_implicit(self):
        """prepare data for recommenders with no need for feature: popularity, iALS, BPR"""
        # train: interaction matrix
        train = pd.read_csv(self.train_path)
        interaction_matrix = train.groupby([self.info['user'], self.info['item']], as_index = False).size()
        interaction_matrix = sps.csr_matrix((interaction_matrix['size'], (interaction_matrix[self.info['user']],
                                                                                   interaction_matrix[self.info['item']])),
                                                    shape = [self.info['m'], self.info['n']])
        # validation: val_user, val_ytrue
        val = pd.read_csv(self.val_path).groupby(self.info['user'])[self.info['item']].agg(lambda x: list(set(x))).reset_index()
        val_user = val[self.info['user']].tolist()
        val_ytrue = val[self.info['item']].tolist()

        # test: test_user, true_test
        test = pd.read_csv(self.test_path).groupby(self.info['user'])[self.info['item']].agg(lambda x: list(set(x))).reset_index()
        test_user = test[self.info['user']].tolist()
        test_ytrue = test[self.info['item']].tolist()

        return interaction_matrix, val_user, val_ytrue, test_user, test_ytrue
    
    
    def _get_data_cornac(self):
        import cornac
        train = pd.read_csv(self.train_path)[[self.info['user'], self.info['item']]]
        train['rating'] = 1
        val = pd.read_csv(self.val_path)[[self.info['user'], self.info['item']]]\
                        .groupby(self.info['user'])[self.info['item']]\
                        .agg(lambda x: list(set(x))).reset_index()
        val_user = val[self.info['user']].tolist()
        val_ytrue = val[self.info['item']].tolist()
        test = pd.read_csv(self.test_path)[[self.info['user'], self.info['item']]]\
                        .groupby(self.info['user'])[self.info['item']]\
                        .agg(lambda x: list(set(x))).reset_index()
        test_user = test[self.info['user']].tolist()
        test_ytrue = test[self.info['item']].tolist()
        
        uid_map_cornac = dict(zip(range(self.info['m']), range(self.info['m'])))
        iid_map_cornac = dict(zip(range(self.info['n']), range(self.info['n'])))
        train_cornac = cornac.data.Dataset.build(
                            data = train.itertuples(index=False), 
                            global_uid_map = uid_map_cornac,
                            global_iid_map = iid_map_cornac,
                            seed=SEED
                        )
        return train_cornac, val_user, val_ytrue, test_user, test_ytrue#, train
    
    
    def _get_data_als_spark(self):
        # prepare spark data schema
        schema = StructType(
            (
                StructField(self.info['user'], IntegerType()),
                StructField(self.info['item'], IntegerType()),
                StructField('rating', FloatType()),
                StructField('timestamp', LongType()),
            )
        )
        spark = start_or_get_spark()
        schema_user = StructType(
            (
                StructField(self.info['user'], IntegerType()),
            )
        )

        train = pd.read_csv(self.train_path)[[self.info['user'], self.info['item'], 'timestamp']]
        train['timestamp'] = pd.to_datetime(train['timestamp']).map(lambda x: int(x.strftime("%s")))
        train['rating'] = 1
        train_spark = spark.createDataFrame(train, schema).cache()

        val = pd.read_csv(self.val_path)[[self.info['user'], self.info['item']]]\
                        .groupby(self.info['user'])[self.info['item']]\
                        .agg(lambda x: list(set(x))).reset_index()
        val_user = spark.createDataFrame(val, schema_user).cache()
        val_ytrue = val[self.info['item']].tolist()

        test = pd.read_csv(self.train_path)[[self.info['user'], self.info['item']]]\
                        .groupby(self.info['user'])[self.info['item']]\
                        .agg(lambda x: list(set(x))).reset_index()
        test_user = spark.createDataFrame(test, schema_user).cache()
        test_ytrue = test[self.info['item']].tolist()
        
        return train_spark, val_user, val_ytrue, test_user, test_ytrue


    def _get_data_lightfm(self, use_text_feature=True, use_no_feature=False, use_only_text=False):
        # get train, val, test data
        interaction_matrix, val_user, val_ytrue, test_user, test_ytrue = self._get_data_implicit()
        # # get item feature
        # with open(self.item_embedding_path, 'rb') as f:
        #     item_embed = pickle.load(f)['item_embedding_matrix']
        assert not (use_text_feature and use_no_feature), "get_data_lightfm: argument conflict"
        assert not (use_only_text and use_no_feature), "get_data_lightfm: argument conflict"
        assert not (not use_text_feature and use_only_text), "get_data_lightfm: argument conflict"
        assert not (use_only_text and self.dataset_name=='movielens_100k'), "get_data_lightfm: argument conflict"

        if self.dataset_name == 'adobe_core5':
            user_feature = pd.read_csv(self.user_feature_path)[[self.info['user'], 'age', 'gender']]\
                .sort_values(self.info['user'],ascending=True)[['age', 'gender']]
            # normalize age
            user_feature['age'] = (user_feature['age']-user_feature['age'].mean())/user_feature['age'].std()
            user_feature = sps.csr_matrix(user_feature.values)
            user_feature = sps.hstack([sps.identity(user_feature.shape[0]), user_feature]).tocsr()
            # use text tf-idf  
            item_feature = sps.load_npz(self.item_feature_path).tocsr()
            if not use_text_feature: 
                # exclude text feature
                item_feature = sps.hstack([item_feature[:,:item_feature.shape[0]], item_feature[:,-82:]]).tocsr()
                assert item_feature.shape[1] == item_feature.shape[0]+82
            if use_only_text: 
                # include only text feature
                item_feature = item_feature[:,:-82]
                user_feature = user_feature[:,:user_feature.shape[0]]


        if self.dataset_name == 'movielens_100k':
            user_feature = pd.read_csv(self.user_feature_path)[[self.info['user'], 'age', 'gender', 'occupation']]\
                .sort_values(self.info['user'], ascending=True).drop(self.info['user'],axis=1)
            user_feature = pd.get_dummies(user_feature, columns=['occupation']).values
            user_feature = sps.csr_matrix(np.concatenate([np.eye(user_feature.shape[0]),user_feature],axis=1))

            item_feature = pd.read_csv(self.item_feature_path)[[self.info['item'], 'genre', 'year']]
            genre_ls = item_feature['genre'].str.split('|').map(lambda x: list(set(x)))
            mlb = MultiLabelBinarizer(sparse_output=True)
            feature_genre = pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(genre_ls),
                index=item_feature.index,
                columns=mlb.classes_
            ).reset_index(drop=True).iloc[:,1:]
            item_feature = pd.concat([item_feature[[self.info['item'], 'year']], feature_genre], axis=1)
            item_feature = pd.get_dummies(item_feature, columns=['year']).sort_values(self.info['item'], ascending=True)
            item_feature = item_feature.drop(self.info['item'], axis=1).values
            item_feature = sps.csr_matrix(np.concatenate([np.eye(item_feature.shape[0]), item_feature],axis=1))

        if use_no_feature:
            user_feature = user_feature[:, :user_feature.shape[0]]
            item_feature = item_feature[:, :item_feature.shape[0]]

        return interaction_matrix, val_user, val_ytrue, test_user, test_ytrue, user_feature, item_feature
    
    
    def _get_data_tensor(self, use_text_feature=True, use_no_feature=False, use_only_text=False):
        return prepare_tensor_data(
            self.dataset_name, self.info, 
            self.train_path, self.val_path, self.test_path, self.item_feature_path,
            use_text_feature,
            use_no_feature,
            use_only_text,
        )
    
    
    def _get_data_wide_deep(self):
        from core.neural_based_methods.wide_deep.wide_deep_utils import prepare_wide_deep_data
        return prepare_wide_deep_data(
            self.dataset_name, self.info, 
            self.train_path, self.val_path, self.test_path, self.user_feature_path, self.item_feature_path
        )


    def _get_data_lightgcn(self):
        from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
        train = pd.read_csv(self.train_path)[[self.info['user'], self.info['item']]]
        train['rating'] = 1
        val = pd.read_csv(self.val_path)[[self.info['user'], self.info['item']]]
        val['rating'] = 1
        
        val_by_user = val[[self.info['user'], self.info['item']]]\
            .groupby(self.info['user'])[self.info['item']]\
            .agg(lambda x: list(set(x))).reset_index()
        val_user = val_by_user[self.info['user']].tolist()
        val_ytrue = val_by_user[self.info['item']].tolist()
        test = pd.read_csv(self.test_path)[[self.info['user'], self.info['item']]]\
            .groupby(self.info['user'])[self.info['item']]\
            .agg(lambda x: list(set(x))).reset_index()
        test_user = test[self.info['user']].tolist()
        test_ytrue = test[self.info['item']].tolist()
        
        # graph data
        train = train.set_axis(['userID', 'itemID', 'rating'], axis=1)
        val = val.set_axis(['userID', 'itemID', 'rating'], axis=1)
        data = ImplicitCF(train=train, test=val, 
                col_user = 'userID', col_item='itemID', col_rating='rating')
        return data, val_user, val_ytrue, test_user, test_ytrue
        
        
    def _get_data_content(self):
        # train
        train = pd.read_csv(self.train_path)

        # validation: val_user, val_ytrue
        val = pd.read_csv(self.val_path).groupby(self.info['user'])[self.info['item']].agg(lambda x: list(set(x))).reset_index()
        val_user = val[self.info['user']].tolist()
        val_ytrue = val[self.info['item']].tolist()

        # test: test_user, true_test
        test = pd.read_csv(self.test_path).groupby(self.info['user'])[self.info['item']].agg(lambda x: list(set(x))).reset_index()
        test_user = test[self.info['user']].tolist()
        test_ytrue = test[self.info['item']].tolist()

        # item embeddings
        with open(os.path.join(self.data_folder_path, self.dataset_name, 'item_embedding.pkl'), 'rb') as f:
            item_embed = pickle.load(f)['item_embedding_matrix']
        return train, val_user, val_ytrue, test_user, test_ytrue, item_embed


    def _get_data_ncf(self):
        train = pd.read_csv(self.train_path)[[self.info['user'], self.info['item']]]
        train['rating'] = 1
        train = train.sort_values(self.info['user'], ascending=True)

        val = pd.read_csv(self.val_path)[[self.info['user'], self.info['item']]]\
                        .groupby(self.info['user'])[self.info['item']]\
                        .agg(lambda x: list(set(x))).reset_index()
        val_user = val[self.info['user']].tolist()
        val_ytrue = val[self.info['item']].tolist()
        test = pd.read_csv(self.test_path)[[self.info['user'], self.info['item']]]\
                        .groupby(self.info['user'])[self.info['item']]\
                        .agg(lambda x: list(set(x))).reset_index()
        test_user = test[self.info['user']].tolist()
        test_ytrue = test[self.info['item']].tolist()

        return train, val_user, val_ytrue, test_user, test_ytrue

    
    def _get_data_pinsage(self, use_text_feature=True, use_no_feature=False, use_only_text=False):
        from core.neural_based_methods.pinsage.data_utils import prepare_data as prepare_pinsage_data
        return prepare_pinsage_data(
            self.info, self.dataset_name, 
            self.train_path, self.val_path, self.test_path, self.user_feature_path, self.item_feature_path,
            use_text_feature, use_no_feature, use_only_text
        )


    def _get_data_merlin(self, use_text_feature=True, use_no_feature=False, use_only_text=False):
        from core.neural_based_methods.two_tower.data_utils import prepare_merlin_data
        return prepare_merlin_data(
            self.dataset_name, self.info, 
            self.train_path, self.val_path, self.test_path, self.user_feature_path, self.item_feature_path, 
            use_text_feature, use_no_feature, use_only_text,
            category_temp_directory=os.path.join(self.data_folder_path, 'merlin_data', self.dataset_name, 'categories'),
        )

    def get_data(self, model_name:str, use_text_feature=True, use_no_feature=False, use_only_text=False):
        """prepare data for the respective model"""
        assert model_name in ['popularity', 'knn', 'bpr_cornac', 
                              'ials', 'als_spark', 'bpr', 
                              'lightfm', 'bivae', 'lightgcn', 'ncf',
                              'content', 'pinsage', 'tensor', 'wide_deep', 'two_tower',
                              ], "DataImporter: model not available"

        if model_name in ['ials', 'bpr']:
            return self._get_data_implicit()

        if model_name == 'lightfm':
            return self._get_data_lightfm(use_text_feature=use_text_feature, use_no_feature=use_no_feature, use_only_text=use_only_text)
        
        if model_name in ['popularity', 'knn', 'bivae', 'bpr_cornac']:
            return self._get_data_cornac()
        
        if model_name == 'als_spark':
            return self._get_data_als_spark()
        
        if model_name == 'lightgcn':
            return self._get_data_lightgcn()

        if model_name == 'content':
            return self._get_data_content()
        
        if model_name == 'ncf':
            return self._get_data_ncf()
        
        if model_name == 'pinsage':
            return self._get_data_pinsage(use_text_feature=use_text_feature, use_no_feature=use_no_feature,
                                          use_only_text=use_only_text)
        
        if model_name == 'tensor':
            return self._get_data_tensor(use_text_feature=use_text_feature, 
                                         use_no_feature=use_no_feature,
                                         use_only_text=use_only_text)
        
        if model_name == 'wide_deep':
            return self._get_data_wide_deep()
        
        if model_name == 'two_tower':
            return self._get_data_merlin(use_text_feature=use_text_feature, use_no_feature=use_no_feature,
                                         use_only_text=use_only_text)