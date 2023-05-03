import tensorflow as tf
import itertools
import pandas as pd
import numpy as np

import os
from core import MAIN_DIRECTORY, SEED
from utils.data_importer import DataImporter
from utils.evaluation import Evaluation
from utils.early_stopping import EarlyStopping

import merlin.models.tf as mm
from merlin.io.dataset import Dataset as merlin_dataset
from merlin.schema.tags import Tags


class TwoTowerRecommender:
    def __init__(self, dataset_name, 
                 data_folder_path=os.path.join(MAIN_DIRECTORY, 'data_clean'),
                 use_text_feature=True, use_no_feature=False, use_only_text=False):
        DI = DataImporter(dataset_name, data_folder_path)
        train, val, train_df, val_user, val_ytrue, test_user, test_ytrue, candidate_features = DI.get_data('two_tower', use_text_feature, use_no_feature, use_only_text)
        self.train = train
        self.val = val
        self.train_df = train_df   # to get interaction data
        self.val_user = val_user
        self.val_ytrue = val_ytrue
        self.test_user = test_user
        self.test_ytrue = test_ytrue
        self.candidate_features = candidate_features

        self.model = None
        self.topk_model = None
        self.params = None


    def _create_model(self):
        # create user schema using USER tag
        schema = self.train.schema
        user_schema = schema.select_by_tag(Tags.USER)
        # create user (query) tower input block
        user_inputs = mm.InputBlockV2(user_schema)
        # create user (query) encoder block
        query = mm.Encoder(user_inputs, mm.MLPBlock([self.params['user_dim'], self.params['tower_dim']], 
                                                    no_activation_last_layer=True))

        # create item schema using ITEM tag
        item_schema = schema.select_by_tag(Tags.ITEM)
        # create item (candidate) tower input block
        item_inputs = mm.InputBlockV2(item_schema)
        # create item (candidate) encoder block
        candidate = mm.Encoder(item_inputs, mm.MLPBlock([self.params['item_dim'], self.params['tower_dim']], 
                                                        no_activation_last_layer=True))

        model = mm.TwoTowerModelV2(query, candidate)

        return model
        

    def train_model(self, tower_dim, user_dim, item_dim, 
                    learning_rate, batch_size, 
                    n_epochs=200, early_stopping=True, eval_step=5, consecutive_eval_threshold=5,
                    seed=SEED, verbose=False, train_k=10):
        self.params = {
            'tower_dim': int(tower_dim),
            'user_dim': int(user_dim),
            'item_dim': int(item_dim), 
            'learning_rate': learning_rate, 
            'batch_size': int(batch_size), 
            'n_epochs': n_epochs, 
            'seed': seed, 
            'verbose': verbose,
        }

        model = self._create_model()
        opt = tf.optimizers.Adam(self.params['learning_rate'])
        model.compile(optimizer=opt, run_eagerly=False, metrics=[mm.RecallAt(train_k), mm.NDCGAt(train_k)])

        if not early_stopping:
            # fit the model at once
            model.fit(self.train, validation_data=self.val, batch_size=self.params['batch_size'], epochs=n_epochs, verbose=verbose)
            self.model = model
        else:
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_ndcg_at_10', patience=consecutive_eval_threshold, 
                            mode='max')
            history = model.fit(self.train, validation_data=self.val, validation_freq=eval_step, batch_size=self.params['batch_size'], epochs=n_epochs, callbacks=[callback], verbose=verbose)
            self.model = model
            self.params['best_eval_score'] = np.max(history.history['val_ndcg_at_10'])
            self.params['best_epoch'] = np.argmax(history.history['val_ndcg_at_10'])
            # ES = EarlyStopping(consecutive_eval_threshold=consecutive_eval_threshold)
            # for i in range(1, n_epochs//eval_step+1):
            #     model.fit(self.train, batch_size=self.params['batch_size'], epochs=eval_step)
            #     self.model = model
            #     epoch_id = eval_step*i

            #     # evaluate
            #     # prepare the topk recommender block
            #     max_interaction = self.train_df.groupby('user_num')['item_num'].count().max()
            #     topk_model = model.to_top_k_encoder(self.candidate_features, k=max_interaction+100, batch_size=128)
            #     topk_model.compile(run_eagerly=False)
            #     self.topk_model = topk_model

            #     eval_score = self.get_validation_ndcg()
            #     if verbose:
            #         print('epoch {}: evaluation score = {}'.format(epoch_id, eval_score))
                
            #     stop_flag = ES.log(epoch_id, eval_score)
            #     if stop_flag:
            #         break
            # self.params['best_epoch'] = ES.best_epoch
            # self.params['iterations'] = epoch_id
            # self.params['best_eval_score'] = ES.best_evaluation_score


    def get_recommendation(self, user_info:merlin_dataset, k):
        """get top-k recommendation list for each user in user_info"""
        if self.model is None:
            print('TwoTowerRecommender: Model not trained yet. Call train_model() first.')
            return

        if self.topk_model is None:
            print('Configuring the top-k model...')
            # prepare the topk recommender block
            max_interaction = self.train_df.groupby('user_num')['item_num'].count().max()
            topk_model = self.model.to_top_k_encoder(self.candidate_features, k=max_interaction+100, batch_size=128)
            topk_model.compile(run_eagerly=False)
            self.topk_model = topk_model
    
        
        eval_loader = mm.Loader(user_info, batch_size=user_info.num_rows)[0]
        pred_user = eval_loader[0]['user_num'].numpy()[:,0].tolist()

        pred = self.topk_model(eval_loader[0])
        pred_item = pred[1].numpy().tolist()
        score = pred[0].numpy().tolist()
        all_pred = pd.DataFrame({'user_num': pred_user, 'item_num': pred_item, 'prediction': score})\
                    .sort_values('user_num', ascending=True)\
                        .explode(['item_num', 'prediction'])

        merged = pd.merge(self.train_df, all_pred, on=["user_num", "item_num"], how="outer")
        merged['item_num'] = merged['item_num'].astype(int)

        all_predictions = merged[merged.rating.isnull()].drop(['rating'], axis=1)
        pred_ls = all_predictions.sort_values('prediction', ascending=False)\
                .groupby('user_num').head(k)[['user_num', 'item_num']]\
                .groupby('user_num')['item_num'].apply(lambda x: list(x)).reset_index()\
                .sort_values('user_num', ascending=True)['item_num'].tolist()

        # pad each recommendation list to length k
        pred_ls = [(p+[1]*k)[:k] for p in pred_ls]
        
        return pred_ls
    

    def get_validation_ndcg(self, k=10):
        pred_ls = self.get_recommendation(self.val_user, k=k)
        val_evaluator = Evaluation(batch_size = self.val_user.num_rows, K=[k], how='val')
        mean_ndcg = val_evaluator.evaluate(pred_ls, self.val_ytrue)
        return mean_ndcg[0]


    def get_test_metrics(self, K):
        pred_ls = self.get_recommendation(self.test_user, max(K))
        test_evaluator = Evaluation(batch_size = self.test_user.num_rows, K=K, how='test')
        mean_precision, mean_recall, HR, mean_ndcg, MRR, MAP = test_evaluator.evaluate(pred_ls, self.test_ytrue)
        return mean_precision, mean_recall, HR, mean_ndcg, MRR, MAP
