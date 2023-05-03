import os
import numpy as np
import shutil

from core import MAIN_DIRECTORY, SEED
from utils.data_importer import DataImporter
from utils.evaluation import Evaluation
from utils.early_stopping import EarlyStopping
from core.neural_based_methods.wide_deep.wide_deep_utils import get_wide_and_deep_columns

from recommenders.utils import tf_utils
from recommenders.datasets.pandas_df_utils import user_item_pairs
import recommenders.models.wide_deep.wide_deep_utils as wide_deep


class WideDeepRecommender:
    def __init__(self, dataset_name, 
                 data_folder_path=os.path.join(MAIN_DIRECTORY, 'data_clean')):
        DI = DataImporter(dataset_name, data_folder_path)
        train, val_user, val_ytrue, test_user, test_ytrue, columns_info, user_feature, item_feature = DI.get_data('wide_deep')
        self.train = train
        self.val_user = val_user
        self.val_ytrue = val_ytrue
        self.test_user = test_user
        self.test_ytrue = test_ytrue
        self.columns_info = columns_info
        self.user_feature = user_feature
        self.item_feature = item_feature
        self.model = None
        self.params = None


    def train_model(self,
            # training
            batch_size, dnn_dropout, 
            embedding_dim, 
            dnn_batch_norm=1,
            # Wide (linear) model hyperparameters
            linear_optimizer="adagrad", linear_optimizer_lr= 0.0621,
            # DNN model hyperparameters
            dnn_optimizer="adadelta", dnn_optimizer_lr=0.1,
            # layer dimensions
            dnn_hidden_layer_num = 3,     # number of hidden layers
            dnn_hidden_layer_1 = 64,    # hidden dimension of the first layer
            # early stopping
            early_stopping=True, eval_step=5, consecutive_eval_threshold=5, 
            n_epochs=200,
            model_dir=os.path.join(MAIN_DIRECTORY, 'res', 'wide_deep_checkpoints'),
            seed=SEED, verbose=False
        ):

        dnn_hidden_units = [int(dnn_hidden_layer_1)*h for h in range(1, int(dnn_hidden_layer_num)+1)]
        self.params = {
            'batch_size': int(batch_size), 'dnn_dropout': dnn_dropout, 'dnn_batch_norm': dnn_batch_norm,
            'embedding_dim': int(embedding_dim),
            'linear_optimizer': linear_optimizer, 'linear_optimizer_lr': linear_optimizer_lr,
            'dnn_optimizer': dnn_optimizer, 'dnn_optimizer_lr': dnn_optimizer_lr, 
            'dnn_hidden_units': dnn_hidden_units,
            'n_epochs': n_epochs,
            'seed': seed, 'verbose': verbose, 
            'model_dir': model_dir
        }
        
        # feature columns
        wide_columns, deep_columns = self._get_wide_deep_columns()

        # training set
        train_fn = self._get_train_set()

        # create model
        model = self._create_model(wide_columns, deep_columns)

        # delete existing checkpoints and train from scratch
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir) 

        self.model = model
        # train
        if not early_stopping: 
            model.train(
                input_fn=train_fn, 
                steps=self.params['batch_size']*n_epochs, 
            )
            self.model = model
        else:
            ES = EarlyStopping(consecutive_eval_threshold=consecutive_eval_threshold)
            for i in range(1,n_epochs//eval_step+1):
                model.train(
                    input_fn=train_fn, 
                    steps=self.params['batch_size']*eval_step, 
                )
                self.model = model
                epoch_id = eval_step*i
                # evaluate
                eval_score = self.get_validation_ndcg()
                if verbose:
                    print('epoch {}: evaluation score = {}'.format(epoch_id, eval_score))
                
                stop_flag = ES.log(epoch_id, eval_score)
                if stop_flag:
                    break
            self.params['best_epoch'] = ES.best_epoch
            self.params['iterations'] = epoch_id
            self.params['best_eval_score'] = ES.best_evaluation_score
        

    def _get_train_set(self):
        """build tf.data.Dataset for training"""
        batch_size = self.params['batch_size']
        seed = self.params['seed']
        # training set
        train_fn = tf_utils.pandas_input_fn(
            df=self.train,
            y_col=self.columns_info['label_columns'],
            batch_size=batch_size,
            num_epochs=None,
            shuffle=True,
            seed=seed,
        )
        return train_fn
    

    def _get_wide_deep_columns(self):
        embedding_dim = self.params['embedding_dim']
        verbose=self.params['verbose']
        # columns
        wide_columns, deep_columns = get_wide_and_deep_columns(
            id_columns=self.columns_info['id_columns'],
            id_columns_vocab_size=self.columns_info['id_columns_vocab_size'],
            embedding_dim=embedding_dim,
            numeric_columns=self.columns_info['numeric_columns'], 
            numeric_columns_len=self.columns_info['numeric_columns_len'],
            categorical_columns=self.columns_info['categorical_columns'], 
            categorical_columns_vocab_list=self.columns_info['categorical_columns_vocab_list'],
            crossed_feat_dim=1000
            )
        if verbose: 
            print("Wide feature specs:")
            for c in wide_columns:
                print("\t", str(c)[:100], "...")
            print("Deep feature specs:")
            for c in deep_columns:
                print("\t", str(c)[:100], "...")
        return wide_columns, deep_columns


    def _create_model(self, wide_columns, deep_columns):
        model = wide_deep.build_model(
            model_dir=self.params['model_dir'],
            wide_columns=wide_columns,
            deep_columns=deep_columns,
            linear_optimizer=tf_utils.build_optimizer(self.params['linear_optimizer'], self.params['linear_optimizer_lr'], **{
                'l1_regularization_strength': 0,
                'l2_regularization_strength': 0,
                'momentum': 0,
            }),
            dnn_optimizer=tf_utils.build_optimizer(self.params['dnn_optimizer'], self.params['dnn_optimizer_lr'], **{
                'l1_regularization_strength': 0,
                'l2_regularization_strength': 0,
                'momentum': 0,  
            }),
            dnn_hidden_units=self.params['dnn_hidden_units'],
            dnn_dropout=self.params['dnn_dropout'],
            dnn_batch_norm=(self.params['dnn_batch_norm']==1),
            log_every_n_iter=self.params['batch_size'],  # log every epoch
            # save_checkpoints_steps=save_checkpoints_steps, 
            seed=self.params['seed']
        )
        return model


    def get_recommendation(self, user_list, k):
        if self.model is None:
            print('WideDeepRecommender: Model not trained yet. Call train_model() first.')
            return
        
        # Prepare ranking evaluation set, i.e. get the cross join of all user-item pairs
        ranking_pool = user_item_pairs(
            user_df=self.user_feature.loc[self.user_feature['userID'].isin(user_list)],
            item_df=self.item_feature,
            user_col='userID',
            item_col='itemID',
            user_item_filter_df=self.train,  # Remove seen items
            shuffle=True,
            seed=SEED
        )
        predictions = list(self.model.predict(input_fn=tf_utils.pandas_input_fn(df=ranking_pool)))
        prediction_df = ranking_pool.copy()
        prediction_df['prediction'] = [p['predictions'][0] for p in predictions]

        pred_ls = prediction_df[['userID', 'itemID', 'prediction']]\
            .sort_values('prediction', ascending=False)\
            .groupby('userID').head(k)[['userID', 'itemID']]\
            .groupby('userID')['itemID'].apply(lambda x: list(x)).reset_index()\
            .sort_values('userID', ascending=True)['itemID'].tolist()
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

        