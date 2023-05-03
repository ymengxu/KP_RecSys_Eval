import os
import pandas as pd

from recommenders.models.ncf.dataset import Dataset as NCFDataset

from core import MAIN_DIRECTORY, SEED
from core.neural_based_methods.ncf.microsoft_recommender_ncf import NCF

from utils.data_importer import DataImporter
from utils.evaluation import Evaluation
from utils.early_stopping import EarlyStopping


class NCFRecommender:
    def __init__(self, dataset_name, data_folder_path=os.path.join(MAIN_DIRECTORY, 'data_clean')):
        DI = DataImporter(dataset_name, data_folder_path)
        train, val_user, val_ytrue, test_user, test_ytrue = DI.get_data('ncf')
        self.dataset_name = dataset_name
        self.train = train
        self.val_user = val_user
        self.val_ytrue = val_ytrue
        self.test_user = test_user
        self.test_ytrue = test_ytrue
        self.model = None
        self.params = None
        self.data = None


    def train_model(self, n_factors, batch_size, learning_rate, n_neg, 
                    epochs=300, early_stopping=True, eval_step=5, consecutive_eval_threshold=5,
                    seed=SEED, verbose=False,
                    train_file_save_folder=os.path.join(MAIN_DIRECTORY, 'data_clean')):
        os.makedirs(train_file_save_folder, exist_ok=True)

        train_file = os.path.join(train_file_save_folder, self.dataset_name, 'ncf', 'train.csv')
        self.train.to_csv(train_file, index=False)
        data = NCFDataset(
            train_file=train_file, 
            # test_file=test_file, 
            n_neg=n_neg,
            seed=seed, 
            # overwrite_test_file_full=True,
            col_user='user_num',
            col_item='item_num',
            col_rating='rating'
        )
        self.data = data
        self.params = {
            'n_factors': int(n_factors), 
            'layer_sizes': [int(n_factors)*4, int(n_factors)*2, int(n_factors)],
            'n_epochs': epochs, 
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        if not early_stopping:
            # train model at once
            model = NCF(
                **self.params, 
                n_users=data.n_users, 
                n_items=data.n_items,
                model_type="NeuMF",
                seed=seed,
                verbose=verbose,
            )
            model.fit(data)
            self.model = model

        else:
            self.params['n_epochs'] = eval_step
            model = NCF(
                **self.params, 
                n_users=data.n_users, 
                n_items=data.n_items,
                model_type="NeuMF",
                seed=seed,
                verbose=verbose
            )
            steps = epochs//eval_step  # maximum evaluation times
            ES = EarlyStopping(consecutive_eval_threshold=consecutive_eval_threshold)
            for i in range(1,steps+1):
                model.fit(data)
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


    def get_recommendation(self, user_list:list, k:int):
        if self.model is None:
            print('NCFRecommender: Model not trained yet. Call train_model() first.')
            return

        users, items, preds = [], [], []
        item = list(self.train['item_num'].unique())
        for user in user_list:
            user = [user] * len(item) 
            users.extend(user)
            items.extend(item)
            preds.extend(list(self.model.predict(user, item, is_list=True)))

        all_predictions = pd.DataFrame(data={"user_num": users, "item_num":items, "prediction":preds})
        merged = pd.merge(self.train, all_predictions, on=["user_num", "item_num"], how="outer")
        all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

        all_predictions = all_predictions.sort_values('prediction', ascending=False)\
                .groupby('user_num').head(k)[['user_num', 'item_num']]\
                .groupby('user_num')['item_num'].apply(lambda x: list(x)).reset_index()\
                .sort_values('user_num', ascending=True)
        pred_ls = all_predictions['item_num'].tolist()
        
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