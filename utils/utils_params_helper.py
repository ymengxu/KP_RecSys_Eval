from core.non_personalized_methods.popularity_recommender import PopularityRecommender
from core.similarity_based_methods.knn_recommender import KNNRecommender
from core.matrix_factorization_methods.i_als_recommender import iALSRecommender
from core.matrix_factorization_methods.bpr_cornac_recommender import BPRCornacRecommender
from core.neural_based_methods.ncf.ncf_recommender import NCFRecommender
from core.neural_based_methods.bivae.bivae_recommender import BiVAERecommender
from core.neural_based_methods.lightgcn_recommender import LightGCNRecommender

from core.light_fm.light_fm_model import LightFMRecommender
from core.neural_based_methods.pinsage.pinsage_recommender import PinSageRecommender
from core.tensor_factorization_methods.tensor_factorization_model import TensorRecommender

from core import SEED, TEST_K, DATA_CLEAN_PATH
from utils.evaluation import get_test_results
import os
from time import time


# optimal params
PARAMS_DICT = {
    'UserKNN': {
        'base': 'user', 
        'metric': 'cosine', 
        'n_neighbors': 96
    }, 
    'ItemKNN': {
        'base': 'user', 
        'metric': 'cosine', 
        'n_neighbors': 50
    }, 
    'iALS': {
        'factors': 19,
        'alpha': 17.053,
        'regularization': 0.0094,
        'iterations': 120, 
        'early_stopping': False, 
        'verbose': False,
        'seed': SEED,
    }, 
    'BPR': {
        'k': 150,
        'learning_rate': 0.04326846,
        'lambda_reg': 0.00113056,
        'max_iter': 185,
        'early_stopping': False, 
        'verbose': False,
        'seed': SEED,
    },
    'NCF': {
        'n_factors': 32,
        'learning_rate': 2.77819E-05,
        'batch_size': 128,
        'n_neg': 4,
        'epochs': 75,
        'early_stopping': False, 
        'verbose': False,
        'seed': SEED,
        'train_file_save_folder': DATA_CLEAN_PATH if os.path.exists(DATA_CLEAN_PATH) else '/tmp/model_evaluation_data/data_clean'
    }, 
    'BiVAE': {
        'k': 95,
        'batch_size': 512,
        'learning_rate': 0.038952562,
        'act_fn': 'tanh',
        'likelihood': 'pois',
        'n_epochs': 300,
        'early_stopping': False, 
        'verbose': False,
        'seed': SEED,
    }, 
    'LightGCN': {
        'embed_size': 62,
        'n_layers': 3,
        'batch_size': 64, 
        'decay': 0.009455433,
        'learning_rate': 0.000745953, 
        'epochs': 110, 
        'early_stopping': False, 
        'verbose': False,
        'seed': SEED,
    }, 
    'LightFM': {
        'loss': 'warp', 
        'no_components': 192, 
        'learning_rate': 0.04043460293502267, 
        'user_alpha': 0.006574446903908164, 
        'item_alpha': 0.00011170676471601027, 
        'epochs': 10,
        'early_stopping': False, 
        'verbose': False,
        'seed': SEED,
    }, 
    'PinSage': {
        'random_walk_length': 4,
        'random_walk_restart_prob': 0.4,
        'num_random_walks': 10,
        'num_neighbors': 6,
        'num_layers': 4,
        'hidden_dims': 16, 
        'lr': 0.000259046,
        'batch_size': 64,
        'epochs': 50,
        'early_stopping': False, 
        'verbose': False,
        'device': 'cpu',
    },

    'Tensor':{
        'user_item_k': 195, 
        'item_time_k': 87, 
        'batch_size': 1024, 
        'lr': 0.008445611008236313, 
        'lambda_c': 0.08363347876469694, 
        'lambda_r': 0.2, 
        'n_neg': 8, 
        'n_epochs': 40,
        'early_stopping': False, 
        'verbose': False,
    } 
}

ABLATION_PARAMS_DICT = {
    'LightFM_allfeature': {
        'loss': 'warp', 
        'no_components': 192, 
        'learning_rate': 0.04043460293502267, 
        'user_alpha': 0.006574446903908164, 
        'item_alpha': 0.00011170676471601027, 
        'epochs': 10,
        'early_stopping': False, 
        'verbose': False,
        'seed': SEED,
    }, 
    'LightFM_nofeature': {
        'loss': 'warp', 
        'no_components': 119, 
        'learning_rate': 0.014114830848757563, 
        'user_alpha': 0.00013230333459852668, 
        'item_alpha': 0.00333524442961255, 
        'epochs': 145,
        'early_stopping': False, 
        'verbose': False,
        'seed': SEED,
    }, 
    'LightFM_notext': {
        'loss': 'warp', 
        'no_components': 48, 
        'learning_rate': 0.007371166806571563, 
        'user_alpha': 0.000428232067636028, 
        'item_alpha': 4.799137238064358e-05, 
        'epochs': 125,
        'early_stopping': False, 
        'verbose': False,
        'seed': SEED,
    }, 
    'LightFM_onlytext': {
        'loss': 'warp', 
        'no_components': 60, 
        'learning_rate': 0.013581997554715388, 
        'user_alpha': 0.0009409737347609889, 
        'item_alpha': 2.0688379989557465e-05, 
        'epochs': 75,
        'early_stopping': False, 
        'verbose': False,
        'seed': SEED,
    }, 
    'PinSage_allfeature': {
        'random_walk_length': 4,
        'random_walk_restart_prob': 0.4,
        'num_random_walks': 10,
        'num_neighbors': 6,
        'num_layers': 4,
        'hidden_dims': 16, 
        'lr': 0.000259046,
        'batch_size': 64,
        'epochs': 50,
        'early_stopping': False, 
        'verbose': False,
        'device': 'cpu',
    },
    'PinSage_nofeature': {
        'random_walk_length': 5,
        'random_walk_restart_prob': 0.6,
        'num_random_walks': 12,
        'num_neighbors': 8,
        'num_layers': 4,
        'hidden_dims': 227, 
        'lr': 0.00024556493427634285,
        'batch_size': 512,
        'epochs': 140,
        'early_stopping': False, 
        'verbose': False,
        'device': 'cpu',
    },
    'PinSage_notext': {
        'random_walk_length': 2,
        'random_walk_restart_prob': 0.6,
        'num_random_walks': 14,
        'num_neighbors': 9,
        'num_layers': 2,
        'hidden_dims': 176, 
        'lr': 0.00024715685313351013,
        'batch_size': 512,
        'epochs': 120,
        'early_stopping': False, 
        'verbose': False,
        'device': 'cpu',
    },
    'PinSage_notext': {
        'random_walk_length': 2,
        'random_walk_restart_prob': 0.6,
        'num_random_walks': 14,
        'num_neighbors': 9,
        'num_layers': 2,
        'hidden_dims': 176, 
        'lr': 0.00024715685313351013,
        'batch_size': 512,
        'epochs': 120,
        'early_stopping': False, 
        'verbose': False,
        'device': 'cpu',
    },
    'PinSage_onlytext': {
        'random_walk_length': 2,
        'random_walk_restart_prob': 0.7,
        'num_random_walks': 14,
        'num_neighbors': 7,
        'num_layers': 2,
        'hidden_dims': 186, 
        'lr': 6.525095758633623e-05,
        'batch_size': 128,
        'epochs': 55,
        'early_stopping': False, 
        'verbose': False,
        'device': 'cpu',
    },

    'Tensor_allfeature':{
        'user_item_k': 195, 
        'item_time_k': 87, 
        'batch_size': 1024, 
        'lr': 0.008445611008236313, 
        'lambda_c': 0.08363347876469694, 
        'lambda_r': 0.2, 
        'n_neg': 8, 
        'n_epochs': 40,
        'early_stopping': False, 
        'verbose': False,
    } 


}


# models 
MODEL_DICT = {
    'Popularity': PopularityRecommender,
    'UserKNN': KNNRecommender,
    'ItemKNN': KNNRecommender,
    'iALS': iALSRecommender, 
    'BPR': BPRCornacRecommender, 
    'NCF': NCFRecommender,
    'BiVAE': BiVAERecommender, 
    'LightGCN': LightGCNRecommender, 
    'LightFM': LightFMRecommender,
    'PinSage': PinSageRecommender,
    'Tensor': TensorRecommender,
}


def get_train_eval_time(
    model_name, 
    model_dict=MODEL_DICT, params_dict=PARAMS_DICT,
    data='adobe_core5', data_folder=DATA_CLEAN_PATH, K:list=TEST_K, verbose=True):
    """
    train and evaluate the recommender, output train and evaluation time
    Args:
        model_name: name of the model 
        model_dict: name to recommender class dictionary, default MODEL_DICT
        params_dict: name to model parameters dictionary, default PARAMS_DICT
        data: data used to train the recommender, default 'adobe_core5'
        data_folder: data folder that stores the data, default DATA_CLEAN_PATH
        K: a list of top-k integers specified for evaluation, default TEST_K
        verbose: whether to print train and eval time, default True
    Returns: train_time, eval_time
    """
    model = model_name.split('_')[0]
    assert model in model_dict.keys(), 'get_train_eval_time: model_name not valid'

    
    if len(model_name.split('_')) > 1: 
        # use different features
        feature_selection = model_name.split('_')[1]
        # print(feature_selection)
        if model != 'Tensor': 
            if feature_selection == 'allfeature': 
                rec = model_dict[model](data, data_folder, use_text_feature=True, use_no_feature=False, use_only_text=False)
            elif feature_selection == 'nofeature': 
                rec = model_dict[model](data, data_folder, use_text_feature=False, use_no_feature=True, use_only_text=False)
            elif feature_selection == 'notext': 
                rec = model_dict[model](data, data_folder, use_text_feature=False, use_no_feature=False, use_only_text=False)
            elif feature_selection == 'onlytext': 
                rec = model_dict[model](data, data_folder, use_text_feature=True, use_no_feature=False, use_only_text=True)
        else:
            if feature_selection == 'allfeature': 
                rec = model_dict[model](data, data_folder, use_text_feature=True, use_only_text=False)
            elif feature_selection == 'notext': 
                rec = model_dict[model](data, data_folder, use_text_feature=False, use_only_text=False)
            elif feature_selection == 'onlytext': 
                rec = model_dict[model](data, data_folder, use_text_feature=True, use_only_text=True)
    else:
        rec = model_dict[model](data, data_folder)
    
    train_start = time()
    if model_name == 'Popularity':
        rec.train_model()
    else:
        rec.train_model(**params_dict[model_name])
    train_end = time()

    eval_start = time()
    res = get_test_results(rec, K)
    eval_end = time()

    train_time = train_end - train_start
    eval_time = eval_end - eval_start

    if verbose:
        print('{} Recommender: Train time = {:.4f}s, Evaluation time = {:.4f}s'.format(model_name, train_time, eval_time))

    return train_time, eval_time, res