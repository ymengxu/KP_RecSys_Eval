import dask.dataframe as dd
import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.preprocessing import MultiLabelBinarizer
import torch

import dgl
import os
from core.neural_based_methods.pinsage.builder import PandasGraphBuilder


def build_train_graph(g, train_indices, utype, itype, etype, etype_rev):
    train_g = g.edge_subgraph(
        {etype: train_indices, etype_rev: train_indices}, relabel_nodes=False
    )
    # copy features
    for ntype in g.ntypes:
        for col, data in g.nodes[ntype].data.items():
            train_g.nodes[ntype].data[col] = data
    for etype in g.etypes:
        for col, data in g.edges[etype].data.items():
            train_g.edges[etype].data[col] = data[
                train_g.edges[etype].data[dgl.EID]
            ]

    return train_g


def linear_normalize(values):
    return (values - values.min(0, keepdims=True)) / (
        values.max(0, keepdims=True) - values.min(0, keepdims=True)
    )


def prepare_data(
    info:dict, dataset_name:str, train_path, val_path, test_path, user_feature_path, item_feature_path,
    use_text_feature=True, use_no_feature=False, use_only_text=False
):
    """prepare data to feed into the pinsage model for adobe_core5, movielens dataset"""

    assert not (use_text_feature and use_no_feature), "get_data_pinsage: argument conflict"
    assert not (use_only_text and use_no_feature), "get_data_pinsage: argument conflict"
    assert not (not use_text_feature and use_only_text), "get_data_pinsage: argument conflict"
    assert not (use_only_text and dataset_name=='movielens_100k'), "get_data_pinsage: argument conflict"

    # train, test, val
    train = pd.read_csv(train_path)[[info['user'], info['item'], info['time']]]
    train['mask'] = 0
    interactions = train.copy()
    train_indices = np.array(interactions.loc[interactions['mask'] == 0].index)
    interactions[info['time']] = pd.to_datetime(interactions[info['time']])
    interactions[info['time']] = interactions[info['time']].map(lambda x: int(x.strftime("%s")))

    val = pd.read_csv(val_path)[[info['user'], info['item']]]\
                    .groupby(info['user'])[info['item']]\
                    .agg(lambda x: list(set(x))).reset_index()
    val_user = val[info['user']].tolist()
    val_ytrue = val[info['item']].tolist()
    test = pd.read_csv(test_path)[[info['user'], info['item']]]\
                    .groupby(info['user'])[info['item']]\
                    .agg(lambda x: list(set(x))).reset_index()
    test_user = test[info['user']].tolist()
    test_ytrue = test[info['item']].tolist()
        
    if dataset_name == 'adobe_core5':
        # item features
        item_feature = sps.load_npz(item_feature_path)
        # text tf-idf feature
        item_feature = item_feature[:,item_feature.shape[0]:]
        item_feature = pd.DataFrame(item_feature.todense()).reset_index().rename(columns={'index':info['item']})
        item_feature_new = item_feature[[info['item']]].copy()
        item_feature_new['text'] = item_feature.iloc[:,1:-82].values.tolist()
        assert len(item_feature_new['text'].iloc[0]) == 7595
        item_feature_new['topics'] = item_feature.iloc[:,-82:].values.tolist()
        assert len(item_feature_new['topics'].iloc[0]) == 82
        # user features
        user_feature = pd.read_csv(user_feature_path)[[info['user'], 'age', 'gender']]\
            .sort_values(info['user'],ascending=True)
        # normalize age
        user_feature['age'] = (user_feature['age']-user_feature['age'].mean())/user_feature['age'].std()

    if dataset_name == 'movielens_100k':
        # item feature
        item_feature = pd.read_csv(item_feature_path)[[info['item'], 'genre', 'year']]
        genre_ls = item_feature['genre'].str.split('|').map(lambda x: list(set(x)))
        mlb = MultiLabelBinarizer(sparse_output=True)
        feature_genre = pd.DataFrame.sparse.from_spmatrix(
            mlb.fit_transform(genre_ls),
            index=item_feature.index,
            columns=mlb.classes_
        ).reset_index(drop=True).iloc[:,1:]
        feature_genre = feature_genre.values.tolist()
        feature_year = item_feature['year'].fillna(item_feature['year'].mean()).astype(int)
        feature_year = pd.get_dummies(feature_year, columns=['year']).values.tolist()
        item_feature_new = item_feature[[info['item']]].copy()
        item_feature_new['year'] = feature_year
        item_feature_new['genre'] = feature_genre
        # user feature
        user_feature = pd.read_csv(user_feature_path)[[info['user'], 'age', 'gender', 'occupation']]\
            .sort_values(info['user'], ascending=True)
        user_feature['occupation'] = pd.get_dummies(user_feature['occupation'], columns=['occupation']).values.tolist()
        
    # build graph
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(user_feature, info['user'], "user")
    graph_builder.add_entities(item_feature, info['item'], "item")
    graph_builder.add_binary_relations(
        interactions, info['user'], info['item'], "clicked"
    )
    graph_builder.add_binary_relations(
        interactions, info['item'], info['user'], "clicked-by"
    )
    g = graph_builder.build()

    # Assign features
    if not use_no_feature:
        if use_only_text:
            if dataset_name=='adobe_core5': 
                g.nodes["item"].data["text"] = torch.FloatTensor(
                    np.array(item_feature_new["text"].tolist())
                )
        else: 
            g.nodes["user"].data["gender"] = torch.LongTensor(
                user_feature["gender"].astype(int).tolist()
            )
            g.nodes["user"].data["age"] = torch.FloatTensor(
                user_feature['age'].tolist()  # Normalized age
            )
            
            if dataset_name == 'adobe_core5':
                # item: text, topic; user: age, gender
                if use_text_feature:
                    g.nodes["item"].data["text"] = torch.FloatTensor(
                        np.array(item_feature_new["text"].tolist())
                    )
                g.nodes["item"].data["topics"] = torch.FloatTensor(
                    np.array(item_feature_new["topics"].tolist())
                )
            if dataset_name == 'movielens_100k':
                # item: genre, year; user: age, gender, occupation
                g.nodes["user"].data["occupation"] = torch.FloatTensor(
                    np.array(user_feature['occupation'].tolist())
                )
                g.nodes["item"].data["genre"] = torch.FloatTensor(
                    np.array(item_feature_new["genre"].tolist())
                )
                g.nodes["item"].data["year"] = torch.FloatTensor(
                    np.array(item_feature_new["year"].tolist())
                )

    g.edges["clicked"].data["timestamp"] = torch.LongTensor(
        interactions[info['time']].values
    )
    g.edges["clicked-by"].data["timestamp"] = torch.LongTensor(
        interactions[info['time']].values
    )
    # Build the graph with training interactions only.
    train_g = build_train_graph(
        g, train_indices, "user", "item", "clicked", "clicked-by"
    )

    # at least one training interaction for each user
    assert train_g.out_degrees(etype="clicked").min() > 0

    dataset = {
        "train-graph": train_g,
        "val_user": val_user,
        "val_ytrue": val_ytrue,
        "test_user": test_user,
        "test_ytrue": test_ytrue,
        "item-texts": None,
        "item-images": None,
        "user-type": "user",
        "item-type": "item",
        "user-to-item-type": "clicked",
        "item-to-user-type": "clicked-by",
        "timestamp-edge-column": "timestamp",
    }
    
    return dataset