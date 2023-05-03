import numpy as np
from numpy import *
import torch
import dgl


class LatestNNRecommender(object):
    def __init__(
        self, user_ntype, item_ntype, user_to_item_etype, timestamp, batch_size
    ):
        self.user_ntype = user_ntype
        self.item_ntype = item_ntype
        self.user_to_item_etype = user_to_item_etype
        self.batch_size = batch_size
        self.timestamp = timestamp

    def recommend(self, full_graph, user_list, K, h_user, h_item):
        """
        Return a (len(user_list), K) matrix of recommended items for each u in user_list
        """
        graph_slice = full_graph.edge_type_subgraph([self.user_to_item_etype])
        latest_interactions = dgl.sampling.select_topk(
            graph_slice, 1, self.timestamp, edge_dir="out"
        )
        _, latest_items = latest_interactions.all_edges(
            form="uv", order="srcdst"
        )
        
        recommended_batches = []
        user_batches = torch.LongTensor(user_list).split(self.batch_size)
        for user_batch in user_batches:
            latest_item_batch = latest_items[user_batch].to(
                device=h_item.device
            )
            dist = h_item[latest_item_batch] @ h_item.t()
            # exclude items that are already interacted
            for i, u in enumerate(user_batch.tolist()):
                interacted_items = full_graph.successors(
                    u, etype=self.user_to_item_etype
                )
                dist[i, interacted_items] = -np.inf
            recommended_batches.append(dist.topk(K, 1)[1])

        recommendations = torch.cat(recommended_batches, 0)
        return recommendations


def evaluate_nn(dataset, user_list, h_item, k, batch_size):
    g = dataset["train-graph"]
    user_ntype = dataset["user-type"]
    item_ntype = dataset["item-type"]
    user_to_item_etype = dataset["user-to-item-type"]
    timestamp = dataset["timestamp-edge-column"]

    rec_engine = LatestNNRecommender(
        user_ntype, item_ntype, user_to_item_etype, timestamp, batch_size
    )
    
    recommendations = rec_engine.recommend(g, user_list, k, None, h_item).cpu().numpy().tolist()
    return recommendations
