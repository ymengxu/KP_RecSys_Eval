import os
from time import time

import torch
import tqdm
from torch.utils.data import DataLoader
# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator

from core import MAIN_DIRECTORY
from core.neural_based_methods.pinsage.model import PinSAGEModel
import core.neural_based_methods.pinsage.sampler as sampler_module
import core.neural_based_methods.pinsage.evaluation as pinsage_evaluation

from utils.data_importer import DataImporter
from utils.evaluation import Evaluation
from utils.early_stopping import EarlyStopping


class PinSageRecommender:
    def __init__(self, dataset_name, data_folder_path=os.path.join(MAIN_DIRECTORY, 'data_clean'), 
                 use_text_feature=True, use_no_feature=False, use_only_text=False):
        DI = DataImporter(dataset_name, data_folder_path)
        dataset = DI.get_data('pinsage', use_text_feature, use_no_feature, use_only_text)
        self.dataset = dataset
        self.val_user = dataset['val_user']
        self.val_ytrue = dataset['val_ytrue']
        self.test_user = dataset['test_user']
        self.test_ytrue = dataset['test_ytrue']
        self.model = None
        self.params = None

        
    def train_model(self, 
                  random_walk_length, random_walk_restart_prob, num_random_walks, 
                  num_neighbors, num_layers, hidden_dims, lr, batch_size, 
                  epochs=200, 
                  early_stopping=False, eval_step=5, consecutive_eval_threshold=5, 
                  device='cuda', num_workers=0, verbose=False):
        # dataset
        g = self.dataset["train-graph"]
        item_texts = self.dataset["item-texts"]
        user_ntype = self.dataset["user-type"]
        item_ntype = self.dataset["item-type"]
        user_to_item_etype = self.dataset["user-to-item-type"]
        timestamp = self.dataset["timestamp-edge-column"]

        # params
        self.params = {
            'random_walk_length': int(random_walk_length),
            'random_walk_restart_prob': random_walk_restart_prob,
            'num_random_walks': int(num_random_walks),
            'num_neighbors': int(num_neighbors),
            'num_layers': int(num_layers),
            'hidden_dims': int(hidden_dims),
            'lr': lr,
            'batch_size': batch_size,
            'num_epochs': epochs,
            'device': device,
            'num_workers': num_workers
        }

        random_walk_length = self.params['random_walk_length']
        num_random_walks = self.params['num_random_walks']
        num_neighbors = self.params['num_neighbors']
        num_layers = self.params['num_layers']
        hidden_dims = self.params['hidden_dims']

        device = torch.device(device)

        # Assign user and item IDs and use them as features (to learn an individual trainable
        # embedding for each entity)
        g.nodes[user_ntype].data["id"] = torch.arange(g.num_nodes(user_ntype))
        g.nodes[item_ntype].data["id"] = torch.arange(g.num_nodes(item_ntype))

        # # Prepare torchtext dataset and Vocabulary (not used here)
        textset = {}
        # tokenizer = get_tokenizer(None)

        # textlist = []
        # batch_first = True

        # if item_texts is not None:
        #     for i in range(g.num_nodes(item_ntype)):
        #         for key in item_texts.keys():
        #             l = tokenizer(item_texts[key][i].lower())
        #             textlist.append(l)
        #     for key, field in item_texts.items():
        #         vocab2 = build_vocab_from_iterator(
        #             textlist, specials=["<unk>", "<pad>"]
        #         )
        #         textset[key] = (
        #             textlist,
        #             vocab2,
        #             vocab2.get_stoi()["<pad>"],
        #             batch_first,
        #         )
        

        # Sampler
        batch_sampler = sampler_module.ItemToItemBatchSampler(
            g, user_ntype, item_ntype, batch_size
        )
        neighbor_sampler = sampler_module.NeighborSampler(
            g,
            user_ntype,
            item_ntype,
            random_walk_length,
            random_walk_restart_prob,
            num_random_walks,
            num_neighbors,
            num_layers,
        )
        collator = sampler_module.PinSAGECollator(
            neighbor_sampler, g, item_ntype, textset
        )
        dataloader = DataLoader(
            batch_sampler,
            collate_fn=collator.collate_train,
            num_workers=num_workers,
        )
        dataloader_test = DataLoader(
            torch.arange(g.num_nodes(item_ntype)),
            batch_size=batch_size,
            collate_fn=collator.collate_test,
            num_workers=num_workers,
        )

        # stored test dataloader to get recommendations
        self.dataloader_test = dataloader_test

        dataloader_it = iter(dataloader)

        # Model
        model = PinSAGEModel(
            g, item_ntype, textset, hidden_dims, num_layers
        ).to(device)
        # Optimizer
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        batches_per_epoch = int(g.num_edges()/(2*batch_size))

        # For each batch of head-tail-negative triplets...
        if not early_stopping:
            for epoch_id in range(1,epochs+1):
                t0 = time()
                tot_loss = 0
                model.train()
                for batch_id in tqdm.trange(batches_per_epoch, disable=True):
                    pos_graph, neg_graph, blocks = next(dataloader_it)
                    # Copy to GPU
                    for i in range(len(blocks)):
                        blocks[i] = blocks[i].to(device)
                    pos_graph = pos_graph.to(device)
                    neg_graph = neg_graph.to(device)

                    loss = model(pos_graph, neg_graph, blocks).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    tot_loss += loss
                t1 = time()
                self.model = model
                if verbose:
                    print(
                        f"epoch {epoch_id} [{round(t1-t0,2)}s]:  training loss={tot_loss/batches_per_epoch}"
                    )
        else: 
            ES = EarlyStopping(consecutive_eval_threshold=consecutive_eval_threshold)
            for epoch_id in range(1,epochs+1):
                t0 = time()
                tot_loss = 0
                model.train()
                for batch_id in tqdm.trange(batches_per_epoch, disable=True):
                    pos_graph, neg_graph, blocks = next(dataloader_it)
                    # Copy to GPU
                    for i in range(len(blocks)):
                        blocks[i] = blocks[i].to(device)
                    pos_graph = pos_graph.to(device)
                    neg_graph = neg_graph.to(device)

                    loss = model(pos_graph, neg_graph, blocks).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    tot_loss += loss
                t1 = time()
                self.model = model
                # Evaluate
                if epoch_id%eval_step == 0:
                    eval_score = self.get_validation_ndcg()
                    t2 = time()
                    if verbose:
                        print(
                           f"epoch {epoch_id} [{round(t1-t0,2)}s]: training loss={tot_loss/batches_per_epoch}  validation ndcg@10={eval_score} [{round(t2-t1,2)}s]"
                        )
                    stop_flag = ES.log(epoch_id, eval_score)
                    if stop_flag:
                        break 
                elif verbose:
                    print(
                        f"epoch {epoch_id} [{round(t1-t0,2)}s]:  training loss={tot_loss/batches_per_epoch}"
                    )
            self.params['best_epoch'] = ES.best_epoch
            self.params['iterations'] = epoch_id
            self.params['best_eval_score'] = ES.best_evaluation_score


    def get_recommendation(self, user_list:list, k:int):
        if self.model is None:
            print('PinSageRecommender: Model not trained yet. Call train_model() first.')
            return

        dataloader_test = self.dataloader_test
        model = self.model

        # eval
        model.eval()
        with torch.no_grad():
            h_item_batches = []
            for blocks in dataloader_test:
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(self.params['device'])

                h_item_batches.append(model.get_repr(blocks))
            h_item = torch.cat(h_item_batches, 0)
            
            pred_ls = pinsage_evaluation.evaluate_nn(self.dataset, user_list, h_item, k, self.params['batch_size'])
        
        return pred_ls
    

    def get_validation_ndcg(self, k=10):
        pred_ls = self.get_recommendation(self.val_user, k=k)
        val_evaluator = Evaluation(batch_size = len(self.val_user), K=[k], how='val')
        mean_ndcg = val_evaluator.evaluate(pred_ls, self.val_ytrue)
        return mean_ndcg[0]
        
        
    def get_test_metrics(self, K):
        pred_ls = self.get_recommendation(self.test_user, max(K))
        test_evaluator = Evaluation(batch_size = len(self.test_user), K=K, how='test')
        mean_precision, mean_recall, HR, mean_ndcg, MRR, MAP = test_evaluator.evaluate(pred_ls, self.test_ytrue)
        return mean_precision, mean_recall, HR, mean_ndcg, MRR, MAP
        