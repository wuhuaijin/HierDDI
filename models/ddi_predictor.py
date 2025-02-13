# -*- coding: utf-8 -*-

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_add_pool
from models.ddi_layers import GINConv, MLP, SubExtractor
from models.knowledge_graph_nn import GIN, GAT
from tqdm import tqdm

def build_knowledge_graph(data):
    edges= []
    print("building knowledge graph......")
    for i in tqdm(range(len(data[0]))):
        node_1 = data[0][i]
        node_2 = data[1][i]
        if [node_1, node_2] in edges:
            continue
        else:
            edges.append([node_1, node_2])
    edges = torch.tensor(edges, dtype = torch.long).t()
    # import pdb; pdb.set_trace()
    return edges

def get_gnn_model(args):
    gnn_model = args.gnn_model
    hidden_dim = args.hidden_dim
    gnn_num_layers = args.gnn_num_layers

    if gnn_model == "GCN":
        return nn.ModuleList([GCNConv(
            hidden_dim, hidden_dim
        ) for i in range(gnn_num_layers)])
    
    elif gnn_model == "GAT":
        gat_num_heads = args.gat_num_heads
        gat_to_concat = args.gat_to_concat
        
        if gat_to_concat:
            return nn.ModuleList([GATConv(
                hidden_dim, hidden_dim // gat_num_heads, gat_num_heads
            ) for i in range(gnn_num_layers)])
        else:
            return nn.ModuleList([GATConv(
                hidden_dim, hidden_dim, gat_num_heads, False
            ) for i in range(gnn_num_layers)])
    
    elif gnn_model == "GIN":
        gin_nn_layers = args.gin_nn_layers
        return nn.ModuleList([GINConv(
            hidden_dim, gin_nn_layers
        ) for i in range(gnn_num_layers)])
    
    else:
        raise NotImplementedError("Unsupported GNN Model")

class InteractionPredictor(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        hidden_dim = args.hidden_dim
        num_node_feats = args.num_node_feats
        num_ddi_types = args.num_ddi_types
        pred_mlp_layers = args.pred_mlp_layers
        
        num_patterns = args.num_patterns
        dropout = args.dropout
        
        self.sub_drop_freq = args.sub_drop_freq
        self.sub_drop_mode = args.sub_drop_mode
        self.device = args.device
        
        self.hidden_dim = hidden_dim
        self.num_patterns = num_patterns
        self.num_ddi_types = num_ddi_types
        
        self.node_fc = nn.Linear(num_node_feats, hidden_dim)
        self.gnn = get_gnn_model(args)
        self.pool = SubExtractor(hidden_dim, num_patterns, args.attn_out_residual)
        self.emb = torch.zeros((args.num_drugs, self.num_patterns + 1, self.hidden_dim)).to(self.device)

        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + num_ddi_types, hidden_dim),
            MLP(hidden_dim, pred_mlp_layers, dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.drop_rate_list = []
        
        self.MLP = nn.Sequential(
                nn.Linear((num_patterns+1) * hidden_dim, hidden_dim),
                MLP(hidden_dim, pred_mlp_layers, dropout),
                nn.Linear(hidden_dim, 1)
            )
        
        if args.inductive:
            self.forward_func = self.forward_inductive
        else:
            self.forward_func = self.forward_transductive
    
    def do_gnn(self, x, edge_index):
        for gnn in self.gnn:
            x = gnn(x, edge_index)
        
        return x
    

    @torch.no_grad()
    def get_sub_to_drop(self, sub_cnt):
        if self.sub_drop_mode == "rand_per_graph":
            drop_prob = torch.rand(sub_cnt.size(0), self.num_patterns).to(self.device)
            drop_prob[sub_cnt == 0] = 0
            sub_to_drop = drop_prob.argmax(dim=-1, keepdim=True)
        
        elif self.sub_drop_mode == "rand_per_batch":
            sub_total = sub_cnt.sum(dim=0)
            # (num_patterns, )
            sub_non_zero = sub_total.nonzero().squeeze(-1).tolist()
            sub_to_drop = random.choice(sub_non_zero)
        
        elif self.sub_drop_mode == "smallest":
            sub_cnt[sub_cnt == 0] = sub_cnt.max().item() + 1
            sub_to_drop = sub_cnt.argmin(dim=-1, keepdim=True)
            # (batch_size, 1)
        
        elif self.sub_drop_mode == "biggest":
            sub_to_drop = sub_cnt.argmax(dim=-1, keepdim=True)
        
        else:
            raise NotImplementedError("Unsupported Sub drop mode")

        return sub_to_drop
    
    @torch.no_grad()
    def do_sub_drop(self, x, edge_index, batch):
        num_nodes = x.size(0)
        
        if num_nodes == 1:
            return x
        
        h = self.node_fc(x)
        h = self.do_gnn(h, edge_index)
        
        _, sub_assign, mask = self.pool(h, batch)
        # sub_assign: (batch_size, max_num_nodes)
        # mask:       (batch_size, max_num_nodes)
        
        sub_assign[~mask] = self.num_patterns
        # padded nodes assigned to a dummy pattern
        
        sub_one_hot = F.one_hot(sub_assign, self.num_patterns + 1)
        # (batch_size, max_num_nodes, num_patterns + 1)
        
        sub_cnt = sub_one_hot.sum(dim=1)[ : , : self.num_patterns]
        # (batch_size, num_patterns)

        sub_to_drop = self.get_sub_to_drop(sub_cnt)
        
        drop_mask = (sub_assign == sub_to_drop)
        # (batch_size, max_num_nodes)
        
        drop_mask = drop_mask[mask]
        # (num_nodes_batch, )
        
        x[drop_mask] = 0
        
        drop_rate = drop_mask.sum().item() / drop_mask.size(0)
        
        return x, drop_rate

    @torch.no_grad()
    def do_node_drop(self, x, drop_rate=0.2):
        if x.size(0) == 1:
            return x
        
        prob = torch.rand(x.size(0), 1).to(self.device)
        # (num_nodes_batch, 1)
        
        drop_mask = (prob < drop_rate)
        keep_mask = ~drop_mask
        # (num_nodes_batch, 1)
        
        drop_mask = drop_mask.float()
        keep_mask = keep_mask.float()
        
        x_out = keep_mask * x
        
        return x_out, drop_rate
    
    def encode_graph(self, graph_batch):
        to_drop = False
        
        if self.training:
            if self.sub_drop_freq == "half":
                to_drop = (random.random() > 0.5)
            elif self.sub_drop_freq == "always":
                to_drop = True
        
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        drug_idx_batch = graph_batch.drug
        
        if to_drop:
            x, drop_rate = self.do_sub_drop(x, edge_index, batch)
            #x, drop_rate = self.do_node_drop(x, drop_rate=0.2)
            self.drop_rate_list.append(drop_rate)
        
        x = self.node_fc(x)
        x = self.do_gnn(x, edge_index)
        
        out = global_add_pool(x, batch)
        
        pool, *_ = self.pool(x, batch)
        pool = F.normalize(pool, dim=-1)
        
        return out, pool, drug_idx_batch
        
    def predict(self, out_1, out_2, pool_1, pool_2, ddi_type):        
        
        ddi_type = list(ddi_type)
        ddi_type = torch.eye(self.num_ddi_types)[ddi_type].to(self.device)
        out = torch.cat([out_1, out_2, ddi_type], dim=-1)
        score = self.mlp(out).squeeze(-1)
        return score
    
    def forward_transductive(self, graph_batch_1, graph_batch_2, ddi_type):

        out_1, pool_1, idx_patch_1 = self.encode_graph(graph_batch_1)
        out_2, pool_2, idx_patch_2 = self.encode_graph(graph_batch_2)

        embedding_1 = torch.cat((out_1.unsqueeze(1), pool_1), dim=1)
        embedding_2 = torch.cat((out_2.unsqueeze(1), pool_2), dim=1)
        with torch.no_grad():
            self.emb[idx_patch_1] = embedding_1
            self.emb[idx_patch_2] = embedding_2
    
        score = self.predict(out_1, out_2, pool_1, pool_2, ddi_type)
        return score
    
    def forward_inductive(self, graph_batch_1, graph_batch_2, graph_batch_old_1, graph_batch_old_2, ddi_type):
        out_1, pool_1, idx_patch_1 = self.encode_graph(graph_batch_1)
        out_2, pool_2, idx_patch_2 = self.encode_graph(graph_batch_2)
        
        embedding_1 = torch.cat((out_1.unsqueeze(1), pool_1), dim=1)
        embedding_2 = torch.cat((out_2.unsqueeze(1), pool_2), dim=1)
        with torch.no_grad():
            self.emb[idx_patch_1] = embedding_1
            self.emb[idx_patch_2] = embedding_2
        
        score = self.predict(out_1, out_2, pool_1, pool_2, ddi_type)

        if self.training:
            return score
        
        old_out_1, old_pool_1, _ = self.encode_graph(graph_batch_old_1)
        old_out_2, old_pool_2, _ = self.encode_graph(graph_batch_old_2)
        
        score_old = self.predict(old_out_1, old_out_2, old_pool_1, old_pool_2, ddi_type)
        
        score = torch.stack([score, score_old]).sigmoid().mean(dim=0)
        
        return score

    # def forward_wo_type(self, graph_batch_1, graph_batch_2, dummy_ddi_type):
    #     out_1, pool_1 = self.encode_graph(graph_batch_1)
    #     out_2, pool_2 = self.encode_graph(graph_batch_2)
        
    #     sim = pool_1 @ pool_2.transpose(-1, -2)
    #     sim = sim.flatten(1)
        
    #     out = torch.cat([out_1, out_2, sim], dim=-1)
    #     score = self.mlp(out).squeeze(-1)
        
    #     return score
    
    

class EmbeddingMlp(torch.nn.Module):
    def __init__(self):
        super(EmbeddingMlp, self).__init__()
        self.fc1 = torch.nn.Linear(61*128, 1024)
        self.fc2 = torch.nn.Linear(1024, 61*128)

    def forward(self, x):
        x = x.view(-1, 61*128)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = x.view(-1, 61, 128)
        return x


class ScoreCombinationNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ScoreCombinationNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ddi_model(nn.Module):
    def __init__(self, args, graph):
        super(ddi_model, self).__init__()
        self.args = args
        self.BGNN = InteractionPredictor(args)
        if args.tgnn_model == 'GIN':
            self.TGNN = GIN(input_embedding=(args.num_patterns+1)*args.hidden_dim, output_embedding =(args.num_patterns + 1)*args.hidden_dim)
        elif args.tgnn_model == 'GAT':
            self.TGNN = GAT(input_embedding=(args.num_patterns+1)*args.hidden_dim, output_embedding =(args.num_patterns + 1)*args.hidden_dim)

        self.mlp = nn.Sequential(
                nn.Linear(2 * args.hidden_dim + args.num_ddi_types, args.hidden_dim),
                MLP(args.hidden_dim, args.pred_mlp_layers, args.dropout),
                nn.Linear(args.hidden_dim, 1)
            )

        self.score_combination_nn = ScoreCombinationNN(input_size=2, hidden_size=64, output_size=1)

        self.knowledge_graph = graph.to(args.device)
        self.embmlp = EmbeddingMlp()
        self.emb = torch.zeros((args.num_drugs, args.num_patterns + 1, args.hidden_dim)).to(args.device)
        
    
    def predict(self, args, out_1, out_2, pool_1, pool_2, ddi_type):

        ddi_type = list(ddi_type)
        ddi_type = torch.eye(args.num_ddi_types)[ddi_type].to(args.device)
        out = torch.cat([out_1, out_2, ddi_type], dim=-1)
        score = self.mlp(out).squeeze(-1)
        return score

    def forward(self, graph_batch_1, graph_batch_2, graph_batch_old_1, graph_batch_old_2, ddi_type):
        if self.args.inductive:
            score_1 = self.BGNN.forward_inductive(graph_batch_1, graph_batch_2, graph_batch_old_1, graph_batch_old_2, ddi_type)
        else:
            score_1 = self.BGNN.forward_transductive(graph_batch_1, graph_batch_2, ddi_type)

        self.emb = self.embmlp(self.BGNN.emb.detach())
        fin_emb = self.TGNN(self.emb.reshape(self.args.num_drugs, -1), self.knowledge_graph, p = 0.5)

        idx_patch_1, idx_patch_2 = graph_batch_1.drug, graph_batch_2.drug
        fin_emb_1 = fin_emb[idx_patch_1].reshape(-1, (self.args.num_patterns+1), self.args.hidden_dim)
        fin_emb_2 = fin_emb[idx_patch_2].reshape(-1, (self.args.num_patterns+1), self.args.hidden_dim)
        fin_out_1, fin_pool_1 = fin_emb_1[:, :1, :].squeeze(1), fin_emb_1[:, 1:, :]
        fin_out_2, fin_pool_2 = fin_emb_2[:, :1, :].squeeze(1), fin_emb_2[:, 1:, :]
        score_2 = self.predict(self.args, fin_out_1, fin_out_2, fin_pool_1, fin_pool_2, ddi_type)

        final = (score_1 + score_2) / 2

        return final