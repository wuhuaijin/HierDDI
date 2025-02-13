import argparse
import torch
import random
import numpy as np
from train.train_transductive import train as train_transductive
from train.train_inductive import train as train_inductive
from train.train_new_type_transductive import train as train_new_type_transductive
from train.train_new_type_inductive import train as train_new_type_inductive
from models.ddi_predictor import ddi_model
import os
from tensorboardX import SummaryWriter
    

dataset_to_abbr = {
    "drugbank" : "drugbank",
    "twosides": "twosides"
}

num_node_feats_dict = {"drugbank" : 75, "twosides" :53}
num_ddi_types_dict = {"drugbank" : 86, "twosides": 963}
num_drugs = {"drugbank":1706, "twosides":645}

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dataset", type=str, choices=[
            "drugbank", "twosides"
        ], default="drugbank"
    )
    parser.add_argument(
        "--kg_graph", type=str, choices=["all", "train"], default="all"
    )
    
    parser.add_argument("--inductive", action="store_true", default=False)
    
    parser.add_argument(
        "--gnn_model", type=str,
        choices=["GCN", "GAT", "GIN"], default="GIN"
    ) 
    
    parser.add_argument(
        "--tgnn_model", type=str,
        choices=["GCN", "GAT", "GIN"], default="GIN"
    ) # GAT for twosides
    
    
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--gnn_num_layers", type=int, default=3)
    
    parser.add_argument("--gat_num_heads", type=int, default=8)
    parser.add_argument("--gat_to_concat", action="store_true", default=False)
    
    parser.add_argument("--gin_nn_layers", type=int, default=5)
    
    parser.add_argument("--num_patterns", type=int, default=60)
    parser.add_argument("--attn_out_residual", action="store_true", default=False)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--pred_mlp_layers", type=int, default=3)
    parser.add_argument("--name", type=str, default=None)

    parser.add_argument(
        "--sub_drop_freq", type=str,
        choices=["half", "always", "never"], default="half"
    )
    parser.add_argument(
        "--sub_drop_mode", type=str, choices=[
            "rand_per_graph", "rand_per_batch",
            "biggest", "smallest"
        ], default="rand_per_graph"
    )

    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--num_epoch", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--nosub", action="store_true", default=False)

    args = parser.parse_args()
    
    args.dataset = dataset_to_abbr[args.dataset.lower()]
    args.num_node_feats = num_node_feats_dict[args.dataset]
    args.num_ddi_types = num_ddi_types_dict[args.dataset]
    args.num_drugs = num_drugs[args.dataset]
    
    args.mode = 'inductive' if args.inductive else 'transductive'

    if args.dataset=='twosides':
        graph = torch.load(f"./dataset/data/{args.dataset}/{args.mode}/all_graph.pt")
    elif args.dataset=='drugbank':
        if args.kg_graph == 'all':
            graph = torch.load(f"./dataset/data/{args.dataset}/{args.mode}/all_graph.pt")
        elif args.kg_graph =='train':
            graph = torch.load(f"/home/bujiazi/workspace/HDDI/dataset/{args.dataset}/{args.mode}/{args.fold}/train_graph.pt")

    set_all_seeds(seed = args.seed)
    
    print(args.device)
    model = ddi_model(args, graph).to(args.device)
    
    path = f"./save/{args.dataset}_{args.kg_graph}_{args.mode}_bgnn{args.gnn_model}_tgnn{args.tgnn_model}_bs{args.batch_size}_lr{args.lr}"
    
    if not os.path.exists(path):
        os.makedirs(path)
        
    log_path = path + "/log"
    summary_writer = SummaryWriter(log_path)
    
    if args.kg_graph == 'train':
        if not args.inductive:
            train_new_type_transductive(model, args, summary_writer)
        else:
            train_new_type_inductive(model, args, summary_writer)
    else:
        if not args.inductive:
            train_transductive(model, args, summary_writer)
        else:
            train_inductive(model, args, summary_writer)
    summary_writer.close()
