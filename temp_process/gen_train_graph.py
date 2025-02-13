import torch
import random
import numpy as np
import json

dataset = 'twosides'
mode = 'transductive'

train_pairs_path = f"./dataset/data/{dataset}/{mode}/{dataset}_train.json"

with open(train_pairs_path, 'r') as f:
    train_pairs = json.load(f)
    
pos_pairs = train_pairs['pos']
unduplicate_set = set()
for pair in pos_pairs:
    unduplicate_set.add(tuple(pair[:2]))
    unduplicate_set.add(tuple(pair[:2:-1]))

unduplicate_list = []
for pair in unduplicate_set:
    if len(pair) != 2:
        print(pair)
    else:
        unduplicate_list.append(list(pair))


ddi_pair = torch.tensor(unduplicate_list)

ddi_pair = ddi_pair.transpose(0, 1)
        
import pdb; pdb.set_trace()

torch.save(ddi_pair, f"./dataset/data/{dataset}/{mode}/train_graph.pt")