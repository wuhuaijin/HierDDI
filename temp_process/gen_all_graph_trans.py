import torch
import json

dataset = 'twosides'

train_pairs_path = f"./dataset/data/{dataset}/transductive/{dataset}_train.json"
valid_pairs_path = f"./dataset/data/{dataset}/transductive/{dataset}_valid.json"
test_pairs_path = f"./dataset/data/{dataset}/transductive/{dataset}_test.json"

with open(train_pairs_path, 'r') as f:
    train_pairs = json.load(f)

with open(valid_pairs_path, 'r') as f:
    valid_pairs = json.load(f)

with open(test_pairs_path, 'r') as f:
    test_pairs = json.load(f)

graph_pairs = train_pairs['pos'] + valid_pairs ['pos'] + test_pairs['pos'] + test_pairs['neg']

unduplicate_set = set()
for pair in graph_pairs:
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

ddi_data_path = f'./dataset/data/{dataset}/transductive/all_graph.pt'
torch.save(ddi_pair, ddi_data_path)