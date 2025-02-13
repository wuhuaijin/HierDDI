import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, JumpingKnowledge, global_mean_pool, SAGEConv, GCNConv
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool

class GIN(torch.nn.Module):
    def __init__(self,  
                 input_embedding=128,
                 output_embedding=128,
                 hidden=512, 
                 train_eps=True):
        super(GIN, self).__init__()
        self.train_eps = train_eps
        self.gin_conv1 = GINConv(
            nn.Sequential(
                nn.Linear(input_embedding, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                # nn.Linear(hidden, hidden),
                # nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                # nn.Linear(hidden, hidden),
                # nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        # self.gin_conv3 = GINConv(
        #     nn.Sequential(
        #         nn.Linear(hidden, hidden),
        #         nn.ReLU(),
        #         nn.Linear(hidden, hidden),
        #         nn.ReLU(),
        #         nn.BatchNorm1d(hidden),
        #     ), train_eps=self.train_eps
        # )

        self.lin1 = nn.Linear(hidden, output_embedding)
        # self.fc1 = nn.Linear(2 * hidden, 7) #clasifier for concat
        # self.fc2 = nn.Linear(hidden, 7)   #classifier for inner product



    def reset_parameters(self):

        self.fc1.reset_parameters()

        self.gin_conv1.reset_parameters()
        self.gin_conv2.reset_parameters()
        # self.gin_conv3.reset_parameters()
        self.lin1.reset_parameters()
        # self.fc1.reset_parameters()
        # self.fc2.reset_parameters()


    def forward(self, x, edge_index, p=0.5):

        x = self.gin_conv1(x, edge_index)
        x = self.gin_conv2(x, edge_index)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=p, training=self.training)

        return x


class GAT(torch.nn.Module):
    def __init__(self, 
                 input_embedding=128,
                 output_embedding=128,
                 hidden=512,
                 num_heads=4,
                 dropout=0.5):
        super(GAT, self).__init__()

        self.gat_conv1 = GATConv(
            in_channels=input_embedding,
            out_channels=hidden,
            heads=num_heads,
            dropout=dropout
        )
        self.gat_conv2 = GATConv(
            in_channels=num_heads * hidden,
            out_channels=hidden,
            heads=num_heads,
            dropout=dropout
        )
        
        self.lin1 = nn.Linear(num_heads * hidden, output_embedding)
        
        self.dropout = nn.Dropout(p=dropout)

    def reset_parameters(self):
        self.gat_conv1.reset_parameters()
        self.gat_conv2.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, x, edge_index, p):
        x = self.gat_conv1(x, edge_index)
        x = self.gat_conv2(x, edge_index)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=p, training=self.training)
        
        return x
