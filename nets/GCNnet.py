import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, GENConv, DeepGCNLayer, SAGPooling, \
    BatchNorm

class GCN_small(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=48):
        super(GCN_small, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels*4)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels*4)
        self.conv2 = GCNConv(hidden_channels*4, hidden_channels*4)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels*4)
        self.conv3 = GCNConv(hidden_channels*4, hidden_channels*4)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels*4)
        self.linear = torch.nn.Linear(hidden_channels*4, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x

class GCN_big(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=64):
        super(GCN_big, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels*2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels*2)
        self.conv3 = GCNConv(hidden_channels*2, hidden_channels*4)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels*4)
        self.conv4 = GCNConv(hidden_channels*4, hidden_channels*4)
        self.bn4 = torch.nn.BatchNorm1d(hidden_channels*4)
        self.linear = torch.nn.Linear(hidden_channels*4, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x