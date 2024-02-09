import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_add_pool

class GraphSage_small(torch.nn.Module):
    def __init__(self, num_features, num_classes, nhid=32, nlayer=5, dropout=0.3):
        super(GraphSage_small, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout
        self.pre = torch.nn.Sequential(torch.nn.Linear(num_features, nhid))
        self.bn_pre = torch.nn.BatchNorm1d(nhid)
        self.graph_convs = torch.nn.ModuleList()
        self.graph_convs.append(SAGEConv(nhid, nhid*2))
        self.bn_convs = torch.nn.ModuleList()
        self.bn_convs.append(torch.nn.BatchNorm1d(nhid*2))
        for l in range(nlayer - 1):
            self.graph_convs.append(SAGEConv(nhid*2, nhid*2))
            self.bn_convs.append(torch.nn.BatchNorm1d(nhid*2))
        self.post = torch.nn.Sequential(torch.nn.Linear(nhid*2, nhid*2), torch.nn.ReLU())
        self.bn_post = torch.nn.BatchNorm1d(nhid*2)
        self.readout = torch.nn.Sequential(torch.nn.Linear(nhid*2, num_classes))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.pre(x)
        x = self.bn_pre(x)
        x = F.relu(x)
        for i in range(len(self.graph_convs)):
            x = self.graph_convs[i](x, edge_index)
            x = self.bn_convs[i](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = global_add_pool(x, batch)
        x = self.post(x)
        x = self.bn_post(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.readout(x)
        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

class GraphSage_big(torch.nn.Module):
    def __init__(self, num_features, num_classes, nhid=64, nlayer=5, dropout=0.3):
        super(GraphSage_big, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout
        self.pre = torch.nn.Sequential(torch.nn.Linear(num_features, nhid))
        self.bn_pre = torch.nn.BatchNorm1d(nhid)
        self.graph_convs = torch.nn.ModuleList()
        self.graph_convs.append(SAGEConv(nhid, nhid*2))
        self.bn_convs = torch.nn.ModuleList()
        self.bn_convs.append(torch.nn.BatchNorm1d(nhid*2))
        for l in range(nlayer - 1):
            self.graph_convs.append(SAGEConv(nhid*2, nhid*2))
            self.bn_convs.append(torch.nn.BatchNorm1d(nhid*2))
        self.post = torch.nn.Sequential(torch.nn.Linear(nhid*2, nhid*2), torch.nn.ReLU())
        self.bn_post = torch.nn.BatchNorm1d(nhid*2)
        self.readout = torch.nn.Sequential(torch.nn.Linear(nhid*2, num_classes))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.pre(x)
        x = self.bn_pre(x)
        x = F.relu(x)
        for i in range(len(self.graph_convs)):
            x = self.graph_convs[i](x, edge_index)
            x = self.bn_convs[i](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = global_add_pool(x, batch)
        x = self.post(x)
        x = self.bn_post(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.readout(x)
        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)