from torch import nn
from torch.nn import ParameterList
from torch.nn.parameter import Parameter
import torch
from torch_geometric.nn import GCNConv, GINConv
from torch_scatter import scatter_mean
import copy


class GNN(nn.Module):
    def __init__(self, args, attrs_dim):
        super(GNN, self).__init__()

        self.args = args
        if args.gnn == 'GCN':
            self.gnn_layers = nn.ModuleList([GCNConv(attrs_dim, attrs_dim) for i in range(args.gnn_layers_num)])
        if args.gnn == 'GIN':
            self.gnn_layers = nn.ModuleList([GINConv(MLP(attrs_dim, [2*attrs_dim, 2*attrs_dim, attrs_dim])) for i in range(args.gnn_layers_num)])
        self.activation = nn.Tanh()

    def forward(self, data):
        x = data.attrs.float()
        for i in range(self.args.gnn_layers_num):
            x = self.gnn_layers[i](x, data.edge_index)
            x = self.activation(x)
        x = scatter_mean(x, data.batch, dim=0)
        return x


class MLP(nn.Module):
    def __init__(self, attrs_dim, dim_list=[16, 8, 2]):
        super(MLP, self).__init__()

        attrs_dim = [attrs_dim]
        attrs_dim.extend(dim_list)
        self.layers = nn.ModuleList([nn.Linear(attrs_dim[i], attrs_dim[i+1]) for i in range(len(dim_list))])
        self.activation = nn.Tanh()

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.activation(x)
        return x


class Predictor(nn.Module):
    def __init__(self, args, attrs_dim):
        super(Predictor, self).__init__()

        self.gnn = GNN(args, attrs_dim)
        self.mlp = MLP(attrs_dim)
        # self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, data):
        x = self.gnn(data)
        graph_embedding = x
        x = self.mlp(x)
        # x = self.logsoftmax(x)
        return x, graph_embedding