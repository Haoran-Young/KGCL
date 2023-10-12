from torch import nn
from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GATConv
import torch
import time
import numpy as np

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


class Encoder(nn.Module):
    def __init__(self, args, attrs_dim, gnn='GCN'):
        super(Encoder, self).__init__()

        self.args = args
        if gnn == 'GCN':
            modulelist = [GCNConv(attrs_dim, args.hidden_dim)]
            for i in range(args.gnn_layers_num-1):
                modulelist.append(GCNConv(args.hidden_dim, args.hidden_dim))
            self.gnn_layers = nn.ModuleList(modulelist)
        if gnn == 'GraphSAGE':
            modulelist = [SAGEConv(attrs_dim, args.hidden_dim)]
            for i in range(args.gnn_layers_num-1):
                modulelist.append(SAGEConv(args.hidden_dim, args.hidden_dim))
            self.gnn_layers = nn.ModuleList(modulelist)
        if gnn == 'GAT':
            modulelist = [GATConv(attrs_dim, args.hidden_dim)]
            for i in range(args.gnn_layers_num-1):
                modulelist.append(GATConv(args.hidden_dim, args.hidden_dim))
            self.gnn_layers = nn.ModuleList(modulelist)
        if gnn == 'GIN':
            # self.gnn_layers = nn.ModuleList([GINConv(MLP(attrs_dim, [2*attrs_dim, 2*attrs_dim, attrs_dim])) for i in range(args.gnn_layers_num)])
            modulelist = [GINConv(MLP(attrs_dim, [2*attrs_dim, 2*attrs_dim, args.hidden_dim]))]
            for i in range(args.gnn_layers_num-1):
                modulelist.append(GINConv(MLP(args.hidden_dim, [2*args.hidden_dim, 2*args.hidden_dim, args.hidden_dim])))
            self.gnn_layers = nn.ModuleList(modulelist)

        self.activation = nn.Tanh()

    def forward(self, data):
        x = data.attrs.float()
        xs = []
        for i in range(self.args.gnn_layers_num):
            x = self.gnn_layers[i](x, data.edge_index)
            x = self.activation(x)
            xs.append(x)
        x_mean = scatter_mean(x, data.batch, dim=0)
        return x, x_mean, xs


class Decoder(nn.Module):
    """
    For proximity decoder, conduct torch.mm(x, x) and implement a sigmoid
    to recover the adjacency matrix.
    For feature decoder, add an extra layer convert hidden_dim to attrs_dim
    """
    def __init__(self, args, attrs_dim, gnn='GCN', mode='proximity'):
        super(Decoder, self).__init__()

        self.mode = mode
        self.args = args
        if gnn == 'GCN':
            modulelist = []
            if mode == 'feature':
                for i in range(args.gnn_layers_num):
                # for i in range(args.gnn_layers_num-1):
                    modulelist.append(GCNConv(args.hidden_dim, args.hidden_dim))
                modulelist.append(GCNConv(args.hidden_dim, attrs_dim))
            else:
                for i in range(args.gnn_layers_num):
                    modulelist.append(GCNConv(args.hidden_dim, args.hidden_dim))
            self.gnn_layers = nn.ModuleList(modulelist)
        if gnn == 'GraphSAGE':
            modulelist = []
            if mode == 'feature':
                for i in range(args.gnn_layers_num):
                # for i in range(args.gnn_layers_num-1):
                    modulelist.append(SAGEConv(args.hidden_dim, args.hidden_dim))
                modulelist.append(SAGEConv(args.hidden_dim, attrs_dim))
            else:
                for i in range(args.gnn_layers_num):
                    modulelist.append(SAGEConv(args.hidden_dim, args.hidden_dim))
            self.gnn_layers = nn.ModuleList(modulelist)
        if gnn == 'GAT':
            modulelist = []
            if mode == 'feature':
                for i in range(args.gnn_layers_num):
                # for i in range(args.gnn_layers_num-1):
                    modulelist.append(GATConv(args.hidden_dim, args.hidden_dim))
                modulelist.append(GATConv(args.hidden_dim, attrs_dim))
            else:
                for i in range(args.gnn_layers_num):
                    modulelist.append(GATConv(args.hidden_dim, args.hidden_dim))
            self.gnn_layers = nn.ModuleList(modulelist)
        if gnn == 'GIN':
            # self.gnn_layers = nn.ModuleList([GINConv(MLP(attrs_dim, [2*attrs_dim, 2*attrs_dim, attrs_dim])) for i in range(args.gnn_layers_num)])
            modulelist = []
            if mode == 'feature':
                for i in range(args.gnn_layers_num):
                # for i in range(args.gnn_layers_num-1):
                    modulelist.append(GINConv(MLP(args.hidden_dim, [2*args.hidden_dim, 2*args.hidden_dim, args.hidden_dim])))
                modulelist.append(GINConv(MLP(args.hidden_dim, [2*args.hidden_dim, 2*args.hidden_dim, attrs_dim])))
            else:
                for i in range(args.gnn_layers_num):
                    modulelist.append(GINConv(MLP(args.hidden_dim, [2*args.hidden_dim, 2*args.hidden_dim, args.hidden_dim])))
            self.gnn_layers = nn.ModuleList(modulelist)

        self.activation = nn.Tanh()

    def forward(self, data, knowledge):
        x = knowledge
        xs = []
        for i in range(self.args.gnn_layers_num):
            x = self.gnn_layers[i](x, data.edge_index)
            x = self.activation(x)
            xs.append(x)
        # x = scatter_mean(x, data.batch, dim=0)

        if self.mode == 'feature':
            x = self.gnn_layers[-1](x, data.edge_index)
            x = self.activation(x)
        
        if self.mode == 'proximity':
            x = torch.sigmoid(torch.mm(x, x.transpose(0, 1)))
        
            batch_index = data.batch.cpu().numpy()
            previous_index = 0
            mask_matrics = []
            for j in range(max(batch_index)+1):
                index = np.where(batch_index == j)[0][-1]+1
                mask_matrics.append(torch.ones(index-previous_index, index-previous_index).cuda())
                previous_index = index
            mask_matrix = torch.block_diag(*mask_matrics)
            x = torch.mul(x, mask_matrix)

        return x, xs
    

class ExtractCL(nn.Module):
    def __init__(self, args, attrs_dim):
        super(ExtractCL, self).__init__()

        self.proximity_encoder = Encoder(args, attrs_dim, gnn=args.structure_gnn)
        self.feature_encoder = Encoder(args, attrs_dim, gnn=args.feature_gnn)
        self.proximity_decoder = Decoder(args, attrs_dim, gnn=args.structure_gnn)
        self.feature_decoder = Decoder(args, attrs_dim, gnn=args.feature_gnn, mode='feature')

    def forward(self, data):
        proximity_knowledge, proximity_knowledge_mean, p_e_xs = self.proximity_encoder(data)
        proximity_recovery, p_d_xs = self.proximity_decoder(data, proximity_knowledge)
        i = data.edge_index
        v = torch.ones(i.size()[1]).cuda()
        original_proximity = torch.sparse_coo_tensor(i, v, proximity_recovery.size()).to_dense().float()

        feature_knowledge, feature_knowledge_mean, f_e_xs = self.feature_encoder(data)
        feature_recovery, f_d_xs = self.feature_decoder(data, feature_knowledge)
        
        return proximity_knowledge_mean, proximity_recovery, original_proximity, \
        feature_knowledge_mean, feature_recovery, data.attrs.float(), \
        p_e_xs, p_d_xs, f_e_xs, f_d_xs