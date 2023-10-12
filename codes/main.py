import argparse
import torch
import numpy as np
from tqdm import tqdm
import torch_geometric
from torch_geometric.data import DataLoader
from torch.nn import MSELoss
import pickle
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from graphCL import ExtractCL
import time
from torch_scatter import scatter_sum

def load_data(args):
    data_folder_path = '../data/'+args.dataset

    adjs_path = data_folder_path+'/'+args.dataset+'_A.txt'
    graph_indicators_path = data_folder_path+'/'+args.dataset+'_graph_indicator.txt'
    graph_labels_path = data_folder_path+'/'+args.dataset+'_graph_labels.txt'
    node_attributes_path = data_folder_path+'/'+args.dataset+'_node_attributes.txt'

    adjs = np.loadtxt(adjs_path, dtype=int, delimiter=',')
    graph_indicators = np.loadtxt(graph_indicators_path, dtype=int, delimiter=',')
    graph_labels = np.loadtxt(graph_labels_path, dtype=int, delimiter=',')
    node_attributes = np.loadtxt(node_attributes_path, delimiter=',')

    return adjs, graph_indicators, graph_labels, node_attributes

def load_nx_graphs(args):
    data_path = '../data/'+args.dataset+'/nx_graphs.pkl'
    with open(data_path, 'rb') as f:
        graphs_list = pickle.load(f)
    return graphs_list

def graph_labels_process(args, graph_labels):
    # processed_labels = [[0, 0] for i in range(len(graph_labels))]
    # for i in range(len(graph_labels)):
    #     if args.dataset == 'PROTEINS_full':
    #         if graph_labels[i] == 1:
    #             processed_labels[i] = 0
    #         else:
    #             processed_labels[i] = 1
    #     elif args.dataset == 'FRANKENSTEIN':
    #         if graph_labels[i] == 1:
    #             processed_labels[i] = 1
    #         else:
    #             processed_labels[i] = 0
    #     elif args.dataset == 'AIDS':
    #         if graph_labels[i] == 1:
    #             processed_labels[i] = 1
    #         else:
    #             processed_labels[i] = 0
    # return processed_labels
    return graph_labels

def extrach_and_cl(args, attrs_dim, original_data):
    original_dataloader = DataLoader(original_data, batch_size=args.batch_size)
    if args.cuda:
        original_dataloader = [batch.cuda() for batch in original_dataloader]
    else:
        original_dataloader = [batch.cpu() for batch in original_dataloader]

    model = ExtractCL(args, attrs_dim)
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.lr
    )
    mse_loss = MSELoss()

    graph_embeddings = []
    graph_labels = []

    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        for i in range(len(original_dataloader)):
            pbar.set_description('Graph Contrastive Learning Epoch %d Batch %d...' % (epoch, i))

            original_batch = original_dataloader[i]

            optimizer.zero_grad()

            proximity_knowledge_mean, proximity_recovery, original_proximity, feature_knowledge_mean, \
            feature_recovery, original_feature, \
            p_e_xs, p_d_xs, f_e_xs, f_d_xs = model(original_batch)

            recovery_loss = (mse_loss(proximity_recovery, original_proximity) + \
            mse_loss(feature_recovery, original_feature))/(2*args.batch_size)

            contrastive_loss = cl_loss(args, proximity_knowledge_mean, feature_knowledge_mean)

            # kd_loss = (distillation_loss(p_e_xs)+distillation_loss(p_d_xs)+\
                       # distillation_loss(f_e_xs)+distillation_loss(f_d_xs))/4

            loss = recovery_loss + contrastive_loss #+ args.gamma*kd_loss

            loss.backward()
            optimizer.step()

            pbar.set_postfix(recovery_loss=recovery_loss.item(), cl_loss=contrastive_loss.item())

            if epoch == args.epochs-1:
                # graph_embeddings.append(proximity_knowledge_mean+feature_knowledge_mean)
                # graph_embeddings.append((proximity_knowledge_mean+feature_knowledge_mean)/2)
                graph_embeddings.append(torch.cat([proximity_knowledge_mean, feature_knowledge_mean], dim=1))
                # graph_embeddings.append(proximity_knowledge_mean)
                # graph_embeddings.append(feature_knowledge_mean)
                graph_labels.append(original_batch.label)

    graph_embeddings = torch.cat(graph_embeddings, dim=0).cpu().detach().numpy()
    graph_labels = np.concatenate(graph_labels)

    return graph_embeddings, graph_labels

def negative_sampling(args, embeddings):
    batch_negative_embeddings = []

    for i in range(len(embeddings)):
        if i == 0:
            pools = embeddings[1:]
        elif i == len(embeddings)-1:
            pools = embeddings[:-1]
        else:
            pools = torch.cat([embeddings[:i], embeddings[i+1:]], dim=0)
        random_indices = list(np.random.randint(0, len(pools), size=args.K))
        negative_embeddings = pools[random_indices, :]
        batch_negative_embeddings.append(negative_embeddings)
        
    batch_negative_embeddings = torch.cat(batch_negative_embeddings, dim=0)

    return batch_negative_embeddings

def cl_loss(args, proximity_knowledge_mean, feature_knowledge_mean):
    pos_pair = torch.cosine_similarity(proximity_knowledge_mean, feature_knowledge_mean)
    pos_pair = torch.exp(pos_pair/args.temp)
    
    p_konwledge_repeat = torch.reshape(proximity_knowledge_mean.repeat(1, args.K), (-1, args.hidden_dim))
    f_knowledge_repeat = torch.reshape(feature_knowledge_mean.repeat(1, args.K), (-1, args.hidden_dim))

    p_negative_samples = negative_sampling(args, proximity_knowledge_mean)
    f_negative_samples = negative_sampling(args, feature_knowledge_mean)

    p_neg_pair = torch.cosine_similarity(p_konwledge_repeat, p_negative_samples)
    f_neg_pair = torch.cosine_similarity(f_knowledge_repeat, f_negative_samples)
    p_neg_pair = torch.exp(p_neg_pair/args.temp)
    f_neg_pair = torch.exp(f_neg_pair/args.temp)

    indicator = [i for i in range(len(proximity_knowledge_mean))]
    indicator = torch.tensor([val for val in indicator for i in range(args.K)]).cuda()
    p_neg_pair = scatter_sum(p_neg_pair, indicator, dim=0)
    f_neg_pair = scatter_sum(f_neg_pair, indicator, dim=0)

    loss = -torch.log(pos_pair/(pos_pair+p_neg_pair))-torch.log(pos_pair/(pos_pair+f_neg_pair))

    return loss.mean()

def distillation_loss(xs):
    l = len(xs)
    loss_func = MSELoss()
    loss = 0
    for k in range(0, l-2):
        for t in range(k+1, l-1):
            loss = loss + loss_func(xs[t], xs[k])
    loss = loss/sum([i for i in range(l)])
    return loss

def svc_classify(args, x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True)
    f1mis = []
    f1mas = []
    for train_index, test_index in kf.split(x, y):

        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='f1_micro', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        
        f1mis.append(f1_score(y_test, classifier.predict(x_test), average="micro"))
        f1mas.append(f1_score(y_test, classifier.predict(x_test), average="macro"))

    return f1mis, f1mas

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--random_seed', type=int, default=12345)
    parser.add_argument('--dataset', type=str, default='PROTEINS_full')
    parser.add_argument('--gnn_layers_num', type=int, default=3)
    parser.add_argument('--structure_gnn', type=str, default='GCN')
    parser.add_argument('--feature_gnn', type=str, default='GCN')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--temp', type=float, default=1)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=1.0)

    args = parser.parse_args()

    adjs, graph_indicators, graph_labels, node_attributes = load_data(args)
    nx_graphs = load_nx_graphs(args)
    pyg_graphs = []

    pbar = tqdm(range(len(graph_labels)))
    for i in pbar:
        pbar.set_description('Converting the %d-th networkx graphs to PyG graphs...' % (i+1))
        pyg_G = torch_geometric.utils.from_networkx(nx_graphs[i])
        pyg_G.id = i
        pyg_G.label = graph_labels[i]
        pyg_graphs.append(pyg_G)

    graphs_num = len(pyg_graphs)
    nodes_num_list = [each.num_nodes for each in pyg_graphs]
    attrs_dim = pyg_graphs[0].attrs.size()[1]

    graph_embeddings, graph_labels = extrach_and_cl(args, attrs_dim, pyg_graphs)

    f1mis, f1mas = svc_classify(args, graph_embeddings, graph_labels, False)
    print('Micros', np.mean(f1mis), np.std(f1mis))
    print('Macros', np.mean(f1mas), np.std(f1mas))