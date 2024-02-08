import os.path as osp
import scipy.stats as stats
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import pandas as pd

import visualization
from aug import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
import sys
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin import Encoder
from evaluate_embedding import evaluate_embedding
from model import *
import time
from arguments import arg_parse
from torch.distributions.bernoulli import Bernoulli
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.utils.data as utils

from torch_geometric.transforms import Constant
import pdb

def renyi(p,q,alpha,get_softmax=True):
    if get_softmax:
        p = F.softmax(p,dim=1)
        q = F.softmax(q,dim=1)
    M=(p+q)/2
    p=torch.sign(p) * torch.pow(torch.abs(p), alpha)
    q=torch.sign(q) * torch.pow(torch.abs(q), alpha)
    M=torch.sign(M) * torch.pow(torch.abs(M), alpha-1)
    x=((p/M).sum().log())/(alpha-1)
    y=((q/M).sum().log())/(alpha-1)
    return (x+y)/2

def js_loss(p_output, q_output,get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output,dim=1)
        q_output = F.softmax(q_output,dim=1)
    log_mean_output = ((p_output + q_output )/2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2


class GcnInfomax(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(GcnInfomax, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # self.prior = args.prior

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs):

        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y, M = self.encoder(x, edge_index, batch)

        g_enc = self.global_d(y)
        l_enc = self.local_d(M)

        mode = 'fd'
        measure = 'JSD'
        local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)

        if self.prior:
            prior = torch.rand_like(y)
            term_a = torch.log(self.prior_d(prior)).mean()
            term_b = torch.log(1.0 - self.prior_d(y)).mean()
            PRIOR = - (term_a + term_b) * self.gamma
        else:
            PRIOR = 0

        return local_global_loss + PRIOR


class simclr(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, dataset_num_features, dataset_num, alpha=0.5, beta=1., gamma=.1,emb_dim=96, hidden=32):
        super(simclr, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.emb_dim = emb_dim

        self.dataset_num = dataset_num
        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))

        self.SDG = nn.Sequential(nn.Linear(self.emb_dim, self.hidden), ReLU(),
                                 nn.Linear(self.hidden), nn.Sigmoid())
        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    @torch.no_grad()
    def sample_negative_index(self, negative_number, epoch, epochs):

        lamda = 1/2
        lower, upper = 0, self.dataset_num
        mu_1 = ((epoch-1) / epochs) ** lamda * (upper - lower)
        mu_2 = ((epoch) / epochs) ** lamda * (upper - lower)


        X = stats.uniform(1,mu_2)
        index = X.rvs(negative_number) 
        index = index.astype(np.int)
        return index


    def SDGG(self, node_representation):
        probs = self.SDG(node_representation).squeeze()
        return probs

    def forward(self, x, edge_index, batch, num_graphs):

        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y, M = self.encoder(x, edge_index, batch)
        y = self.proj_head(y)
        return y

    def rank_negative_queue(self, x1, x2):

        x2 = x2.t()
        x = x1.mm(x2)

        x1_frobenius = x1.norm(dim=1).unsqueeze(0).t()
        x2_frobenins = x2.norm(dim=0).unsqueeze(0)
        x_frobenins = x1_frobenius.mm(x2_frobenins)

        final_value = x.mul(1 / x_frobenins)
        sort_queue, _ = torch.sort(final_value, dim=0, descending=False)
        return sot_queue

    def loss_cal(self, q_batch, q_aug_batch, negative_sim):

        T = 0.5

        positive_sim = torch.cosine_similarity(q_batch, q_aug_batch, dim=1)  # 维度有时对不齐

        positive_exp = torch.exp(positive_sim / T)

        negative_exp = torch.exp(negative_sim / T)

        negative_sum = torch.sum(negative_exp, dim=0)

        loss = positive_exp / (positive_exp+negative_sum)

        loss = -torch.log(loss).mean()

        return loss


import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':

    args = arg_parse()
    setup_seed(args.seed)

    accuracies = {'val': [], 'test': []}
    log_loss = [float("inf")]
    epochs = 20
    log_interval = 1
    batch_size = args.batch_size
    lr = args.lr # 0.01
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)

    dataset = TUDataset(path, name=DS, aug=args.aug)
    dataset_eval = TUDataset(path, name=DS, aug='none')
    dataset_num = len(dataset)
    negative_number = 8

    print(dataset_num)
    print(dataset.get_num_feature())
    print(negative_number)
    print('====dataset========')
    print(dataset)
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = simclr(args.hidden_dim, args.num_gc_layers, dataset_num_features, dataset_num).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))

    model.eval()
    emb, y = model.encoder.get_embeddings(dataloader_eval)

    """
    acc_val, acc = evaluate_embedding(emb, y)
    accuracies['val'].append(acc_val)
    accuracies['test'].append(acc)
    """
    stop_counter = 0
    patience = 3

    state = 0.01
    gamma = 1.99
    nata = 0.1

    for epoch in range(1, epochs+1): # epoch=20

        loss_all = 0
        sigma = 0
        start_time = time.time()
        model.train()
        dataset_embedding, _ = model.encoder.get_embeddings(dataloader)
        dataset_embedding = torch.from_numpy(dataset_embedding).to(device)
        n = torch.nn.BatchNorm1d(96, eps=1e-05, momentum=0.1, affine=True).to(device)

        for data in dataloader:
            sigma_r = 0
            data, data_aug = data
            optimizer.zero_grad()
            node_num, _ = data.x.size()
            data = data.to(device)
            q_batch = model(data.x, data.edge_index, data.batch, data.num_graphs)
            if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == 'random2' or args.aug == 'random3' or args.aug == 'random4':
                edge_idx = data_aug.edge_index.numpy()
                _, edge_num = edge_idx.shape
                idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]
                node_num_aug = len(idx_not_missing)
                data_aug.x = data_aug.x[idx_not_missing]
                data_aug.batch = data.batch[idx_not_missing]
                idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
                edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if
                            not edge_idx[0, n] == edge_idx[1, n]]
                data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)
            q_aug_batch = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)
            q_aug_batch_1 = n(q_aug_batch)
            sort_probs = model.SDGG(q_aug_batch_1)
            m = Bernoulli(sort_probs)
            action = m.sample()
            D_select = action.bool()
            xx1 = q_batch[D_select]
            xx2 = q_aug_batch[D_select]
            x1 = torch.mean(xx1, dim=0)
            x2 = torch.mean(xx2, dim=0)
            ddm = js_loss(x1.unsqueeze(dim=0),x2.unsqueeze(dim=0))
            pi = -m.log_prob(action)
            q_batch = q_batch[: q_aug_batch.size()[0]]
            sort_queue = model.rank_negative_queue(dataset_embedding, xx1)
            sample_index = model.sample_negative_index(negative_number, epoch, epochs)
            sample_index = torch.tensor(sample_index).to(device)
            negative_sim = sort_queue.index_select(0, sample_index)
            r = state + gamma * ddm
            state = ddm.item()
            sigma_r = sigma_r + r * gamma * pi.mean()
            loss = model.loss_cal(xx1, xx2, negative_sim)
            loss_all = sigma_r + loss
            loss_all += loss.item()
            loss.backward()
            optimizer.step()
            sigma += sigma_r
        end_time = time.time()
        print('Epoch {}, Loss {}'.format(epoch, loss_all))
        print('time: {} s'.format(end_time - start_time))

        log_loss.append(loss_all)

        if log_loss[-1] > log_loss[-2]:  # early stop
            stop_counter += 1
            negative_number = int(negative_number/2)
            if stop_counter > patience or negative_number <= 2:
                model.eval()
                emb = model.encoder.get_embeddings(dataloader_eval)
                acc_val, acc = evaluate_embedding(emb, y)
                accuracies['val'].append(acc_val)
                accuracies['test'].append(acc)
                break

        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader_eval)
            acc_val, acc = evaluate_embedding(emb, y)
            accuracies['val'].append(acc_val)
            accuracies['test'].append(acc)

    tpe = ('local' if args.local else '') + ('prior' if args.prior else '')

    pd.DataFrame(accuracies).to_csv('Result/' + args.DS + '_result.csv', index=False)

    with open('logs/log_' + args.DS + '_' + args.aug, 'a+') as f:
        s = json.dumps(accuracies)
        f.write('{},{},{},{},{},{},{},{},{}\n'.format(args.DS, tpe, args.num_gc_layers, epochs, log_interval, lr,
                                                      args.batch_size, negative_number, s))
        f.write('\n')
