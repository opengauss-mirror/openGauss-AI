# serial
from __future__ import division
from __future__ import print_function
from model.simple_gcn import GCN
from load_data.load_training_sample import load_data, accuracy

import time
import os
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from config import parse_cmd_args, model_parameters


def train(epoch, labels, params):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    # print(output[idx_train])

    loss_train = F.mse_loss(output[idx_train], labels[idx_train])

    # loss_train = nn.CrossEntropyLoss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not params.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.mse_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test(labels):
    model.eval()
    output = model(features, adj)

    loss_test = F.mse_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))


if __name__ == '__main__':

    '''
    feature_num = 2
    iteration_num = 230
    # workload_size = 10
    workload_num = 3184
    edge_dim = 30
    node_dim = 30
    '''
    argus = parse_cmd_args()
    base_dir = os.path.abspath(os.curdir)
    data_path = os.path.join(base_dir, 'data/query_plan/job-pg')

    params = model_parameters()
    model = GCN(nfeat=argus['feature_num'],
                nhid=params.hidden,
                nclass=argus['node_dim'],
                dropout=params.dropout)

    optimizer = optim.Adam(model.parameters(),
                           lr=params.lr, weight_decay=params.weight_decay)

    for wid in range(argus['iteration_num']):
        # Load data
        adj, features, labels, idx_train, idx_val, idx_test = load_data(path=os.path.join(data_path, "serial/graph/"),
                                                                        dataset="sample-plan-" + str(wid))

        # Train model
        t_total = time.time()
        for epoch in range(params.epochs):
            train(epoch, labels, params)
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        test(labels)

    for wid in range(argus['iteration_num'], argus['iteration_num'] + 55):
        # Load data
        adj, features, labels, idx_train, idx_val, idx_test = load_data(path=os.path.join(data_path, "serial/graph/"),
                                                                        dataset="sample-plan-" + str(wid))

        # test model
        t_total = time.time()

        # Testing
        test(labels)
