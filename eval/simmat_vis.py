#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# this file is for hyper-parameter tuning of ERD. 
# which loads the pretrained local weights.
import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch import nn

from options import args_parser
from update import local_update_weights_infer_on_public, global_update_ERD, global_update_ERD_CL
from models import my_resnet18
# from Flerd import Flerd
from utils import get_dataset, get_dataset_ssl, average_weights, exp_details

from torch.multiprocessing import Process, Manager

from utils import build_model


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')

    args = args_parser()
    # without training the local models.
    args.local_ep = 0
    os.makedirs(args.ckptdir, exist_ok=True)
    exp_details(args)

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset_ssl(args)

    # public dataset
    inferset = get_dataset(args)[0]
    pubset = inferset if not args.flerd_cl else get_dataset_ssl(args)[0]
    pubidx = user_groups['0']

    global_model = build_model(args.model, num_classes=args.mlp_dim)

    # brute force replacement for the mlp 
    dim_mlp = global_model.fc.weight.shape[1]
    global_model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), global_model.fc)

    # Set the model to train and send it to device.
    # global_model.to(device) # we don't need to load the global model into the gpu yet. 
    global_model.train()
    # print(global_model)

    # local_models
    local_models = [build_model(args.model, num_classes=args.mlp_dim) for _ in range(1, args.num_users)]
    local_root = '/raid/fedssl-checkpoints/locals/cifar100-alpha1-client6/local_1-client_'
    for c in range(1,6):
        local_path = f"{local_root}{c}.pth.tar"
        local_models[c-1].fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), local_models[c-1].fc)
        local_models[c-1].load_state_dict(torch.load(local_path))

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    
    local_weights, local_losses = [], []

    global_model.train()
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)

    #########################################################################
    ########### Multi-Processing for training the local clients. ############
    manager = Manager()
    return_dict = manager.dict()

    # all the child processes
    processes = [
        Process(target=local_update_weights_infer_on_public,
                args=(gpu, args, local_models[gpu], train_dataset, inferset, pubidx, 1, user_groups[str(idx)], return_dict))
        for (gpu, idx) in enumerate(idxs_users[1:])
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
    
    # get a list of the local models and the local losses.
    reps = []
    for key in return_dict:
        local_weights.append(copy.deepcopy(return_dict[key][0]))
        local_losses.append(copy.deepcopy(return_dict[key][1]))
        reps.append(return_dict[key][2].cpu().numpy())

    with open('reps.npy', 'wb') as f:
        np.save(f, np.array(reps))

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))