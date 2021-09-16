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

    # load the global_model
    global_path = '/raid/fedssl-checkpoints/flerd-T/stu0.1-target0.1/global-ep200.pth.tar'
    global_model.load_state_dict(torch.load(global_path))
    print(global_model)
    global_model.eval()

    # load the local model
    local_model = build_model(args.model, num_classes=args.mlp_dim)
    local_model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), local_model.fc)
    local_path = '/raid/fedssl-checkpoints/locals/cifar100-alpha1-client6/local_1-client_1.pth.tar'
    local_model.load_state_dict(torch.load(local_path))
    local_model.eval()

    #########################################################################
    ########### Multi-Processing for training the local clients. ############
    manager = Manager()
    return_dict = manager.dict()

    # all the child processes
    processes = [Process(
        target=local_update_weights_infer_on_public, 
        args=(gpu, args, copy.deepcopy(model), train_dataset, inferset, pubidx, gpu+1, user_groups[str(1)], return_dict)) 
        for (gpu, model) in enumerate([local_model, global_model])
        ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    # get a list of the local models and the local losses.
    local_reps = return_dict[0][2].cpu().numpy()
    global_reps = return_dict[1][2].cpu().numpy()

    with open('global_local_reps.npy', 'wb') as f:
        np.save(f, [global_reps, local_reps])

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))