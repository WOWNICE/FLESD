#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import json

import torch
from torch import nn
from torchvision import transforms

from options import args_parser
from update import local_update_weights_infer_on_public, global_update_ERD, global_update_ERD_CL
from models import my_resnet18
# from Flerd import Flerd
from utils import get_dataset, get_dataset_ssl, average_weights, exp_details, normalize, TwoCropsTransform, GaussianBlur

from torch.multiprocessing import Process, Manager

from utils import build_model


def main():
    start_time = time.time()

    # avoid re-initialize CUDA error.
    # torch.multiprocessing.set_start_method('spawn')

    # define paths
    path_project = os.path.abspath('..')

    args = args_parser()
    os.makedirs(args.ckptdir, exist_ok=True)
    exp_details(args)

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset_ssl(args)

    # public dataset, support 
    if not args.pubset:
        inferset = get_dataset(args)[0]
        pubset = inferset if not args.flerd_cl else get_dataset_ssl(args)[0]
        pubidx = user_groups['0']
    else:
        # use the current api to get the 
        tmp_args = copy.deepcopy(args)
        ds, percent = args.pubset.split('.')
        tmp_args.num_users = 100 // int(percent)
        tmp_args.dataset = ds
        tmp_args.dirichlet = '100' # uniformly sample the dataset.

        inferset, _, new_user_groups = get_dataset(tmp_args)
        # brute-force replace the data augmentation
        if ds == 'imagenet100':
            inferset.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(32),
                transforms.ToTensor(), 
                normalize
            ])
        elif ds == 'tiny-imagenet':
            inferset.transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(), 
                normalize
            ])
        
        if not tmp_args.flerd_cl:
            pubset = inferset 
        else: 
            pubset = get_dataset_ssl(tmp_args)[0]
            if ds in ['imagenet100', 'tiny-imagnet']:
                pubset.transform = TwoCropsTransform(transforms.Compose([
                    transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.8, 0.8, 0.8, 0.4)
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.5),
                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]))
            elif ds == 'cifar10':
                pass # already resized to 32x32. no need to change.
        pubidx = new_user_groups['0']

        print(f'OOD pubset: {len(pubidx)}')        

    global_model = build_model(args.model, num_classes=args.mlp_dim)

    # brute force replacement for the mlp 
    dim_mlp = global_model.fc.weight.shape[1]
    global_model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), global_model.fc)

    # load the global model if specified.
    if args.global_ckpt: 
        print(f'loading global checkpoint at {args.global_ckpt}')
        global_model.load_state_dict(torch.load(args.global_ckpt))

    # Set the model to train and send it to device.
    # global_model.to(device) # we don't need to load the global model into the gpu yet. 
    global_model.train()
    # print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    local_losses = defaultdict(list)

    for epoch in tqdm(range(args.epochs)):
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        idxs_users.sort()

        #########################################################################
        ########### Multi-Processing for training the local clients. ############
        manager = Manager()
        return_dict = manager.dict()

        # all the child processes
        print(global_model.bn1.weight)
        print(copy.deepcopy(global_model).bn1.weight)
        processes = [
            Process(target=local_update_weights_infer_on_public,
                    args=(gpu+1, args, copy.deepcopy(global_model), train_dataset, inferset, pubidx, epoch, user_groups[str(idx)], return_dict))
            for (gpu, idx) in enumerate(idxs_users[1:])
        ]

        for p in processes:
            p.start()

        for p in processes:
            p.join()
        
        # get a list of the local models and the local losses.
        reps = []
        for key in return_dict:
            # local_weights.append(copy.deepcopy(return_dict[key][0]))
            local_losses[key].append(copy.deepcopy(return_dict[key][1]))
            # print(return_dict[key][2])
            reps.append(return_dict[key][2])
        
        # update global weights via knowledge distillation 
        # global_weights = global_update_ERD(0, args, model=copy.deepcopy(global_model), pubset=pubset, pubidxs=pubidx, reps=copy.deepcopy(reps))
        manager = Manager()
        global_return_dict = manager.dict()

        # all the child processes
        processes = [
            Process(target=global_update_ERD_CL if args.flerd_cl else global_update_ERD,
                    args=(0, args, copy.deepcopy(global_model), pubset, pubidx, copy.deepcopy(reps), global_return_dict))
        ]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        global_weights = global_return_dict[0]

        # print(global_weights)

        # # update global weights
        # train from scratch or continue training from the latest checkpoint, need to further investigate it.
        global_model.load_state_dict(global_weights)

        # save the global model
        if epoch % args.save_freq == 0:
            save_path = args.ckptdir 
            os.makedirs(save_path, exist_ok=True)
            
            global_path = os.path.join(save_path, f"global_{epoch}.pth.tar")
            torch.save(global_weights, global_path)

            # save all the local model
            for key in return_dict:
                local_path = os.path.join(save_path, f"local_{epoch}-client_{key}.pth.tar")
                torch.save(return_dict[key][0], local_path)


    # save the final global-and-local model
    save_path = args.ckptdir 
    os.makedirs(save_path, exist_ok=True)
    
    global_path = os.path.join(save_path, f"global_{epoch+1}.pth.tar")
    torch.save(global_weights, global_path)

    with open(os.path.join(save_path, "local_losses.json"), 'w') as f:
        json.dump(local_losses, f)

    # save all the local model
    for key in return_dict:
        local_path = os.path.join(save_path, f"local_{epoch+1}-client_{key}.pth.tar")
        torch.save(return_dict[key][0], local_path)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))



if __name__ == '__main__':
    main()