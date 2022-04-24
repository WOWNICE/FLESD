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
from sampling import get_user_groups
from local_training import local_training
from global_aggregate import esd, average_weights
from models import my_resnet18
from data import get_dataset
from utils import exp_details, build_model

from torch.multiprocessing import Process, Manager

from utils import build_model


def main():
    # Miscs before training.
    args = args_parser()
    start_time = time.time()

    os.makedirs(args.ckptdir, exist_ok=True)
    exp_details(args)

    # load dataset and user groups
    train_dataset, test_dataset = get_dataset(args)

    # get user groups
    user_groups = get_user_groups(args)

    # public dataset 
    if not args.pubset:
        # use the supervised dataset. 
        tmp_args = copy.deepcopy(args)
        tmp_args.train_mode = 'test' # hack it, so the training data will be augmented as the testing data.
        inferset = get_dataset(tmp_args)[0]

        pubset = get_dataset(args)[0] # public dataset for global ssl training.
        pubidx = user_groups['0']
    else:
        # # use the current api to get the 
        # tmp_args = copy.deepcopy(args)
        # ds, percent = args.pubset.split('.')
        # tmp_args.num_users = 100 // int(percent)
        # tmp_args.dataset = ds
        # tmp_args.dirichlet = '100' # uniformly sample the dataset.

        # inferset, _, new_user_groups = get_dataset(tmp_args)
        # # brute-force replace the data augmentation
        # if ds == 'imagenet100':
        #     inferset.transform = transforms.Compose([
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.Resize(32),
        #         transforms.ToTensor(), 
        #         normalize
        #     ])
        # elif ds == 'tiny-imagenet':
        #     inferset.transform = transforms.Compose([
        #         transforms.Resize(32),
        #         transforms.ToTensor(), 
        #         normalize
        #     ])
        
        # if not tmp_args.flesd_cl:
        #     pubset = inferset 
        # else: 
        #     pubset = get_dataset_ssl(tmp_args)[0]
        #     if ds in ['imagenet100', 'tiny-imagnet']:
        #         pubset.transform = TwoCropsTransform(transforms.Compose([
        #             transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
        #             transforms.RandomApply([
        #                 transforms.ColorJitter(0.8, 0.8, 0.8, 0.4)
        #             ], p=0.8),
        #             transforms.RandomGrayscale(p=0.5),
        #             transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        #             transforms.RandomHorizontalFlip(),
        #             transforms.ToTensor(),
        #             normalize
        #         ]))
        #     elif ds == 'cifar10':
        #         pass # already resized to 32x32. no need to change.
        # pubidx = new_user_groups['0']

        # print(f'OOD pubset: {len(pubidx)}')     
        pass   

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

    local_losses = []

    # Global Rounds: One loop is a Communication Round
    for epoch in tqdm(range(1, args.epochs+1)):
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        idxs_users.sort()

        ###########################################################
        # Multi-Processing for training the local clients. 
        # support any client number, gpu number >= 1
        ###########################################################
        manager = Manager()
        return_dict = manager.dict()

        start = 0 if args.global_agg == 'wa' else 1
        while start < len(idxs_users):
            # all the child processes
            processes = [
                Process(target=local_training,
                        args=(idx, i, args, copy.deepcopy(global_model), train_dataset, epoch, user_groups[str(idx)], return_dict, inferset, pubidx, None))
                for (i, idx) in enumerate(idxs_users[start:start+args.gpus])
            ]

            for p in processes:
                p.start()

            for p in processes:
                p.join()
            
            start += args.gpus

        ###########################################################
        # Global Aggregation Scheme
        ###########################################################
        # weight-averaging global aggregation 
        if args.global_agg == 'wa': 
            local_weights = []
            for res in return_dict.values():
                local_weights.append(res['model'])

            global_weights = average_weights(local_weights)

        elif args.global_agg == 'esd':
            # get a list of the local models and the local losses.
            reps = [res['reps'] for res in return_dict.values()]
            
            # update global weights via knowledge distillation 
            manager = Manager()
            global_return_dict = manager.dict()

            # launch the ensemble similarity distillation as a child process,
            # to avoid cuda reinitialization error.
            processes = [
                Process(target=esd,
                        args=(0, args, copy.deepcopy(global_model), pubset, pubidx, copy.deepcopy(reps), global_return_dict, epoch))
            ]

            for p in processes:
                p.start()

            for p in processes:
                p.join()

            # update global weights
            # train from scratch or continue training from the latest checkpoint, need to further investigate it.
            global_weights = global_return_dict[0]
        
        global_model.load_state_dict(global_weights)

        ###########################################################
        # Checkpointing & Recording
        ###########################################################
        # save the global model
        if epoch % args.save_freq == 0:
            save_path = args.ckptdir 
            os.makedirs(save_path, exist_ok=True)
            
            global_path = os.path.join(save_path, f"global_{epoch}.pth.tar")
            torch.save(global_weights, global_path)

            # save all the local model
            for key in return_dict:
                local_path = os.path.join(save_path, f"local_{epoch}-client_{key}.pth.tar")
                torch.save(return_dict[key]['model'], local_path)

        # record the training losses of each client.
        loss_dict = {}
        for uid in return_dict.keys():
            loss_dict[uid] = return_dict[uid]['epoch_loss']
        
        local_losses.append(loss_dict)

    # save the final global-and-local model
    save_path = args.ckptdir 
    os.makedirs(save_path, exist_ok=True)
    
    global_path = os.path.join(save_path, f"global_{epoch}.pth.tar")
    print(global_path, 'checkpointing')
    torch.save(global_weights, global_path)

    with open(os.path.join(save_path, "local_losses.json"), 'w') as f:
        json.dump(local_losses, f)

    # save all the local model
    for key in return_dict:
        local_path = os.path.join(save_path, f"local_{epoch+1}-client_{key}.pth.tar")
        torch.save(return_dict[key]['model'], local_path)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))



if __name__ == '__main__':
    main()