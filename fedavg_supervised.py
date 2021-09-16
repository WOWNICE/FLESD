#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import json

import torch
from torch import nn

from options import args_parser
from update import local_update_weights_supervised
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, my_resnet18
from utils import get_dataset, average_weights, exp_details

from torch.multiprocessing import Process, Manager

from utils import build_model


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')

    args = args_parser()
    os.makedirs(args.ckptdir, exist_ok=True)
    exp_details(args)

    # load dataset and user groups
    # for supervised learning.
    train_dataset, test_dataset, user_groups = get_dataset(args)

    global_model = build_model(args.model, num_classes=args.mlp_dim)

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

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
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
        processes = [
            Process(target=local_update_weights_supervised,
                    args=(gpu, args, copy.deepcopy(global_model), train_dataset, test_dataset, epoch, user_groups[str(idx)], return_dict))
            for (gpu, idx) in enumerate(idxs_users)
        ]

        for p in processes:
            p.start()

        for p in processes:
            p.join()
        
        # get a list of the local models and the local losses.
        for key in return_dict:
            local_weights.append(copy.deepcopy(return_dict[key][0]))
            local_losses.append(copy.deepcopy(return_dict[key][1]))

        #########################################################################

        # for idx in idxs_users:
        #     local_model = l(args=args, dataset=train_dataset,
        #                               idxs=user_groups[idx], logger=logger)
        #     w, loss = local_model.update_weights(
        #         model=copy.deepcopy(global_model), global_round=epoch)
        #     local_weights.append(copy.deepcopy(w))
        #     local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')

        # Eval the global model using the 1-NN method for every global epoch.
        # TODO: 1-NN method. 

        # Save all the local models, and the global model.
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

    # gather the val accuracy of every clients.
    val_accs = {}
    for key in return_dict:
        val_accs[key] = return_dict[key][-1]

    if args.local_val_freq == 0:
        with open('local_acc.json', 'w') as f:
            json.dump(val_accs, f)

    # save the final global-and-local model
    save_path = args.ckptdir 
    os.makedirs(save_path, exist_ok=True)
    
    global_path = os.path.join(save_path, f"global_{epoch+1}.pth.tar")
    torch.save(global_weights, global_path)

    # save all the local model
    for key in return_dict:
        local_path = os.path.join(save_path, f"local_{epoch+1}-client_{key}.pth.tar")
        torch.save(return_dict[key][0], local_path)

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.dirichlet,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
