#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import time
import numpy as np
from losses import SimCLR, SupCls
from main.flesd import Flesd
import copy

from data import DatasetSplit

from utils import AllMeter, ProgressMeter, AverageMeter, accuracy, checkpointing, validate


def local_training(
    user_id, 
    gpu,
    args, 
    model, 
    dataset,  
    global_round,   # current global epochs
    idxs,           # indices file for the client
    return_dict,    # where different clients comm.
    pubset=None,    # FURL, public dataset
    pubidxs=None,   # FURL, public dataset indices
    testset=None,   # Supervised Learning, evaluation
    ):
    """
    A general local training function that supports:
        1. SimCLR local training, 
        2. Supervised training.
    Returns a dictionary consisting of:
        1. Local model weights,
        2. Inferred representations on the public dataset (if specified),
        3. losses 
    """
    # set the default cuda device.
    torch.cuda.set_device(f'cuda:{gpu}')

    # dataloader
    trainloader = DataLoader(
        DatasetSplit(dataset, list(idxs)), 
        batch_size=args.local_bs, 
        shuffle=True, 
        num_workers=args.num_workers
    )

    # old model for FedProx and other weight-restricted methods
    old_model = copy.deepcopy(model).cuda()
    old_model.eval()

    # Set mode to train model
    model = model.cuda()
    model.train()

    # criterion to generate the loss, used for both supervised and unsupervised. 
    criterion = nn.CrossEntropyLoss().cuda()

    # Set optimizer for the local updates
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                        weight_decay=1e-6)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # LOCAL TRAINING STARTS
    local_losses = []
    for lep in range(1, args.local_ep+1):
        # set the meters for debugging.
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data Time', ':6.3f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        losses = AllMeter('Loss', ':.4e')

        progress = ProgressMeter(
            len(trainloader),
            [batch_time, data_time, losses, top1],
            prefix=f'Client ({user_id}), CommRound ({global_round}), LocalEp ({lep}/{args.local_ep}): '
        )

        #######################################
        # Real Training Code
        #######################################
        end = time.time()
        for batch_idx, (input_data, semantic_labels, _) in enumerate(trainloader):
            data_time.update(time.time() - end)
            model.zero_grad()

            with torch.cuda.amp.autocast(enabled=True):
                if args.train_mode == 'ssl':    # SimCLR 
                    loss, acc1 = SimCLR(model, input_data, tau=args.T, criterion=criterion)
                elif args.train_mode == 'sup':
                    loss, acc1 = SupCls(model, input_data, labels=semantic_labels, criterion=criterion)
                
                # fedprox
                if args.mu > 0:
                    prox_loss = 0
                    for param_q, param_k in zip(model.parameters(), old_model.parameters()):
                        prox_loss += args.mu * torch.sum(torch.pow(param_k - param_q, 2))
                    loss += prox_loss

            # BP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update the training statistics
            batch_time.update(time.time()-end)
            losses.update(loss.detach().item())
            top1.update(acc1.detach().item())
            end = time.time()

            if args.verbose and (batch_idx % args.local_print_freq == 0):
                progress.display(batch_idx)
        
        # store the epoch losses.
        local_losses.append(losses.trace)

        ###########################################
        # Supervised: evaluate on the testset. 
        ###########################################
        if args.local_val_freq == 0: pass
        elif args.train_mode == 'sup' and lep % args.local_val_freq == 0 and testset:
            testloader = DataLoader(testset, batch_size=args.local_bs, shuffle=False, num_workers=args.num_workers)
            print(f"Local validating. Client No: {gpu}, Local Epoch: {iter}, Top1-Acc: {validate(testloader, model, criterion):.1f}")

        # checkpointing the local models
        if args.local_save_freq != 0 and lep % args.local_save_freq == 0:
            checkpointing(model, save_path=args.ckptdir, user_id=user_id, local_round=lep, global_round=global_round)


    #######################################################
    # Evaluate after supervised local training is done.
    #######################################################
    if args.train_mode == 'sup' and testset:
        testloader = DataLoader(testset, batch_size=args.local_bs, shuffle=False, num_workers=args.num_workers)
        print(f"Local validating. Client No: {gpu}, Local Epoch: {iter}, Top1-Acc: {validate(testloader, model, criterion):.1f}")
    

    #######################################################
    # Infer on the public data.
    #######################################################
    reps = None  # default value.
    if args.train_mode == 'ssl' and pubset and pubidxs:
        reps = pubset_infer(model, pubset, pubidxs)


    # write the results back to the main process.
    return_dict[user_id] = {
        'model': model.cpu().state_dict(),
        'reps': reps,
        'epoch_loss': local_losses
    }


###############################################
# Helper Functions
###############################################
def validate(val_loader, model, criterion):
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))
            top1.update(acc1[0], images.size(0))

    return top1.avg

def pubset_infer(model, pubset, pubidxs, bs=128, num_workers=8):
    # infer on the public dataset.
    model.eval()

    publoader = DataLoader(DatasetSplit(pubset, list(pubidxs)), batch_size=bs, shuffle=False, num_workers=num_workers)

    reps = []
    with torch.no_grad():
        for _, (x, _, _) in enumerate(publoader):
            x = x.cuda(non_blocking=True)
            rep = model(x)
            reps.append(rep.cpu())
        
    reps = torch.cat(reps, 0)

    return reps
