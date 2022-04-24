#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
from tabnanny import check
import torch
from torch import nn
from torchvision import datasets, transforms
from data import DatasetSplit
from torch.utils.data import DataLoader

from sampling import read_inds

from PIL import ImageFilter
from PIL import Image
import random
import os
import time

from main.flesd import Flesd

# the models
from models import *
from utils import AllMeter, AverageMeter, ProgressMeter, checkpointing


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], float(len(w)))
    return w_avg


def esd(gpu, args, model, pubset, pubidxs, reps, return_dict, round):
    """
    Implementation of Ensemble Similarity Distillation.
    Here we use augmenation and the contrastive loss as an auxiliary during distillation.
    However, by default, the weight of contrastive learning loss is 0.
    """
    # set the default device for global aggregating.
    torch.cuda.set_device(f'cuda:{gpu}')

    sim_mat = ensemble_sims(reps, percent=int(args.flesd_percent), tau=args.flesd_T)

    # build the model.
    flesd_model = Flesd(
        model, 
        K=args.flesd_K, 
        m=args.flesd_m, 
        T=args.flesd_T, 
        sim_mat=sim_mat, 
        dataset=args.dataset
    ).cuda()

    trainloader = DataLoader(
        DatasetSplit(pubset, list(pubidxs)), 
        batch_size=args.flesd_bs, 
        shuffle=True, 
        num_workers=args.num_workers, 
        drop_last=True
    )

    # Set mode to train model
    flesd_model.train()

    # Set optimizer for the local updates
    if args.flesd_optimizer == 'sgd':
        optimizer = torch.optim.SGD(flesd_model.parameters(), lr=args.flesd_lr,
                                    momentum=args.momentum)
    elif args.flesd_optimizer == 'adam':
        optimizer = torch.optim.Adam(flesd_model.parameters(), lr=args.flesd_lr,
                                        weight_decay=args.flesd_wd)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for iter in range(1, args.flesd_epochs+1):
        # set the meters for debugging.
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data Time', ':6.3f')
        losses = AllMeter('Loss', ':.4e') # we want the whole trace of the losses

        progress = ProgressMeter(
            len(trainloader),
            [batch_time, data_time, losses],
            prefix=f'ESD Round ({round}), Ep ({iter}/{args.flesd_epochs}): '
        )

        end = time.time()
        for i, (input_data, _, batch_idx) in enumerate(trainloader):
        # for i, dataterm in enumerate(trainloader):
            x1, x2 = input_data[0].cuda(non_blocking=True), input_data[1].cuda(non_blocking=True)
            data_time.update(time.time()-end)

            flesd_model.zero_grad()

            # amp acceleration.
            with torch.cuda.amp.autocast(enabled=True):
                # forward pass will automatically update the queue.
                logits, labels, q, k = flesd_model(x1, x2, batch_idx)
                labels = labels.cuda()

                # fill the queue first. 
                if i < args.flesd_K // args.flesd_bs:
                    continue

                # computational stability, logsumexp for small temperature. 
                ##################### KD loss #######################
                logits = logits - logits.max(axis=1, keepdims=True).values.detach()

                # soft-label loss
                log_prob = torch.nn.functional.log_softmax(logits, dim=1)
                loss = -torch.sum(log_prob * labels) / labels.shape[0]
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update the training statistics
            batch_time.update(time.time()-end)
            losses.update(loss.detach().item())
            end = time.time()

            if args.verbose and (i % args.print_freq == 0):
                progress.display(i)

        # only used for the flesd global. 
        # user_id == -1 represents the global aggregation checkpointing.
        if iter % args.save_freq == 0:
            checkpointing(flesd_model, save_path=args.ckptdir, global_round=round, user_id=-1, local_round=iter)

    return_dict[0] = flesd_model.cpu().encoder_q.state_dict()


def ensemble_sims(reps, percent=100, tau=0.1):
    """
    Ensemble the Similarity Matrices.
        1. Denoise (hard filtering) the similarity matrices.
        2. Sharpen the similarity matrices by the temperature Tau.
        3. Ensemble the preprocessed matrices.
    """
    # Compute the denoised similarity matrix.
    comms = []
    for rep in reps:
        rep = nn.functional.normalize(rep)
        mat = rep.mm(rep.T)

        # n% percent of the similarity is preserved.  
        if percent == 100: pass # by default there is no filtering.
        else: 
            # self similarity is not counted as communication overhead, thus plus 1.
            k = int(percent / 100 * reps[0].shape[0]) + 1 

            # filter the n% of the data
            row_smallest = torch.topk(mat, k).values[:, -1:]
            mat = (mat > row_smallest - 1e-6).float() * mat # similarity values that are filtered is replaced by 0.

        comms.append(mat)
    
    # exp(s/t) normalization to get the distribution. 
    # this should be done globally, since then on the local batch, you can simply re-scale it to probability simplex.
    def normalize_sim(mat, tau):
        # prevent exp overflow.
        mat = mat - mat.max(axis=1, keepdims=True).values 
        mat = torch.exp(mat / tau) 
        # mat = mat / mat.sum(axis=1)
        return mat

    # get the ensembled similarity matrix
    sim_mats = [normalize_sim(com, tau) for com in comms]
    sim_mat = sum(sim_mats) / len(sim_mats)

    return sim_mat