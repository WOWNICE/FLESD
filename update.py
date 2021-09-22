#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os
import time
import numpy as np
from Flesd import Flesd
import copy

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        (x1, x2), label = self.dataset[self.idxs[item]]
        return (x1, x2), torch.tensor(label)


class SupervisedDatasetSplit(DatasetSplit):
    """For supervised learning of fedavg."""
    def __getitem__(self, item):
        x, label = self.dataset[self.idxs[item]]
        return x, torch.tensor(label)


class DatasetSplitIdx(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        x, label = self.dataset[self.idxs[item]]
        return x, item

class DatasetSplitTwoCropsIdx(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        (x1, x2), _ = self.dataset[self.idxs[item]]
        return x1, x2, item


def local_update_weights(gpu, args, model, dataset, global_round, idxs, return_dict):
    """Use the SimCLR loss to train the model, which supports the fedprox training"""
    # set the default cuda device. 
    torch.cuda.set_device(gpu)
    trainloader = DataLoader(DatasetSplit(dataset, list(idxs)), batch_size=args.local_bs, shuffle=True, num_workers=args.num_workers)
    criterion = nn.CrossEntropyLoss().cuda()

    # old model
    old_model = copy.deepcopy(model).cuda()
    old_model.eval()

    # Set mode to train model
    model = model.cuda()
    model.train()
    epoch_loss = []

    # Set optimizer for the local updates
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                        weight_decay=1e-6)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for iter in range(args.local_ep):
        batch_loss = []
        for batch_idx, ((x1, x2), _) in enumerate(trainloader):
            x1, x2 = x1.cuda(), x2.cuda()

            model.zero_grad()

            with torch.cuda.amp.autocast(enabled=True):
                loss = 0
                ###################### not sure whether this change the performance.
                # q, k = model(x1), model(x2)
                q, k = nn.functional.normalize(model(x1), dim=1), nn.functional.normalize(model(x2), dim=1)

                # adopted from the mocov2 code, not exactly SimCLR
                logits = torch.einsum('nc,ck->nk', [q, k.T])

                # apply temperature
                logits /= args.T

                # labels: positive key indicators
                labels = torch.arange(logits.shape[0], dtype=torch.long).cuda()

                loss += criterion(logits, labels)
                
                # fedprox
                if args.mu > 0:
                    prox_loss = 0
                    for param_q, param_k in zip(model.parameters(), old_model.parameters()):
                        prox_loss += args.mu * torch.sum(torch.pow(param_k - param_q, 2))
                    loss += prox_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if args.verbose and (batch_idx % 10 == 0):
                print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    global_round, iter, batch_idx * len(x1),
                    len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # checkpointing the local models
        if args.local_save_freq != 0 and iter % args.local_save_freq == 0:
            save_path = args.ckptdir 
            os.makedirs(save_path, exist_ok=True)

            local_path = os.path.join(save_path, f"local_{global_round}-client_{gpu}-localep_{iter}.pth.tar")
            torch.save(model.state_dict(), local_path)
    
    # final checkpoints
    if args.local_save_freq != 0:
        save_path = args.ckptdir 
        os.makedirs(save_path, exist_ok=True)

        local_path = os.path.join(save_path, f"local_{global_round}-client_{gpu}-localep_{iter+1}.pth.tar")
        torch.save(model.state_dict(), local_path)

    return_dict[gpu] = [model.cpu().state_dict(), epoch_loss]
    # return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


def local_update_weights_supervised(gpu, args, model, dataset, testset, global_round, idxs, return_dict):
    """Supervised Learning Classification Loss."""
    # set the default cuda device. 
    torch.cuda.set_device(gpu)
    trainloader = DataLoader(SupervisedDatasetSplit(dataset, list(idxs)), batch_size=args.local_bs, shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.local_bs, shuffle=False, num_workers=args.num_workers)

    criterion = nn.CrossEntropyLoss().cuda()

    # Set mode to train model
    model = model.cuda()
    model.train()
    epoch_loss, epoch_val_accs = [], []

    # Set optimizer for the local updates
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                        weight_decay=1e-6)

    for iter in range(args.local_ep):
        batch_loss = []
        for batch_idx, (x, labels) in enumerate(trainloader):
            x, labels = x.cuda(non_blocking=True), labels.cuda(non_blocking=True)

            model.zero_grad()

            logits = model(x)

            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            if args.verbose and (batch_idx % 100 == 0):
                # local 
                print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    global_round, iter, batch_idx * len(x),
                    len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # checkpointing the local models
        if args.local_save_freq != 0 and iter % args.local_save_freq == 0:
            save_path = args.ckptdir 
            os.makedirs(save_path, exist_ok=True)

            local_path = os.path.join(save_path, f"local_{global_round}-client_{gpu}-localep_{iter}.pth.tar")
            torch.save(model.state_dict(), local_path)

        # validate on the testset
        if args.local_val_freq == 0: 
            epoch_val_accs.append(validate(testloader, model, criterion).item())
        elif iter % args.local_val_freq == 0:
            print(f"Local validating. ClientNo: {gpu}, Local Epoch: {iter}, Top1-Acc: {validate(testloader, model, criterion):.1f}")
    
    # final checkpoints
    if args.local_save_freq != 0:
        save_path = args.ckptdir 
        os.makedirs(save_path, exist_ok=True)

        local_path = os.path.join(save_path, f"local_{global_round}-client_{gpu}-localep_{iter+1}.pth.tar")
        torch.save(model.state_dict(), local_path)
    
    # final validation
    if args.local_val_freq != 0 and (iter+1) % args.local_val_freq == 0:
        print(f"Local validating. ClientNo: {gpu}, Local Epoch: {iter}, Top1-Acc: {validate(testloader, model, criterion):.1f}")

    return_dict[gpu] = [model.cpu().state_dict(), sum(epoch_loss) / len(epoch_loss), epoch_val_accs]
    # return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


#####################################################################################################################
def local_update_weights_infer_on_public(gpu, args, model, dataset, pubset, pubidxs, global_round, idxs, return_dict):
    """Use the SimCLR loss to train the model.
    After training for local epochs, return the similarity matrix of the public dataset.
    """
    # set the default cuda device. 
    # torch.cuda.set_device(gpu)
    trainloader = DataLoader(DatasetSplit(dataset, list(idxs)), batch_size=args.local_bs, shuffle=True, num_workers=args.num_workers)
    publoader = DataLoader(SupervisedDatasetSplit(pubset, list(pubidxs)), batch_size=args.local_bs, shuffle=False, num_workers=args.num_workers)
    criterion = nn.CrossEntropyLoss().cuda(gpu)

    # Set mode to train model
    model = model.cuda(gpu)
    model.train()
    epoch_loss = []

    # Set optimizer for the local updates
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                        weight_decay=1e-6)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for iter in range(args.local_ep):
        batch_loss = []
        for batch_idx, ((x1, x2), _) in enumerate(trainloader):

            x1, x2 = x1.cuda(gpu, non_blocking=True), x2.cuda(gpu, non_blocking=True)

            model.zero_grad()

            # amp to accelerate the computation and enlarge the batch size.
            with torch.cuda.amp.autocast(enabled=True):
                q, k = nn.functional.normalize(model(x1), dim=1), nn.functional.normalize(model(x2), dim=1)
                # adopted from the mocov2 code, not exactly SimCLR
                # compute logitsq
                # Einstein sum is more intuitive
                logits = torch.einsum('nc,ck->nk', [q, k.T])

                # apply temperature
                logits /= args.T

                # labels: positive key indicators
                labels = torch.arange(logits.shape[0], dtype=torch.long).cuda(gpu)

                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if args.verbose and (batch_idx % 10 == 0):
                print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    global_round, iter, batch_idx * len(x1),
                    len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))

    # infer on the public dataset.
    model.eval()
    reps = []
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(publoader):
            x = x.cuda(gpu, non_blocking=True)
            rep = model(x)
            reps.append(rep.cpu())
        
    reps = torch.cat(reps, 0)
        
    return_dict[gpu] = [model.cpu().state_dict(), epoch_loss, reps]
    # return_dict[gpu] = [model.cpu().state_dict(), 1, reps]


def global_update_ERD(gpu, args, model, pubset, pubidxs, reps, return_dict):
    """
    Implementation of Ensemble Relational(Similarity) Distillation.
    Here no augmentation is used, but original images. 
    """
    # set the default cuda device. 
    # cannot set the default gpu
    # torch.cuda.set_device(gpu)

    if args.communication == 'sim-full':
        comms = []
        for rep in reps:
            rep = nn.functional.normalize(rep)
            comms.append(rep.mm(rep.T))  # delete the x-x similrarity 
    elif args.communication.startswith('sim-'):
        # top-k similar samples are preserved. 
        k = int(args.communication.split('-')[-1])

    # exp(s/t) normalization to get the distribution. 
    # this should be done globally, since then on the local batch, you can simply linearly scale the probability.
    def spike(mat, tau):
        mat = mat - mat.max(axis=1, keepdims=True).values # prevent exp overflow.
        mat = torch.exp(mat / tau) 
        # mat = mat / mat.sum(axis=1)
        return mat

    sim_mats = [spike(com, args.flesd_targetT) for com in comms]
    sim_mat = sum(sim_mats) / len(sim_mats)

    # print(sim_mat)

    flesd_model = Flesd(
        model, 
        K=args.flesd_K, 
        m=args.flesd_m, 
        T=args.flesd_T, 
        sim_mat=sim_mat, 
        dataset=args.dataset
    ).cuda(gpu)

    trainloader = DataLoader(
        DatasetSplitIdx(pubset, list(pubidxs)), 
        batch_size=args.flesd_bs, 
        shuffle=True, 
        num_workers=args.num_workers, 
        drop_last=True
    )
    # criterion = nn.CrossEntropyLoss().cuda()

    # Set mode to train model
    flesd_model.train()
    epoch_loss = []

    # Set optimizer for the local updates
    if args.flesd_optimizer == 'sgd':
        optimizer = torch.optim.SGD(flesd_model.parameters(), lr=args.flesd_lr,
                                    momentum=args.momentum)
    elif args.flesd_optimizer == 'adam':
        optimizer = torch.optim.Adam(flesd_model.parameters(), lr=args.flesd_lr,
                                        weight_decay=1e-6)

    for iter in range(args.flesd_epochs):
        batch_loss = []
        for i, (x, batch_idx) in enumerate(trainloader):
            x = x.cuda(gpu)

            flesd_model.zero_grad()

            # forward pass will automatically update the queue.
            logits, labels, _, _ = flesd_model(x, x, batch_idx)
            labels = labels.cuda(gpu)

            # fill the queue first. 
            if i < args.flesd_K // args.flesd_bs:
                continue

            # hard-label loss
            # loss = criterion(logits, labels)

            # computational stability, logsumexp
            # for small temperature. 
            logits = logits - logits.max(axis=1, keepdims=True).values.detach()

            # soft-label loss
            log_prob = torch.nn.functional.log_softmax(logits, dim=1)
            loss = -torch.sum(log_prob * labels) / labels.shape[0]
            
            loss.backward()
            optimizer.step()

            if args.verbose and (i % 10 == 0):
                print('| ERD epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    iter, i * len(x),
                    len(trainloader.dataset),
                    100. * i / len(trainloader), loss.item()))
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))

    torch.cuda.empty_cache()

    return_dict[gpu] = flesd_model.cpu().encoder_q.state_dict()

def global_update_ERD_CL(gpu, args, model, pubset, pubidxs, reps, return_dict):
    """
    Implementation of Ensemble Relational(Similarity) Distillation.
    Here we use augmenation and the contrastive loss as an auxiliary during distillation.
    However, by default, the weight of contrastive learning loss is 0.
    """
    # Compute the target similarity matrix.
    # by default the communication is sim-full.
    if args.communication.startswith('sim-full'):
        comms = []
        for rep in reps:
            rep = nn.functional.normalize(rep)
            mat = rep.mm(rep.T)
            if args.communication.startswith('sim-full-clip'):
                # (x - x.mean())/x.std(), clip(x), normalize first. 
                mat = mat - mat.mean(axis=1, keepdims=True)
                mat = mat / mat.max(axis=1, keepdims=True).values

            comms.append(mat) 
    elif args.communication.startswith('sim-'):
        # sim-n: n% percent of the similarity is preserved.  
        percent = float(args.communication.split('-')[1])
        k = int(percent / 100 * reps[0].shape[0]) + 1 # self similarity is not counted as communication overhead, thus plus 1.
        comms = []
        for rep in reps:
            rep = nn.functional.normalize(rep)
            mat = rep.mm(rep.T)
            if 'clip' in args.communication:
                mat = mat - mat.mean(axis=1, keepdims=True)
                mat = mat / mat.max(axis=1, keepdims=True).values
                mat = torch.clamp(mat, min=-1, max=1)
            
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
    sim_mats = [normalize_sim(com, args.flesd_targetT) for com in comms]
    sim_mat = sum(sim_mats) / len(sim_mats)

    # build the model.
    flesd_model = Flesd(
        model, 
        K=args.flesd_K, 
        m=args.flesd_m, 
        T=args.flesd_T, 
        sim_mat=sim_mat, 
        dataset=args.dataset
    ).cuda(gpu)

    trainloader = DataLoader(
        DatasetSplitTwoCropsIdx(pubset, list(pubidxs)), 
        batch_size=args.flesd_bs, 
        shuffle=True, 
        num_workers=args.num_workers, 
        drop_last=True
    )

    # hard-label loss for CL.
    criterion = nn.CrossEntropyLoss().cuda(gpu)

    # Set mode to train model
    flesd_model.train()
    epoch_loss = []

    # Set optimizer for the local updates
    if args.flesd_optimizer == 'sgd':
        optimizer = torch.optim.SGD(flesd_model.parameters(), lr=args.flesd_lr,
                                    momentum=args.momentum)
    elif args.flesd_optimizer == 'adam':
        optimizer = torch.optim.Adam(flesd_model.parameters(), lr=args.flesd_lr,
                                        weight_decay=args.flesd_wd)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for iter in range(args.flesd_epochs):
        for i, (x1, x2, batch_idx) in enumerate(trainloader):
        # for i, dataterm in enumerate(trainloader):
            x1, x2 = x1.cuda(gpu, non_blocking=True), x2.cuda(gpu, non_blocking=True)

            flesd_model.zero_grad()

            # amp acceleration.
            with torch.cuda.amp.autocast(enabled=True):
                # forward pass will automatically update the queue.
                logits, labels, q, k = flesd_model(x1, x2, batch_idx)
                labels = labels.cuda(gpu)

                # fill the queue first. 
                if i < args.flesd_K // args.flesd_bs:
                    continue

                # computational stability, logsumexp
                # for small temperature. 
                # the distillation loss.
                loss = 0

                ##################### KD loss #######################
                logits = logits - logits.max(axis=1, keepdims=True).values.detach()

                # soft-label loss
                log_prob = torch.nn.functional.log_softmax(logits, dim=1)
                loss += -torch.sum(log_prob * labels) / labels.shape[0]
                
                ############### CL loss #####################
                # the contrastive loss
                cl_logits = torch.einsum('nc,ck->nk', [q, k.T])

                # apply temperature
                cl_logits /= args.T

                # labels: positive key indicators
                cl_labels = torch.arange(cl_logits.shape[0], dtype=torch.long).cuda()

                loss += args.flesd_cl_weight * criterion(cl_logits, cl_labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if args.verbose and (i % 2 == 0):
                print('| CL-ERD epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    iter, i * len(x1),
                    len(trainloader.dataset),
                    100. * i / len(trainloader), loss.item()))

        # only used for the flesd global. 
        if iter % 100 == 0:
            save_path = args.ckptdir 
            os.makedirs(save_path, exist_ok=True)

            local_path = os.path.join(save_path, f"global-ep{iter}.pth.tar")
            torch.save(copy.deepcopy(flesd_model).cpu().encoder_q.state_dict(), local_path)

    return_dict[gpu] = flesd_model.cpu().encoder_q.state_dict()

###############################################
# Evaluation Helper Functions
###############################################
def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % 100 == 0:
            #     progress.display(i)

        # # TODO: this should also be done with the ProgressMeter
        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #       .format(top1=top1, top5=top5))

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res