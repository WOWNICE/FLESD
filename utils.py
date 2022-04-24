#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms

from sampling import read_inds

from PIL import ImageFilter
from PIL import Image
import random
import os
import time

# the models
from models import *
 

def exp_details(args):
    """TODO: To Be More specific."""
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')
    print(f'    Dataset   : {args.dataset}\n')
    print(f'    Training Mode   : {args.train_mode}\n')
    print(f'    Num of GPUS     : {args.gpus}\n')

    print('    Federated parameters:')
    print(f'    Dirichlet:         : {args.dirichlet}')
    print(f'    Num of users:      : {args.num_users}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    print(f'    Data Index File    : {args.ind_file}\n')


def build_model(model_name, num_classes=10):
    # BUILD MODEL
    if model_name == 'resnet18':
        global_model = my_resnet18(num_classes=num_classes)
    elif model_name == 'resnet50':
        global_model = resnet50(num_classes=num_classes)
    else:
        exit('Error: unrecognized model')

    return global_model

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


class AllMeter(AverageMeter):
    """Additionally stores the traces of the value"""
    def __init__(self, name, fmt=':f'):
        super().__init__(name, fmt)

    def reset(self):
        super().reset()
        self.trace = []

    def update(self, val, n=1):
        super().update(val, n=n)
        self.trace.append(val)
        

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


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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

def checkpointing(model, save_path, global_round, user_id, local_round):
    """Checkponts the local model"""
    os.makedirs(save_path, exist_ok=True)

    file_name = f"global_{global_round}-client_{user_id}-localep_{local_round}.pth.tar" if user_id != -1 else f"global_{global_round}-ep_{local_round}.pth.tar"

    local_path = os.path.join(save_path, file_name)
    torch.save(model.state_dict(), local_path)


###############################################
# Evaluation Helper Functions
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


if __name__ == '__main__':
    # model = build_model('resnet18')
    # print(model)
    pass