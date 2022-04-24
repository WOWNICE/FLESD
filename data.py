#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sampling import read_inds
from transforms import custom_transforms

from PIL import ImageFilter
from PIL import Image
import random
import os

# the models
from models import *

class DatasetSplit(Dataset):
    """
    A Dataset Wrapper that returns everything needed.
    1: (a pair of) input data, # when in SSL situation, it's a pair of augmented views.
    2: corresponding labels,
    3: the index of the data.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        input_data, label = self.dataset[self.idxs[item]]
        
        return input_data, torch.tensor(label), item


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    data_dir = args.data_dir
    train_mode = args.train_mode 

    # test: give the original training data. 
    assert train_mode in ['ssl', 'sup', 'test'], f"train_mode '{train_mode}' is not supported."
    

    # SSL/SL are distinguished by train_mode argument
    if args.dataset in ['cifar10', 'cifar100']:
        # transforms
        train_transform = custom_transforms[f'{train_mode}-cifar']
        test_transform = custom_transforms['test-cifar']

        dataset_function = datasets.CIFAR10 if args.dataset == 'cifar10' else datasets.CIFAR100
        train_dataset = dataset_function(data_dir, train=True, download=True, transform=train_transform)
        test_dataset = dataset_function(data_dir, train=False, download=True, transform=test_transform)


    elif args.dataset in ['imagenet100', 'tiny-imagenet']:
        train_transform = custom_transforms[f'{train_mode}-in'] if args.dataset == 'imagenet100' else custom_transforms[f'{train_mode}-ti']
        test_transform = custom_transforms['test-in'] if args.dataset == 'imagenet100' else custom_transforms['test-ti']

        dataset_function = datasets.ImageFolder
        train_data_dir = os.path.join(data_dir, 'train')
        test_data_dir = os.path.join(data_dir, 'val')

        train_dataset = dataset_function(train_data_dir, train=True, download=True, transform=train_transform)
        test_dataset = dataset_function(test_data_dir, train=False, download=True, transform=test_transform)

    else:
        raise Exception(f'Dataset name {args.dataset} is currently not supported.')

    return train_dataset, test_dataset