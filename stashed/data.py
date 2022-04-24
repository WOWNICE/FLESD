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

def _get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    data_dir = args.data_dir
    if args.dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=train_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)
    elif args.dataset == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                       transform=train_transform)

        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=apply_transform)
    elif args.dataset == 'imagenet100':
        apply_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(), 
            normalize
        ])

        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=apply_transform)
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=apply_transform)

    elif args.dataset == 'tiny-imagenet':
        apply_transform = transforms.Compose([
            transforms.ToTensor(), 
            normalize
            ])

        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=apply_transform)
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=apply_transform)
        
    # sample training data amongst users
    user_groups = read_inds(
        dataset_name=args.dataset,
        data_path=data_dir,
        client_num=str(args.num_users),
        alpha=str(args.dirichlet),
        generate_ifnonexist=args.gen_data_if_nonexist
    )

    return train_dataset, test_dataset, user_groups


def get_dataset_ssl(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    data_dir = args.data_dir
    if args.dataset == 'cifar10':
        apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
        ssl_transform = TwoCropsTransform(base_transform=small_transform)

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=ssl_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)
    if args.dataset == 'cifar100':
        apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
        ssl_transform = TwoCropsTransform(base_transform=small_transform)

        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                       transform=ssl_transform)

        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=apply_transform)

    elif args.dataset == 'imagenet100':
        apply_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(), 
            normalize
        ])
        ssl_transform = TwoCropsTransform(base_transform=imagenet_transform)

        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=ssl_transform)
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=apply_transform)
    
    elif args.dataset == 'tiny-imagenet':
        apply_transform = transforms.Compose([transforms.ToTensor(), normalize])
        ssl_transform = TwoCropsTransform(base_transform=tiny_imagenet_transform)

        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=ssl_transform)
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=apply_transform)

    # sample training data amongst users
    user_groups = read_inds(
        dataset_name=args.dataset,
        data_path=data_dir,
        client_num=str(args.num_users),
        alpha=str(args.dirichlet),
        generate_ifnonexist=args.gen_data_if_nonexist
    )

    return train_dataset, test_dataset, user_groups


def get_eval_dataset(args):
    data_dir = args.data
    if args.dataset in ['cifar10', 'cifar100']:
        # train dataset
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=32),
                transforms.RandomHorizontalFlip(),  # with 0.5 probability
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
        # test dataset
        test_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )

        if args.dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                    transform=train_transform)
            
            test_dataset = datasets.CIFAR10(data_dir, train=False, download=False,
                                        transform=test_transform)
        elif args.dataset == 'cifar100':
            train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                    transform=train_transform)
            
            test_dataset = datasets.CIFAR100(data_dir, train=False, download=False,
                                        transform=test_transform)
    
    elif args.dataset == 'tiny-imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])

        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=test_transform)

    elif args.dataset == 'imagenet100':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=test_transform)
    
    return train_dataset, test_dataset