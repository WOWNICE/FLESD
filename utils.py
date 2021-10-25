#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms

from sampling import noniid

from PIL import ImageFilter
from PIL import Image
import random
import os

# the models
from models import *

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

# for tiny_imagenet
tiny_imagenet_transform = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.4)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

imagenet_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.4)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

small_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


def get_dataset(args):
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
    user_groups = noniid(
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
    user_groups = noniid(
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


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    print(f'    Dirichlet:         : {args.dirichlet}')
    print(f'    Num of users:      : {args.num_users}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def build_model(model_name, num_classes=10):
    # BUILD MODEL
    if model_name == 'resnet18':
        global_model = my_resnet18(num_classes=num_classes)
    elif model_name == 'resnet50':
        global_model = resnet50(num_classes=num_classes)
    else:
        exit('Error: unrecognized model')

    return global_model


if __name__ == '__main__':
    # model = build_model('resnet18')
    # print(model)

    data_dir = '/raid/imagenet-100'
    apply_transform = transforms.Compose([transforms.ToTensor(), normalize])
    ssl_transform = TwoCropsTransform(base_transform=imagenet_transform)

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=ssl_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=apply_transform)

    print(len(train_dataset))
    
    print(train_dataset.__dir__())