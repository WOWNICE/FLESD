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

# the models
from models import *

normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalize_cifar = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
custom_transforms = {}

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

###############################################################
# Self-Supervised Training Transforms
###############################################################

custom_transforms['ssl-ti'] = TwoCropsTransform(transforms.Compose([
    transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.4)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.5),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_imagenet
]))

custom_transforms['ssl-in'] = TwoCropsTransform(transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.4)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.5),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_imagenet
]))

custom_transforms['ssl-cifar'] = TwoCropsTransform(transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_cifar
]))


###############################################################
# Self-Supervised Linear-Eval/Supervised Training Transforms
###############################################################

custom_transforms['sup-cifar'] = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_cifar
])

custom_transforms['sup-ti'] = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_imagenet,
])

custom_transforms['sup-in'] = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_imagenet,
])

###############################################################
# Testing Transforms
###############################################################
custom_transforms['test-in'] = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize_imagenet
])

custom_transforms['test-ti'] = transforms.Compose([
    transforms.ToTensor(), 
    normalize_imagenet
])

custom_transforms['test-cifar'] = transforms.Compose([
    transforms.ToTensor(), 
    normalize_imagenet
])