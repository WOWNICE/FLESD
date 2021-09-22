#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50 

__all__ = ['my_resnet18', 'resnet50']


# resnets 
def my_resnet18(num_classes=10):
    net = resnet18(num_classes=num_classes)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    return net