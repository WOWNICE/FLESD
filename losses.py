#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import logit, nn
from torch.utils.data import DataLoader
import os
import time
import numpy as np
from main.flesd import Flesd
import copy
from utils import accuracy


def SimCLR(model, input_data, tau, criterion):
    """returns the simclr loss and pretext task accuracy."""
    x1, x2 = input_data[0].cuda(non_blocking=True), input_data[1].cuda(non_blocking=True)
    ###################### not sure whether this change the performance.
    # q, k = model(x1), model(x2)
    q, k = nn.functional.normalize(model(x1), dim=1), nn.functional.normalize(model(x2), dim=1)

    # adopted from the mocov2 code, not exactly SimCLR
    logits = torch.einsum('nc,ck->nk', [q, k.T])

    # apply temperature
    logits /= tau

    # labels: positive key indicators
    labels = torch.arange(logits.shape[0], dtype=torch.long).cuda(non_blocking=True)

    return criterion(logits, labels), accuracy(output=logits, target=labels)[0]


def SupCls(model, input_data, labels, criterion):
    """returns the simclr loss and training accuracy."""
    x = input_data.cuda(non_blocking=True)
    logits = model(x)
    return criterion(logits, labels), accuracy(output=logits, target=labels)[0]