#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('-j', '--num-workers', type=int, default=8,
                        help="num workers per user(gpu).")
    parser.add_argument('--save-freq', type=int, default=10,
                        help="the frequency of checkpointing the global model")
    parser.add_argument('--local-save-freq', type=int, default=0,
                        help="the frequency of checkpointing the local model")
    parser.add_argument('--local-val-freq', type=int, default=0,
                        help="the frequency of checkpointing the local model")
    parser.add_argument('--gen-data-if-nonexist', action='store_true',
                        help="generate the dataset if the current required dataset does not exist.")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--mlp_dim', type=int, default=128, help="number \
                        of classes, originally. Now it's the mlp final dimension.")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--dirichlet', type=str, default='1',
                        help='dirichlet alpha, str.')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    # simclr setting
    parser.add_argument('--T', type=float, default=0.2,
                    help='temperature.')
    parser.add_argument('--ckptdir', type=str, default='/raid/fedssl-checkpoints', help="name \
                of the checkpoint directory")
    parser.add_argument('--mu', type=float, default=-1, 
                        help="fedprox setting. <0 disable and degenerate to fedavg.")

    # flesd setting.
    parser.add_argument('--communication', type=str, default='sim-full', 
                        help="the communication result back to the main thread.")
    parser.add_argument('--flesd-K', type=int, default=4096, 
                        help="queue size for RKD")
    parser.add_argument('--flesd-m', type=float, default=0.999, 
                        help="momentum encoder for flesd.")
    parser.add_argument('--flesd-T', type=float, default=0.07, 
                        help="temperature for flesd.")
    parser.add_argument('--flesd-targetT', type=float, default=0.07, 
                        help="temperature before ensembling the similarity matrix.")
    parser.add_argument('--ensemble-smoothT', type=float, default=0, 
                        help="temperature after ensembling the similarity matrix.")
    parser.add_argument('--flesd-epochs', type=int, default=200, 
                        help="epochs of distillation")
    parser.add_argument('--flesd-optimizer', type=str, default='adam', 
                        help="the optimizer for adam.")
    parser.add_argument('--flesd-lr', type=float, default=1e-3, 
                        help="the ERD learning rate for the global network.")
    parser.add_argument('--flesd-bs', type=int, default=256, 
                        help="the batch size of ensemble distillation")
    parser.add_argument('--flesd-cl-weight', type=float, default=1., 
                        help="how much the CL loss applies during distillation.")
    parser.add_argument('--flesd-cl', action='store_true', 
                        help="whether contrastive learning")
    parser.add_argument('--flesd-wd', type=float, default=1e-6, 
                        help="flesd weight decay.")
    parser.add_argument('--pubset', type=str, default='', 
                        help="if not set, then use the client[0]'s data as the public dataset \
                        for global aggregation. use dataset_name.percentage to indicate which and what fraction the public dataset to use.")
    
    # initialization of global model.
    parser.add_argument('--global-ckpt', type=str, default='', 
                        help="the global checkpoint")

    args = parser.parse_args()
    return args


def lincls_parser():
    parser = argparse.ArgumentParser(description='Linear Classification Evaluation')

    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--dataset', type=str, default='cifar10', help="name \
                        of dataset")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        help='model architecture.')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=30., type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by a ratio)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                        metavar='W', help='weight decay (default: 0.)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')

    parser.add_argument('--pretrained', default='', type=str,
                        help='path to moco pretrained checkpoint')

    return parser.parse_args()