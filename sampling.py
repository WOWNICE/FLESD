#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import numpy as np
from torchvision import datasets, transforms
import json


def read_inds(
        ind_file='inds/noniid.json', 
        dataset_name='cifar10', 
        data_path='/home/shz/data/cifar10', 
        client_num='6', 
        alpha='0.01', 
        generate_ifnonexist=False
    ):
    """
    helper function that readsthe provided index file.
    return: indices of the data samples
    """
    try:
        # read the json file
        with open(ind_file, 'r') as f:
            inds_dist = json.load(f)
    except:
        inds_dist = {}
    
    key = '/'.join([dataset_name, client_num, alpha])
    if key not in inds_dist and not generate_ifnonexist:
        print(f"The keys in the current noniid distribution file: {inds_dist.keys()}. Consider generate it by setting <generate_ifnonexist>.")
    elif key not in inds_dist:
        # generate the non-existing data distribution.
        generate_noniid_inds(
            ind_file=ind_file,
            dataset_name=dataset_name, 
            data_path=data_path, 
            alpha=alpha, 
            client_num=client_num, 
            overwrite=False
        )
        # reload the index file 
        with open(ind_file, 'r') as f:
            inds_dist = json.load(f)

    # return the user dict.
    return inds_dist[key]


def generate_noniid_inds(
        ind_file='inds/noniid.json', 
        dataset_name='cifar10', 
        data_path='/home/shz/data/cifar10', 
        client_num='6', 
        alpha='0.01', 
        min_prop=0.5, # minimal proportion of the data one client have compared to the average data.
        overwrite=False
    ):
    # if empty ind file, write a empty json file into the file location.
    if not os.path.exists(ind_file): 
        with open(ind_file, 'w') as f:
            json.dump({}, f)
    
    # read the json file
    with open(ind_file, 'r') as f:
        inds_dist = json.load(f)
    
    # new key. 
    key = '/'.join([dataset_name, client_num, alpha])
    if key in inds_dist and not overwrite:
        print(f"Data distribution {key} has already been generated. To overwirte, please set the parameter <overwrite>.")
        return 
    
    # get the dataset.
    # support cifar and imagenet.
    client_num, alpha = int(client_num), float(alpha)

    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(data_path, train=True, download=False)
    elif dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(data_path, train=True, download=False)
    elif dataset_name == 'tiny-imagenet':
        dataset = datasets.ImageFolder(os.path.join(data_path, 'train'))
    elif dataset_name == 'imagenet100':
        dataset = datasets.ImageFolder(os.path.join(data_path, 'train'))
    
    # dict_users is the target output 
    dict_users = {i: np.array([]) for i in range(client_num)}
    idxs = np.arange(len(dataset))
    labels = np.array(dataset.targets)

    # after sorting, 
    # idxs, labels are sorted.
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs, labels = idxs_labels[0, :], idxs_labels[1, :]

    # count the class and corresponding number of samples 
    classes, class_nums = np.unique(labels, return_counts=True)
    
    # splitted idxs
    splitted_idxs = np.split(idxs, [sum(class_nums[:i]) for i in range(1, len(class_nums))])

    # sample the dirichlet distribution, 
    # but also satisfy the min_prop condition that 
    # indicates the minimal number of the samples the client gets.
    while True:
        d = np.random.dirichlet([alpha for _ in range(client_num)], len(classes)).T
        prop = d.sum(axis=1).min()
        if prop > min_prop / client_num:
            break

    # shuffle the idxs
    for c_idx in splitted_idxs:
        np.random.shuffle(c_idx)

    # d_i: class distribution of user i
    num_distribution = []
    for user_i, d_i in enumerate(d):
        num_i = (class_nums * d_i).astype(int)
        num_distribution.append(np.array(num_i))
        idx_res = []
        for c in range(len(classes)):
            idx_res.extend(splitted_idxs[c][:num_i[c]])
            splitted_idxs[c] = splitted_idxs[c][num_i[c]:]
        dict_users[user_i] = [int(x) for x in idx_res]
    
    # write into the .json file
    inds_dist[key] = dict_users
    with open(ind_file, 'w') as f:
        json.dump(inds_dist, f)


def get_user_groups(args):
    """same function as read_inds"""
    # sample training data amongst users
    return read_inds(
        ind_file=args.ind_file,
        dataset_name=args.dataset,
        data_path=args.data_dir,
        client_num=str(args.num_users),
        alpha=str(args.dirichlet),
        generate_ifnonexist=args.gen_data_if_nonexist
    )


if __name__ == '__main__':
    for alpha in ['0.01', '0.1', '1', '10', '100']:
        generate_noniid_inds(dataset_name='cifar10', data_path='/raid/CIFAR', alpha=alpha, client_num='6', overwrite=True)
