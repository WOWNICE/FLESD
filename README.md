# FLESD: Federated Self-Supervised Learning via Ensemble Similarity Distillation
## Credit to previous work
This repo is built upon the amazing code base of [federated learning](https://github.com/AshwinRJ/Federated-Learning) and [self-supervised learning method MoCo](https://github.com/facebookresearch/moco). The self-supervised contrastive local learning follows the idea of [SimCLR](https://arxiv.org/abs/2002.05709).

## Preparation
This repo runs under the pytorch automatic mixed precision and GPU support, so please make sure the version of pytorch >= 1.6.0 and proper cuda setup on your device. 

## Non-i.i.d-ness simulation
We simulate the non-i.i.d distribution across different clients by applying dirichelet distribution. The indices of samples are stored in `noniid.json` and this part of code can be found in `sampling.py`. We provide our sampled data distribution in this repo for the sake of reproductivity.

## Baselines
### Local training
In the paper, we typically benchmarked the local training of the supervised/self-supervised method in the federated learning scenario. To achieve this, run `fedavg_supervised`/`fedavg_simclr.py` with global epochs set to 1 and local checkpoint frequency to non-zero. Take CIFAR10 dataset as an example:
```sh
python fedavg_simclr.py \
    --epochs 1 --num_users 6 --frac 1. --local_ep 200 --local_bs 1024 --lr 1e-3 \
    --model resnet18 --dataset cifar10 --mlp_dim 128 --optimizer adam --verbose 1 \
    --dirichlet 0.01 --T 0.4 --ckptdir <path/to/checkpoint/directory> \
    --local-save-freq 100 --gen-data-if-nonexist \
    --data-dir <path/to/data/directory>
```
### Federated baselines
We benchmarked two federated baselines [FedAvg](https://arxiv.org/abs/1602.05629v2) and [FedProx](https://arxiv.org/abs/1812.06127) applied to self-supervised learning in the paper. Training FedAvg+SimCLR on CIFAR100, 6 clients, with direclet value  $\alpha=0.01$ for 200 epochs: 
```sh
python fedavg_simclr.py \
    --epochs 20 --num_users 6 --frac 1. --local_ep 10 --local_bs 1024 --lr 1e-3 \
    --model resnet18 --dataset cifar100 --mlp_dim 128 --optimizer adam --verbose 1 \
    --dirichlet 0.01 --T 0.4 --ckptdir <path/to/checkpoint/directory> \
    --local-save-freq 0 --save-freq 1 --gen-data-if-nonexist \
    --data-dir <path/to/data/directory>
```
If you want to train FedProx, specify the `mu` argument using the command above. The default $\mu$=1e-4 in the original paper.
```sh
python fedavg_simclr.py --mu 1e-4 \
    --epochs 20 --num_users 6 --frac 1. --local_ep 10 --local_bs 1024 --lr 1e-3 \
    --model resnet18 --dataset cifar100 --mlp_dim 128 --optimizer adam --verbose 1 \
    --dirichlet 0.01 --T 0.4 --ckptdir <path/to/checkpoint/directory> \
    --local-save-freq 0 --save-freq 1 --gen-data-if-nonexist \
    --data-dir <path/to/data/directory>
```
## FLESD training
The default training of FLESD:
```sh
python main.py \
    --epochs 2 --num_users 6 --frac 1. --local_ep 100 --local_bs 1024 --lr 1e-3 \
    --model resnet18 --dataset cifar100 --mlp_dim 128 --optimizer adam --verbose 1 \
    --dirichlet 0.01 --T 0.4 --ckptdir <path/to/checkpoint/directory> \
    --local-save-freq 0 --save-freq 1 --communication sim-full \
    --flesd-K 2048 --flesd-m 0.999 --flesd-T 0.1 \
    --flesd-targetT 0.1 --flesd-epochs 200 \
    --flesd-optimizer adam --flesd-lr 1e-3 \
    --flesd-cl --flesd-cl-weight 0 \
    --data-dir <path/to/data/directory>
```
the arguments of which represents:
- `communication`: which type of similarity matrix to pass and aggregate, supporting:
    - sim-full: original similarity;
    - sim-full-clip: normalize and clip the similarity matrix before ensembling (empirically not helpful, not shown in the paper);
    - sim-\<n\>: \<n\>% of the top similarity values are kept during communication;
- `flesd-K`: anchor set size (momentum queue size);
- `flesd-m`: momentum factor $\zeta$;
- `flesd-T`: student temperature;
- `flesd-targetT`: temperature for ensembling similarity matrix;
- `flesd-epochs`: epochs of Ensemble Similarity Distillation;
- `flesd-optimizer`: optimizer used during distillation;
- `flesd-lr`: learning rate during distillation;
- `flesd-cl`: set to enable data augmentation during distillation;
- `flesd-cl-weight`: weight of the contrastive loss in addition to similarity distillation loss.

## Evaluation
The linear evaluation is performed under the standard setting: fixing the backbone network, and train a new linear classifier. In this repo, we set all the networks for 60 epochs with SGD optmizer. 
```sh
python eval.lincls <path/to/dataset> \
    -a resnet18 --lr 3e-2 --batch-size 256 \
    --pretrained <path/to/checkpoint/file> --gpu 0 \
    --dataset cifar100 --epochs 60 --schedule 30 50
```
## License
Since we developed some of our code based on MoCo repo, this project is under the CC-BY-NC 4.0 license, kept the same as MoCo.
