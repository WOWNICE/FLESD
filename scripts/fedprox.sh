globalep=10
localep=20
alpha=1
for mu in 1e-3 1e-2 
do
	python fedavg_simclr.py --epochs $globalep --num_users 6 --frac 1. --local_ep $localep --local_bs 1024 --lr 1e-3 --model resnet18 --dataset cifar100 --mlp_dim 128 --optimizer adam --verbose 1 --T 0.4 --dirichlet $alpha --ckptdir /raid/newfeds/fedprox/alpha$alpha-globalep$globalep-localep$localep-mu$mu --local-save-freq 0 --gen-data-if-nonexist --mu $mu
done
