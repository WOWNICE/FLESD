for alpha in 0.01 0.1 1 10 100
do
	python fedavg_simclr.py --epochs 1 --num_users 6 --frac 1. --local_ep 200 --local_bs 1024 --lr 1e-3 --model resnet18 --dataset cifar100 --mlp_dim 128 --optimizer adam --verbose 1 --T 0.4 --dirichlet $alpha --ckptdir /raid/newfeds/local/alpha$alpha --local-save-freq 100 --gen-data-if-nonexist
done
