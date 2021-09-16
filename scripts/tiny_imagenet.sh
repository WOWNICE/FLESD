for alpha in 0.01 1 100 ; do python fedavg_simclr.py --epochs 10 --num_users 6 --frac 1. --local_ep 20  --local_bs 256 --lr 1e-3 --model resnet18 --dataset tiny-imagenet --mlp_dim 128 --optimizer adam --verbose 0 --T 0.4 --dirichlet $alpha --ckptdir /raid/newfeds-tiny-imagenet/fedavg/alpha$alpha-globalep10-localep20 --local-save-freq 0 --gen-data-if-nonexist ; done

globalep=2
localep=100
for alpha in 0.01 1 100
do 
	python flerd_main.py --epochs $globalep --num_users 6 --frac 1. --local_ep $localep --local_bs 256 --lr 1e-3 --model resnet18 --dataset tiny-imagenet --mlp_dim 128 --optimizer adam --verbose 0 --dirichlet $alpha --gen-data-if-nonexist --T 0.4 --ckptdir /raid/newfeds-tiny-imagenet/flerd/alpha${alpha}-globalep${globalep}-localep${localep} --local-save-freq 0 --communication sim-full --flerd-K 2048 --flerd-m 0.999 --flerd-T 0.1 --flerd-targetT 0.1 --flerd-epochs 200 --flerd-optimizer adam --flerd-bs 128 --flerd-cl --flerd-cl-weight 0 --flerd-lr 1e-3
done
