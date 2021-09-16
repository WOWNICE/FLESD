alpha=$1
globalep=$2
localep=$3
python flerd_main.py --epochs $globalep --num_users 6 --frac 1. --local_ep $localep --local_bs 128 --lr 1e-3 --model resnet50 --dataset imagenet100 --mlp_dim 128 --optimizer adam --verbose 0 --dirichlet $alpha --gen-data-if-nonexist --T 0.2 --ckptdir /raid/newfeds-imagenet100/flerd/alpha${alpha}-globalep${globalep}-localep${localep} --local-save-freq 0 --communication sim-full --flerd-K 2048 --flerd-m 0.999 --flerd-T 0.1 --flerd-targetT 0.1 --flerd-epochs 200 --flerd-optimizer adam --flerd-bs 128 --flerd-cl --flerd-cl-weight 0 --flerd-lr 1e-3
