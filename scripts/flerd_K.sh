globalep=2
localep=100
gpu=0
for alpha in 0.01 1 100
do
	for anchor in 128 256 512 1024 2048 4096
	do
		python flerd_main.py --epochs $globalep --num_users 6 --frac 1. --local_ep $localep --local_bs 1024 --lr 1e-3 --model resnet18 --dataset cifar100 --mlp_dim 128 --optimizer adam --verbose 0 --dirichlet $alpha --gen-data-if-nonexist --T 0.4 --ckptdir /raid/newfeds/flerd/anchor_size/alpha${alpha}-globalep${globalep}-localep${localep}-anchor${anchor} --local-save-freq 0 --communication sim-full --flerd-K $anchor --flerd-m 0.999 --flerd-T 0.1 --flerd-targetT 0.1 --flerd-epochs 200 --flerd-optimizer adam --flerd-bs 128 --flerd-cl --flerd-cl-weight 0 --flerd-lr 1e-3 &
		if test $gpu -eq 1
		then
			wait
		fi
		gpu=$(($gpu+1))
		gpu=$(($gpu%2))
	done
done

gpu=0
for alpha in 0.01 1 100
do
	for anchor in 128 256 512 1024 2048 4096
	do
		gpu=$(($gpu+1))
		gpu=$(($gpu%8))
		ckpt_path=/raid/newfeds/flerd/anchor_size/alpha${alpha}-globalep${globalep}-localep${localep}-anchor${anchor}/global_$globalep.pth.tar
		python -m eval.lincls -a resnet18 --lr 3e-2 --batch-size 256 --pretrained $ckpt_path --gpu $gpu /raid/CIFAR100 --dataset cifar100 --epochs 60 --schedule 30 50 > /raid/newfeds/flerd/anchor_size/lincls-alpha${alpha}-anchor${anchor}.txt &
	done
done

