globalep=5
for localep in 100 200 400
do
	for alpha in 0.01 1 100
	do
		python flerd_main.py --epochs $globalep --num_users 6 --frac 1. --local_ep $localep --local_bs 1024 --lr 1e-3 --model resnet18 --dataset cifar100 --mlp_dim 128 --optimizer adam --verbose 0 --dirichlet $alpha --gen-data-if-nonexist --T 0.4 --ckptdir /raid/newfeds/flerd/alpha${alpha}-globalep${globalep}-localep${localep} --local-save-freq 0 --communication sim-full --flerd-K 2048 --flerd-m 0.999 --flerd-T 0.1 --flerd-targetT 0.1 --flerd-epochs 200 --flerd-optimizer adam --flerd-bs 128 --flerd-cl --flerd-cl-weight 0 --flerd-lr 1e-3
	done
done

gpu=0
for localep in 100 200 400
do
	for alpha in 0.01 1 100
	do
		gpu=$(($gpu+1))
		gpu=$(($gpu%8))
		ckpt_path=/raid/newfeds/flerd/alpha$alpha-globalep$globalep-localep$localep/global_$globalep.pth.tar
		python -m eval.lincls -a resnet18 --lr 3e-2 --batch-size 256 --pretrained $ckpt_path --gpu $gpu /raid/CIFAR100 --dataset cifar100 --epochs 60 --schedule 30 50 > /raid/newfeds/flerd/alpha$alpha-globalep$globalep-localep$localep/lincls.txt &
	done
done

