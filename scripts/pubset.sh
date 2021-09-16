dataset=$1
for percent in 1 5 10 20
do
	for alpha in 0.01 1 100
	do
		python flerd_main.py --epochs 2 --num_users 6 --frac 1. --local_ep 100 --local_bs 1024 --lr 1e-3 --model resnet18 --dataset cifar100 --mlp_dim 128 --optimizer adam --verbose 0 --dirichlet $alpha --gen-data-if-nonexist --T 0.4 --ckptdir /raid/newfeds/flerd-pubset/$dataset-alpha$alpha-percent$percent --local-save-freq 0 --communication sim-full --flerd-K 256 --flerd-m 0.999 --flerd-T 0.1 --flerd-targetT 0.1 --flerd-epochs 200 --flerd-optimizer adam --flerd-bs 128 --flerd-cl --flerd-cl-weight 0 --flerd-lr 1e-3 --pubset $dataset.$percent
	done
done


# evaluation
gpu=0
for percent in 1 5 10 20
do
	for alpha in 0.01 1 100
	do
		python -m eval.lincls -a resnet18 --lr 3e-2 --batch-size 256 --pretrained /raid/newfeds/flerd-pubset/$dataset-alpha$alpha-percent$percent/global_2.pth.tar --gpu $gpu /raid/CIFAR100 --dataset cifar100 --epochs 60 --schedule 30 50 > /raid/newfeds/flerd-pubset/$dataset-alpha$alpha-percent$percent.txt &
		gpu=$(($gpu+1))
		gpu=$(($gpu%8))
	done
done



