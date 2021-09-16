root_path=/raid/newfeds-c10/full
for gpu in $(seq 1 8)
do
	ep=$(($gpu * 100))
	real_gpu=$(($gpu-1))
	python -m eval.lincls -a resnet18 --lr 3e-2 --batch-size 256 --pretrained $root_path/local_0-client_0-localep_$ep.pth.tar --gpu $real_gpu /raid/CIFAR --dataset cifar10 --epochs 60 --schedule 30 50 > $root_path/lincls-ep$ep.txt &
done
