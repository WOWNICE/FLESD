root_path=/raid/newfeds-c10/local
for alpha in 0.01 0.1 1 10 100
do
	for client in 0 1 2 3 4 5
	do
		python -m eval.lincls -a resnet18 --lr 3e-2 --batch-size 256 --pretrained $root_path/alpha$alpha/local_0-client_${client}-localep_200.pth.tar --gpu $client /raid/CIFAR --dataset cifar10 --epochs 60 --schedule 30 50 > $root_path/alpha$alpha/lincls-client$client.txt &
	done
	wait
done
