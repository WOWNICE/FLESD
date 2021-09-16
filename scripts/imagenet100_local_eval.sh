root_path=/raid/newfeds-imagenet100/local
gpu=0
for alpha in 0.01 1 100
do
	for client in 0 1 2 3 4 5
	do
		python -m eval.lincls -a resnet50 --lr 3e-1 --batch-size 256 --pretrained $root_path/alpha$alpha/local_1-client_${client}.pth.tar --gpu $gpu /raid/imagenet-100 --dataset imagenet100 --epochs 60 --schedule 30 50 > $root_path/alpha$alpha/lincls-client$client.txt &
		if test $gpu -eq 7
		then
			wait
		fi
		gpu=$(($gpu+1))
		gpu=$(($gpu%8))
	done
done
