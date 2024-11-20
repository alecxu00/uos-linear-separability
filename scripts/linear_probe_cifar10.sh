#!/bin/bash

#SBATCH --job-name=linear_probe_cifar10
#SBATCH --account=qingqu1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --mem-per-gpu=64G
#SBATCH --time=07-00:00:00
#SBATCH --output=../logs/linear_probe_cifar10_init.log

declare -a widths=1024
declare -a data_types=("cifar10")

declare -a trials=(0 1 2 3 4 5 6 7 8 9)

for trial in "${trials[@]}"
do
	for width in "${widths[@]}"
	do
		for data_type in "${data_types[@]}"
		do
			model_path="../save/hybrid/cifar10_init/width_${width}_depth_7_nonlinear_depth_6_10_classes_relu_activation_seed_0/trial_${trial}"
			python3 ../linear_probe.py \
			--model_path $model_path \
			--data_dim 3072 \
			--hidden_dim $width \
			--depth 7 \
			--nonlinear_depth 6 \
			--data_type $data_type \
			--num_classes 10 \
			--samples_per_class 1000
		done
	done
done
