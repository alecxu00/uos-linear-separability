#!/bin/bash

#SBATCH --job-name=linear_probe_cifar10
#SBATCH --account=qingqu1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=01-00:00:00
#SBATCH --output=../logs/linear_probe_cifar10_nonlinear.log

declare -a widths=1024
declare -a data_types=("cifar10")

for width in "${widths[@]}"
do
	for data_type in "${data_types[@]}"
	do
		model_path="../save/hybrid/cifar10/width_${width}_depth_7_nonlinear_depth_6_10_classes_relu_activation_seed_0"
		python3 ../linear_probe.py \
		--model_path $model_path \
		--data_dim 3072 \
		--hidden_dim $width \
		--depth 7 \
		--nonlinear_depth 6 \
		--data_type $data_type \
		--num_classes 10 \
		--samples_per_class 1000
		#--save_dir ../save/hybrid/cifar10
	done
done

