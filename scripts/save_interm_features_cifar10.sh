#!/bin/bash

#SBATCH --job-name=save_interm_features_cifar10
#SBATCH --account=qingqu1
#SBATCH --partition=standard
#SBATCH --mem-per-cpu=64G
#SBATCH --output=../logs/save_interm_features_cifar10_nonlinear.log

declare -a widths=1024
declare -a data_types=("cifar10")

for width in "${widths[@]}"
do
	for data_type in "${data_types[@]}"
	do
		model_path="../save/hybrid/cifar10/width_${width}_depth_7_nonlinear_depth_3_10_classes_relu_activation_seed_0"
		python3 ../save_interm_features.py \
		--model_path $model_path \
		--data_dim 3072 \
		--hidden_dim $width \
		--depth 7 \
		--nonlinear_depth 3 \
		--data_type $data_type \
		--num_classes 10 \
		--samples_per_class 1000
		#--save_dir ../save/hybrid/cifar10
	done
done

