#!/bin/bash

#SBATCH --job-name=linear_probe_data
#SBATCH --account=qingqu1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output=../logs/linear_probe_data.log

declare -a widths=(32 64 128 256 512 1024 2048)
declare -a data_types=("uos" "mog")

for width in "${widths[@]}"
do
	for data_type in "${data_types[@]}"
	do
		model_path="../save/hybrid/data_types/width_${width}_depth_2_nonlinear_depth_1_gaussian_init_${data_type}_data_relu_activation_seed_0"
		python3 ../linear_probe.py \
		--model_path $model_path \
		--hidden_dim $width \
		--depth 2 \
		--nonlinear_depth 1 \
		--data_type $data_type \
		--save_dir ../save/hybrid/data_types
	done
done

