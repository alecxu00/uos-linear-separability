#!/bin/bash

#SBATCH --job-name=linear_probe_activations
#SBATCH --account=qingqu1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output=../logs/linear_probe_activations.log

declare -a widths=(32 64 128 256 512 1024 2048)
declare -a activations=("relu" "elu" "gelu" "step" "quadratic" "sigmoid" "leaky-relu")

for width in "${widths[@]}"
do
	for activation in "${activations[@]}"
	do
		model_path="../save/hybrid/activations/width_${width}_depth_2_nonlinear_depth_1_gaussian_init_uos_data_${activation}_activation_seed_0"
		python3 ../linear_probe.py \
		--model_path $model_path \
		--hidden_dim $width \
		--depth 2 \
		--nonlinear_depth 1 \
		--activation $activation \
		--save_dir ../save/hybrid/activations
	done
done

