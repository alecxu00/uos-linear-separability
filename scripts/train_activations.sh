#!/bin/bash

#SBATCH --job-name=train_activations
#SBATCH --account=qingqu1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output=../logs/train_activations.log


declare -a widths=(32 64 128 256 512 1024 2048)
declare -a activations=("relu" "elu" "gelu" "step" "quadratic" "sigmoid" "leaky-relu")

for width in "${widths[@]}"
do
	for activation in "${activations[@]}"
	do
		python3 ../train.py \
		--hidden_dim $width \
		--depth 2 \
		--nonlinear_depth 1 \
		--activation $activation \
		--lr 1e-1 \
		--save_dir ../save/hybrid/activations
	done
done

