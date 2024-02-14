#!/bin/bash

#SBATCH --job-name=train_inits
#SBATCH --account=qingqu1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output=../logs/train_inits.log

declare -a widths=(32 64 128 256 512 1024 2048)
declare -a inits=("gaussian" "uniform")

for width in "${widths[@]}"
do
	for init in "${inits[@]}"
	do
		python3 ../train.py \
		--hidden_dim $width \
		--depth 2 \
		--nonlinear_depth 1 \
		--init $init \
		--lr 1e-1 \
		--save_dir ../save/hybrid/inits
	done
done

