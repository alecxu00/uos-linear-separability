#!/bin/bash

#SBATCH --job-name=train_data
#SBATCH --account=qingqu1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output=../logs/train_data.log

declare -a widths=(32 64 128 256 512 1024 2048)
declare -a data_types=("uos" "mog")

for width in "${widths[@]}"
do
	for data_type in "${data_types[@]}"
	do
		lr=$([ "$data_type" = "uos" ] && echo 1e-1 || echo 1e-2)

		python3 ../train.py \
		--hidden_dim $width \
		--depth 2 \
		--nonlinear_depth 1 \
		--data_type $data_type \
		--lr $lr \
		--save_dir ../save/hybrid/data_types
	done
done

