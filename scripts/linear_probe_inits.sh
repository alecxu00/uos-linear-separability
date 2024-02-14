#!/bin/bash

#SBATCH --job-name=linear_probe_inits
#SBATCH --account=qingqu1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output=../logs/linear_probe_inits.log

declare -a widths=(32 64 128 256 512 1024 2048)
declare -a inits=("gaussian" "uniform")

for width in "${widths[@]}"
do
	for init in "${inits[@]}"
	do
		model_path="../save/hybrid/inits/width_${width}_depth_2_nonlinear_depth_1_${init}_init_uos_data_relu_activation_seed_0"
		python3 ../linear_probe.py \
		--model_path $model_path \
		--hidden_dim $width \
		--depth 2 \
		--nonlinear_depth 1 \
		--init $init \
		--save_dir ../save/hybrid/inits
	done
done

