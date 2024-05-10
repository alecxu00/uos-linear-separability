#!/bin/bash

#SBATCH --job-name=run_ranks
#SBATCH --account=qingqu1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=00-08:00:00
#SBATCH --output=./logs/run_ranks.log

declare -a ranks=(64) #(1 2 4 8 16 32 64)
declare -a widths=(32 64 128 256 512 1024 2048)
declare -a activations=("relu" "elu" "gelu" "leaky-relu" "quadratic")

DEPTH=2
NONLINEAR_DEPTH=1

INIT="gaussian"
INIT_VAR=1e-2

SEED=0

DATA_TYPE="uos"
NUM_CLASSES=2
SAMPLES_PER_CLASS=1000
DATA_DIM=128
ANGLE=90

EPOCHS=1000
BATCH_SIZE=128
LR=1e-1
PATIENCE=100

SAVE_DIR="./save/hybrid/ranks"

for RANK in "${ranks[@]}"
do
	for WIDTH in "${widths[@]}"
	do
		for ACTIVATION in "${activations[@]}"
		do
			python3 train.py \
			--hidden_dim $WIDTH --depth $DEPTH --nonlinear_depth $NONLINEAR_DEPTH --activation $ACTIVATION \
			--init $INIT --init_var $INIT_VAR \
			--seed $SEED --data_type $DATA_TYPE --num_classes $NUM_CLASSES --samples_per_class $SAMPLES_PER_CLASS \
			--data_dim $DATA_DIM --rank $RANK --angle $ANGLE \
			--epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --patience $PATIENCE \
			--save_dir $SAVE_DIR

			MODEL_PATH="${SAVE_DIR}/width_${WIDTH}_depth_${DEPTH}_nonlinear_depth_${NONLINEAR_DEPTH}_${INIT}_init_${DATA_TYPE}_data_dim_${DATA_DIM}_${NUM_CLASSES}_classes_rank_${RANK}_angle_${ANGLE}_${ACTIVATION}_activation_seed_${SEED}"

			python3 test.py \
			--model_path $MODEL_PATH \
			--hidden_dim $WIDTH --depth $DEPTH --nonlinear_depth $NONLINEAR_DEPTH --activation $ACTIVATION \
			--init $INIT --init_var $INIT_VAR \
			--seed $SEED --data_type $DATA_TYPE --num_classes $NUM_CLASSES --samples_per_class $SAMPLES_PER_CLASS \
			--data_dim $DATA_DIM --rank $RANK --angle $ANGLE \
			--batch_size $BATCH_SIZE
		done
	done
done
