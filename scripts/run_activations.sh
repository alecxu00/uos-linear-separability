#!/bin/bash

#SBATCH --job-name=run_activations
#SBATCH --account=qingqu1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --mem-per-gpu=16G
#SBATCH --time=01-00:00:00
#SBATCH --output=../logs/run_activations_d16_r4_K5_quadratic.log

#declare -a widths=(32 64 128 256 512 1024)
declare -a widths=(8 16 32 64 128 256)
declare -a activations=("quadratic") #"elu" "gelu" "leaky-relu" "quadratic")

DEPTH=2
NONLINEAR_DEPTH=1

INIT="gaussian"
INIT_VAR=1e-2

SEED=0

DATA_TYPE="uos"
NUM_CLASSES=5
SAMPLES_PER_CLASS=5000
DATA_DIM=16
RANK=4
ANGLE=0

EPOCHS=500
BATCH_SIZE=128
LR=1e-1
PATIENCE=100

SAVE_DIR="../save/hybrid/activations"

for WIDTH in "${widths[@]}"
do
	for ACTIVATION in "${activations[@]}"
	do
		python3 ../train.py \
		--hidden_dim $WIDTH --depth $DEPTH --nonlinear_depth $NONLINEAR_DEPTH --activation $ACTIVATION \
		--init $INIT --init_var $INIT_VAR \
		--seed $SEED --data_type $DATA_TYPE --num_classes $NUM_CLASSES --samples_per_class $SAMPLES_PER_CLASS \
		--data_dim $DATA_DIM --rank $RANK --angle $ANGLE \
		--epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --patience $PATIENCE \
		--save_dir $SAVE_DIR

		MODEL_PATH="${SAVE_DIR}/width_${WIDTH}_depth_${DEPTH}_nonlinear_depth_${NONLINEAR_DEPTH}_${NUM_CLASSES}_classes_rank_${RANK}_angle_${ANGLE}_${ACTIVATION}_activation_seed_${SEED}"

		python3 ../test.py \
		--model_path $MODEL_PATH \
		--hidden_dim $WIDTH --depth $DEPTH --nonlinear_depth $NONLINEAR_DEPTH --activation $ACTIVATION \
		--init $INIT --init_var $INIT_VAR \
		--seed $SEED --data_type $DATA_TYPE --num_classes $NUM_CLASSES --samples_per_class $SAMPLES_PER_CLASS \
		--data_dim $DATA_DIM --rank $RANK --angle $ANGLE \
		--batch_size $BATCH_SIZE
	done
done
