#!/bin/bash

#SBATCH --job-name=num_classes_sweep
#SBATCH --account=qingqu1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --mem-per-gpu=16G
#SBATCH --time=01-00:00:00
#SBATCH --output=../logs/num_classes_sweep_d128_r16_K3_relu.log

#declare -a widths=(32 64 128 256 512 1024)
declare -a widths=(8 16 32 64 128 256 512 1024)
declare -a activations=("relu") #"elu" "gelu" "leaky-relu" "quadratic")

DEPTH=2
NONLINEAR_DEPTH=1

INIT="gaussian"
INIT_VAR=1e-2

SEED=0

DATA_TYPE="uos"
NUM_CLASSES=3
SAMPLES_PER_CLASS=5000
DATA_DIM=128
RANK=16
#ANGLE=0
NOISE_STD=0

EPOCHS=500
BATCH_SIZE=128
LR=1e-1
PATIENCE=100

SAVE_DIR="../save/num_classes_sweep"

for WIDTH in "${widths[@]}"
do
	for ACTIVATION in "${activations[@]}"
	do
		python3 ../train.py \
		--hidden_dim $WIDTH --depth $DEPTH --nonlinear_depth $NONLINEAR_DEPTH --activation $ACTIVATION \
		--init $INIT --init_var $INIT_VAR \
		--seed $SEED --data_type $DATA_TYPE --num_classes $NUM_CLASSES --samples_per_class $SAMPLES_PER_CLASS \
		--data_dim $DATA_DIM --rank $RANK --noise_std $NOISE_STD \
		--epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --patience $PATIENCE \
		--save_dir $SAVE_DIR

		MODEL_PATH="${SAVE_DIR}/width_${WIDTH}_depth_${DEPTH}_nonlinear_depth_${NONLINEAR_DEPTH}_${NUM_CLASSES}_classes_rank_${RANK}_${ACTIVATION}_activation_seed_${SEED}"

		python3 ../test.py \
		--model_path $MODEL_PATH \
		--hidden_dim $WIDTH --depth $DEPTH --nonlinear_depth $NONLINEAR_DEPTH --activation $ACTIVATION \
		--init $INIT --init_var $INIT_VAR \
		--seed $SEED --data_type $DATA_TYPE --num_classes $NUM_CLASSES --samples_per_class $SAMPLES_PER_CLASS \
		--data_dim $DATA_DIM --rank $RANK --noise_std $NOISE_STD \
		--batch_size $BATCH_SIZE
	done
done
