#!/bin/bash

#SBATCH --job-name=run_inits
#SBATCH --account=qingqu1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --output=../logs/run_inits.log

# sweep through network widths and activation functions
declare -a widths=(32 64 128 256 512 1024 2048)
declare -a inits=("gaussian" "uniform")

# model, data, and training settings
DEPTH=2
NONLINEAR_DEPTH=1
ACTIVATION="relu"

INIT_VAR=1e-2

SEED=0

DATA_TYPE="uos"
NUM_CLASSES=5
SAMPLES_PER_CLASS=100
DATA_DIM=16
RANK=4
ANGLE=0

EPOCHS=1000
BATCH_SIZE=128
LR=1e-1
PATIENCE=100

SAVE_DIR="../save/hybrid/inits"

for WIDTH in "${widths[@]}"
do
	for INIT in "${inits[@]}"
	do
		python3 ../train.py \
		--hidden_dim $WIDTH --depth $DEPTH --nonlinear_depth $NONLINEAR_DEPTH --activation $ACTIVATION \
		--init $INIT --init_var $INIT_VAR \
		--seed $SEED --data_type $DATA_TYPE --num_classes $NUM_CLASSES --samples_per_class $SAMPLES_PER_CLASS \
		--data_dim $DATA_DIM --rank $RANK --angle $ANGLE \
		--epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --patience $PATIENCE \
		--save_dir $SAVE_DIR

		MODEL_PATH="${SAVE_DIR}/width_${WIDTH}_depth_${DEPTH}_nonlinear_depth_${NONLINEAR_DEPTH}_${INIT}_init_${DATA_TYPE}_data_${NUM_CLASSES}_classes_rank_${RANK}_angle_${ANGLE}_${ACTIVATION}_activation_seed_${SEED}"

		python3 ../test.py \
		--model_path $MODEL_PATH \
		--hidden_dim $WIDTH --depth $DEPTH --nonlinear_depth $NONLINEAR_DEPTH --activation $ACTIVATION \
		--init $INIT --init_var $INIT_VAR \
		--seed $SEED --data_type $DATA_TYPE --num_classes $NUM_CLASSES --samples_per_class $SAMPLES_PER_CLASS \
		--data_dim $DATA_DIM --rank $RANK --angle $ANGLE \
		--batch_size $BATCH_SIZE
	done
done

