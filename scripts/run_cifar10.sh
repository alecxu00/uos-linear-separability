#!/bin/bash

#SBATCH --job-name=run_cifar10_mlp
#SBATCH --account=qingqu1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=01-00:00:00
#SBATCH --output=../logs/run_cifar10_mlp.log

declare -a widths=(1024)
declare -a activations=("relu")

DEPTH=7
NONLINEAR_DEPTH=6

INIT="gaussian"
INIT_VAR=1e-2

SEED=0

DATA_TYPE="cifar10"
NUM_CLASSES=10
TRAIN_SAMPLES_PER_CLASS=1000
TEST_SAMPLES_PER_CLASS=1000
DATA_DIM=3072

EPOCHS=500
BATCH_SIZE=128
LR=1e-3
PATIENCE=100

SAVE_DIR="../save/cifar10_mlp"

for WIDTH in "${widths[@]}"
do
	for ACTIVATION in "${activations[@]}"
	do
		python3 ../train.py --num_trials 1 \
		--hidden_dim $WIDTH --depth $DEPTH --nonlinear_depth $NONLINEAR_DEPTH --activation $ACTIVATION \
		--init $INIT --init_var $INIT_VAR \
		--seed $SEED --data_type $DATA_TYPE --num_classes $NUM_CLASSES --samples_per_class $TRAIN_SAMPLES_PER_CLASS \
		--data_dim $DATA_DIM \
		--epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --patience $PATIENCE \
		--save_dir $SAVE_DIR

		MODEL_PATH="${SAVE_DIR}/width_${WIDTH}_depth_${DEPTH}_nonlinear_depth_${NONLINEAR_DEPTH}_${NUM_CLASSES}_classes_${ACTIVATION}_activation_seed_${SEED}"

		#python3 ../test.py \
		#--model_path $MODEL_PATH \
		#--hidden_dim $WIDTH --depth $DEPTH --nonlinear_depth $NONLINEAR_DEPTH --activation $ACTIVATION \
		#--init $INIT --init_var $INIT_VAR \
		#--seed $SEED --data_type $DATA_TYPE --num_classes $NUM_CLASSES --samples_per_class $TEST_SAMPLES_PER_CLASS \
		#--data_dim $DATA_DIM \
		#--batch_size $BATCH_SIZE
	done
done
