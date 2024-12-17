#!/bin/bash

#SBATCH --job-name=linear_probe_during_train
#SBATCH --account=qingqu1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=07-00:00:00
#SBATCH --output=../logs/linear_probe_during_train_d16_r4_shallow_relu.log

#declare -a widths=(32 64 128 256 512 1024)
declare -a widths=(128)
declare -a activations=("relu") #"elu" "gelu" "leaky-relu" "quadratic")

DEPTH=3
NONLINEAR_DEPTH=2

INIT="gaussian"
INIT_VAR=1e-2

NUM_TRIALS=5
SEED=0

DATA_TYPE="uos"
NUM_CLASSES=2
SAMPLES_PER_CLASS=5000
DATA_DIM=16
RANK=4
#ANGLE=0
NOISE_STD=0

EPOCHS=100
BATCH_SIZE=128
LR=1e-1
PATIENCE=100

LAYERS_TO_PROBE=2
LINEAR_EPOCHS=200
LINEAR_BATCH_SIZE=128
LINEAR_LR=1e-1
LINEAR_PATIENCE=100

SAVE_DIR="../save/linear_probe_during_train"

for WIDTH in "${widths[@]}"
do
	for ACTIVATION in "${activations[@]}"
	do
		python3 ../linear_probe_during_train.py --num_trials $NUM_TRIALS \
		--hidden_dim $WIDTH --depth $DEPTH --nonlinear_depth $NONLINEAR_DEPTH --activation $ACTIVATION \
		--init $INIT --init_var $INIT_VAR \
		--seed $SEED --data_type $DATA_TYPE --num_classes $NUM_CLASSES --samples_per_class $SAMPLES_PER_CLASS \
		--data_dim $DATA_DIM --rank $RANK --noise_std $NOISE_STD \
		--epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR --patience $PATIENCE \
		--layers_to_probe $LAYERS_TO_PROBE --linear_epochs $LINEAR_EPOCHS --linear_batch_size $LINEAR_BATCH_SIZE --linear_lr $LINEAR_LR --linear_patience $LINEAR_PATIENCE \
                --save_dir $SAVE_DIR
	done
done
