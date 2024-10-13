#!/bin/bash

#SBATCH --job-name=run_dependence
#SBATCH --account=qingqu1
#SBATCH --partition=standard
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=00-16:00:00
#SBATCH --output=./logs/run_dependence.log

python dependence.py
