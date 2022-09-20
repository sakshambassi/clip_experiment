#!/bin/bash
#SBATCH --job-name=546_assign1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --output=546_assign1.out
#SBATCH --error=error_output.err
#SBATCH --mem=24GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

eval "$(conda shell.bash hook)"
conda activate clip
python 546_assign1.py