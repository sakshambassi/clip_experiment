#!/bin/bash
#SBATCH --job-name=hw2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --output=hw2.out
#SBATCH --error=hw2.err
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1

eval "$(conda shell.bash hook)"
conda activate clip
python hw2.py