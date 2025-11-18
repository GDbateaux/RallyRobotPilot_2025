#!/bin/bash
#SBATCH --job-name=rally_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=00:30:00
#SBATCH --tasks=1
#SBATCH --gpus-per-node=1 
#SBATCH --nodelist=calypso8
#SBATCH --cpus-per-task=10

source .venv/bin/activate
python3 src/train.py
