#!/bin/bash
#SBATCH --job-name=ciffy-train
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Usage: sbatch train.sh config.yaml
# Or: sbatch --gres=gpu:2 train.sh config.yaml  # for multi-GPU

set -e

CONFIG=${1:-config.yaml}

echo "Starting training with config: $CONFIG"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

python scripts/train_latent_diffusion.py --config "$CONFIG"
