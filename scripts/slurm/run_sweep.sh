#!/bin/bash
#SBATCH --job-name=wandb-agent
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err

# Usage:
#   wandb sweep configs/sweeps/latent_diffusion.yaml
#   # Returns: Created sweep with ID: user/project/abc123
#   sbatch run_sweep.sh user/project/abc123

set -e

SWEEP_ID=${1:?Error: sweep ID required. Usage: sbatch run_sweep.sh user/project/sweep-id}

echo "Starting W&B agent for sweep: $SWEEP_ID"
echo "Job array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

wandb agent "$SWEEP_ID"
