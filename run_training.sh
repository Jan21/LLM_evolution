#!/bin/bash
#SBATCH --job-name=llm-train
#SBATCH --account=OPEN-XX-XX
#SBATCH --partition=qgpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

set -e

# Set your WandB API key here
export WANDB_API_KEY=""

module purge
module load Apptainer/1.1.5-GCCcore-11.3.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF_FILE="${SCRIPT_DIR}/llm_evolution.sif"

cd "$SCRIPT_DIR"

# Single run
apptainer exec --nv "$SIF_FILE" python train.py

# Hyperparameter sweep (uncomment to use)
# apptainer exec --nv "$SIF_FILE" python train.py --multirun
