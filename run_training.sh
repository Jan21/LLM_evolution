#!/bin/bash
#SBATCH --job-name=llm-train
#SBATCH --account=OPEN-XX-XX
#SBATCH --partition=qgpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=training_%j.out
#SBATCH --error=training_%j.err

# Exit on any error
set -e

echo "======================================"
echo "LLM Evolution Training Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "======================================"

# Load required modules
module purge
module load Apptainer/1.1.5-GCCcore-11.3.0

# Define paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF_FILE="${SCRIPT_DIR}/llm_evolution.sif"

# Check if container exists
if [ ! -f "$SIF_FILE" ]; then
    echo "ERROR: Container not found: $SIF_FILE"
    echo "Please run build_container.sh first"
    exit 1
fi

# Set WandB API key if needed (set this in your environment or here)
# export WANDB_API_KEY="your_wandb_api_key_here"

# Print GPU information
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Run training
echo "Starting training..."
echo "Working directory: $SCRIPT_DIR"
echo ""

# Change to project directory and run training
cd "$SCRIPT_DIR"

# Option 1: Single run
apptainer exec --nv "$SIF_FILE" python train.py

# Option 2: Hyperparameter sweep with Optuna (uncomment to use)
# apptainer exec --nv "$SIF_FILE" python train.py --multirun

echo ""
echo "======================================"
echo "Training completed!"
echo "End time: $(date)"
echo "======================================"
