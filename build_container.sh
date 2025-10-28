#!/bin/bash
#SBATCH --job-name=build-apptainer
#SBATCH --account=OPEN-XX-XX
#SBATCH --partition=qcpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=build_container_%j.out
#SBATCH --error=build_container_%j.err

# Exit on any error
set -e

echo "======================================"
echo "Building Apptainer Container"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "======================================"

# Load required modules on Karolina
module purge
module load Apptainer/1.1.5-GCCcore-11.3.0

# Define paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEF_FILE="${SCRIPT_DIR}/llm_evolution.def"
SIF_FILE="${SCRIPT_DIR}/llm_evolution.sif"

# Check if definition file exists
if [ ! -f "$DEF_FILE" ]; then
    echo "ERROR: Definition file not found: $DEF_FILE"
    exit 1
fi

# Remove old container if it exists
if [ -f "$SIF_FILE" ]; then
    echo "Removing existing container: $SIF_FILE"
    rm -f "$SIF_FILE"
fi

# Set temporary directory for build
export APPTAINER_TMPDIR="${SCRIPT_DIR}/tmp_build"
mkdir -p "$APPTAINER_TMPDIR"

# Build the container
echo ""
echo "Building container from: $DEF_FILE"
echo "Output file: $SIF_FILE"
echo ""

apptainer build --fakeroot "$SIF_FILE" "$DEF_FILE"

# Check if build was successful
if [ $? -eq 0 ] && [ -f "$SIF_FILE" ]; then
    echo ""
    echo "======================================"
    echo "Build completed successfully!"
    echo "Container location: $SIF_FILE"
    echo "Container size: $(du -h "$SIF_FILE" | cut -f1)"
    echo "End time: $(date)"
    echo "======================================"

    # Clean up temporary directory
    rm -rf "$APPTAINER_TMPDIR"

    # Test the container
    echo ""
    echo "Testing container..."
    apptainer exec "$SIF_FILE" python --version
    apptainer exec "$SIF_FILE" python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
else
    echo ""
    echo "======================================"
    echo "ERROR: Build failed!"
    echo "End time: $(date)"
    echo "======================================"
    exit 1
fi
