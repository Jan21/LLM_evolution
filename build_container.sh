#!/bin/bash
#SBATCH --job-name=build-container
#SBATCH --account=OPEN-XX-XX
#SBATCH --partition=qcpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=build_%j.out
#SBATCH --error=build_%j.err

set -e

module purge
module load apptainer

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF_FILE="${SCRIPT_DIR}/llm_evolution.sif"

[ -f "$SIF_FILE" ] && rm -f "$SIF_FILE"

export APPTAINER_TMPDIR="${SCRIPT_DIR}/tmp_build"
mkdir -p "$APPTAINER_TMPDIR"

apptainer build --fakeroot "$SIF_FILE" "${SCRIPT_DIR}/llm_evolution.def"

rm -rf "$APPTAINER_TMPDIR"

echo "Build complete: $SIF_FILE"
