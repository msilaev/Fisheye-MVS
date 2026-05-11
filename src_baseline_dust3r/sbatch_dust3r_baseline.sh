#!/bin/bash

#SBATCH --job-name=dust3r_baseline
#SBATCH --output=dust3r_baseline-%j.log
#SBATCH --error=dust3r_baseline-%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1

set -e
set -x

REMOTE_LOG_DIR="$1"
IMAGE_DIR="$2"
RESULTS_DIR="$3"
SCRIPT_DIR="$4"
DUST3R_DIR="$5"          # path to the dust3r repo root (for PYTHONPATH)

module load mamba
eval "$(mamba shell hook --shell bash)"
source activate dust3r_env

export PYTHONPATH="${DUST3R_DIR}:${DUST3R_DIR}/croco:${PYTHONPATH:-}"

mkdir -p "$RESULTS_DIR/dust3r"
mkdir -p "$RESULTS_DIR/dust3r_rect"

image0="${IMAGE_DIR}/image0.jpg"
image1="${IMAGE_DIR}/image1.jpg"

# Raw fisheye
python "$SCRIPT_DIR/run_dust3r_baseline.py" \
    --image0 "$image0" \
    --image1 "$image1" \
    --output "$RESULTS_DIR/dust3r/result.json" \
    --device cuda \
    > "$REMOTE_LOG_DIR/dust3r_raw.log" 2>&1

# Rectified fisheye (projectaria_tools, nominal Aria RGB calibration)
python "$SCRIPT_DIR/run_dust3r_baseline.py" \
    --image0 "$image0" \
    --image1 "$image1" \
    --output "$RESULTS_DIR/dust3r_rect/result.json" \
    --device cuda \
    --rectify \
    > "$REMOTE_LOG_DIR/dust3r_rect.log" 2>&1

echo "[REMOTE] DUSt3R baseline (raw + rectified) finished."
