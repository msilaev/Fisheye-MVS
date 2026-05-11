#!/bin/bash

#SBATCH --job-name=pinhole_baseline
#SBATCH --output=pinhole_baseline-%j.log
#SBATCH --error=pinhole_baseline-%j.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -e
set -x

REMOTE_LOG_DIR="$1"
IMAGE_DIR="$2"
RESULTS_DIR="$3"
SCRIPT_DIR="$4"
CALIB_PATH="${5:-}"   # optional: JSON with fisheye624_params / rect_size / rect_focal

module load mamba
eval "$(mamba shell hook --shell bash)"
source activate 3d_pose_env

mkdir -p "$RESULTS_DIR/pinhole"

# ADT images are stored as image0.jpg / image1.jpg in the pair dir
image0="${IMAGE_DIR}/image0.jpg"
image1="${IMAGE_DIR}/image1.jpg"
output="${RESULTS_DIR}/pinhole/result.json"

CALIB_ARG=""
if [ -n "$CALIB_PATH" ]; then
    CALIB_ARG="--calib $CALIB_PATH"
fi

python "$SCRIPT_DIR/run_pinhole_baseline.py" \
    --image0 "$image0" \
    --image1 "$image1" \
    --output "$output" \
    $CALIB_ARG \
    > "$REMOTE_LOG_DIR/pinhole_baseline.log" 2>&1

echo "[REMOTE] Pinhole baseline finished."
