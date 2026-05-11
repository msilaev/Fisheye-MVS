#!/bin/bash

#SBATCH --job-name=fisheye_emat
#SBATCH --output=fisheye_emat-%j.log
#SBATCH --error=fisheye_emat-%j.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -e
set -x

REMOTE_LOG_DIR="$1"
RESULTS_DIR="$2"       # per-pair results dir (contains mkpts1.npy, mkpts2.npy)
SCRIPT_DIR="$3"
MASK_PATH="$4"
CALIB_TYPE="${5:-aria}"          # aria | kitti360
KITTI_CALIB="${6:-}"             # path to image_02.yaml (kitti360 only)
SIZE_X="${7:-1408}"
SIZE_Y="${8:-1408}"

module load mamba
eval "$(mamba shell hook --shell bash)"
source activate 3d_pose_env

OUTPUT="${RESULTS_DIR}/result_fisheye_emat.json"

KITTI_ARG=""
if [ -n "$KITTI_CALIB" ]; then
    KITTI_ARG="--kitti-calib $KITTI_CALIB"
fi

python "$SCRIPT_DIR/run_fisheye_emat_baseline.py" \
    --mkpts0  "${RESULTS_DIR}/mkpts1.npy" \
    --mkpts1  "${RESULTS_DIR}/mkpts2.npy" \
    --output  "$OUTPUT" \
    --calib-type "$CALIB_TYPE" \
    --mask    "$MASK_PATH" \
    --size-x  "$SIZE_X" \
    --size-y  "$SIZE_Y" \
    $KITTI_ARG \
    > "$REMOTE_LOG_DIR/fisheye_emat.log" 2>&1

echo "[REMOTE] Fisheye E-mat baseline finished: $OUTPUT"
