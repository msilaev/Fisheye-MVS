#!/bin/bash
set -e
set -x

REMOTE_LOG_DIR="$1"
REMOTE_SCRIPT_DIR_unik3d="$2"
REMOTE_SCRIPT_DIR_superglue="$3"
REMOTE_SCRIPT_DIR_procrustes="$4"
REMOTE_IMAGE_DIR="$5"
REMOTE_RESULTS_DIR="$6"
remote_transform_result_path="${7}"
DISTANCE_THRESHOLD="${8}"

CONDA_SETUP="/home/mikhail/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SETUP" ]; then
    source "$CONDA_SETUP"
else
  eval "$(conda shell.bash hook)"
fi

conda deactivate || true
conda activate mvf-unik3d

echo "[REMOTE] Running Procrustes pose matching ..."

#LOCAL_RESULTS_DIR="/worktmp/THESES/GAUSSIAN-SPLATTING/experiments/RESULT_MADPOSE/result_madpose"
LOCAL_RESULTS_DIR=$REMOTE_RESULTS_DIR
LOCAL_IMAGE_DIR=$REMOTE_DIR_EXP


cd $REMOTE_SCRIPT_DIR_procrustes
python pose_estimation_procrustes.py \
    --point1 "${LOCAL_RESULTS_DIR}/image0_points.npy" \
    --point2 "${LOCAL_RESULTS_DIR}/image1_points.npy" \
    --mkpts1 "${LOCAL_RESULTS_DIR}/mkpts1.npy" \
    --mkpts2 "${LOCAL_RESULTS_DIR}/mkpts2.npy" \
    --img1 "${LOCAL_IMAGE_DIR}/image0.png" \
    --img2 "${LOCAL_IMAGE_DIR}/image1.png" \
    --remote_transform_result_path "${remote_transform_result_path}" \
    --distance_threshold $DISTANCE_THRESHOLD \
    > "$REMOTE_LOG_DIR/pose_estimation.log" 2>&1