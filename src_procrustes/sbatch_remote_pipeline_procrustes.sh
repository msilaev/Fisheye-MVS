#!/bin/bash
set -e
set -x

#SBATCH --job-name=procrustes_inference
#SBATCH --output=procrustes_inference-%j.log
#SBATCH --error=procrustes_inference-%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

REMOTE_LOG_DIR="$1"
REMOTE_SCRIPT_DIR_unik3d="$2"
REMOTE_SCRIPT_DIR_superglue="$3"
REMOTE_SCRIPT_DIR_procrustes="$4"
REMOTE_IMAGE_DIR="$5"
REMOTE_RESULTS_DIR="$6"
remote_fisheye_mask_path="$7"
remote_transform_result_path="$8"
DISTANCE_THRESHOLD="$9"
SIZE_X="${10}"
SIZE_Y="${11}"
EXTRA_ARGS="${12}"    # optional, e.g. "--use-ransac"

module load mamba
eval "$(mamba shell hook --shell bash)"
source activate 3d_pose_env

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
    --remote_fisheye_mask_path "${remote_fisheye_mask_path}" \
    --remote_transform_result_path "${remote_transform_result_path}" \
    --distance_threshold $DISTANCE_THRESHOLD \
    --size_x $SIZE_X \
    --size_y $SIZE_Y \
    $EXTRA_ARGS \
    > "$REMOTE_LOG_DIR/pose_estimation.log" 2>&1