#!/bin/bash
# GPU sub-pipeline for the MADPose+rect baseline:
#   1. Rectify fisheye images → pinhole 512×512
#   2. UniK3D depth on rectified images
#   3. SuperGlue matching on rectified images
# Saves: image0_rect_depth.npy, image1_rect_depth.npy, mkpts1_rect.npy, mkpts2_rect.npy
#
# Args:
#   $1  PAIRS_LIST_FILE
#   $2  RESULTS_ROOT
#   $3  LOG_ROOT
#   $4  PROJ_DIR
#   $5  UNIK3D_DIR
#   $6  SUPERGLUE_DIR
#   $7  CALIB_TYPE    aria (default) or kitti360
#   $8  KITTI_CALIB   path to image_02.yaml (required if kitti360)

#SBATCH --job-name=rect_gpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1

set -x

export HF_HOME=/scratch/work/<remote-user>/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/work/<remote-user>/.cache/huggingface/hub
export TORCH_HOME=/scratch/work/<remote-user>/.cache/torch
export XDG_CACHE_HOME=/scratch/work/<remote-user>/.cache

PAIRS_LIST_FILE="$1"
RESULTS_ROOT="$2"
LOG_ROOT="$3"
PROJ_DIR="$4"
UNIK3D_DIR="$5"
SUPERGLUE_DIR="$6"
CALIB_TYPE="${7:-aria}"
KITTI_CALIB="${8:-}"

RECT_SIZE=512

module load mamba
eval "$(mamba shell hook --shell bash)"

run_pair_rect() {
    local PAIR_DIR="$1"
    local PAIR_NAME
    PAIR_NAME=$(basename "$PAIR_DIR")
    local RESULTS_DIR="${RESULTS_ROOT}/${PAIR_NAME}"
    local LOG_DIR="${LOG_ROOT}/${PAIR_NAME}"
    local IMAGE0
    local IMAGE1
    if [ -f "${PAIR_DIR}/image0.jpg" ]; then
        IMAGE0="${PAIR_DIR}/image0.jpg"; IMAGE1="${PAIR_DIR}/image1.jpg"
    else
        IMAGE0="${PAIR_DIR}/image0.png"; IMAGE1="${PAIR_DIR}/image1.png"
    fi
    local RECT_DIR="${RESULTS_DIR}/rect_tmp"

    mkdir -p "$RESULTS_DIR" "$LOG_DIR"

    # Skip if rect intermediates already exist
    if [ -f "${RESULTS_DIR}/image0_rect_depth.npy" ] && \
       [ -f "${RESULTS_DIR}/image1_rect_depth.npy" ] && \
       [ -f "${RESULTS_DIR}/mkpts1_rect.npy" ] && \
       [ -f "${RESULTS_DIR}/mkpts2_rect.npy" ]; then
        echo "[SKIP] $PAIR_NAME rect intermediates already exist"
        return 0
    fi

    echo "[START] $PAIR_NAME"

    # ── 1. Rectify images ─────────────────────────────────────────────────────
    source activate 3d_pose_env
    mkdir -p "$RECT_DIR"
    python "${PROJ_DIR}/src_madpose/rectify_images.py" \
        --image0     "$IMAGE0" \
        --image1     "$IMAGE1" \
        --output-dir "$RECT_DIR" \
        --calib-type "$CALIB_TYPE" \
        ${KITTI_CALIB:+--kitti-calib "$KITTI_CALIB"} \
        > "${LOG_DIR}/${PAIR_NAME}_rectify.log" 2>&1 \
        || { echo "[ERROR] $PAIR_NAME rectify failed"; return 1; }

    # ── 2. UniK3D depth on rectified images ───────────────────────────────────
    source deactivate || true
    source activate mvf-unik3d
    CONFIG="${UNIK3D_DIR}/configs/train/vitb.json"
    cp "${PROJ_DIR}/src_unik3d/custom_unik3d_inference.py" "${UNIK3D_DIR}/scripts/"
    cd "${UNIK3D_DIR}/scripts"
    python custom_unik3d_inference.py \
        --config-file "$CONFIG" \
        --data-source-dir "$RECT_DIR" \
        --output-dir "$RECT_DIR" \
        --images-output-dir "$RECT_DIR" \
        --camera-path "${UNIK3D_DIR}/assets/demo/scannet.json" \
        > "${LOG_DIR}/${PAIR_NAME}_unik3d_rect.log" 2>&1 \
        || { echo "[ERROR] $PAIR_NAME unik3d_rect failed"; return 1; }

    # ── 3. SuperGlue matching on rectified images ─────────────────────────────
    source deactivate || true
    source activate superglue38
    cd "$SUPERGLUE_DIR"
    python match_pairs.py \
        --resize 1600 \
        --superglue outdoor \
        --max_keypoints 2048 \
        --nms_radius 3 \
        --resize_float \
        --input_dir   "$RECT_DIR" \
        --input_pairs "${RECT_DIR}/image_pairs.txt" \
        --output_dir  "$RECT_DIR" \
        > "${LOG_DIR}/${PAIR_NAME}_superglue_rect.log" 2>&1 \
        || { echo "[ERROR] $PAIR_NAME superglue_rect match failed"; return 1; }

    source deactivate || true
    source activate 3d_pose_env
    cd "${PROJ_DIR}/src_superglue"
    python superglue_extract_pairs.py \
        --matches-file-npz "${RECT_DIR}/image0_image1_matches.npz" \
        --size_x "$RECT_SIZE" \
        --size_y "$RECT_SIZE" \
        --mkpts1-file "${RESULTS_DIR}/mkpts1_rect.npy" \
        --mkpts2-file "${RESULTS_DIR}/mkpts2_rect.npy" \
        >> "${LOG_DIR}/${PAIR_NAME}_superglue_rect.log" 2>&1 \
        || { echo "[ERROR] $PAIR_NAME superglue_rect extract failed"; return 1; }

    # ── Move depth and rectified images out of temp dir ──────────────────────
    mv "${RECT_DIR}/image0_depth.npy" "${RESULTS_DIR}/image0_rect_depth.npy" \
        || { echo "[ERROR] $PAIR_NAME missing image0_depth.npy"; return 1; }
    mv "${RECT_DIR}/image1_depth.npy" "${RESULTS_DIR}/image1_rect_depth.npy" \
        || { echo "[ERROR] $PAIR_NAME missing image1_depth.npy"; return 1; }
    mv "${RECT_DIR}/image0.jpg"       "${RESULTS_DIR}/image0_rect.jpg"
    mv "${RECT_DIR}/image1.jpg"       "${RESULTS_DIR}/image1_rect.jpg"

    # ── Cleanup temp rect dir ─────────────────────────────────────────────────
    rm -rf "$RECT_DIR"

    echo "[DONE] $PAIR_NAME"
}

while IFS= read -r PAIR_DIR; do
    [ -z "$PAIR_DIR" ] && continue
    run_pair_rect "$PAIR_DIR" || echo "[ERROR] $PAIR_DIR failed, continuing"
done < "$PAIRS_LIST_FILE"

echo "[BATCH DONE] $(wc -l < "$PAIRS_LIST_FILE") pairs processed"
