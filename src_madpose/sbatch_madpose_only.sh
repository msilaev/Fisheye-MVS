#!/bin/bash
# Run MADPose-only on a batch of pairs (CPU partition).
# Assumes depth.npy and mkpts.npy already exist in RESULTS_DIR per pair.
#
# Args:
#   $1  PAIRS_LIST_FILE  text file with one PAIR_DIR path per line
#   $2  RESULTS_ROOT     root dir; results go into $RESULTS_ROOT/<pair_name>/
#   $3  LOG_ROOT         root dir; logs go into $LOG_ROOT/<pair_name>/
#   $4  PROJ_DIR         Fisheye-MVS project root on remote
#   $5  CALIB_TYPE       aria (default) or kitti360
#   $6  KITTI_CALIB      path to image_02.yaml (required if kitti360)

#SBATCH --job-name=madpose_only
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=batch-skl

set -x

# Redirect model caches to scratch to avoid filling home quota
export HF_HOME=/scratch/work/<remote-user>/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/work/<remote-user>/.cache/huggingface/hub
export TORCH_HOME=/scratch/work/<remote-user>/.cache/torch
export XDG_CACHE_HOME=/scratch/work/<remote-user>/.cache

PAIRS_LIST_FILE="$1"
RESULTS_ROOT="$2"
LOG_ROOT="$3"
PROJ_DIR="$4"
CALIB_TYPE="${5:-aria}"
KITTI_CALIB="${6:-}"

module load mamba
eval "$(mamba shell hook --shell bash)"
source activate madpose_env

run_pair_madpose() {
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

    mkdir -p "$LOG_DIR"

    # Ensure GT relative pose JSON exists for this pair.
    if [ ! -f "${RESULTS_DIR}/result_gt.json" ]; then
        python "${PROJ_DIR}/src_eval/generate_result_gt.py" \
            --cam2w-0 "${PAIR_DIR}/image0.txt" \
            --cam2w-1 "${PAIR_DIR}/image1.txt" \
            --output "${RESULTS_DIR}/result_gt.json" \
            > "${LOG_DIR}/${PAIR_NAME}_gt.log" 2>&1 || \
            echo "[WARN] $PAIR_NAME failed to generate result_gt.json"
    fi

    # ── MADPose (fisheye) ─────────────────────────────────────────────────────
    if [ -f "${RESULTS_DIR}/result_madpose.json" ] && \
       ! grep -q '"R_est": null' "${RESULTS_DIR}/result_madpose.json"; then
        echo "[SKIP] $PAIR_NAME madpose already valid"
    elif [ ! -f "${RESULTS_DIR}/image0_depth.npy" ] || \
         [ ! -f "${RESULTS_DIR}/image1_depth.npy" ] || \
         [ ! -f "${RESULTS_DIR}/mkpts1.npy" ] || \
         [ ! -f "${RESULTS_DIR}/mkpts2.npy" ]; then
        echo "[SKIP] $PAIR_NAME missing fisheye depth/mkpts"
    else
        echo "[START madpose] $PAIR_NAME"
        python "${PROJ_DIR}/src_madpose/run_madpose_baseline.py" \
            --image0  "$IMAGE0" \
            --image1  "$IMAGE1" \
            --mkpts0  "${RESULTS_DIR}/mkpts1.npy" \
            --mkpts1  "${RESULTS_DIR}/mkpts2.npy" \
            --depth0  "${RESULTS_DIR}/image0_depth.npy" \
            --depth1  "${RESULTS_DIR}/image1_depth.npy" \
            --output  "${RESULTS_DIR}/result_madpose.json" \
            > "${LOG_DIR}/${PAIR_NAME}_madpose.log" 2>&1 || \
            echo '{"R_est": null, "error": "madpose failed"}' > "${RESULTS_DIR}/result_madpose.json"

        if ! grep -q '"R_est": null' "${RESULTS_DIR}/result_madpose.json"; then
            rm -f "${RESULTS_DIR}/image0_depth.npy" "${RESULTS_DIR}/image1_depth.npy" \
                  "${RESULTS_DIR}/mkpts1.npy"        "${RESULTS_DIR}/mkpts2.npy"
        fi
    fi

    # ── MADPose+rect ───────────────────────────────────────────────────────────
    # Preferred path: use rect intermediates from sbatch_rect_gpu.sh.
    # Fallback path: run run_madpose_rect_baseline.py directly from fisheye
    # images + fisheye depth/mkpts so result_madpose_rect.json is still produced.
    if [ -f "${RESULTS_DIR}/result_madpose_rect.json" ] && \
       ! grep -q '"R_est": null' "${RESULTS_DIR}/result_madpose_rect.json"; then
        echo "[SKIP] $PAIR_NAME madpose_rect already valid"
    else
        if [ -f "${RESULTS_DIR}/image0_rect_depth.npy" ] && \
           [ -f "${RESULTS_DIR}/image1_rect_depth.npy" ] && \
           [ -f "${RESULTS_DIR}/mkpts1_rect.npy" ] && \
           [ -f "${RESULTS_DIR}/mkpts2_rect.npy" ] && \
           [ -f "${RESULTS_DIR}/image0_rect.jpg" ] && \
           [ -f "${RESULTS_DIR}/image1_rect.jpg" ]; then
            echo "[START madpose_rect] $PAIR_NAME (rect intermediates)"
            python "${PROJ_DIR}/src_madpose/run_madpose_baseline.py" \
                --image0  "${RESULTS_DIR}/image0_rect.jpg" \
                --image1  "${RESULTS_DIR}/image1_rect.jpg" \
                --mkpts0  "${RESULTS_DIR}/mkpts1_rect.npy" \
                --mkpts1  "${RESULTS_DIR}/mkpts2_rect.npy" \
                --depth0  "${RESULTS_DIR}/image0_rect_depth.npy" \
                --depth1  "${RESULTS_DIR}/image1_rect_depth.npy" \
                --output  "${RESULTS_DIR}/result_madpose_rect.json" \
                > "${LOG_DIR}/${PAIR_NAME}_madpose_rect.log" 2>&1 || \
                echo '{"R_est": null, "error": "madpose_rect failed"}' > "${RESULTS_DIR}/result_madpose_rect.json"

            if ! grep -q '"R_est": null' "${RESULTS_DIR}/result_madpose_rect.json"; then
                rm -f "${RESULTS_DIR}/image0_rect_depth.npy" "${RESULTS_DIR}/image1_rect_depth.npy" \
                      "${RESULTS_DIR}/mkpts1_rect.npy"        "${RESULTS_DIR}/mkpts2_rect.npy" \
                      "${RESULTS_DIR}/image0_rect.jpg"        "${RESULTS_DIR}/image1_rect.jpg"
                echo "[DONE] $PAIR_NAME"
            else
                echo "[FAIL] $PAIR_NAME - madpose_rect produced null result"
            fi
        elif [ -f "${RESULTS_DIR}/image0_depth.npy" ] && \
             [ -f "${RESULTS_DIR}/image1_depth.npy" ] && \
             [ -f "${RESULTS_DIR}/mkpts1.npy" ] && \
             [ -f "${RESULTS_DIR}/mkpts2.npy" ]; then
            echo "[START madpose_rect] $PAIR_NAME (direct fallback from fisheye inputs)"
            python "${PROJ_DIR}/src_madpose/run_madpose_rect_baseline.py" \
                --image0  "$IMAGE0" \
                --image1  "$IMAGE1" \
                --mkpts0  "${RESULTS_DIR}/mkpts1.npy" \
                --mkpts1  "${RESULTS_DIR}/mkpts2.npy" \
                --depth0  "${RESULTS_DIR}/image0_depth.npy" \
                --depth1  "${RESULTS_DIR}/image1_depth.npy" \
                --output  "${RESULTS_DIR}/result_madpose_rect.json" \
                --calib-type "$CALIB_TYPE" \
                ${KITTI_CALIB:+--kitti-calib "$KITTI_CALIB"} \
                > "${LOG_DIR}/${PAIR_NAME}_madpose_rect.log" 2>&1 || \
                echo '{"R_est": null, "error": "madpose_rect failed"}' > "${RESULTS_DIR}/result_madpose_rect.json"

            if ! grep -q '"R_est": null' "${RESULTS_DIR}/result_madpose_rect.json"; then
                echo "[DONE] $PAIR_NAME"
            else
                echo "[FAIL] $PAIR_NAME - madpose_rect produced null result"
            fi
        else
            echo "[SKIP] $PAIR_NAME missing both rect intermediates and fisheye depth/mkpts"
        fi
    fi
}

while IFS= read -r PAIR_DIR; do
    [ -z "$PAIR_DIR" ] && continue
    run_pair_madpose "$PAIR_DIR" || echo "[ERROR] $PAIR_DIR failed, continuing"
done < "$PAIRS_LIST_FILE"

echo "[BATCH DONE] $(wc -l < "$PAIRS_LIST_FILE") pairs processed"
