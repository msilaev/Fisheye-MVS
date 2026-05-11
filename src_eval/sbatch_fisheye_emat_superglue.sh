#!/bin/bash
# Re-run SuperGlue to regenerate mkpts, then run Fisheye E-mat baseline.
# Skips pairs that already have a valid result_fisheye_emat.json.
# Deletes regenerated mkpts after use to reclaim space.
#
# Args:
#   $1  PAIRS_LIST_FILE
#   $2  RESULTS_ROOT
#   $3  LOG_ROOT
#   $4  PROJ_DIR
#   $5  SUPERGLUE_DIR
#   $6  MASK_PATH
#   $7  CALIB_TYPE       aria (default) or kitti360
#   $8  KITTI_CALIB      path to image_02.yaml (kitti360 only)
#   $9  SIZE_X           (default 1408)
#   $10 SIZE_Y           (default 1408)

#SBATCH --job-name=femat_sg
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
SUPERGLUE_DIR="$5"
MASK_PATH="$6"
CALIB_TYPE="${7:-aria}"
KITTI_CALIB="${8:-}"
SIZE_X="${9:-1408}"
SIZE_Y="${10:-1408}"

module load mamba
eval "$(mamba shell hook --shell bash)"

run_pair() {
    local PAIR_DIR="$1"
    local PAIR_NAME
    PAIR_NAME=$(basename "$PAIR_DIR")
    local RESULTS_DIR="${RESULTS_ROOT}/${PAIR_NAME}"
    local LOG_DIR="${LOG_ROOT}/${PAIR_NAME}"

    mkdir -p "$LOG_DIR"

    # Skip if already valid
    if [ -f "${RESULTS_DIR}/result_fisheye_emat.json" ] && \
       ! grep -q '"R_est": null' "${RESULTS_DIR}/result_fisheye_emat.json"; then
        echo "[SKIP] $PAIR_NAME already done"
        return
    fi

    local IMAGE0 IMAGE1
    if [ -f "${PAIR_DIR}/image0.jpg" ]; then
        IMAGE0="${PAIR_DIR}/image0.jpg"; IMAGE1="${PAIR_DIR}/image1.jpg"
    else
        IMAGE0="${PAIR_DIR}/image0.png"; IMAGE1="${PAIR_DIR}/image1.png"
    fi

    MKPTS_EXISTED=false
    if [ -f "${RESULTS_DIR}/mkpts1.npy" ] && [ -f "${RESULTS_DIR}/mkpts2.npy" ]; then
        MKPTS_EXISTED=true
    fi

    # Re-run SuperGlue if mkpts missing
    if [ "$MKPTS_EXISTED" = false ]; then
        echo "[SUPERGLUE] $PAIR_NAME"
        mamba activate superglue38
        cd "$SUPERGLUE_DIR"
        python match_pairs.py \
            --resize 1600 \
            --superglue outdoor \
            --max_keypoints 2048 \
            --nms_radius 3 \
            --resize_float \
            --input_dir   "$PAIR_DIR" \
            --input_pairs "${PAIR_DIR}/image_pairs.txt" \
            --output_dir  "$RESULTS_DIR" \
            > "${LOG_DIR}/${PAIR_NAME}_superglue_rerun.log" 2>&1

        mamba activate 3d_pose_env
        cd "${PROJ_DIR}/src_superglue"
        python superglue_extract_pairs.py \
            --matches-file-npz "${RESULTS_DIR}/image0_image1_matches.npz" \
            --size_x "$SIZE_X" \
            --size_y "$SIZE_Y" \
            --mkpts1-file "${RESULTS_DIR}/mkpts1.npy" \
            --mkpts2-file "${RESULTS_DIR}/mkpts2.npy" \
            >> "${LOG_DIR}/${PAIR_NAME}_superglue_rerun.log" 2>&1

        rm -f "${RESULTS_DIR}/image0_image1_matches.npz"
    fi

    if [ ! -f "${RESULTS_DIR}/mkpts1.npy" ] || [ ! -f "${RESULTS_DIR}/mkpts2.npy" ]; then
        echo "[ERROR] $PAIR_NAME — mkpts still missing after SuperGlue"
        return
    fi

    # Run fisheye E-mat
    echo "[FISHEYE_EMAT] $PAIR_NAME"
    mamba activate 3d_pose_env
    python "${PROJ_DIR}/src_baseline_fisheye/run_fisheye_emat_baseline.py" \
        --mkpts0     "${RESULTS_DIR}/mkpts1.npy" \
        --mkpts1     "${RESULTS_DIR}/mkpts2.npy" \
        --output     "${RESULTS_DIR}/result_fisheye_emat.json" \
        --calib-type "$CALIB_TYPE" \
        --mask       "$MASK_PATH" \
        --size-x     "$SIZE_X" \
        --size-y     "$SIZE_Y" \
        ${KITTI_CALIB:+--kitti-calib "$KITTI_CALIB"} \
        > "${LOG_DIR}/${PAIR_NAME}_fisheye_emat.log" 2>&1 || \
        echo '{"R_est": null, "t_est": null, "scale_est": 1.0, "error": "failed"}' \
            > "${RESULTS_DIR}/result_fisheye_emat.json"

    # Clean up regenerated mkpts (only if we created them; preserve pre-existing ones)
    if [ "$MKPTS_EXISTED" = false ] && [[ "$PAIR_NAME" != pair_custom_* ]]; then
        rm -f "${RESULTS_DIR}/mkpts1.npy" "${RESULTS_DIR}/mkpts2.npy"
    fi

    if grep -q '"R_est": null' "${RESULTS_DIR}/result_fisheye_emat.json"; then
        echo "[FAIL] $PAIR_NAME"
    else
        echo "[DONE] $PAIR_NAME"
    fi
}

while IFS= read -r PAIR_DIR; do
    [ -z "$PAIR_DIR" ] && continue
    run_pair "$PAIR_DIR" || echo "[ERROR] $PAIR_DIR failed, continuing"
done < "$PAIRS_LIST_FILE"

echo "[BATCH DONE] $(wc -l < "$PAIRS_LIST_FILE") pairs processed"
