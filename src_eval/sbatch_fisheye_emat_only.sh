#!/bin/bash
# Run Fisheye Essential Matrix baseline on a batch of pairs (CPU partition).
# Requires mkpts1.npy / mkpts2.npy already produced by sbatch_eval_batch.sh.
#
# Args:
#   $1  PAIRS_LIST_FILE  text file with one PAIR_DIR path per line
#   $2  RESULTS_ROOT     root dir; results go into $RESULTS_ROOT/<pair_name>/
#   $3  LOG_ROOT         root dir; logs go into $LOG_ROOT/<pair_name>/
#   $4  PROJ_DIR         Fisheye-MVS project root on remote
#   $5  MASK_PATH        fisheye validity mask .png
#   $6  CALIB_TYPE       aria (default) or kitti360
#   $7  KITTI_CALIB      path to KITTI-360 image_02.yaml (kitti360 only)
#   $8  SIZE_X           image width  (default 1408)
#   $9  SIZE_Y           image height (default 1408)

#SBATCH --job-name=fisheye_emat
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=batch-skl

set -x

PAIRS_LIST_FILE="$1"
RESULTS_ROOT="$2"
LOG_ROOT="$3"
PROJ_DIR="$4"
MASK_PATH="$5"
CALIB_TYPE="${6:-aria}"
KITTI_CALIB="${7:-}"
SIZE_X="${8:-1408}"
SIZE_Y="${9:-1408}"

module load mamba
eval "$(mamba shell hook --shell bash)"
source activate 3d_pose_env

run_pair_fisheye_emat() {
    local PAIR_DIR="$1"
    local PAIR_NAME
    PAIR_NAME=$(basename "$PAIR_DIR")
    local RESULTS_DIR="${RESULTS_ROOT}/${PAIR_NAME}"
    local LOG_DIR="${LOG_ROOT}/${PAIR_NAME}"

    mkdir -p "$LOG_DIR"

    # Skip if already computed
    if [ -f "${RESULTS_DIR}/result_fisheye_emat.json" ] && \
       ! grep -q '"R_est": null' "${RESULTS_DIR}/result_fisheye_emat.json"; then
        echo "[SKIP] $PAIR_NAME fisheye_emat already valid"
        return
    fi

    # Requires mkpts from the eval step
    if [ ! -f "${RESULTS_DIR}/mkpts1.npy" ] || \
       [ ! -f "${RESULTS_DIR}/mkpts2.npy" ]; then
        echo "[SKIP] $PAIR_NAME missing mkpts"
        return
    fi

    echo "[START fisheye_emat] $PAIR_NAME"
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
        echo '{"R_est": null, "t_est": null, "scale_est": 1.0, "error": "fisheye_emat failed"}' \
            > "${RESULTS_DIR}/result_fisheye_emat.json"

    if grep -q '"R_est": null' "${RESULTS_DIR}/result_fisheye_emat.json"; then
        echo "[FAIL] $PAIR_NAME"
    else
        echo "[DONE] $PAIR_NAME"
    fi
}

while IFS= read -r PAIR_DIR; do
    [ -z "$PAIR_DIR" ] && continue
    run_pair_fisheye_emat "$PAIR_DIR" || echo "[ERROR] $PAIR_DIR failed, continuing"
done < "$PAIRS_LIST_FILE"

echo "[BATCH DONE] $(wc -l < "$PAIRS_LIST_FILE") pairs processed"
