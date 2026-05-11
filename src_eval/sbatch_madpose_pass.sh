#!/bin/bash
# Second-pass: run MADPose on already-computed depth+match files.
# Reads results dirs that already have mkpts*.npy + image*_depth.npy
# but are missing result_madpose.json.
#
# Args:
#   $1  PAIRS_LIST_FILE  text file with one PAIR_DIR path per line
#   $2  RESULTS_ROOT     root dir containing per-pair result subdirs
#   $3  LOG_ROOT         root dir for logs
#   $4  PROJ_DIR         Fisheye-MVS project root on remote

#SBATCH --job-name=madpose_pass
#SBATCH --output=madpose_pass-%j.log
#SBATCH --error=madpose_pass-%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1

set -x

PAIRS_LIST_FILE="$1"
RESULTS_ROOT="$2"
LOG_ROOT="$3"
PROJ_DIR="$4"

module load mamba
eval "$(mamba shell hook --shell bash)"
source activate madpose_env

while IFS= read -r PAIR_DIR; do
    [ -z "$PAIR_DIR" ] && continue
    PAIR_NAME=$(basename "$PAIR_DIR")
    RESULTS_DIR="${RESULTS_ROOT}/${PAIR_NAME}"
    LOG_DIR="${LOG_ROOT}/${PAIR_NAME}"
    mkdir -p "$LOG_DIR"

    if [ -f "${RESULTS_DIR}/result_madpose.json" ]; then
        echo "[SKIP] $PAIR_NAME"
        continue
    fi

    if [ ! -f "${RESULTS_DIR}/mkpts1.npy" ] || [ ! -f "${RESULTS_DIR}/image0_depth.npy" ]; then
        echo "[MISSING inputs] $PAIR_NAME — skipping"
        continue
    fi

    echo "[START] $PAIR_NAME"
    python "${PROJ_DIR}/src_madpose/run_madpose_baseline.py" \
        --image0  "${PAIR_DIR}/image0.jpg" \
        --image1  "${PAIR_DIR}/image1.jpg" \
        --mkpts0  "${RESULTS_DIR}/mkpts1.npy" \
        --mkpts1  "${RESULTS_DIR}/mkpts2.npy" \
        --depth0  "${RESULTS_DIR}/image0_depth.npy" \
        --depth1  "${RESULTS_DIR}/image1_depth.npy" \
        --output  "${RESULTS_DIR}/result_madpose.json" \
        > "${LOG_DIR}/${PAIR_NAME}_madpose.log" 2>&1 && \
        echo "[DONE] $PAIR_NAME" || \
        { echo "[ERROR] $PAIR_NAME"; echo '{"R_est": null, "error": "madpose failed"}' > "${RESULTS_DIR}/result_madpose.json"; }

done < "$PAIRS_LIST_FILE"

echo "[PASS DONE]"
