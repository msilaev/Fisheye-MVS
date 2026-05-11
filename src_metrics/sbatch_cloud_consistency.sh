#!/bin/bash
# Process a batch of evaluated pairs: re-run UniK3D to regenerate point clouds,
# then compute symmetric NN cloud-consistency metrics for every result method.
# Output: <results_root>/<pair>/cloud_consistency.json  (one per pair)
#
# Args:
#   $1  PAIRS_LIST_FILE   one full path to a test_pairs/pair_rot_* dir per line
#   $2  RESULTS_ROOT      eval_full/results  (result_*.json files live here)
#   $3  LOG_ROOT          directory for per-pair logs
#   $4  PROJ_DIR          Fisheye-MVS project root on remote
#   $5  UNIK3D_DIR        UniK3D repo root
#   $6  FISHEYE_MASK      path to fisheye mask PNG
#   $7  DISTANCE_THRESH   distance threshold for cloud filtering (default 1000)
#   $8  OVERLAP_THRESH    NN distance threshold for inlier ratio (default 0.05)
#   $9  SIZE_X            image width (default 1408)
#  $10  SIZE_Y            image height (default 1408)
#  $11  DATASET           adt or kitti (default adt)

#SBATCH --job-name=cloud_cc
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1

set -x

PAIRS_LIST_FILE="$1"
RESULTS_ROOT="$2"
LOG_ROOT="$3"
PROJ_DIR="$4"
UNIK3D_DIR="$5"
FISHEYE_MASK="$6"
DISTANCE_THRESH="${7:-1000}"
OVERLAP_THRESH="${8:-0.05}"
SIZE_X="${9:-1408}"
SIZE_Y="${10:-1408}"
DATASET="${11:-adt}"

export HF_HOME=/scratch/work/<remote-user>/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/work/<remote-user>/.cache/huggingface/hub
export TORCH_HOME=/scratch/work/<remote-user>/.cache/torch
export XDG_CACHE_HOME=/scratch/work/<remote-user>/.cache

module load mamba
eval "$(mamba shell hook --shell bash)"

run_pair() {
    local PAIR_DIR="$1"
    local PAIR_NAME
    PAIR_NAME=$(basename "$PAIR_DIR")
    local RESULTS_DIR="${RESULTS_ROOT}/${PAIR_NAME}"
    local LOG_DIR="${LOG_ROOT}/${PAIR_NAME}"

    mkdir -p "$LOG_DIR"

    # Skip if already done
    if [ -f "${RESULTS_DIR}/cloud_consistency.json" ]; then
        echo "[SKIP] $PAIR_NAME already has cloud_consistency.json"
        return 0
    fi

    # Skip if evaluation results are missing
    if [ ! -f "${RESULTS_DIR}/result_procrustes.json" ]; then
        echo "[SKIP] $PAIR_NAME has no result_procrustes.json, skipping"
        return 0
    fi

    echo "[START] $PAIR_NAME"

    # ── 1. Re-run UniK3D to regenerate point clouds ──────────────────────────
    mamba activate mvf-unik3d
    cp "${PROJ_DIR}/src_unik3d/custom_unik3d_inference.py" "${UNIK3D_DIR}/scripts/"
    cd "${UNIK3D_DIR}/scripts"
    python custom_unik3d_inference.py \
        --config-file "${UNIK3D_DIR}/configs/train/vitb.json" \
        --data-source-dir "$PAIR_DIR" \
        --output-dir "$RESULTS_DIR" \
        --images-output-dir "$RESULTS_DIR" \
        --camera-path "${UNIK3D_DIR}/assets/demo/scannet.json" \
        > "${LOG_DIR}/${PAIR_NAME}_unik3d_cc.log" 2>&1

    if [ ! -f "${RESULTS_DIR}/image0_points.npy" ]; then
        echo "[ERROR] $PAIR_NAME UniK3D did not produce image0_points.npy"
        return 1
    fi

    # ── 2. Stage GT poses into results dir (compute_cloud_consistency.py
    #       expects image0.txt / image1.txt in the same folder as the clouds) ─
    cp "${PAIR_DIR}/image0.txt" "${RESULTS_DIR}/image0.txt"
    cp "${PAIR_DIR}/image1.txt" "${RESULTS_DIR}/image1.txt"

    # ── 3. Compute cloud consistency metrics ─────────────────────────────────
    mamba activate 3d_pose_env
    cd "${PROJ_DIR}/src_metrics"
    python compute_cloud_consistency.py \
        --pair-dir           "$RESULTS_DIR" \
        --dataset            "$DATASET" \
        --mask               "$FISHEYE_MASK" \
        --distance-threshold "$DISTANCE_THRESH" \
        --overlap-threshold  "$OVERLAP_THRESH" \
        --max-points         200000 \
        --output             "${RESULTS_DIR}/cloud_consistency.json" \
        > "${LOG_DIR}/${PAIR_NAME}_cloud_consistency.log" 2>&1

    # ── 4. Cleanup regenerated intermediates ─────────────────────────────────
    rm -f "${RESULTS_DIR}/image0_points.npy" "${RESULTS_DIR}/image0_rays.npy" \
          "${RESULTS_DIR}/image1_points.npy" "${RESULTS_DIR}/image1_rays.npy" \
          "${RESULTS_DIR}/image0_depth.npy"  "${RESULTS_DIR}/image1_depth.npy" \
          "${RESULTS_DIR}/image0_depth.png"  "${RESULTS_DIR}/image1_depth.png" \
          "${RESULTS_DIR}/image0.txt"        "${RESULTS_DIR}/image1.txt"

    echo "[DONE] $PAIR_NAME"
}

while IFS= read -r PAIR_DIR; do
    [ -z "$PAIR_DIR" ] && continue
    run_pair "$PAIR_DIR" || echo "[ERROR] $PAIR_DIR failed, continuing"
done < "$PAIRS_LIST_FILE"

echo "[BATCH DONE] $(wc -l < "$PAIRS_LIST_FILE") pairs processed"
