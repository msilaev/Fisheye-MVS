#!/bin/bash
# Process a batch of pair directories sequentially in one SLURM job.
# Each pair runs all evaluation methods: UniK3D -> SuperGlue -> Procrustes + Pinhole + DUSt3R.
#
# Args:
#   $1  PAIRS_LIST_FILE  text file with one PAIR_DIR path per line
#   $2  RESULTS_ROOT     root dir; results go into $RESULTS_ROOT/<pair_name>/
#   $3  LOG_ROOT         root dir; logs go into $LOG_ROOT/<pair_name>/
#   $4  PROJ_DIR         Fisheye-MVS project root on remote
#   $5  UNIK3D_DIR       UniK3D repo root
#   $6  SUPERGLUE_DIR    SuperGluePretrainedNetwork repo root
#   $7  DUST3R_DIR       DUSt3R repo root
#   $8  FISHEYE_MASK     path to fisheye mask png
#   $9  DISTANCE_THRESH  distance threshold for procrustes (default 1000)
#   $10 SIZE_X           image width (default 1408)
#   $11 SIZE_Y           image height (default 1408)
#   $12 CALIB_TYPE       rectification backend: aria (default) or kitti360
#   $13 KITTI_CALIB      path to KITTI-360 image_02.yaml (required if kitti360)

#SBATCH --job-name=eval_batch
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:1

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
UNIK3D_DIR="$5"
SUPERGLUE_DIR="$6"
DUST3R_DIR="$7"
FISHEYE_MASK="$8"
DISTANCE_THRESH="${9:-1000}"
SIZE_X="${10:-1408}"
SIZE_Y="${11:-1408}"
CALIB_TYPE="${12:-aria}"
KITTI_CALIB="${13:-}"

module load mamba
eval "$(mamba shell hook --shell bash)"

run_pair() {
    local PAIR_DIR="$1"
    local PAIR_NAME
    PAIR_NAME=$(basename "$PAIR_DIR")
    local RESULTS_DIR="${RESULTS_ROOT}/${PAIR_NAME}"
    local LOG_DIR="${LOG_ROOT}/${PAIR_NAME}"
    local IMAGE0
    local IMAGE1
    # Support both .jpg (ADT) and .png (KITTI-360)
    if [ -f "${PAIR_DIR}/image0.jpg" ]; then
        IMAGE0="${PAIR_DIR}/image0.jpg"; IMAGE1="${PAIR_DIR}/image1.jpg"
    else
        IMAGE0="${PAIR_DIR}/image0.png"; IMAGE1="${PAIR_DIR}/image1.png"
    fi

    mkdir -p "$RESULTS_DIR" "$LOG_DIR"

    # Skip if Procrustes+RANSAC already done (the only missing result)
    if [ -f "${RESULTS_DIR}/result_procrustes_ransac.json" ]; then
        echo "[SKIP] $PAIR_NAME already complete"
        return 0
    fi

    echo "[START] $PAIR_NAME"

    # ── 1. UniK3D depth inference ─────────────────────────────────────────────
    mamba activate mvf-unik3d
    CONFIG="${UNIK3D_DIR}/configs/train/vitb.json"
    cp "${PROJ_DIR}/src_unik3d/custom_unik3d_inference.py" "${UNIK3D_DIR}/scripts/"
    cd "${UNIK3D_DIR}/scripts"
    python custom_unik3d_inference.py \
        --config-file "$CONFIG" \
        --data-source-dir "$PAIR_DIR" \
        --output-dir "$RESULTS_DIR" \
        --images-output-dir "$RESULTS_DIR" \
        --camera-path "${UNIK3D_DIR}/assets/demo/scannet.json" \
        > "${LOG_DIR}/${PAIR_NAME}_unik3d.log" 2>&1

    # ── 2. SuperGlue matching ─────────────────────────────────────────────────
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
        > "${LOG_DIR}/${PAIR_NAME}_superglue.log" 2>&1

    cd "${PROJ_DIR}/src_superglue"
    python superglue_extract_pairs.py \
        --matches-file-npz "${RESULTS_DIR}/image0_image1_matches.npz" \
        --size_x "$SIZE_X" \
        --size_y "$SIZE_Y" \
        --mkpts1-file "${RESULTS_DIR}/mkpts1.npy" \
        --mkpts2-file "${RESULTS_DIR}/mkpts2.npy" \
        >> "${LOG_DIR}/${PAIR_NAME}_superglue.log" 2>&1

    # ── 3. Procrustes pose estimation (ours) ─────────────────────────────────
    mamba activate 3d_pose_env

    # GT relative pose (needs numpy — generate here inside 3d_pose_env)
    # --rotate converts raw cam-to-world GT into the rotated-image frame used by
    # Procrustes/MADPose (ADT images are rotated 90 deg CW before UniK3D).
    if [ ! -f "${RESULTS_DIR}/result_gt.json" ]; then
        GT_ROTATE_FLAG=""
        [ "$CALIB_TYPE" = "aria" ] && GT_ROTATE_FLAG="--rotate"
        python "${PROJ_DIR}/src_eval/generate_result_gt.py" \
            --cam2w-0 "${PAIR_DIR}/image0.txt" \
            --cam2w-1 "${PAIR_DIR}/image1.txt" \
            --output "${RESULTS_DIR}/result_gt.json" \
            $GT_ROTATE_FLAG \
            > "${LOG_DIR}/${PAIR_NAME}_gt.log" 2>&1 || \
            echo "[WARN] $PAIR_NAME failed to generate result_gt.json"
    fi

    cd "${PROJ_DIR}/src_procrustes"
    python pose_estimation_procrustes.py \
        --point1  "${RESULTS_DIR}/image0_points.npy" \
        --point2  "${RESULTS_DIR}/image1_points.npy" \
        --mkpts1  "${RESULTS_DIR}/mkpts1.npy" \
        --mkpts2  "${RESULTS_DIR}/mkpts2.npy" \
        --remote_fisheye_mask_path "$FISHEYE_MASK" \
        --remote_transform_result_path "${RESULTS_DIR}/result_procrustes.json" \
        --distance_threshold "$DISTANCE_THRESH" \
        --size_x "$SIZE_X" --size_y "$SIZE_Y" \
        > "${LOG_DIR}/${PAIR_NAME}_procrustes.log" 2>&1

    # ── 3b. Procrustes + RANSAC (ablation) ───────────────────────────────────
    python pose_estimation_procrustes.py \
        --point1  "${RESULTS_DIR}/image0_points.npy" \
        --point2  "${RESULTS_DIR}/image1_points.npy" \
        --mkpts1  "${RESULTS_DIR}/mkpts1.npy" \
        --mkpts2  "${RESULTS_DIR}/mkpts2.npy" \
        --remote_fisheye_mask_path "$FISHEYE_MASK" \
        --remote_transform_result_path "${RESULTS_DIR}/result_procrustes_ransac.json" \
        --distance_threshold "$DISTANCE_THRESH" \
        --size_x "$SIZE_X" --size_y "$SIZE_Y" \
        --use-ransac \
        >> "${LOG_DIR}/${PAIR_NAME}_procrustes.log" 2>&1

    # Remove large intermediates (keep mkpts + depth for MADPose second pass).
    # For custom pairs (pair_custom_*) keep point clouds for local visualisation.
    rm -f "${RESULTS_DIR}/image0_rays.npy" \
          "${RESULTS_DIR}/image1_rays.npy" \
          "${RESULTS_DIR}/image0_image1_matches.npz" \
          "${RESULTS_DIR}/image0_depth.png"  "${RESULTS_DIR}/image1_depth.png"
    if [[ "$PAIR_NAME" != pair_custom_* ]]; then
        rm -f "${RESULTS_DIR}/image0_points.npy" "${RESULTS_DIR}/image1_points.npy"
    fi

    # ── 4. Pinhole baseline ───────────────────────────────────────────────────
    if [ -f "${RESULTS_DIR}/result_pinhole.json" ]; then
        echo "[SKIP-PINHOLE] $PAIR_NAME"
    else
    python "${PROJ_DIR}/src_baseline_pinhole/run_pinhole_baseline.py" \
        --image0 "$IMAGE0" \
        --image1 "$IMAGE1" \
        --output "${RESULTS_DIR}/result_pinhole.json" \
        --calib-type "$CALIB_TYPE" \
        ${KITTI_CALIB:+--kitti-calib "$KITTI_CALIB"} \
        > "${LOG_DIR}/${PAIR_NAME}_pinhole.log" 2>&1
    fi

    # ── 5. DUSt3R raw fisheye ─────────────────────────────────────────────────
    if [ -f "${RESULTS_DIR}/result_dust3r.json" ] && [ -f "${RESULTS_DIR}/result_dust3r_rect.json" ]; then
        echo "[SKIP-DUST3R] $PAIR_NAME"
    else
    mamba activate dust3r_env
    export PYTHONPATH="${DUST3R_DIR}:${DUST3R_DIR}/croco:${PYTHONPATH:-}"
    if [ ! -f "${RESULTS_DIR}/result_dust3r.json" ]; then
    python "${PROJ_DIR}/src_baseline_dust3r/run_dust3r_baseline.py" \
        --image0 "$IMAGE0" \
        --image1 "$IMAGE1" \
        --output "${RESULTS_DIR}/result_dust3r.json" \
        --device cuda \
        > "${LOG_DIR}/${PAIR_NAME}_dust3r.log" 2>&1
    fi

    # ── 6. DUSt3R + rectification ─────────────────────────────────────────────
    if [ ! -f "${RESULTS_DIR}/result_dust3r_rect.json" ]; then
    python "${PROJ_DIR}/src_baseline_dust3r/run_dust3r_baseline.py" \
        --image0 "$IMAGE0" \
        --image1 "$IMAGE1" \
        --output "${RESULTS_DIR}/result_dust3r_rect.json" \
        --device cuda \
        --rectify \
        --calib-type "$CALIB_TYPE" \
        ${KITTI_CALIB:+--kitti-calib "$KITTI_CALIB"} \
        > "${LOG_DIR}/${PAIR_NAME}_dust3r_rect.log" 2>&1
    fi
    fi  # end dust3r block

    # ── 7. MADPose — skipped here; run via sbatch_madpose_only.sh on batch-skl ──
    # MADPose crashes with SIGILL on GPU nodes (CPU instruction set mismatch).
    # Keep depth/mkpts so sbatch_madpose_only.sh can pick them up on batch-skl.
    echo "[SKIP-MADPOSE-GPU] $PAIR_NAME — submit sbatch_madpose_only.sh on batch-skl"

    # ── Final cleanup: keep depth/mkpts for chained MADPose job ─────────────────
    # sbatch_madpose_only.sh is submitted with --dependency=afterok on this job's
    # ID and handles its own cleanup after MADPose succeeds.
    # Only clean up here if MADPose already ran and succeeded (e.g. first-pass).
    if [ -f "${RESULTS_DIR}/result_madpose.json" ] && \
       ! grep -q '"R_est": null' "${RESULTS_DIR}/result_madpose.json"; then
        rm -f "${RESULTS_DIR}/image0_depth.npy" "${RESULTS_DIR}/image1_depth.npy" \
              "${RESULTS_DIR}/mkpts1.npy"        "${RESULTS_DIR}/mkpts2.npy"
    fi

    echo "[DONE] $PAIR_NAME"
}

# Process all pairs in this batch sequentially
while IFS= read -r PAIR_DIR; do
    [ -z "$PAIR_DIR" ] && continue
    run_pair "$PAIR_DIR" || echo "[ERROR] $PAIR_DIR failed, continuing"
done < "$PAIRS_LIST_FILE"

echo "[BATCH DONE] $(wc -l < "$PAIRS_LIST_FILE") pairs processed"
