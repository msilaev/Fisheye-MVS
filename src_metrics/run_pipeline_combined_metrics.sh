#!/bin/bash
# Run one full pipeline iteration (unik3d → superglue → pose_est+GT) for a single pair.
set -e
set -x

REMOTE_DATA_DIR="$1"
REMOTE_OUT_DIR="$2"
REMOTE_LOG_DIR="$3"
REMOTE_SCRIPT_DIR_metrics="$4"
REMOTE_SCRIPT_DIR_unik3d="$5"
REMOTE_SCRIPT_DIR_superglue="$6"
REMOTE_IMAGE_DIR="$7"           # scratch dir for this pair's images
REMOTE_RESULTS_DIR="$8"
SEQUENCE_NAME="$9"
IND="${10}"
SAMPLE_DB_PATH="${11}"
FISHEYE_MASK_PATH="${12}"
DISTANCE_THRESHOLD="${13}"
SIZE_X="${14}"
SIZE_Y="${15}"
ROTATE_GT="${16}"                # "true" for ADT, "" for KITTI
REMOTE_UNIK3D_DIR="${17}"
REMOTE_SUPER_GLUE_DIR="${18}"

# ── UniK3D ────────────────────────────────────────────────────────────────────
module load mamba
eval "$(mamba shell hook --shell bash)"
source activate mvf-unik3d

cp "${REMOTE_SCRIPT_DIR_unik3d}/custom_unik3d_inference.py" "$REMOTE_UNIK3D_DIR/scripts/"
cd "$REMOTE_UNIK3D_DIR/scripts"

python custom_unik3d_inference.py \
    --config-file "${REMOTE_UNIK3D_DIR}/configs/train/vitb.json" \
    --data-source-dir "$REMOTE_IMAGE_DIR" \
    --output-dir "$REMOTE_RESULTS_DIR" \
    --images-output-dir "$REMOTE_RESULTS_DIR" \
    --camera-path "${REMOTE_UNIK3D_DIR}/assets/demo/scannet.json" \
    > "$REMOTE_LOG_DIR/unik3d_metrics_${IND}.log" 2>&1

# ── SuperGlue ─────────────────────────────────────────────────────────────────
source deactivate
source activate superglue38

cd "$REMOTE_SUPER_GLUE_DIR"

python match_pairs.py \
    --resize 1600 \
    --superglue outdoor \
    --max_keypoints 2048 \
    --nms_radius 3 \
    --resize_float \
    --input_dir "$REMOTE_IMAGE_DIR" \
    --input_pairs "${REMOTE_IMAGE_DIR}/image_pairs.txt" \
    --output_dir "$REMOTE_RESULTS_DIR" \
    --viz \
    > "$REMOTE_LOG_DIR/superglue_metrics_${IND}.log" 2>&1

cd "$REMOTE_SCRIPT_DIR_superglue"
python superglue_extract_pairs.py \
    --matches-file-npz "${REMOTE_RESULTS_DIR}/image0_image1_matches.npz" \
    --size_x "$SIZE_X" \
    --size_y "$SIZE_Y" \
    --mkpts1-file "${REMOTE_RESULTS_DIR}/mkpts1.npy" \
    --mkpts2-file "${REMOTE_RESULTS_DIR}/mkpts2.npy" \
    >> "$REMOTE_LOG_DIR/superglue_metrics_${IND}.log" 2>&1

# ── Pose estimation + GT evaluation ───────────────────────────────────────────
source deactivate
source activate 3d_pose_env

ROTATE_FLAG=""
if [ "$ROTATE_GT" = "true" ]; then
    ROTATE_FLAG="--rotate-gt"
fi

cd "$REMOTE_SCRIPT_DIR_metrics"
python pose_estimation_with_gt.py \
    --point1 "${REMOTE_RESULTS_DIR}/image0_points.npy" \
    --point2 "${REMOTE_RESULTS_DIR}/image1_points.npy" \
    --mkpts1 "${REMOTE_RESULTS_DIR}/mkpts1.npy" \
    --mkpts2 "${REMOTE_RESULTS_DIR}/mkpts2.npy" \
    --cam2w-1 "${REMOTE_IMAGE_DIR}/image0.txt" \
    --cam2w-2 "${REMOTE_IMAGE_DIR}/image1.txt" \
    --fisheye-mask-path "$FISHEYE_MASK_PATH" \
    --distance-threshold "$DISTANCE_THRESHOLD" \
    --size-x "$SIZE_X" \
    --size-y "$SIZE_Y" \
    --sample-t-R-test-db-path "$SAMPLE_DB_PATH" \
    --sample-ind "$IND" \
    $ROTATE_FLAG \
    > "$REMOTE_LOG_DIR/pose_est_metrics_${IND}.log" 2>&1

echo "[REMOTE] Pair ${IND} done."
