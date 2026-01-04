#!/bin/bash
set -e
set -x

REMOTE_LOG_DIR="$1"
REMOTE_SCRIPT_DIR_unik3d="$2"
REMOTE_SCRIPT_DIR_superglue="$3"
REMOTE_SCRIPT_DIR_procrustes="$4"
REMOTE_IMAGE_DIR="$5"
REMOTE_RESULTS_DIR="$6"
REMOTE_SUPER_GLUE_DIR="$7"
SIZE_X="$8"
SIZE_Y="$9"

image_pair_file="${REMOTE_IMAGE_DIR}/image_pairs.txt"
image_input_dir="${REMOTE_IMAGE_DIR}"

CONDA_SETUP="/home/mikhail/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SETUP" ]; then
    source "$CONDA_SETUP"
else
    echo "ERROR: Could not find conda.sh at $CONDA_SETUP"
    exit 1
fi

conda deactivate || true
conda activate superglue38

cd "$REMOTE_SUPER_GLUE_DIR"

time_start=$(date +%s)
echo "[REMOTE] Running SuperGlue matching ..."

python match_pairs.py \
    --resize 1600 \
    --superglue outdoor \
    --max_keypoints 2048 \
    --nms_radius 3 \
    --resize_float \
    --input_dir "$image_input_dir" \
    --input_pairs "$image_pair_file" \
    --output_dir "$REMOTE_RESULTS_DIR" \
    --viz \
    > "$REMOTE_LOG_DIR/remote_pipeline_superglue_metrics.log" 2>&1

echo "[REMOTE] SuperGlue inference finished."
echo "SuperGlue exit code = $?"

###############################
# 3. Extract valid match pairs
###############################

npz="$REMOTE_RESULTS_DIR/image0_image1_matches.npz"

mkpts1="$REMOTE_RESULTS_DIR/mkpts1.npy"
mkpts2="$REMOTE_RESULTS_DIR/mkpts2.npy"

cd $REMOTE_SCRIPT_DIR_superglue

python superglue_extract_pairs.py \
    --matches-file-npz "$npz" \
    --size_x "$SIZE_X" \
    --size_y "$SIZE_Y" \
    --mkpts1-file "$mkpts1" \
    --mkpts2-file "$mkpts2" \
    > "$REMOTE_LOG_DIR/valid_pairs.log" 2>&1

echo "superglue_extract_pairs exit code = $?"




