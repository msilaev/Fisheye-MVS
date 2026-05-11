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
REMOTE_UNIK3D_DIR="$8"
remote_fisheye_mask_path="$9"
remote_transform_result_path="${10}"
DISTANCE_THRESHOLD="${11}"
SIZE_X="${12}"
SIZE_Y="${13}"

CONDA_SETUP="/home/<remote-user>/miniconda3/etc/profile.d/conda.sh"

if [ -f "$CONDA_SETUP" ]; then
    source "$CONDA_SETUP"
else
    eval "$(conda shell.bash hook)"
fi

conda deactivate || true
conda activate mvf-unik3d

echo unik3d
cd $REMOTE_SCRIPT_DIR_unik3d
./remote_pipeline_unik3d.sh \
  "$REMOTE_LOG_DIR" \
  "$REMOTE_SCRIPT_DIR_unik3d" \
  "$REMOTE_SCRIPT_DIR_superglue" \
  "$REMOTE_SCRIPT_DIR_procrustes" \
  "$REMOTE_IMAGE_DIR" \
  "$REMOTE_RESULTS_DIR" \
  "$REMOTE_UNIK3D_DIR" \
  > "$REMOTE_LOG_DIR/remote_pipeline_unik3d.log" 2>&1

# Run superglue script
cd $REMOTE_SCRIPT_DIR_superglue
echo superglue
./remote_pipeline_superglue.sh \
  "$REMOTE_LOG_DIR" \
  "$REMOTE_SCRIPT_DIR_unik3d" \
  "$REMOTE_SCRIPT_DIR_superglue" \
  "$REMOTE_SCRIPT_DIR_procrustes" \
  "$REMOTE_IMAGE_DIR" \
  "$REMOTE_RESULTS_DIR" \
  "$REMOTE_SUPER_GLUE_DIR" \
  "$SIZE_X" \
  "$SIZE_Y" \
  > "$REMOTE_LOG_DIR/remote_pipeline_superglue.log" 2>&1

# Run visualise script
cd $REMOTE_SCRIPT_DIR_procrustes
echo procrustes
./remote_pipeline_procrustes.sh \
  "$REMOTE_LOG_DIR" \
  "$REMOTE_SCRIPT_DIR_unik3d" \
  "$REMOTE_SCRIPT_DIR_superglue" \
  "$REMOTE_SCRIPT_DIR_procrustes" \
  "$REMOTE_IMAGE_DIR" \
  "$REMOTE_RESULTS_DIR" \
  "$remote_fisheye_mask_path" \
  "$remote_transform_result_path" \
  "$DISTANCE_THRESHOLD" \
  "$SIZE_X" \
  "$SIZE_Y" \
  > "$REMOTE_LOG_DIR/remote_pipeline_procrustes.log" 2>&1
