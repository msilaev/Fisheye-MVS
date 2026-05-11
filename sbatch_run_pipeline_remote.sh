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

JOB1=$(sbatch --parsable \
  "$REMOTE_SCRIPT_DIR_unik3d/sbatch_remote_pipeline_unik3d.sh" \
  "$REMOTE_LOG_DIR" \
  "$REMOTE_SCRIPT_DIR_unik3d" \
  "$REMOTE_SCRIPT_DIR_superglue" \
  "$REMOTE_SCRIPT_DIR_procrustes" \
  "$REMOTE_IMAGE_DIR" \
  "$REMOTE_RESULTS_DIR" \
  "$REMOTE_UNIK3D_DIR")
echo "Submitted unik3d job: $JOB1"

JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 \
  "$REMOTE_SCRIPT_DIR_superglue/sbatch_remote_pipeline_superglue.sh" \
  "$REMOTE_LOG_DIR" \
  "$REMOTE_SCRIPT_DIR_unik3d" \
  "$REMOTE_SCRIPT_DIR_superglue" \
  "$REMOTE_SCRIPT_DIR_procrustes" \
  "$REMOTE_IMAGE_DIR" \
  "$REMOTE_RESULTS_DIR" \
  "$REMOTE_SUPER_GLUE_DIR" \
  "$SIZE_X" \
  "$SIZE_Y")
echo "Submitted superglue job: $JOB2 (depends on $JOB1)"

JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 \
  "$REMOTE_SCRIPT_DIR_procrustes/sbatch_remote_pipeline_procrustes.sh" \
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
  "$SIZE_Y")
echo "Submitted procrustes job: $JOB3 (depends on $JOB2)"
