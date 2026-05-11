#!/bin/bash
set -e
set -x

REMOTE_DATA_DIR="$1"
REMOTE_OUT_DIR="$2"
REMOTE_LOG_DIR="$3"

REMOTE_SCRIPT_DIR_metrics="$4"
REMOTE_SCRIPT_DIR_unik3d="$5"
REMOTE_SCRIPT_DIR_superglue="$6"
REMOTE_SCRIPT_DIR_visualise="$7"

REMOTE_DIR_EXP="$8"
REMOTE_RESULTS_DIR="$9"
SEQUENCE_NAME="${10}"
ind="${11}"
sample_t_R_test_db_path="${12}"

echo unik3d
cd $REMOTE_SCRIPT_DIR_unik3d
./remote_pipeline_unik3d_metrics.sh \
  "$REMOTE_DATA_DIR" \
  "$REMOTE_OUT_DIR" \
  "$REMOTE_LOG_DIR" \
  "$REMOTE_SCRIPT_DIR_metrics" \
  "$REMOTE_SCRIPT_DIR_unik3d" \
  "$REMOTE_SCRIPT_DIR_superglue" \
  "$REMOTE_SCRIPT_DIR_visualise" \
  "$REMOTE_DIR_EXP" \
  "$REMOTE_RESULTS_DIR" \
  "$SEQUENCE_NAME" \
  > "$REMOTE_LOG_DIR/remote_pipeline_unik3d_metrics.log" 2>&1

# Run superglue script
cd $REMOTE_SCRIPT_DIR_superglue
echo superglue
./remote_pipeline_superglue_metrics.sh \
  "$REMOTE_DATA_DIR" \
  "$REMOTE_OUT_DIR" \
  "$REMOTE_LOG_DIR" \
  "$REMOTE_SCRIPT_DIR_metrics" \
  "$REMOTE_SCRIPT_DIR_unik3d" \
  "$REMOTE_SCRIPT_DIR_superglue" \
  "$REMOTE_SCRIPT_DIR_visualise" \
  "$REMOTE_DIR_EXP" \
  "$REMOTE_RESULTS_DIR" \
  "$SEQUENCE_NAME" \
  > "$REMOTE_LOG_DIR/remote_pipeline_superglue_metrics.log" 2>&1

# Run visualise script
cd $REMOTE_SCRIPT_DIR_visualise
echo visualize
./remote_pipeline_kitti_visualise_metrics.sh \
  "$REMOTE_DATA_DIR" \
  "$REMOTE_OUT_DIR" \
  "$REMOTE_LOG_DIR" \
  "$REMOTE_SCRIPT_DIR_metrics" \
  "$REMOTE_SCRIPT_DIR_unik3d" \
  "$REMOTE_SCRIPT_DIR_superglue" \
  "$REMOTE_SCRIPT_DIR_visualise" \
  "$REMOTE_DIR_EXP" \
  "$REMOTE_RESULTS_DIR" \
  "$SEQUENCE_NAME" \
  "$ind" \
  "$sample_t_R_test_db_path" \
  > "$REMOTE_LOG_DIR/remote_pipeline_visualise_metrics.log" 2>&1
