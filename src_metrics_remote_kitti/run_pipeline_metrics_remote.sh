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
REMOTE_RESULTS_DIR="${9}"
SEQUENCE_NAME="${10}"


REMOTE_USER="mikhail"
REMOTE_HOST="<remote-workstation>"
CONDA_SETUP="/home/<remote-user>/miniconda3/etc/profile.d/conda.sh"

if [ -f "$CONDA_SETUP" ]; then
    source "$CONDA_SETUP"
else
    eval "$(conda shell.bash hook)"
fi

conda deactivate || true
conda activate mvf-unik3d

cd "$REMOTE_SCRIPT_DIR_metrics"

#./run_image_pairs_db_gen_remote.sh \
#  "$REMOTE_DATA_DIR" \
#  "$REMOTE_OUT_DIR" \
#  "$REMOTE_LOG_DIR" \
#  "$REMOTE_SCRIPT_DIR" \
#  "$REMOTE_DIR_EXP" \
#  "$SEQUENCE_NAME" \
#  > "$REMOTE_LOG_DIR/image_pairs_db_gen.log" 2>&1

#./run_sample_t_R_test_gen_remote.sh \
#  "$REMOTE_DATA_DIR" \
#  "$REMOTE_OUT_DIR" \
#  "$REMOTE_LOG_DIR" \
#  "$REMOTE_SCRIPT_DIR" \
#  "$REMOTE_DIR_EXP" \
#  "$SEQUENCE_NAME" \
#  > "$REMOTE_LOG_DIR/sample_t_R_test.log" 2>&1

./run_sample_t_R_test_fetch_pairs_remote.sh \
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
  > "$REMOTE_LOG_DIR/run_sample_t_R_test_fetch_pairs_remote.log" 2>&1