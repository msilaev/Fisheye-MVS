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

# Initialize Conda for bash (robust for non-interactive shells)
CONDA_SETUP="/home/<remote-user>/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SETUP" ]; then
    source "$CONDA_SETUP"
else
  eval "$(conda shell.bash hook)"
fi
conda activate mvf-unik3d

sample_t_R_test_db_path="${REMOTE_DATA_DIR}/sample_t_R_pair_dataset.json"

num_samples=$(jq length "$sample_t_R_test_db_path")
echo "Found $num_samples entries in sample list"

TIME_LOG="${REMOTE_LOG_DIR}/processing_time.log"
echo "ind,seconds" > "$TIME_LOG"

for ((ind=0; ind < num_samples; ind++)); do
  echo "Processing image pair ${ind}"

  start_time=$(date +%s)

  python sample_t_R_test_fetch_pairs.py \
    --sample_ind "$ind" \
    --sequence_name "$SEQUENCE_NAME" \
    --pose_data_dir "${REMOTE_DATA_DIR}/poses" \
    --image_data_dir "${REMOTE_DATA_DIR}/images" \
    --local_exp_dir "$REMOTE_DIR_EXP" \
    --sample_t_R_test_db_path "$sample_t_R_test_db_path"

  ./run_pipeline_combined_metrics.sh \
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
    > "$REMOTE_LOG_DIR/run_pipeline_combined_metrics.log" 2>&1

  end_time=$(date +%s)
  elapsed=$((end_time - start_time))

  echo "${ind},${elapsed}" >> "$TIME_LOG"
  echo "Finished ${ind} in ${elapsed}s"
done
