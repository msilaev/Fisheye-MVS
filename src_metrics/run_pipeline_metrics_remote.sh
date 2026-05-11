#!/bin/bash
# Remote orchestrator: build pair DB → sample test set → run pipeline for each pair.
set -e
set -x

REMOTE_DATA_DIR="$1"
REMOTE_OUT_DIR="$2"
REMOTE_LOG_DIR="$3"
REMOTE_SCRIPT_DIR_metrics="$4"
REMOTE_SCRIPT_DIR_unik3d="$5"
REMOTE_SCRIPT_DIR_superglue="$6"
REMOTE_IMAGE_DIR="$7"
REMOTE_RESULTS_DIR="$8"
SEQUENCE_NAME="$9"
IMG_EXT="${10}"                  # jpg (ADT) or png (KITTI)
FISHEYE_MASK_PATH="${11}"
DISTANCE_THRESHOLD="${12}"
SIZE_X="${13}"
SIZE_Y="${14}"
ROTATE_GT="${15}"                # "true" for ADT, "" for KITTI
REMOTE_UNIK3D_DIR="${16}"
REMOTE_SUPER_GLUE_DIR="${17}"

PAIR_DB_PATH="${REMOTE_OUT_DIR}/pair_dataset.json"
SAMPLE_DB_PATH="${REMOTE_OUT_DIR}/sample_t_R_pair_dataset.json"

module load mamba
eval "$(mamba shell hook --shell bash)"
source activate 3d_pose_env

cd "$REMOTE_SCRIPT_DIR_metrics"

# ── Step 1: Build pair database ───────────────────────────────────────────────
python image_pairs_db_gen.py \
    --sequence_name    "$SEQUENCE_NAME" \
    --pose_data_dir    "${REMOTE_DATA_DIR}/poses" \
    --image_data_dir   "${REMOTE_DATA_DIR}/images" \
    --pair_database_path "$PAIR_DB_PATH" \
    --img-ext          "$IMG_EXT" \
    > "$REMOTE_LOG_DIR/image_pairs_db_gen.log" 2>&1

# ── Step 2: Sample test set ───────────────────────────────────────────────────
python sample_t_R_test_gen.py \
    --sequence_name    "$SEQUENCE_NAME" \
    --pose_data_dir    "${REMOTE_DATA_DIR}/poses" \
    --image_data_dir   "${REMOTE_DATA_DIR}/images" \
    --pair_database_path  "$PAIR_DB_PATH" \
    --sample_t_R_test_db_path "$SAMPLE_DB_PATH" \
    > "$REMOTE_LOG_DIR/sample_t_R_test_gen.log" 2>&1

# ── Step 3: Run pipeline for each pair ────────────────────────────────────────
NUM_SAMPLES=$(python -c "import json; d=json.load(open('$SAMPLE_DB_PATH')); print(len(d))")
echo "Running pipeline for $NUM_SAMPLES pairs"

TIME_LOG="${REMOTE_LOG_DIR}/processing_time.log"
echo "ind,seconds" > "$TIME_LOG"

ROTATE_ARG=""
[ "$ROTATE_GT" = "true" ] && ROTATE_ARG="--rotate"

for ((ind=0; ind < NUM_SAMPLES; ind++)); do
    echo "=== Pair ${ind}/${NUM_SAMPLES} ==="
    t0=$(date +%s)

    # Fetch the pair (images + pose txt files)
    python sample_t_R_test_fetch_pairs.py \
        --sample-ind "$ind" \
        --sample-t-R-test-db-path "$SAMPLE_DB_PATH" \
        --output-dir "$REMOTE_IMAGE_DIR" \
        $ROTATE_ARG \
        > "$REMOTE_LOG_DIR/fetch_pair_${ind}.log" 2>&1

    # Run full pipeline + GT evaluation
    bash "$REMOTE_SCRIPT_DIR_metrics/run_pipeline_combined_metrics.sh" \
        "$REMOTE_DATA_DIR" \
        "$REMOTE_OUT_DIR" \
        "$REMOTE_LOG_DIR" \
        "$REMOTE_SCRIPT_DIR_metrics" \
        "$REMOTE_SCRIPT_DIR_unik3d" \
        "$REMOTE_SCRIPT_DIR_superglue" \
        "$REMOTE_IMAGE_DIR" \
        "$REMOTE_RESULTS_DIR" \
        "$SEQUENCE_NAME" \
        "$ind" \
        "$SAMPLE_DB_PATH" \
        "$FISHEYE_MASK_PATH" \
        "$DISTANCE_THRESHOLD" \
        "$SIZE_X" \
        "$SIZE_Y" \
        "$ROTATE_GT" \
        "$REMOTE_UNIK3D_DIR" \
        "$REMOTE_SUPER_GLUE_DIR"

    t1=$(date +%s)
    echo "${ind},$((t1 - t0))" >> "$TIME_LOG"
    echo "Pair ${ind} done in $((t1-t0))s"
done

echo "[REMOTE] All pairs processed. Results: $SAMPLE_DB_PATH"
