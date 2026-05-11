#!/bin/bash
# Fetch cloud_consistency.json files from remote and aggregate into a summary CSV.
#
# Usage:
#   bash fetch_cloud_consistency_local.sh
#   DATASET=kitti bash fetch_cloud_consistency_local.sh
#   EXPERIMENT_NAME=ADT_seq136 bash fetch_cloud_consistency_local.sh
set -e
set -x

source .env
source path_config.sh

_EXPERIMENT_NAME_OVERRIDE="${EXPERIMENT_NAME:-}"

DATASET="${DATASET:-adt}"
if [ "$DATASET" = "kitti" ]; then
    source experiment_config_kitti.sh
else
    source experiment_config_adt.sh
fi

[ -n "$_EXPERIMENT_NAME_OVERRIDE" ] && EXPERIMENT_NAME="$_EXPERIMENT_NAME_OVERRIDE"

REMOTE_EVAL_DIR="${REMOTE_EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/eval_full"
REMOTE_RESULTS_DIR="${REMOTE_EVAL_DIR}/results"

LOCAL_EVAL_DIR="${LOCAL_ROOT}/experiments/${EXPERIMENT_NAME}/eval_full"
LOCAL_RESULTS_DIR="${LOCAL_EVAL_DIR}/results"
LOCAL_PAIRS_DIR="${LOCAL_EVAL_DIR}/test_pairs"

mkdir -p "$LOCAL_RESULTS_DIR"

# ── Check for incomplete results ──────────────────────────────────────────────
echo "[LOCAL] Checking for missing cloud_consistency.json on remote..."
ssh "$REMOTE_USER@$REMOTE_HOST" "
for d in '${REMOTE_RESULTS_DIR}'/pair_rot_*; do
    PAIR=\$(basename \$d)
    [ -f \"\$d/result_procrustes.json\" ] || continue
    [ -f \"\$d/cloud_consistency.json\" ] || echo \"  MISSING \$PAIR\"
done
echo 'Check done.'
"

# ── Fetch cloud_consistency.json files via single tar ─────────────────────────
echo "[LOCAL] Fetching cloud_consistency.json files..."
RESULTS_STRIP=$(ssh "$REMOTE_USER@$REMOTE_HOST" \
    "echo '${REMOTE_RESULTS_DIR}' | tr '/' '\n' | grep -c .")
ssh "$REMOTE_USER@$REMOTE_HOST" \
    "find '${REMOTE_RESULTS_DIR}' -name 'cloud_consistency.json' | tar czf - -T -" \
    | tar xzf - -C "$LOCAL_RESULTS_DIR" --strip-components="$RESULTS_STRIP"

# ── Aggregate into summary CSV ────────────────────────────────────────────────
echo "[LOCAL] Aggregating metrics..."
LOCAL_SUMMARY_CSV="${LOCAL_EVAL_DIR}/cloud_consistency_summary.csv"

python "${LOCAL_ROOT}/Fisheye-MVS/src_metrics/aggregate_cloud_consistency.py" \
    --pairs-dir   "$LOCAL_PAIRS_DIR" \
    --results-dir "$LOCAL_RESULTS_DIR" \
    --output-csv  "$LOCAL_SUMMARY_CSV" \
    --rotation-bins    ${ROTATION_BINS} \
    --translation-bins ${TRANSLATION_BINS}

echo ""
echo "[LOCAL] Done. Summary at: $LOCAL_SUMMARY_CSV"
