#!/bin/bash
# Fetch evaluation results from remote and compute error table.
# Uses tar+scp for fast bulk transfer instead of per-file scp.
#
# Usage:
#   bash fetch_eval_results_local.sh                                        # ADT seq133 (default)
#   DATASET=kitti bash fetch_eval_results_local.sh                          # KITTI-360
#   EXPERIMENT_NAME=ADT_seq136 bash fetch_eval_results_local.sh             # other ADT sequence
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
REMOTE_PAIRS_DIR="${REMOTE_EVAL_DIR}/test_pairs"
REMOTE_RESULTS_DIR="${REMOTE_EVAL_DIR}/results"

LOCAL_EVAL_DIR="${LOCAL_ROOT}/experiments/${EXPERIMENT_NAME}/eval_full"
LOCAL_PAIRS_DIR="${LOCAL_EVAL_DIR}/test_pairs"
LOCAL_RESULTS_DIR="${LOCAL_EVAL_DIR}/results"

mkdir -p "$LOCAL_PAIRS_DIR" "$LOCAL_RESULTS_DIR"

# ── Check any failed jobs ─────────────────────────────────────────────────────
echo "[LOCAL] Checking for incomplete results on remote..."
ssh "$REMOTE_USER@$REMOTE_HOST" "
for d in '${REMOTE_RESULTS_DIR}'/pair_rot_*; do
    PAIR=\$(basename \$d)
    missing=''
    for f in result_procrustes.json result_pinhole.json result_dust3r.json result_dust3r_rect.json result_madpose.json; do
        [ -f \"\$d/\$f\" ] || missing=\"\$missing \$f\"
    done
    [ -n \"\$missing\" ] && echo \"  MISSING \$PAIR:\$missing\"
done
echo 'Check done.'
"

# ── Fetch pair metadata (info.json + pose txts) via single tar ────────────────
echo "[LOCAL] Fetching pair metadata..."
PAIRS_STRIP=$(ssh "$REMOTE_USER@$REMOTE_HOST" \
    "echo '${REMOTE_PAIRS_DIR}' | tr '/' '\n' | grep -c .")
ssh "$REMOTE_USER@$REMOTE_HOST" \
    "find '${REMOTE_PAIRS_DIR}' -name 'info.json' -o -name '*.txt' | tar czf - -T -" \
    | tar xzf - -C "$LOCAL_PAIRS_DIR" --strip-components="$PAIRS_STRIP"

# ── Fetch result JSONs via single tar ─────────────────────────────────────────
echo "[LOCAL] Fetching result JSONs..."
RESULTS_STRIP=$(ssh "$REMOTE_USER@$REMOTE_HOST" \
    "echo '${REMOTE_RESULTS_DIR}' | tr '/' '\n' | grep -c .")
ssh "$REMOTE_USER@$REMOTE_HOST" \
    "find '${REMOTE_RESULTS_DIR}' -name 'result_*.json' | tar czf - -T -" \
    | tar xzf - -C "$LOCAL_RESULTS_DIR" --strip-components="$RESULTS_STRIP"

# ── Compute errors ────────────────────────────────────────────────────────────
echo "[LOCAL] Computing errors..."
LOCAL_ERRORS_CSV="${LOCAL_EVAL_DIR}/errors.csv"

python "${LOCAL_ROOT}/Fisheye-MVS/src_eval/compute_errors.py" \
    --pairs-dir   "$LOCAL_PAIRS_DIR" \
    --results-dir "$LOCAL_RESULTS_DIR" \
    --output-csv  "$LOCAL_ERRORS_CSV" \
    --rotation-bins    ${ROTATION_BINS} \
    --translation-bins ${TRANSLATION_BINS}

echo ""
echo "[LOCAL] Done. CSV at: $LOCAL_ERRORS_CSV"
