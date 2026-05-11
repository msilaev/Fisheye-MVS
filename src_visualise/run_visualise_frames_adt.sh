#!/bin/bash
# Submit a SLURM job to render merged-cloud comparison figures for ADT.
# Selects characteristic pairs across rotation bins and renders
# GT | Ours+RANSAC | MADPose | MADPose+rect side-by-side.
#
# Usage:
#   bash run_visualise_frames_adt.sh                            # ARC-133 (default)
#   SEQUENCE_NAME=Apartment_release_clean_seq136_M1292 \
#     EXPERIMENT_NAME=ADT_seq136 bash run_visualise_frames_adt.sh
#   PAIRS="pair_rot_20-30_trans_0.5-1.0_7 pair_rot_40-50_trans_1.0-1.5_3" \
#     bash run_visualise_frames_adt.sh
set -e
set -x

source "$(dirname "$0")/../.env"
source "$(dirname "$0")/../path_config.sh"
source "$(dirname "$0")/../experiment_config_adt.sh"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-ADT_seq133}"
REMOTE_EVAL_DIR="${REMOTE_EXPERIMENTS_DIR}/${EXPERIMENT_NAME}/eval_full"
REMOTE_PAIRS_DIR="${REMOTE_EVAL_DIR}/test_pairs"
REMOTE_RESULTS_DIR="${REMOTE_EVAL_DIR}/results"
REMOTE_OUTPUT_DIR="${REMOTE_EVAL_DIR}/rendered_example"
REMOTE_VIS_DIR="${REMOTE_PROJECT_DIR}/src_visualise"

LOCAL_OUTPUT_DIR="${LOCAL_ROOT}/experiments/${EXPERIMENT_NAME}/rendered_example"
mkdir -p "$LOCAL_OUTPUT_DIR"

# ── Characteristic + requested frame pairs ─────────────────────────────────
DEFAULT_PAIRS=(
    # Small rotation (~5°), small translation
    #"pair_rot_00-10_trans_0.0-0.5_5"
    # Small rotation (~5°), medium translation
    #"pair_rot_00-10_trans_0.5-1.0_5"
    # Medium rotation (~15°), small translation
    #"pair_rot_10-20_trans_0.0-0.5_3"
    # Medium rotation (~15°), medium translation
    #"pair_rot_10-20_trans_0.5-1.0_3"
    # Large rotation (~25°), small translation
    #"pair_rot_20-30_trans_0.0-0.5_4"
    # Large rotation (~25°), medium translation
    #"pair_rot_20-30_trans_0.5-1.0_4"
    # Large rotation, small translation
    #"pair_rot_30-40_trans_0.0-0.5_2"
    # Large rotation, medium translation
    #"pair_rot_30-40_trans_0.5-1.0_2"
    # Very large rotation (~45°), small translation
    #"pair_rot_40-50_trans_0.0-0.5_3"
    # Very large rotation (~45°), medium translation
    #"pair_rot_40-50_trans_0.5-1.0_3"

    # Additional requested ADT frame pairs (resolved or prepared on demand)
    "frames:Apartment_release_clean_seq136_M1292_frame000900|Apartment_release_clean_seq136_M1292_frame000840"
    "frames:Apartment_release_clean_seq136_M1292_frame002563|Apartment_release_clean_seq136_M1292_frame002537"
    "frames:Apartment_release_decoration_seq136_M1292_frame002000|Apartment_release_decoration_seq136_M1292_frame001965"
    "frames:Apartment_release_multiuser_cook_seq141_M1292_frame002033|Apartment_release_multiuser_cook_seq141_M1292_frame002090"
)

if [ -n "${PAIRS:-}" ]; then
    IFS=' ' read -r -a SELECTED <<< "$PAIRS"
else
    SELECTED=("${DEFAULT_PAIRS[@]}")
fi

append_unique_pair() {
    local pair_name="$1"
    local existing
    for existing in "${VERIFIED[@]}"; do
        if [ "$existing" = "$pair_name" ]; then
            return 0
        fi
    done
    VERIFIED+=("$pair_name")
}

# Verify existing pair dirs; keep frame-based requests for on-demand preparation.
VERIFIED=()
for P in "${SELECTED[@]}"; do
    if [[ "$P" == frames:* ]]; then
        append_unique_pair "$P"
    elif ssh "$REMOTE_USER@$REMOTE_HOST" "[ -d '${REMOTE_PAIRS_DIR}/${P}' ]" 2>/dev/null; then
        append_unique_pair "$P"
    else
        echo "[WARN] pair not found on remote, skipping: $P"
    fi
done

if [ ${#VERIFIED[@]} -eq 0 ]; then
    echo "[ERROR] No valid pairs found. Check REMOTE_PAIRS_DIR=${REMOTE_PAIRS_DIR}"
    exit 1
fi

# ── Write pairs list on remote ──────────────────────────────────────────────
REMOTE_LIST="${REMOTE_EVAL_DIR}/render_pairs_adt.txt"
ssh "$REMOTE_USER@$REMOTE_HOST" "
    mkdir -p '${REMOTE_OUTPUT_DIR}'
    printf '%s\n' $(printf "'%s' " "${VERIFIED[@]}") > '${REMOTE_LIST}'
    cat '${REMOTE_LIST}'
"

LOCAL_CAMERA_JSON="$(dirname "$0")/../src_visualise_local/camera_adt.json"
REMOTE_CAMERA_JSON="${REMOTE_VIS_DIR}/camera_adt.json"

LOCAL_MASK="$(dirname "$0")/../assets/fisheye_masks/MaskADT_rot.png"
REMOTE_MASK="${REMOTE_VIS_DIR}/MaskADT_rot.png"

# ── Sync vis scripts ────────────────────────────────────────────────────────
scp "$(dirname "$0")/render_merged_clouds.py" \
    "$REMOTE_USER@$REMOTE_HOST:${REMOTE_VIS_DIR}/"
scp "$(dirname "$0")/prepare_render_pairs.py" \
    "$REMOTE_USER@$REMOTE_HOST:${REMOTE_VIS_DIR}/"
scp "$(dirname "$0")/sbatch_render_examples.sh" \
    "$REMOTE_USER@$REMOTE_HOST:${REMOTE_VIS_DIR}/"
if [ -f "$LOCAL_CAMERA_JSON" ]; then
    scp "$LOCAL_CAMERA_JSON" "$REMOTE_USER@$REMOTE_HOST:${REMOTE_CAMERA_JSON}"
    echo "[LOCAL] Uploaded camera pose → ${REMOTE_CAMERA_JSON}"
else
    REMOTE_CAMERA_JSON=""
    echo "[LOCAL] No camera_adt.json found — remote renderer will use auto-fit view"
fi
if [ -f "$LOCAL_MASK" ]; then
    scp "$LOCAL_MASK" "$REMOTE_USER@$REMOTE_HOST:${REMOTE_MASK}"
    echo "[LOCAL] Uploaded fisheye mask → ${REMOTE_MASK}"
else
    REMOTE_MASK=""
    echo "[LOCAL] No MaskADT_rot.png found — rendering without fisheye mask"
fi
ssh "$REMOTE_USER@$REMOTE_HOST" "
    sed -i 's/\r//' '${REMOTE_VIS_DIR}/sbatch_render_examples.sh'
    chmod +x '${REMOTE_VIS_DIR}/sbatch_render_examples.sh'
"

# ── Submit SLURM job ─────────────────────────────────────────────────────────
SLURM_LOG="${REMOTE_EVAL_DIR}/slurm_logs"
ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p '$SLURM_LOG'"

SBATCH_CMD="sbatch --parsable --output='${SLURM_LOG}/render_adt-%j.log' --error='${SLURM_LOG}/render_adt-%j.err' '${REMOTE_VIS_DIR}/sbatch_render_examples.sh' '${REMOTE_LIST}' '${REMOTE_PAIRS_DIR}' '${REMOTE_RESULTS_DIR}' '${REMOTE_OUTPUT_DIR}' '${REMOTE_PROJECT_DIR}' '${REMOTE_UNIK3D_DIR}' 'adt' '${DISTANCE_THRESHOLD:-1000}' '${REMOTE_CAMERA_JSON}' '${REMOTE_MASK}'"
JOB_ID=$(ssh "$REMOTE_USER@$REMOTE_HOST" "$SBATCH_CMD")
echo "[LOCAL] Submitted SLURM job: $JOB_ID"
echo "[LOCAL] Monitor: ssh $REMOTE_USER@$REMOTE_HOST 'squeue -j $JOB_ID'"
echo "[LOCAL] When done, download with:"
echo "  bash $(dirname "$0")/fetch_rendered_examples.sh adt $JOB_ID"
