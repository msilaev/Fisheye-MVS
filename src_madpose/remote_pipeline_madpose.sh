#!/bin/bash
set -e
set -x

image_pair_file="$1"
image_input_dir="$2"
DATA_TYPE="$3"
REMOTE_LOG_DIR="$4"
REMOTE_RESULTS_DIR="$5"
REMOTE_SCRIPT_DIR="$6"

# ------------------------------
# 1. Load Conda
# ------------------------------
if [ -f "/home/<remote-user>/miniconda3/etc/profile.d/conda.sh" ]; then
    source /home/<remote-user>/miniconda3/etc/profile.d/conda.sh
else
    echo "ERROR: Could not find conda.sh"
    exit 1
fi

# ------------------------------
# 2. Activate environment
# ------------------------------
conda deactivate || true
conda activate madpose_env

# ------------------------------
# 3. Run MADPose
# ------------------------------
cd /home/<remote-user>/GAUSSIAN-SPLATTING/madpose/examples
time_start=$(date +%s)

echo "[REMOTE] Running madpose ..."
echo "$REMOTE_LOG_DIR/madpose_example.log"
#python calibrated.py
python shared_focal.py > "$REMOTE_LOG_DIR/madpose_example.log" 2>&1

cd $REMOTE_DIR

mkpts1="${REMOTE_RESULTS_DIR}/mkpts1.npy"
mkpts2="${REMOTE_RESULTS_DIR}/mkpts2.npy"

#mkpts1="${REMOTE_RESULTS_DIR}/lightglue_matches_0.npy"
#mkpts2="${REMOTE_RESULTS_DIR}/lightglue_matches_1.npy"

depth1="${REMOTE_RESULTS_DIR}/image0_depth.npy"
depth2="${REMOTE_RESULTS_DIR}/image1_depth.npy"

madpose_est_file="$REMOTE_RESULTS_DIR/madpose_est_file.json"

cd $REMOTE_SCRIPT_DIR

# image0_mp_depth.npy" \
python madpose_inference.py \
  --mp_depth_0_file "${REMOTE_RESULTS_DIR}/image0_mp_depth.npy" \
  --mp_depth_1_file "${REMOTE_RESULTS_DIR}/image1_mp_depth.npy" \
  --matches_0_file "${mkpts1}" \
  --matches_1_file "${mkpts2}" \
  --depth_0_file "${depth1}" \
  --depth_1_file "${depth2}" \
  --image_input_dir "${image_input_dir}" \
  --image_pair_file "${image_pair_file}" \
  --madpose_est_file "${madpose_est_file}" \
  > "$REMOTE_LOG_DIR/madpose_inference.log" 2>&1

#  --image0 "${image_input_dir}/image0.jpg" \
#  --image1 "${image_input_dir}/image1.jpg" > "${REMOTE_DIR_EXP}/madpose_inference.log" 2>&1

echo "[REMOTE] madpose finished successfully"

# ------------------------------
# 4. Timing
# ------------------------------
time_end=$(date +%s)
time_diff=$((time_end - time_start))
echo "Madpose took ${time_diff} sec"

# ------------------------------
# 5. Cleanup
# ------------------------------
conda deactivate || true
echo "[REMOTE] madpose complete."


