#!/bin/bash
set -e

eval "$(conda shell.bash hook)"
conda activate unik3d

LOCAL_SRC_DIR="/worktmp/THESES/GAUSSIAN-SPLATTING/experiments/output_unik3D_pinhole_1"

mkpts1="${LOCAL_SRC_DIR}/mkpts1.npy"
mkpts2="${LOCAL_SRC_DIR}/mkpts2.npy"

mkpts_lightglue1="${LOCAL_SRC_DIR}/lightglue_matches_0.npy"
mkpts_lightglue2="${LOCAL_SRC_DIR}/lightglue_matches_1.npy"

python compare_points.py \
  --mkpts1 "${mkpts1}" \
  --mkpts2 "${mkpts2}" \
  --mkpts_lightglue1 "${mkpts_lightglue1}" \
  --mkpts_lightglue2 "${mkpts_lightglue2}"



  #--point1 "${LOCAL_SRC_DIR}/mkpts1.npy" \
  #--point2 "${LOCAL_SRC_DIR}/mkpts2.npy"

