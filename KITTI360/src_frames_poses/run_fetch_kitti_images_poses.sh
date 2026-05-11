#!/bin/bash
set -e
set -x
# Script to visualize ADT (Aria Digital Twin) frames

# --- Load environment variables ---
if [ -f ".env" ]; then
  source .env
fi

SEQUENCE_NAME="2013_05_28_drive_0000_sync"
CAM_NAME="02"
LOCAL_DIR_ROOT="/worktmp/THESES/GAUSSIAN-SPLATTING/KITTI-360"
root_dir=$LOCAL_DIR_ROOT #KITTI-360
data_2d_dir=data_2d_raw
poses_dir=poses_dir

#POSES_DIR=root_dir/$poses_dir/$SEQUENCE_NAME/"image_${CAM_NAME}"

POSES_DIR=$root_dir/$poses_dir/$SEQUENCE_NAME/image_${CAM_NAME}


mkdir -p $POSES_DIR

# Initialize Conda for bash
eval "$(conda shell.bash hook)"
conda activate unik3d

python extract_kitti_poses.py \
  --kitti_root $LOCAL_DIR_ROOT \
  --sequence_name $SEQUENCE_NAME \
  --output_dir $POSES_DIR
