#!/bin/bash
# Sets up all required mamba environments on the remote machine.
# Usage: bash setup_remote_envs.sh <REMOTE_UNIK3D_DIR> <REMOTE_SUPER_GLUE_DIR> <REMOTE_SCRIPT_DIR_procrustes>
#
# Clone dependencies first:
#   cd /scratch/work/<remote-user>/3d
#   git clone https://github.com/lpiccinelli-eth/UniK3D.git UniK3D
#   git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git SuperGluePretrainedNetwork
#
# Then run:
#   bash setup_remote_envs.sh \
#     /scratch/work/<remote-user>/3d/UniK3D \
#     /scratch/work/<remote-user>/3d/SuperGluePretrainedNetwork \
#     /scratch/work/<remote-user>/3d/Fisheye-MVS/src_procrustes
set -e
set -x

REMOTE_UNIK3D_DIR="${1}"
REMOTE_SUPER_GLUE_DIR="${2}"
REMOTE_SCRIPT_DIR_procrustes="${3}"

module load mamba
eval "$(mamba shell hook --shell bash)"

# --- mvf-unik3d ---
mamba create -n mvf-unik3d python=3.10 -y
source activate mvf-unik3d
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e "$REMOTE_UNIK3D_DIR"
source deactivate

# --- superglue38 ---
mamba create -n superglue38 python=3.8 -y
source activate superglue38
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python matplotlib numpy
source deactivate

# --- 3d_pose_env ---
mamba create -n 3d_pose_env python=3.10 -y
source activate 3d_pose_env
pip install -r "$REMOTE_SCRIPT_DIR_procrustes/requirements.txt"
pip install pandas matplotlib
source deactivate

echo "All environments set up successfully."
