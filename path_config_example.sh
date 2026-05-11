# Root of the Fisheye-MVS project clone on the remote machine
REMOTE_PROJECT_DIR="/scratch/work/<username>/3d/Fisheye-MVS"

# Root directory for experiment data (images, results, logs) on the remote
REMOTE_EXPERIMENTS_DIR="/scratch/work/<username>/3d/experiments"

# Root of the shared dataset storage (ADT, KITTI-360, etc.)
REMOTE_DATASETS_DIR="/scratch/elec/t412-speechcom/symptonic-r2b/simulation-r2b/data_ms/3d"

# External dependency repos on the remote
REMOTE_SUPER_GLUE_DIR="/scratch/work/<username>/3d/SuperGluePretrainedNetwork"
REMOTE_UNIK3D_DIR="/scratch/work/<username>/3d/UniK3D"

# Local machine: parent directory containing both the project clone and experiments/
# WSL:      /mnt/c/Users/<username>/Documents
# Git Bash: /c/Users/<username>/Documents
LOCAL_ROOT="/mnt${LOCAL_ROOT}"