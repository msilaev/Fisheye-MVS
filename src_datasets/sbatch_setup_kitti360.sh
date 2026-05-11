#!/bin/bash
#SBATCH --job-name=kitti360_setup
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=batch-skl

SEQUENCE="${1:-2013_05_28_drive_0000_sync}"
PROJ_DIR="$2"

bash "${PROJ_DIR}/src_datasets/setup_kitti360.sh" "$SEQUENCE"
