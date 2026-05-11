from kitti360scripts.helpers.project import CameraFisheye
import argparse
import os

import yaml
import os

# Fix for newer PyYAML versions
import kitti360scripts.helpers.project as project_module
original_readYAMLFile = project_module.readYAMLFile

def patched_readYAMLFile(fileName):
    '''make OpenCV YAML file compatible with python'''
    ret = {}
    skip_lines = 1
    with open(fileName) as fin:
        for i in range(skip_lines):
            fin.readline()
        yamlFileOut = fin.read()
        import re
        myRe = re.compile(r": ([^ ])")
        yamlFileOut = myRe.sub(r':  \1', yamlFileOut)
        ret = yaml.load(yamlFileOut, Loader=yaml. FullLoader)  # Add Loader here
    return ret

# Apply the patch
project_module. readYAMLFile = patched_readYAMLFile

from kitti360scripts.helpers.project import CameraFisheye

# Set your KITTI-360 dataset path
kitti360Path = '/path/to/KITTI-360'


def main():
    # Set your KITTI-360 dataset path
    kitti360Path = "/worktmp/THESES/GAUSSIAN-SPLATTING/KITTI-360" #'/path/to/KITTI-360'  # or use os.environ['KITTI360_DATASET']

    # Initialize fisheye camera
    seq = 0
    sequence = '2013_05_28_drive_%04d_sync' % seq
    cam_id = 2  # fisheye camera id:  2 or 3

    camera = CameraFisheye(kitti360Path, sequence, cam_id)

    # Access poses for each frame
    for frame in camera. frames:
        # Get camera-to-world transformation matrix (4x4)
        cam2world_pose = camera.cam2world[frame]
        print(f"Frame {frame}: pose shape {cam2world_pose.shape}")
        # cam2world_pose is a 4x4 transformation matrix
if __name__ == "__main__":
    #parser = argparse.ArgumentParser()

    #parser.add_argument("--sequence_name", required=True, type=str)

    #main(parser.parse_args())
    main()