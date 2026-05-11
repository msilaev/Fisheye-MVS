import argparse
import yaml
import numpy as np
import  os

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


def main(args):
    # Set your KITTI-360 dataset path
    kitti360Path = args.kitti_root

    # Initialize fisheye camera
    sequence = args.sequence_name
    cam_id = 2  # fisheye camera id:  2 or 3

    camera = CameraFisheye(kitti360Path, sequence, cam_id)

    # Access poses for each frame
    total_frames = 0
    for frame_idx in camera.frames:
        # Get camera-to-world transformation matrix (4x4)
        cam2world_pose = camera.cam2world[frame_idx]
        #print(f"Frame {frame_idx}: pose shape {cam2world_pose.shape}")
        #input()
        # cam2world_pose is a 4x4 transformation matrix

        # --- Saving Logic ---

        print(f"frame ids {frame_idx}")

        frame_name = f"{int(frame_idx):010d}"

        # Save Pose (as plain text .txt)
        pose_filename = os.path.join(args.output_dir, f"{frame_name}.txt")
        np.savetxt(pose_filename, cam2world_pose, fmt='%.8f')

        total_frames += 1

    print(f"total frames {total_frames}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence_name", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--kitti_root", required=True, type=str)


    main(parser.parse_args())
