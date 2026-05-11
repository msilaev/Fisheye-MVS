import argparse
import open3d as o3d
import numpy as np
import os
import imageio.v2 as imageio
from utils import compute_pose_error

import json
from pathlib import Path

#(args.sequence_name, args.sequence_path, args.database_path)

def build_pair_database(sequence_name, sequence_pose_path, sequence_image_path,
                        max_frame_sep = 100) :
    """
    Forms the database of image pairs with relative pose distances.

    Args:
        sequence_path: Path to the specific ADT sequence.
        output_root: Root directory for output.
        max_frame_sep: Optional, to limit the search space for pairs (e.g., N^2 -> N*max_frame_sep).

    Returns:
        A dictionary where keys are (frame_idx1, frame_idx2) and values are pair data.
    """
    sequence_pose_path = Path(sequence_pose_path)
    sequence_image_path = Path(sequence_image_path)

    # Assuming pose files are named sequentially (e.g., 000000.txt, 000001.txt)
    #pose_dir = output_root / seq_path.name.replace("/", "_").replace("\\", "_") / "poses"
    #img_dir = output_root / seq_path.name.replace("/", "_").replace("\\", "_") / "images"

    print(f"pose_dir {sequence_pose_path}")
    print(f"image_dir {sequence_image_path}")

    pose_files = sorted(sequence_pose_path.glob("*.txt"))
    img_files = sorted(sequence_image_path.glob("*.png"))  # Assuming PNG, adjust if needed

    img_file_pruned = []

    for pose_name in pose_files:
        path = Path(pose_name)

        # Extract frame number
        pose_number = int(path.stem)

        pose_root = path.parent

        # Correct zero-padding (10 digits) + extension
        img_path = sequence_image_path / f"{pose_number:010d}.png"

        img_file_pruned.append(img_path)




    print(f"len pose files {len(pose_files)}")
    print(f"len img files {len(img_file_pruned)}")

    print(f"pose files 1 {pose_files[0]}")
    print(f"image files 1 {img_file_pruned[0]}")




    if not pose_files: # or len(pose_files) != len(img_files):
        print("Error: Pose and/or image files not found or counts mismatch.")
        return {}

    # Load all poses into a list for quick access
    # Poses are assumed to be T_world_camera (T_wc)
    poses = [np.loadtxt(f) for f in pose_files]

    pair_database = {}

    N = len(poses)

    # Iterate over all frame pairs (i, j) where i < j to avoid redundant calculations
    # and self-comparisons (i=j)
    for i in range(N):
        # Limit j to a window for efficiency (optional, remove max_frame_sep for full N^2)
        # We start j from i + 1 to only compute unique, ordered pairs
        for j in range(i + 1, min(N, i + 1 + max_frame_sep)):
            T_W_C1 = poses[i]  # World to Camera 1 pose
            T_W_C2 = poses[j]  # World to Camera 2 pose

            # Relative pose: T_1 -> 2
            # T_1->2 = T_W->C2 * T_C1->W = T_W_C2^-1 * T_W_C1
            # Note: np.linalg.inv(T_W_C2) @ T_W_C1 is the transformation from C1 frame to C2 frame.
            T_1to2_gt = np.linalg.inv(T_W_C2) @ T_W_C1

            R_gt = T_1to2_gt[:3, :3]
            t_gt = T_1to2_gt[:3, 3]

            d_t, d_R = compute_pose_error(R_gt, t_gt, np.eye(3), 0*t_gt)

            # --- Populate the database ---
            pair_database[(i, j)] = {
                "frame_idx_1": i,
                "frame_idx_2": j,
                "pose_path_1": str(pose_files[i]),
                "pose_path_2": str(pose_files[j]),
                "img_path_1": str(img_file_pruned[i]),
                "img_path_2": str(img_file_pruned[j]),
                "translation_dist": d_t,  # meters
                "rotation_angle": d_R  # degrees
            }

    return pair_database

def save_pair_database_to_json(pair_database: dict, pair_database_path: str):
    """
    Saves the pair_database dictionary to a JSON file.
    """

    # JSON requires string keys, so we convert the (i, j) tuple keys to "i_j" strings.
    serializable_dict = {}

    ind=0
    for (idx1, idx2), data in pair_database.items():
        ind+=1
        key_str = f"{idx1}_{idx2}"
        serializable_dict[key_str] = data


    print(ind)
    # Save the dictionary to JSON
    print(f"Saving pair database to JSON: {pair_database_path}")
    print(serializable_dict[f"{0}_{1}"])
    with open(pair_database_path, 'w') as f:
        # Use indent=4 for human readability
        json.dump(serializable_dict, f, indent=4)


def load_pair_database_from_json(pair_database_path: str) -> dict:
    """
    Loads the JSON file and converts string keys back to (int, int) tuples.
    """
    with open(pair_database_path, 'r') as f:
        data = json.load(f)

    # Convert string keys back to tuple keys
    pair_database = {}
    for key_str, data_item in data.items():
        idx1, idx2 = map(int, key_str.split('_'))
        pair_database[(idx1, idx2)] = data_item

    return pair_database

def main(args):

    # Example usage:

    #sequence_name = Path(sequence_path).name.replace("/", "_")

    #def build_pair_database(sequence_name, sequence_pose_path, sequence_image_path,
    #                        max_frame_sep=100):

    pair_database = build_pair_database(args.sequence_name,
                                        args.pose_data_dir,
                                        args.image_data_dir)


    save_pair_database_to_json(pair_database, args.pair_database_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence_name", required=True, type=str)
    parser.add_argument("--pose_data_dir", required=True, type=str)
    parser.add_argument("--image_data_dir", required=True, type=str)
    parser.add_argument("--pair_database_path", required=True, type=str)

    main(parser.parse_args())
