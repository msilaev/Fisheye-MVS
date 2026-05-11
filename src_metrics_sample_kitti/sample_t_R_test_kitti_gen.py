import numpy as np
import random
from typing import Dict, List, Tuple
import json
import argparse
import matplotlib.pyplot as plt
from PIL import Image

from pathlib import Path
import shutil

import os

# --- Configuration for Binning ---

# Define the boundaries for the Translation (dt) bins (in meters)
TRANSLATION_BINS = [0.1, 1.0, 2.0, 3.0, 4, 5]  # Bins: [0, 0.2), [0.2, 0.8), [0.8, 2.0), [2.0, inf)
ROTATION_BINS = [0.0, 5.0, 10.0, 20]  # Bins: [0, 5), [5, 15), [15, 30), [30, inf)

# Target number of pairs to sample from each populated bin
SAMPLES_PER_BIN = 50


def load_pair_database_from_json(input_filepath: str) -> dict:
    """
    Loads the JSON file and converts string keys back to (int, int) tuples.
    """
    with open(input_filepath, 'r') as f:
        data = json.load(f)

    # Convert string keys back to tuple keys
    pair_database = {}
    for key_str, data_item in data.items():
        idx1, idx2 = map(int, key_str.split('_'))
        pair_database[(idx1, idx2)] = data_item

    return pair_database


def get_bin_index(value: float, bins: List[float]) -> int:
    """Finds the index of the bin a value falls into."""
    for i in range(len(bins) - 1):
        # Check if value is in the range [bins[i], bins[i+1])
        if bins[i] <= value < bins[i + 1]:
            return i

    return -1  # Should not happen if inf is used correctly


def sample_test_set(pair_database: Dict[Tuple[int, int], dict]) -> List[dict]:
    """
    Groups pairs into 2D (Translation x Rotation) bins and samples uniformly.
    """

    # Use a dictionary of dictionaries to store the bin contents
    binned_pairs: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    # 2. Assign every pair to a 2D bin
    for key, data in pair_database.items():
        dt = data["translation_dist"]
        dr = data["rotation_angle"]

        t_idx = get_bin_index(dt, TRANSLATION_BINS)
        r_idx = get_bin_index(dr, ROTATION_BINS)

        # Ensure a valid bin was found
        if t_idx != -1 and r_idx != -1:
            bin_key = (t_idx, r_idx)

            if bin_key not in binned_pairs:
                binned_pairs[bin_key] = []

            binned_pairs[bin_key].append(key)

    # 3. Uniformly Sample from each bin
    final_test_set = []

    print(f"\n--- Sampling Summary ---")
    print(f"Target Samples per Bin: {SAMPLES_PER_BIN}")

    for bin_key, pair_keys in binned_pairs.items():
        t_idx, r_idx = bin_key

        # Describe the bin for logging
        t_range = f"[{TRANSLATION_BINS[t_idx]:.2f}m, {TRANSLATION_BINS[t_idx + 1]:.2f}m)"
        r_range = f"[{ROTATION_BINS[r_idx]:.1f}°, {ROTATION_BINS[r_idx + 1]:.1f}°)"

        num_available = len(pair_keys)

        # Determine how many pairs to sample
        num_to_sample = min(SAMPLES_PER_BIN, num_available)

        # Randomly sample the required number of keys
        sampled_keys = random.sample(pair_keys, num_to_sample)

        # Retrieve the full data for the sampled pairs
        for key in sampled_keys:

            pair_entry = pair_database[key]

            final_test_set.append(pair_entry)

        print(f"Bin T{t_idx} ({t_range}) x R{r_idx} ({r_range}): Available={num_available}, Sampled={num_to_sample}")

    print(f"\nTotal Pairs in Final Test Set: {len(final_test_set)}")

    return final_test_set

def change_paths_remote(final_test_set) -> List[dict]:
    """
    Groups pairs into 2D (Translation x Rotation) bins and samples uniformly.
    """

    final_test_set_rename=[]
    for pair_entry in final_test_set:

        pair_entry_rename = {}
        pair_entry_rename['frame_idx_1'] = pair_entry['frame_idx_1']
        pair_entry_rename['frame_idx_2'] = pair_entry['frame_idx_2']
        pair_entry_rename['translation_dist'] = pair_entry['translation_dist']
        pair_entry_rename['rotation_angle'] = pair_entry['rotation_angle']

        s = pair_entry['pose_path_1'].split("THESES")[1]
        s_new = "/home/<remote-user>" + s
        pair_entry_rename['pose_path_1'] = s_new

        s = pair_entry['pose_path_2'].split("THESES")[1]
        s_new = "/home/<remote-user>" + s
        pair_entry_rename['pose_path_2'] = s_new

        s = pair_entry['img_path_1'].split("THESES")[1]
        s_new = "/home/<remote-user>" + s
        pair_entry_rename['img_path_1'] = s_new

        s = pair_entry['img_path_2'].split("THESES")[1]
        s_new = "/home/<remote-user>" + s
        pair_entry_rename['img_path_2'] = s_new

        #print("s_new=", s_new)

        #'pose_path_1': '/worktmp/THESES/GAUSSIAN-SPLATTING/KITTI-360/poses_dir/2013_05_28_drive_0000_sync/image_02/0000000134.txt'
        #"pose_path_1": "/home/<remote-user>/GAUSSIAN-SPLATTING/experiments/KITTI-360_data/2013_05_28_drive_0000_sync/poses/0000000134.txt"

        #{'frame_idx_1': 43, 'frame_idx_2': 44,
        # 'pose_path_1': '/worktmp/THESES/GAUSSIAN-SPLATTING/KITTI-360/poses_dir/2013_05_28_drive_0000_sync/image_02/0000000134.txt',
        # 'pose_path_2': '/worktmp/THESES/GAUSSIAN-SPLATTING/KITTI-360/poses_dir/2013_05_28_drive_0000_sync/image_02/0000000135.txt',
        # 'img_path_1': '/worktmp/THESES/GAUSSIAN-SPLATTING/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_02/data_rgb/0000000134.png',
        # 'img_path_2': '/worktmp/THESES/GAUSSIAN-SPLATTING/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_02/data_rgb/0000000135.png',
        # 'translation_dist': 0.3849459443291553, 'rotation_angle': 0.17721071859339058}

        final_test_set_rename.append(pair_entry_rename)

    return final_test_set_rename


def same_images_poses_test_set(args, final_test_set):

    #args.pose_data_dir.mkdir(parents=True, exist_ok=True)
    #args.image_data_dir.mkdir(parents=True, exist_ok=True)

    for sample in final_test_set:

        print(f"saving sample {sample}")

        pose_path_1 = Path(sample["pose_path_1"])
        pose_path_2 = Path(sample["pose_path_2"])
        img_path_1 = Path(sample["img_path_1"])
        img_path_2 = Path(sample["img_path_2"])

        # Preserve filename + extension
        pose_path_1_filter = Path(args.pose_data_dir) / pose_path_1.name
        pose_path_2_filter = Path(args.pose_data_dir) / pose_path_2.name
        img_path_1_filter = Path(args.image_data_dir) / img_path_1.name
        img_path_2_filter = Path(args.image_data_dir) / img_path_2.name

        shutil.copy2(pose_path_1, pose_path_1_filter)
        shutil.copy2(pose_path_2, pose_path_2_filter)
        shutil.copy2(img_path_1, img_path_1_filter)
        shutil.copy2(img_path_2, img_path_2_filter)

def main(args):

    pair_database = load_pair_database_from_json(args.pair_database_path)

    final_test_set = sample_test_set(pair_database)

    final_test_set_rename = change_paths_remote(final_test_set)

    with open(args.sample_t_R_test_db_path, "w") as f:
        json.dump(final_test_set_rename, f, indent=2)

    same_images_poses_test_set(args, final_test_set)

    #final_test_set = np.array(final_test_set)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence_name", required=True, type=str)
    parser.add_argument("--pose_data_dir", required=True, type=str)
    parser.add_argument("--image_data_dir", required=True, type=str)
    parser.add_argument("--pair_database_path", required=True, type=str)
    parser.add_argument("--sample_t_R_test_db_path", required=True, type=str)

    main(parser.parse_args())