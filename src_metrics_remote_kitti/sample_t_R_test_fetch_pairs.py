import numpy as np
import random
from typing import Dict, List, Tuple
import json
import argparse
import matplotlib.pyplot as plt
from PIL import Image

import os
from utils import compute_pose_error

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
    # Handle the case where the value is in the final (inf) bin
    if value >= bins[-2] and bins[-1] == np.inf:
        return len(bins) - 2

    return -1  # Should not happen if inf is used correctly


def fetch_pose_image( test_set, ind):

    test_set_entry = test_set[ind]

    # Load images
    img1 = Image.open(test_set_entry["img_path_1"])
    img2 = Image.open(test_set_entry["img_path_2"])

    # Rotate clockwise by 90 degrees
    #img1_rot = img1.rotate(-90, expand=True)
    #img2_rot = img2.rotate(-90, expand=True)

    img1_rot = img1
    img2_rot = img2

    # Convert to NumPy arrays (if needed)
    img1_rot_np = np.array(img1_rot) / 255.0
    img2_rot_np = np.array(img2_rot) / 255.0

    pose1 = np.loadtxt(test_set_entry["pose_path_1"])
    pose2 = np.loadtxt(test_set_entry["pose_path_2"])

    return img1_rot_np, img2_rot_np, pose1, pose2

def save_poses(args, pose1, pose2):

    pose1_path = os.path.join(args.local_exp_dir, f"image0.txt")
    pose2_path = os.path.join(args.local_exp_dir, f"image1.txt")

    print(f"Saved: {pose1_path}")
    print(f"Saved: {pose2_path}")

    np.savetxt(pose1_path, pose1, fmt='%.8f')
    np.savetxt(pose2_path, pose2, fmt='%.8f')

def save_images(args, img1, img2):

    os.makedirs(args.local_exp_dir, exist_ok=True)

    # Convert numpy arrays (0–1 floats) to PIL images (uint8)
    img1_pil = Image.fromarray((img1 * 255).astype(np.uint8))
    img2_pil = Image.fromarray((img2 * 255).astype(np.uint8))

    img1_path = os.path.join(args.local_exp_dir, f"image0.png")
    img2_path = os.path.join(args.local_exp_dir, f"image1.png")

    img_description_path=os.path.join(args.local_exp_dir, f"image_pairs.txt")
    with open(img_description_path, "w") as f:
        description_string=f"image0.png image1.png"
        f.write(description_string)

    img1_pil.save(img1_path)
    img2_pil.save(img2_path)

    print(f"Saved: {img1_path}")
    print(f"Saved: {img2_path}")

def main(args):

    print(f"json path {args.sample_t_R_test_db_path}")

    with open(args.sample_t_R_test_db_path, "r") as f:
        final_test_set = json.load(f)

    final_test_set = np.array(final_test_set)

    print(f"example entry {final_test_set[0]}")

    img1, img2, pose1, pose2 = fetch_pose_image(final_test_set, args.sample_ind)

    save_images(args, img1, img2)
    save_poses(args, pose1, pose2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence_name", required=True, type=str)
    parser.add_argument("--pose_data_dir", required=True, type=str)
    parser.add_argument("--image_data_dir", required=True, type=str)
    parser.add_argument("--sample_t_R_test_db_path", required=True, type=str)
    parser.add_argument("--sample_ind", type=int)
    parser.add_argument("--local_exp_dir", required=True, type=str)
    main(parser.parse_args())