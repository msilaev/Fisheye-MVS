import numpy as np
import random
from typing import Dict, List, Tuple
import json
import argparse
import matplotlib.pyplot as plt
from PIL import Image

import os

# --- Configuration for Binning ---

# Define the boundaries for the Translation (dt) bins (in meters)
TRANSLATION_BINS = [0.0, 0.2, 0.8, 2.0, 3.0]  # Bins: [0, 0.2), [0.2, 0.8), [0.8, 2.0), [2.0, inf)

# Define the boundaries for the Rotation (dR) bins (in degrees)
ROTATION_BINS = [0.0, 5.0, 15.0, 30.0, 45.0, 60.0,  70.0]  # Bins: [0, 5), [5, 15), [15, 30), [30, inf)

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
    # Handle the case where the value is in the final (inf) bin
    if value >= bins[-2] and bins[-1] == np.inf:
        return len(bins) - 2

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
            final_test_set.append(pair_database[key])

        print(f"Bin T{t_idx} ({t_range}) x R{r_idx} ({r_range}): Available={num_available}, Sampled={num_to_sample}")

    print(f"\nTotal Pairs in Final Test Set: {len(final_test_set)}")

    return final_test_set

def plt_dist_dR_dt(final_test_set):

    dR_arr = []
    dt_arr = []

    dRt =[]

    for values in final_test_set:

        dR_arr.append(values["rotation_angle"])
        dt_arr.append(values["translation_dist"])

    dR_arr = np.array(dR_arr)
    dt_arr = np.array(dt_arr)

    plt.figure(figsize=(12, 5))

    # --- 1D histograms ---
    plt.subplot(1, 3, 1)
    plt.hist(dR_arr, bins=50)
    plt.title("Rotation angle distribution")
    plt.xlabel("Rotation angle")
    plt.ylabel("Count")

    plt.subplot(1, 3, 2)
    plt.hist(dt_arr, bins=50)
    plt.title("Translation distance distribution")
    plt.xlabel("Translation distance")
    plt.ylabel("Count")

    # --- 2D histogram (angle vs distance) ---
    plt.subplot(1, 3, 3)
    #plt.hist2d(dR_arr, dt_arr, bins=50, cmap="viridis")
    #H, xedges, yedges, img = plt.hist2d(dR_arr, dt_arr, bins=50)
    plt.hist2d(dR_arr, dt_arr, bins=[ROTATION_BINS, TRANSLATION_BINS], cmap='viridis')

    plt.colorbar(label="Count")
    plt.xlabel("Rotation angle")
    plt.ylabel("Translation distance")
    plt.title("2D Histogram")

    plt.tight_layout()
    plt.show()


def fetch_pose_image( test_set, ind):

    test_set_entry = test_set[ind]

    # Load images
    img1 = Image.open(test_set_entry["img_path_1"])
    img2 = Image.open(test_set_entry["img_path_2"])

    # Rotate clockwise by 90 degrees
    img1_rot = img1.rotate(-90, expand=True)
    img2_rot = img2.rotate(-90, expand=True)

    # Convert to NumPy arrays (if needed)
    img1_rot_np = np.array(img1_rot) / 255.0
    img2_rot_np = np.array(img2_rot) / 255.0

    '''
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img1_rot_np)
    plt.title("Image 1")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img2_rot_np)
    plt.title("Image 2")
    plt.axis("off")

    plt.show()
    '''

    return img1_rot_np, img2_rot_np



def save_images_poses(args, img1, img2, pose1, pose2, ind):
    os.makedirs(args.local_exp_dir, exist_ok=True)

    print(f"Saving images for index {ind} into {args.local_exp_dir}")

    # Convert numpy arrays (0–1 floats) to PIL images (uint8)
    img1_pil = Image.fromarray((img1 * 255).astype(np.uint8))
    img2_pil = Image.fromarray((img2 * 255).astype(np.uint8))

    img1_path = os.path.join(args.local_exp_dir, f"image0_{ind}.jpg")
    img2_path = os.path.join(args.local_exp_dir, f"image1_{ind}.jpg")

    img1_pil.save(img1_path)
    img2_pil.save(img2_path)

    print(f"Saved: {img1_path}")
    print(f"Saved: {img2_path}")


def main(args):
    pair_database = load_pair_database_from_json(args.pair_database_path)

    final_test_set = sample_test_set(pair_database)

    with open(args.sample_t_R_test_db_path, "w") as f:
        json.dump(final_test_set, f, indent=2)

    #final_test_set = np.array(final_test_set)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence_name", required=True, type=str)
    parser.add_argument("--pose_data_dir", required=True, type=str)
    parser.add_argument("--image_data_dir", required=True, type=str)
    parser.add_argument("--pair_database_path", required=True, type=str)
    parser.add_argument("--sample_t_R_test_db_path", required=True, type=str)


    main(parser.parse_args())