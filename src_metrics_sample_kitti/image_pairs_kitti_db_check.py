import argparse
import open3d as o3d
import numpy as np
import os
import imageio.v2 as imageio
from utils import compute_pose_error

import matplotlib.pyplot as plt

import json
from pathlib import Path

#(args.sequence_name, args.sequence_path, args.database_path)

def save_pair_database_to_json(pair_database: dict, pair_database_path: str):
    """
    Saves the pair_database dictionary to a JSON file.
    """

    # JSON requires string keys, so we convert the (i, j) tuple keys to "i_j" strings.
    serializable_dict = {}
    for (idx1, idx2), data in pair_database.items():
        key_str = f"{idx1}_{idx2}"
        serializable_dict[key_str] = data

    # Save the dictionary to JSON
    print(f"Saving pair database to JSON: {pair_database_path}")
    with open(pair_database_path, 'r') as f:
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


def plt_dist_dR_dt(pair_database):

    dR_arr = []
    dt_arr = []

    for key, values in pair_database.items():
        dR_arr.append(values["rotation_angle"])
        dt_arr.append(values["translation_dist"])

    dR_arr = np.array(dR_arr)
    dt_arr = np.array(dt_arr)

    # Plot histograms
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(dR_arr, bins=50)
    plt.title("Rotation angle distribution")
    plt.xlabel("Rotation angle")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(dt_arr, bins=50)
    plt.title("Translation distance distribution")
    plt.xlabel("Translation distance")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()


def main(args):

    pair_database = load_pair_database_from_json(args.pair_database_path)
    plt_dist_dR_dt(pair_database)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence_name", required=True, type=str)
    parser.add_argument("--pose_data_dir", required=True, type=str)
    parser.add_argument("--image_data_dir", required=True, type=str)
    parser.add_argument("--pair_database_path", required=True, type=str)

    main(parser.parse_args())
