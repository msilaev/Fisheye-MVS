"""
Sample a balanced test set from the pair database by binning into a 2D
(translation × rotation) grid and drawing up to SAMPLES_PER_BIN pairs per cell.
"""

import argparse
import json
import random
import numpy as np
from typing import Dict, List, Tuple

TRANSLATION_BINS = [0.0, 0.5, 1.0, 1.5, 2.0]
ROTATION_BINS    = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
SAMPLES_PER_BIN  = 50


def load_pair_database_from_json(path):
    with open(path) as f:
        data = json.load(f)
    return {tuple(map(int, k.split("_"))): v for k, v in data.items()}


def get_bin_index(value, bins):
    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i + 1]:
            return i
    return -1


def sample_test_set(pair_database):
    binned: Dict[Tuple[int, int], list] = {}
    for key, data in pair_database.items():
        t_idx = get_bin_index(data["translation_dist"], TRANSLATION_BINS)
        r_idx = get_bin_index(data["rotation_angle"],   ROTATION_BINS)
        if t_idx != -1 and r_idx != -1:
            binned.setdefault((t_idx, r_idx), []).append(key)

    final = []
    print(f"\n--- Sampling Summary (target {SAMPLES_PER_BIN}/bin) ---")
    for (t_idx, r_idx), keys in binned.items():
        n = min(SAMPLES_PER_BIN, len(keys))
        for k in random.sample(keys, n):
            final.append(pair_database[k])
        t_range = f"[{TRANSLATION_BINS[t_idx]:.2f}, {TRANSLATION_BINS[t_idx+1]:.2f})m"
        r_range = f"[{ROTATION_BINS[r_idx]:.1f}, {ROTATION_BINS[r_idx+1]:.1f})°"
        print(f"  T{t_idx} {t_range} x R{r_idx} {r_range}: {len(keys)} avail, {n} sampled")

    print(f"\nTotal pairs in test set: {len(final)}")
    return final


def main(args):
    db = load_pair_database_from_json(args.pair_database_path)
    test_set = sample_test_set(db)
    with open(args.sample_t_R_test_db_path, "w") as f:
        json.dump(test_set, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sequence_name",          required=True)
    p.add_argument("--pose_data_dir",          required=True)
    p.add_argument("--image_data_dir",         required=True)
    p.add_argument("--pair_database_path",     required=True)
    p.add_argument("--sample_t_R_test_db_path", required=True)
    main(p.parse_args())
