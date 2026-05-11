"""
Convert ADT ground-truth poses to the JSON format used by evaluate_all.py.

ADT format: one .txt file per frame, containing a 4x4 camera-to-world matrix.
  Filename example: Apartment_release_clean_seq133_M1292_frame000000.txt

Usage:
  python convert_gt_adt.py \\
      --pose0 /path/to/frameA.txt \\
      --pose1 /path/to/frameB.txt \\
      --output /path/to/gt/pair_0001.json

For batch conversion of many pairs, pass --pairs-list, a text file where
each line is:  <pose0.txt> <pose1.txt> <output.json>
"""

import argparse
import json
import numpy as np


def load_pose(path):
    """Load a 4x4 camera-to-world matrix from an ADT .txt file."""
    return np.loadtxt(path)  # (4, 4)


def relative_pose(T0, T1):
    """
    Compute relative pose from camera 0 to camera 1.
    T0, T1: 4x4 camera-to-world matrices.
    Returns R (3x3), t (3,)  such that  p1 = R @ p0 + t
    """
    T_rel = np.linalg.inv(T0) @ T1
    R = T_rel[:3, :3]
    t = T_rel[:3, 3]
    return R, t


def convert_pair(pose0_path, pose1_path, output_path):
    T0 = load_pose(pose0_path)
    T1 = load_pose(pose1_path)
    R, t = relative_pose(T0, T1)

    gt = {"R_gt": R.tolist(), "t_gt": t.tolist()}
    with open(output_path, "w") as f:
        json.dump(gt, f, indent=2)
    print(f"Saved {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pose0",  default=None, help="Path to frame 0 pose .txt")
    p.add_argument("--pose1",  default=None, help="Path to frame 1 pose .txt")
    p.add_argument("--output", default=None, help="Output JSON path")
    p.add_argument("--pairs-list", default=None,
                   help="Text file with lines: <pose0.txt> <pose1.txt> <output.json>")
    args = p.parse_args()

    if args.pairs_list:
        with open(args.pairs_list) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                pose0, pose1, output = line.split()
                convert_pair(pose0, pose1, output)
    elif args.pose0 and args.pose1 and args.output:
        convert_pair(args.pose0, args.pose1, args.output)
    else:
        p.error("Provide either --pose0/--pose1/--output or --pairs-list")


if __name__ == "__main__":
    main()
