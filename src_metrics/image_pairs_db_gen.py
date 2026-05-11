"""
Build a database of all frame pairs within a sequence, with their GT relative poses.

Works for both ADT (--img-ext jpg) and KITTI-360 (--img-ext png).

Expected dataset layout on remote:
  <sequence_dir>/
    poses/    000000.txt  000001.txt  ...   (4x4 cam-to-world matrices)
    images/   000000.jpg  000001.jpg  ...   (or .png for KITTI)

Output: pair_dataset.json
  {
    "0_5": {
      "frame_idx_1": 0, "frame_idx_2": 5,
      "pose_path_1": "...", "pose_path_2": "...",
      "img_path_1":  "...", "img_path_2":  "...",
      "translation_dist": 0.42,
      "rotation_angle":   12.3,
      "translation_error_est": null,
      "rotation_error_est":    null
    }, ...
  }
"""

import argparse
import json
import numpy as np
from pathlib import Path
from utils import compute_pose_error


def build_pair_database(sequence_name, sequence_pose_path, sequence_image_path,
                        img_ext="jpg", max_frame_sep=100):
    sequence_pose_path  = Path(sequence_pose_path)
    sequence_image_path = Path(sequence_image_path)

    print(f"pose_dir  {sequence_pose_path}")
    print(f"image_dir {sequence_image_path}")

    pose_files = sorted(sequence_pose_path.glob("*.txt"))
    img_files  = sorted(sequence_image_path.glob(f"*.{img_ext}"))

    if not pose_files or len(pose_files) != len(img_files):
        print(f"Error: pose count={len(pose_files)}, image count={len(img_files)} — mismatch or empty.")
        return {}

    poses = [np.loadtxt(f) for f in pose_files]
    N = len(poses)
    pair_database = {}

    for i in range(N):
        for j in range(i + 1, min(N, i + 1 + max_frame_sep)):
            T_W_C1 = poses[i]
            T_W_C2 = poses[j]

            T_1to2_gt = np.linalg.inv(T_W_C2) @ T_W_C1
            R_gt = T_1to2_gt[:3, :3]
            t_gt = T_1to2_gt[:3, 3]

            d_t, d_R = compute_pose_error(R_gt, t_gt, np.eye(3), np.zeros(3))

            pair_database[(i, j)] = {
                "frame_idx_1": i,
                "frame_idx_2": j,
                "pose_path_1": str(pose_files[i]),
                "pose_path_2": str(pose_files[j]),
                "img_path_1":  str(img_files[i]),
                "img_path_2":  str(img_files[j]),
                "translation_dist":      d_t,
                "rotation_angle":        d_R,
                "translation_error_est": None,
                "rotation_error_est":    None,
            }

    return pair_database


def save_pair_database_to_json(pair_database, path):
    serializable = {f"{i}_{j}": v for (i, j), v in pair_database.items()}
    print(f"Saving {len(serializable)} pairs to {path}")
    with open(path, "w") as f:
        json.dump(serializable, f, indent=4)


def main(args):
    db = build_pair_database(
        args.sequence_name,
        args.pose_data_dir,
        args.image_data_dir,
        img_ext=args.img_ext,
        max_frame_sep=args.max_frame_sep,
    )
    save_pair_database_to_json(db, args.pair_database_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sequence_name",      required=True)
    p.add_argument("--pose_data_dir",      required=True)
    p.add_argument("--image_data_dir",     required=True)
    p.add_argument("--pair_database_path", required=True)
    p.add_argument("--img-ext",            default="jpg",
                   help="Image file extension: jpg (ADT) or png (KITTI)")
    p.add_argument("--max-frame-sep",      type=int, default=100)
    main(p.parse_args())
