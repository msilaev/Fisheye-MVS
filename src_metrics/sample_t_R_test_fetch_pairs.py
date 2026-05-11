"""
Fetch a single sampled pair from the test set JSON:
  - saves image0.jpg / image1.jpg  (rotated 90° CW if --rotate)
  - saves image0.txt / image1.txt  (GT cam-to-world poses)
  - writes image_pairs.txt         (required by SuperGlue)

Usage:
  python sample_t_R_test_fetch_pairs.py \\
      --sample-ind 0 \\
      --sample-t-R-test-db-path /path/sample_t_R_pair_dataset.json \\
      --output-dir /path/to/experiment_images \\
      [--rotate]   # pass for ADT (images stored rotated 90° CW)
"""

import argparse
import json
import os
import numpy as np
from PIL import Image


def fetch_pair(test_set, ind, rotate):
    entry = test_set[ind]
    img1 = Image.open(entry["img_path_1"])
    img2 = Image.open(entry["img_path_2"])
    if rotate:
        img1 = img1.rotate(-90, expand=True)
        img2 = img2.rotate(-90, expand=True)
    return (np.array(img1) / 255.0,
            np.array(img2) / 255.0,
            np.loadtxt(entry["pose_path_1"]),
            np.loadtxt(entry["pose_path_2"]))


def save_pair(output_dir, img1_np, img2_np, pose1, pose2):
    os.makedirs(output_dir, exist_ok=True)

    img_ext = "jpg"
    for stem, arr in [("image0", img1_np), ("image1", img2_np)]:
        pil = Image.fromarray((arr * 255).astype(np.uint8))
        pil.save(os.path.join(output_dir, f"{stem}.{img_ext}"))
        print(f"Saved {stem}.{img_ext}")

    np.savetxt(os.path.join(output_dir, "image0.txt"), pose1, fmt="%.8f")
    np.savetxt(os.path.join(output_dir, "image1.txt"), pose2, fmt="%.8f")

    with open(os.path.join(output_dir, "image_pairs.txt"), "w") as f:
        f.write(f"image0.{img_ext} image1.{img_ext}")


def main(args):
    with open(args.sample_t_R_test_db_path) as f:
        test_set = json.load(f)

    print(f"Test set: {len(test_set)} pairs. Fetching index {args.sample_ind}.")

    img1, img2, pose1, pose2 = fetch_pair(test_set, args.sample_ind, args.rotate)
    save_pair(args.output_dir, img1, img2, pose1, pose2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample-ind",              type=int, required=True)
    p.add_argument("--sample-t-R-test-db-path", required=True)
    p.add_argument("--output-dir",              required=True)
    p.add_argument("--rotate", action="store_true",
                   help="Rotate images 90° CW (use for ADT)")
    main(p.parse_args())
