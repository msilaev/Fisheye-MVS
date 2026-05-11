"""
Select N representative image pairs per (rotation × translation) joint bin,
and write each pair into a separate output directory ready for the evaluation pipeline.

Expected dataset layout:
  <sequence_dir>/
    poses/    000000.txt  000001.txt  ...  (4x4 cam-to-world)
    images/   000000.jpg  000001.jpg  ...  (or .png for KITTI)

Output (one folder per selected pair):
  <output_root>/
    pair_rot_00-10_trans_0.0-0.5_1/
      image0.jpg
      image1.jpg
      image0.txt        <- GT cam-to-world pose (for evaluation)
      image1.txt
      image_pairs.txt   <- "image0.jpg image1.jpg"
      info.json         <- rotation_angle, translation_dist, frame indices
    ...
    selected_pairs.json  <- flat list of all selected pairs with metadata

Usage:
  python select_pairs_by_rotation.py \\
      --pose-dir   /data/ADT/sequence/poses \\
      --image-dir  /data/ADT/sequence/images \\
      --output-dir /data/pairs_for_testing \\
      [--img-ext jpg]            # jpg (ADT) or png (KITTI)
      [--rotate]                 # rotate 90 deg CW (ADT images are stored sideways)
      [--max-frame-sep 100]      # limit pair search window
      [--min-frame-sep 3]        # minimum frame gap (avoid near-identical frames)
      [--n-per-bin 50]           # pairs to select per rotation x translation joint bin
      [--seed 42]                # random seed for reproducibility
      [--no-copy-images]         # use symlinks instead of copying (saves disk)
      [--rotation-bins 0 10 20 30 40 50]       # bin edges in degrees (ADT default)
      [--translation-bins 0.0 0.5 1.0 1.5 2.0] # bin edges in metres (ADT default)
      # KITTI-360 example:
      [--rotation-bins 0 10 20 30 40 50]
      [--translation-bins 0 5 10 20 50]
"""

import argparse
import json
import os
import random
import numpy as np
from pathlib import Path
from PIL import Image


def angle_between(R1, R2):
    cos = (np.trace(R1.T @ R2) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def relative_pose(T0, T1):
    T_rel = np.linalg.inv(T1) @ T0
    return T_rel[:3, :3], T_rel[:3, 3]


def rotation_angle(T0, T1):
    R_rel, _ = relative_pose(T0, T1)
    return angle_between(R_rel, np.eye(3))


def translation_dist(T0, T1):
    _, t = relative_pose(T0, T1)
    return float(np.linalg.norm(t))


def rot_bin_index(angle, bins):
    for b in range(len(bins) - 1):
        if bins[b] <= angle < bins[b + 1]:
            return b
    return -1


def trans_bin_index(dist, bins):
    for b in range(len(bins) - 1):
        if bins[b] <= dist < bins[b + 1]:
            return b
    return -1


def joint_bin_label(r_lo, r_hi, t_lo, t_hi):
    return f"pair_rot_{int(r_lo):02d}-{int(r_hi):02d}_trans_{t_lo:.1f}-{t_hi:.1f}"


def save_pair(output_dir, img_path0, img_path1, pose0, pose1,
              rot_angle, trans_dist, frame_idx0, frame_idx1,
              rotate, img_ext, copy_images=True):
    os.makedirs(output_dir, exist_ok=True)

    if copy_images:
        for stem, src_path in [("image0", img_path0), ("image1", img_path1)]:
            img = Image.open(src_path)
            if rotate:
                img = img.rotate(-90, expand=True)
            img.save(os.path.join(output_dir, f"{stem}.{img_ext}"))
    else:
        for stem, src_path in [("image0", img_path0), ("image1", img_path1)]:
            dst = os.path.join(output_dir, f"{stem}.{img_ext}")
            if not os.path.exists(dst):
                os.symlink(src_path, dst)

    np.savetxt(os.path.join(output_dir, "image0.txt"), pose0, fmt="%.8f")
    np.savetxt(os.path.join(output_dir, "image1.txt"), pose1, fmt="%.8f")

    with open(os.path.join(output_dir, "image_pairs.txt"), "w") as f:
        f.write(f"image0.{img_ext} image1.{img_ext}")

    info = {
        "frame_idx_0":      frame_idx0,
        "frame_idx_1":      frame_idx1,
        "rotation_angle":   round(rot_angle, 3),
        "translation_dist": round(trans_dist, 4),
        "source_image_0":   str(img_path0),
        "source_image_1":   str(img_path1),
    }
    with open(os.path.join(output_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"  Saved -> {os.path.basename(output_dir)}  "
          f"(R={rot_angle:.1f}°, t={trans_dist:.3f}m, frames {frame_idx0}-{frame_idx1})")


def main(args):
    rotation_bins    = args.rotation_bins
    translation_bins = args.translation_bins

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.no_copy_images and args.rotate:
        print("WARNING: --no-copy-images uses symlinks; --rotate will be ignored. "
              "Images will point to the original (unrotated) files.")

    pose_dir  = Path(args.pose_dir)
    image_dir = Path(args.image_dir)

    pose_files = sorted(pose_dir.glob("*.txt"))
    img_files  = sorted(image_dir.glob(f"*.{args.img_ext}"))

    if not pose_files:
        raise FileNotFoundError(f"No pose .txt files in {pose_dir}")
    if len(pose_files) != len(img_files):
        raise ValueError(f"Pose count ({len(pose_files)}) != image count ({len(img_files)})")

    print(f"Found {len(pose_files)} frames.")
    print(f"Building pair candidates (max_frame_sep={args.max_frame_sep}, "
          f"min_frame_sep={args.min_frame_sep}) ...")

    poses = [np.loadtxt(f) for f in pose_files]
    N = len(poses)

    # Bin all valid pairs by (rot_bin, trans_bin)
    joint_bins = {}   # (rb, tb) -> list of (i, j, rot_angle, trans_dist)
    for i in range(N):
        for j in range(i + args.min_frame_sep,
                       min(N, i + 1 + args.max_frame_sep)):
            rot = rotation_angle(poses[i], poses[j])
            rb = rot_bin_index(rot, rotation_bins)
            if rb < 0:
                continue
            trans = translation_dist(poses[i], poses[j])
            tb = trans_bin_index(trans, translation_bins)
            if tb < 0:
                continue
            joint_bins.setdefault((rb, tb), []).append((i, j, rot, trans))

    n_rot  = len(rotation_bins) - 1
    n_trans = len(translation_bins) - 1

    print(f"\nBin summary (target {args.n_per_bin} per joint bin):")
    os.makedirs(args.output_dir, exist_ok=True)
    all_selected = []

    for rb in range(n_rot):
        r_lo, r_hi = rotation_bins[rb], rotation_bins[rb + 1]
        for tb in range(n_trans):
            t_lo, t_hi = translation_bins[tb], translation_bins[tb + 1]

            candidates = joint_bins.get((rb, tb), [])
            if not candidates:
                print(f"  R[{r_lo:.0f}-{r_hi:.0f}°) T[{t_lo:.1f}-{t_hi:.1f}m): no pairs")
                continue

            n = min(args.n_per_bin, len(candidates))
            chosen = random.sample(candidates, n)
            chosen.sort(key=lambda x: x[2])  # sort by rotation angle

            print(f"  R[{r_lo:.0f}-{r_hi:.0f}°) T[{t_lo:.1f}-{t_hi:.1f}m): "
                  f"{len(candidates):5d} candidates -> {len(chosen)} selected")

            label = joint_bin_label(r_lo, r_hi, t_lo, t_hi)
            for k, (i, j, rot, trans) in enumerate(chosen):
                out_dir = os.path.join(args.output_dir, f"{label}_{k+1}")
                save_pair(
                    out_dir,
                    img_files[i], img_files[j],
                    poses[i], poses[j],
                    rot, trans, i, j,
                    args.rotate, args.img_ext,
                    copy_images=not args.no_copy_images,
                )
                all_selected.append({
                    "pair_dir":         out_dir,
                    "rotation_angle":   round(rot, 3),
                    "translation_dist": round(trans, 4),
                    "frames":           [i, j],
                    "rot_bin":          f"{r_lo}-{r_hi}",
                    "trans_bin":        f"{t_lo:.1f}-{t_hi:.1f}",
                })

    summary_path = os.path.join(args.output_dir, "selected_pairs.json")
    with open(summary_path, "w") as f:
        json.dump(all_selected, f, indent=2)

    print(f"\nSelected {len(all_selected)} pairs total.")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pose-dir",       required=True,
                   help="Directory with per-frame pose .txt files")
    p.add_argument("--image-dir",      required=True,
                   help="Directory with per-frame images")
    p.add_argument("--output-dir",     required=True,
                   help="Root output directory for selected pairs")
    p.add_argument("--img-ext",        default="jpg",
                   help="Image extension: jpg (ADT) or png (KITTI)")
    p.add_argument("--rotate",         action="store_true",
                   help="Rotate images 90 deg CW (use for ADT)")
    p.add_argument("--max-frame-sep",  type=int, default=100,
                   help="Max frame index gap to consider for pairing")
    p.add_argument("--min-frame-sep",  type=int, default=3,
                   help="Min frame index gap (avoids near-identical frames)")
    p.add_argument("--n-per-bin",      type=int, default=50,
                   help="Number of pairs to select per rotation x translation joint bin")
    p.add_argument("--seed",           type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--no-copy-images", action="store_true",
                   help="Use symlinks instead of copying images (saves disk space; "
                        "--rotate is ignored when set)")
    p.add_argument("--rotation-bins",    nargs="+", type=float,
                   default=[0, 10, 20, 30, 40, 50],
                   help="Rotation bin edges in degrees (default: ADT 0 10 20 30 40 50; "
                        "KITTI example: 0 10 20 30 40 50)")
    p.add_argument("--translation-bins", nargs="+", type=float,
                   default=[0.0, 0.5, 1.0, 1.5, 2.0],
                   help="Translation bin edges in metres (default: ADT 0.0 0.5 1.0 1.5 2.0; "
                        "KITTI example: 0 5 10 20 50)")
    main(p.parse_args())
