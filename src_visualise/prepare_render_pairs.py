#!/usr/bin/env python3
"""Resolve or prepare render pairs for visualisation.

Input pair specs can be either:
- an existing pair directory name, e.g. ``pair_rot_10-20_trans_5.0-10.0_4``
- or an explicit frame request, e.g. ``frames:0000006360|0000006367``

For each invocation, the script prints exactly one TSV line to stdout:
    <pair_name>\t<pair_dir>\t<results_dir>

If an exact pre-generated pair already exists anywhere under the relevant remote
experiment directories, it is reused so the corresponding method result JSONs
can also be reused. Otherwise a custom pair directory is created on demand from
``dataset/images`` + ``dataset/poses`` for the matching experiment/sequence.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def frame_stem(value: str) -> str:
    return Path(str(value)).stem


def angle_between(R1: np.ndarray, R2: np.ndarray) -> float:
    cos = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def relative_pose(T0: np.ndarray, T1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T_rel = np.linalg.inv(T1) @ T0
    return T_rel[:3, :3], T_rel[:3, 3]


def pair_name_from_frames(frame0: str, frame1: str) -> str:
    def clean(text: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)

    return f"pair_custom_{clean(frame0)}__{clean(frame1)}"


def iter_experiment_dirs(dataset: str, experiments_root: Path, default_pairs_dir: Path):
    seen: set[Path] = set()

    try:
        default_exp_dir = default_pairs_dir.resolve().parents[1]
    except (IndexError, OSError):
        default_exp_dir = None

    if default_exp_dir and default_exp_dir.exists():
        seen.add(default_exp_dir)
        yield default_exp_dir

    if dataset == "kitti":
        candidate = experiments_root / "KITTI-360"
        if candidate.exists():
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield candidate
    else:
        for candidate in sorted(experiments_root.glob("ADT*")):
            if not candidate.is_dir():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield candidate


def find_exact_pair(experiments: list[Path], wanted: set[str]) -> tuple[str, Path, Path] | None:
    for exp_dir in experiments:
        pairs_dir = exp_dir / "eval_full" / "test_pairs"
        if not pairs_dir.exists():
            continue

        for info_path in sorted(pairs_dir.glob("*/info.json")):
            try:
                info = json.loads(info_path.read_text())
            except (OSError, ValueError, json.JSONDecodeError):
                continue

            found = {
                frame_stem(info.get("source_image_0", "")),
                frame_stem(info.get("source_image_1", "")),
            }
            if found == wanted:
                pair_dir = info_path.parent
                results_dir = exp_dir / "eval_full" / "results" / pair_dir.name
                print(
                    f"[INFO] Reusing existing pair {pair_dir.name} from {exp_dir.name}",
                    file=sys.stderr,
                )
                return pair_dir.name, pair_dir, results_dir
    return None


def dataset_has_frames(dataset_dir: Path, frame0: str, frame1: str, image_ext: str) -> bool:
    image_dir = dataset_dir / "images"
    pose_dir = dataset_dir / "poses"
    needed = [
        image_dir / f"{frame0}.{image_ext}",
        image_dir / f"{frame1}.{image_ext}",
        pose_dir / f"{frame0}.txt",
        pose_dir / f"{frame1}.txt",
    ]
    return all(path.exists() for path in needed)


def find_source_dataset(
    dataset: str,
    experiments: list[Path],
    shared_datasets_root: Path,
    frame0: str,
    frame1: str,
    image_ext: str,
) -> tuple[Path, Path | None] | None:
    for exp_dir in experiments:
        dataset_dir = exp_dir / "dataset"
        if dataset_has_frames(dataset_dir, frame0, frame1, image_ext):
            return dataset_dir, exp_dir

    if dataset == "kitti":
        search_roots = [shared_datasets_root / "KITTI-360"]
    else:
        search_roots = [shared_datasets_root / "ADT"]

    for root in search_roots:
        if not root.exists():
            continue
        if dataset_has_frames(root, frame0, frame1, image_ext):
            return root, None
        for seq_dir in sorted(root.glob("*")):
            if seq_dir.is_dir() and dataset_has_frames(seq_dir, frame0, frame1, image_ext):
                return seq_dir, None

    return None


def copy_or_rotate(src: Path, dst: Path, rotate: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if rotate:
        img = Image.open(src)
        img = img.rotate(-90, expand=True)
        img.save(dst)
    else:
        shutil.copy2(src, dst)


# 90° CW image rotation maps camera axes: X→Y, Y→-X, Z→Z
# i.e. p_rotated = Rz_cw90 @ p_original
Rz_cw90 = np.array([[ 0, 1, 0],
                     [-1, 0, 0],
                     [ 0, 0, 1]], dtype=float)


def rotate_cam2world(T: np.ndarray) -> np.ndarray:
    """Adjust a cam-to-world 4x4 pose for a 90° CW image rotation.

    When the image is rotated 90° CW before being fed to UniK3D, the camera
    coordinate frame rotates accordingly. The new cam-to-world pose is:
        R_new = R_old @ Rz_cw90.T   (rotate camera frame)
        t_new = t_old               (translation unchanged)
    """
    T_new = T.copy()
    T_new[:3, :3] = T[:3, :3] @ Rz_cw90.T
    return T_new


def build_custom_pair(
    pair_dir: Path,
    dataset_dir: Path,
    frame0: str,
    frame1: str,
    image_ext: str,
    rotate: bool,
) -> None:
    pair_dir.mkdir(parents=True, exist_ok=True)

    img_dir = dataset_dir / "images"
    pose_dir = dataset_dir / "poses"

    img0_src = img_dir / f"{frame0}.{image_ext}"
    img1_src = img_dir / f"{frame1}.{image_ext}"
    pose0_src = pose_dir / f"{frame0}.txt"
    pose1_src = pose_dir / f"{frame1}.txt"

    copy_or_rotate(img0_src, pair_dir / f"image0.{image_ext}", rotate)
    copy_or_rotate(img1_src, pair_dir / f"image1.{image_ext}", rotate)

    T0 = np.loadtxt(pose0_src)
    T1 = np.loadtxt(pose1_src)
    # Poses are always saved as raw cam-to-world (same convention as standard test_pairs).
    # The image rotation (for UniK3D) does not affect the GT pose convention used by
    # procrustes / MADPose — those methods work in the rotated image space and their
    # output R aligns with the raw GT poses, matching what standard pairs store.
    np.savetxt(pair_dir / "image0.txt", T0)
    np.savetxt(pair_dir / "image1.txt", T1)

    with open(pair_dir / "image_pairs.txt", "w", encoding="utf-8") as f:
        f.write(f"image0.{image_ext} image1.{image_ext}")

    R_rel, t_rel = relative_pose(T0, T1)

    info = {
        "frame_idx_0": frame0,
        "frame_idx_1": frame1,
        "rotation_angle": round(angle_between(R_rel, np.eye(3)), 3),
        "translation_dist": round(float(np.linalg.norm(t_rel)), 4),
        "source_image_0": str(img0_src),
        "source_image_1": str(img1_src),
        "source_pose_0": str(pose0_src),
        "source_pose_1": str(pose1_src),
        "generated_from_frames": True,
    }
    (pair_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve or prepare render pair directories")
    parser.add_argument("--dataset", required=True, choices=["adt", "kitti"])
    parser.add_argument("--pair-spec", required=True)
    parser.add_argument("--default-pairs-dir", required=True)
    parser.add_argument("--default-results-dir", required=True)
    parser.add_argument("--output-root", required=True,
                        help="Where to place generated custom pair folders")
    parser.add_argument("--experiments-root", required=True)
    parser.add_argument("--shared-datasets-root", required=True)
    parser.add_argument("--image-ext", default="jpg")
    parser.add_argument("--rotate", action="store_true",
                        help="Rotate copied images 90 degrees clockwise (ADT)")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Delete and rebuild custom pair dir even if it already exists")
    args = parser.parse_args()

    pair_spec = args.pair_spec.strip()
    default_pairs_dir = Path(args.default_pairs_dir)
    default_results_dir = Path(args.default_results_dir)
    experiments_root = Path(args.experiments_root)
    shared_datasets_root = Path(args.shared_datasets_root)

    if not pair_spec.startswith("frames:"):
        pair_name = pair_spec
        print(f"{pair_name}\t{default_pairs_dir / pair_name}\t{default_results_dir / pair_name}")
        return 0

    spec = pair_spec[len("frames:"):]
    if "|" not in spec:
        print(f"[ERROR] Invalid frame spec: {pair_spec}", file=sys.stderr)
        return 1

    raw0, raw1 = spec.split("|", 1)
    frame0 = frame_stem(raw0)
    frame1 = frame_stem(raw1)
    wanted = {frame0, frame1}

    experiments = list(iter_experiment_dirs(args.dataset, experiments_root, default_pairs_dir))

    exact = find_exact_pair(experiments, wanted)
    if exact is not None:
        pair_name, pair_dir, results_dir = exact
        print(f"{pair_name}\t{pair_dir}\t{results_dir}")
        return 0

    dataset_match = find_source_dataset(
        args.dataset,
        experiments,
        shared_datasets_root,
        frame0,
        frame1,
        args.image_ext,
    )
    if dataset_match is None:
        print(
            f"[ERROR] Could not find both frames in any dataset directory: {frame0}, {frame1}",
            file=sys.stderr,
        )
        return 1

    dataset_dir, exp_dir = dataset_match
    pair_name = pair_name_from_frames(frame0, frame1)
    pair_dir = Path(args.output_root) / pair_name

    needs_build = (
        args.force_rebuild
        or not (pair_dir / f"image0.{args.image_ext}").exists()
        or not (pair_dir / "image0.txt").exists()
    )
    if needs_build:
        if args.force_rebuild and pair_dir.exists():
            shutil.rmtree(pair_dir)
            print(f"[INFO] Removed stale custom pair dir {pair_name}", file=sys.stderr)
        build_custom_pair(pair_dir, dataset_dir, frame0, frame1, args.image_ext, args.rotate)
        print(f"[INFO] Prepared custom pair {pair_name} from {dataset_dir}", file=sys.stderr)
    else:
        print(f"[INFO] Reusing prepared custom pair {pair_name}", file=sys.stderr)

    if exp_dir is not None:
        results_dir = exp_dir / "eval_full" / "results" / pair_name
    else:
        results_dir = default_results_dir / pair_name

    print(f"{pair_name}\t{pair_dir}\t{results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
