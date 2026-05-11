import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from PIL import Image


def process_kitti360_sequence(
    kitti_root: str,
    sequence_name: str,
    output_root: str,
    camera: str = "image_00"
) -> int:
    """
    Extracts RGB images and cam2world poses from KITTI-360.
    """

    kitti_root = Path(kitti_root)
    output_root = Path(output_root)

    seq_path = kitti_root / "data_2d_raw" / sequence_name
    pose_path = kitti_root / "data_poses" / sequence_name / "cam0_to_world.txt"

    img_dir = seq_path / camera / "data_rect"

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not pose_path.exists():
        raise FileNotFoundError(f"Pose file not found: {pose_path}")

    # --- Load poses ---
    poses = np.loadtxt(pose_path)
    # poses shape: [N, 13] → frame_id + 12 pose values

    pose_dict = {}
    for row in poses:
        frame_id = int(row[0])
        T = np.eye(4)
        T[:3, :] = row[1:].reshape(3, 4)
        pose_dict[frame_id] = T

    # --- Setup output dirs ---
    seq_out = output_root / sequence_name
    img_out = seq_out / "images"
    pose_out = seq_out / "poses"

    img_out.mkdir(parents=True, exist_ok=True)
    pose_out.mkdir(parents=True, exist_ok=True)

    image_files = sorted(img_dir.glob("*.png"))

    total_frames = 0

    for img_path in tqdm(image_files, desc=sequence_name):
        frame_id = int(img_path.stem)

        if frame_id not in pose_dict:
            continue

        # --- Load image ---
        img = Image.open(img_path).convert("RGB")

        # --- Get pose ---
        cam2world = pose_dict[frame_id]

        frame_name = f"{sequence_name}_frame{frame_id:06d}"

        img.save(img_out / f"{frame_name}.jpg")
        np.savetxt(pose_out / f"{frame_name}.txt", cam2world, fmt="%.8f")

        total_frames += 1

    print(f"Finished {sequence_name}: {total_frames} frames")
    return total_frames


def main():
    parser = argparse.ArgumentParser(
        description="Extract KITTI-360 images and camera poses"
    )
    parser.add_argument("--kitti_root", type=str, required=True)
    parser.add_argument("--sequence", type=str, required=True,
                        help="e.g. 2013_05_28_drive_0000_sync")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--camera", type=str, default="image_00")

    args = parser.parse_args()

    process_kitti360_sequence(
        kitti_root=args.kitti_root,
        sequence_name=args.sequence,
        output_root=args.output,
        camera=args.camera
    )


if __name__ == "__main__":
    main()
