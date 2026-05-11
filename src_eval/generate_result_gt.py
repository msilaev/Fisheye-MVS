#!/usr/bin/env python3
"""Generate result_gt.json for one pair directory.

The output format matches other method result JSONs:
{
  "R_est": [[...],[...],[...]],
  "t_est": [...],
  "scale_est": 1.0
}

For ADT (--rotate), images are rotated 90 deg CW before being fed to UniK3D,
so all methods (Procrustes, MADPose) output R/t in the *rotated* camera frame.
The GT relative pose (from raw cam-to-world poses) must be converted to the same
rotated frame so that errors are computed consistently:

    R_gt_rot = Rz_ccw90 @ R_gt_raw @ Rz_cw90
    t_gt_rot = Rz_ccw90 @ t_gt_raw

where Rz_ccw90 rotates 90 deg CCW around Z (= inverse of the 90 deg CW image
rotation applied to ADT images).
"""

import argparse
import json
from pathlib import Path

import numpy as np

# 90-degree rotation matrices around Z
_Rz_ccw90 = np.array([[0, -1, 0], [1,  0, 0], [0, 0, 1]], dtype=float)
_Rz_cw90  = np.array([[0,  1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam2w-0", required=True, help="Path to image0.txt (4x4 cam-to-world)")
    parser.add_argument("--cam2w-1", required=True, help="Path to image1.txt (4x4 cam-to-world)")
    parser.add_argument("--output", required=True, help="Output result_gt.json path")
    parser.add_argument("--rotate", action="store_true",
                        help="Apply 90 deg CW image-rotation frame correction (use for ADT)")
    args = parser.parse_args()

    t0 = np.loadtxt(args.cam2w_0)
    t1 = np.loadtxt(args.cam2w_1)
    rel = np.linalg.inv(t1) @ t0

    R = rel[:3, :3]
    t = rel[:3, 3]

    if args.rotate:
        R = _Rz_ccw90 @ R @ _Rz_cw90
        t = _Rz_ccw90 @ t

    out = {
        "R_est": R.tolist(),
        "t_est": t.tolist(),
        "scale_est": 1.0,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
