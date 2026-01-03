import argparse
import numpy as np

RESIZE_LENGTH = 1600

def main():
    # Load arguments
    p = argparse.ArgumentParser()
    p.add_argument("--matches-file-npz", type=str, required=True)
    p.add_argument("--size_x", type=int, required=True)   # original width
    p.add_argument("--size_y", type=int, required=True)   # original height
    p.add_argument("--mkpts1-file", type=str, required=True)
    p.add_argument("--mkpts2-file", type=str, required=True)
    args = p.parse_args()

    # Load SuperGlue output
    data = np.load(args.matches_file_npz)
    matches = data["matches"]
    kpts0 = data["keypoints0"]
    kpts1 = data["keypoints1"]

    # Keep valid matches
    valid = matches > -1
    mkpts1 = kpts0[valid]
    mkpts2 = kpts1[matches[valid]]

    # Determine scaling used by SuperGlue
    max_dim = max(args.size_x, args.size_y)
    scale = RESIZE_LENGTH / max_dim
    print("SuperGlue resize scale:", scale)

    # Rescale back to original image coordinates
    mkpts1_orig = (mkpts1 / scale).astype(int)
    mkpts2_orig = (mkpts2 / scale).astype(int)

    # Save results
    np.save(args.mkpts1_file, mkpts1_orig)
    np.save(args.mkpts2_file, mkpts2_orig)

if __name__ == "__main__":
    main()
