import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mkpts1", type=str, required=True,
                        help="SuperGlue points for image 1")
    parser.add_argument("--mkpts2", type=str, required=True,
                        help="SuperGlue points for image 2")

    parser.add_argument("--mkpts_lightglue1", type=str, required=True,
                        help="LightGlue points for image 1")
    parser.add_argument("--mkpts_lightglue2", type=str, required=True,
                        help="LightGlue points for image 2")

    args = parser.parse_args()

    # Load all point sets
    mkpts1 = np.load(args.mkpts1)                # (N,2)
    mkpts2 = np.load(args.mkpts2)                # (N,2)
    lgpts1 = np.load(args.mkpts_lightglue1)      # (M,2)
    lgpts2 = np.load(args.mkpts_lightglue2)      # (M,2)

    print("SuperGlue mkpts1:", mkpts1.shape)
    print("SuperGlue mkpts2:", mkpts2.shape)
    print("LightGlue mkpts1:", lgpts1.shape)
    print("LightGlue mkpts2:", lgpts2.shape)

    # Plot SuperGlue vs LightGlue (image 1)
    plt.figure(figsize=(10, 8))
    plt.scatter(mkpts1[:, 0], mkpts1[:, 1], s=8, c='red', label="SuperGlue", alpha=0.6)
    plt.scatter(lgpts1[:, 0], lgpts1[:, 1], s=8, c='blue', label="LightGlue", alpha=0.6)
    plt.gca().invert_yaxis()
    plt.title("Matched Keypoints on Image 1")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")

    # Plot SuperGlue vs LightGlue (image 2)
    plt.figure(figsize=(10, 8))
    plt.scatter(mkpts2[:, 0], mkpts2[:, 1], s=8, c='red', label="SuperGlue", alpha=0.6)
    plt.scatter(lgpts2[:, 0], lgpts2[:, 1], s=8, c='blue', label="LightGlue", alpha=0.6)
    plt.gca().invert_yaxis()
    plt.title("Matched Keypoints on Image 2")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()


if __name__ == "__main__":
    main()
