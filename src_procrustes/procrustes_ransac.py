import numpy as np
from scipy.linalg import orthogonal_procrustes


def umeyama_sim3(src, dst):
    """
    Estimate Sim(3): scale s, rotation R, translation t such that
    dst ≈ s * R @ src[i] + t  for each point i.
    src, dst: (N, 3)
    Returns: s (float), R (3x3), t (3,)
    """
    mu_src = src.mean(0)
    mu_dst = dst.mean(0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    norm_src = np.linalg.norm(src_c)
    norm_dst = np.linalg.norm(dst_c)
    if norm_src == 0 or norm_dst == 0:
        raise ValueError("Degenerate point set (zero norm)")

    src_c = src_c / norm_src
    dst_c = dst_c / norm_dst

    R, s = orthogonal_procrustes(dst_c, src_c)
    scale = s * norm_dst / norm_src
    t = mu_dst - scale * (R @ mu_src)

    return scale, R, t


def ransac_sim3(pts_src, pts_dst,
                n_iter=1000,
                inlier_thresh=0.5,
                min_inliers=6,
                seed=42):
    """
    RANSAC wrapper around Umeyama Sim(3) alignment.

    pts_src, pts_dst : (N, 3) matched 3D point arrays (same units as
                        the distance_threshold used in the main pipeline,
                        i.e. metres for KITTI / ADT).
    inlier_thresh    : per-point residual threshold in the same units.
    Returns: s, R (3x3), t (3,), inlier_mask (bool, length N)
    """
    rng = np.random.default_rng(seed)
    N = len(pts_src)
    best_inliers = np.zeros(N, dtype=bool)
    best_s, best_R, best_t = None, None, None

    for _ in range(n_iter):
        idx = rng.choice(N, 3, replace=False)
        try:
            s, R, t = umeyama_sim3(pts_src[idx], pts_dst[idx])
        except ValueError:
            continue

        pts_aligned = s * (R @ pts_src.T).T + t
        residuals = np.linalg.norm(pts_aligned - pts_dst, axis=1)
        inliers = residuals < inlier_thresh

        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_s, best_R, best_t = s, R, t

    if best_s is None:
        raise RuntimeError("RANSAC failed to find any valid hypothesis")

    # Refit on full inlier set
    if best_inliers.sum() >= min_inliers:
        best_s, best_R, best_t = umeyama_sim3(
            pts_src[best_inliers], pts_dst[best_inliers]
        )

    print(f"RANSAC: {best_inliers.sum()}/{N} inliers")
    return best_s, best_R, best_t, best_inliers
