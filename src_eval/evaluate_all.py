"""
Evaluate and compare pose estimation methods.

Expected directory layout:
  <results_dir>/
    <method_name>/
      pair_0001.json    # {"R_est":..., "t_est":...}
      pair_0002.json
      ...
  <gt_dir>/
    pair_0001.json      # {"R_gt":..., "t_gt":...}
    pair_0002.json
    ...

Usage:
  python evaluate_all.py \\
      --gt-dir /path/to/gt \\
      --results-dir /path/to/results \\
      --methods ours ours_ransac pinhole dust3r dust3r_rect \\
      --output table.csv
"""

import argparse
import json
import os
import numpy as np


# ── Metric helpers ────────────────────────────────────────────────────────────

def rotation_error_deg(R_est, R_gt):
    """Geodesic rotation error in degrees."""
    R_rel = R_est.T @ R_gt
    cos = np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def translation_error_deg(t_est, t_gt):
    """Angular error between translation directions (degrees)."""
    n = np.linalg.norm(t_est) * np.linalg.norm(t_gt)
    if n < 1e-8:
        return 180.0
    cos = np.clip(np.dot(t_est, t_gt) / n, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def auc(errors, threshold):
    return float(np.mean(np.array(errors) < threshold))


def summarise(rot_errs, tra_errs):
    return {
        "R_mean":  float(np.mean(rot_errs)),
        "R_med":   float(np.median(rot_errs)),
        "t_mean":  float(np.mean(tra_errs)),
        "t_med":   float(np.median(tra_errs)),
        "AUC@5":   auc(rot_errs, 5),
        "AUC@10":  auc(rot_errs, 10),
        "AUC@20":  auc(rot_errs, 20),
        "n_pairs": len(rot_errs),
    }


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_result(path):
    with open(path) as f:
        d = json.load(f)
    R_key = "R_est" if "R_est" in d else "R_gt"
    t_key = "t_est" if "t_est" in d else "t_gt"
    R = np.array(d[R_key])
    t = np.array(d[t_key]).ravel()
    return R, t


def evaluate_method(method_dir, gt_dir):
    pair_files = sorted(
        f for f in os.listdir(method_dir) if f.endswith(".json")
    )
    rot_errs, tra_errs = [], []
    missing = 0
    for fname in pair_files:
        gt_path  = os.path.join(gt_dir, fname)
        est_path = os.path.join(method_dir, fname)
        if not os.path.exists(gt_path):
            missing += 1
            continue
        R_est, t_est = load_result(est_path)
        R_gt,  t_gt  = load_result(gt_path)
        rot_errs.append(rotation_error_deg(R_est, R_gt))
        tra_errs.append(translation_error_deg(t_est, t_gt))

    if missing:
        print(f"  WARNING: {missing} GT files missing")
    if not rot_errs:
        return None
    return summarise(rot_errs, tra_errs)


# ── Formatting ────────────────────────────────────────────────────────────────

COLS = ["R_mean", "R_med", "t_mean", "t_med", "AUC@5", "AUC@10", "AUC@20", "n_pairs"]
HEADERS = {
    "R_mean":  "R° mean", "R_med":  "R° med",
    "t_mean":  "t° mean", "t_med":  "t° med",
    "AUC@5":   "AUC@5°", "AUC@10": "AUC@10°", "AUC@20": "AUC@20°",
    "n_pairs": "N",
}

def fmt(v, key):
    if key == "n_pairs":
        return str(int(v))
    if key.startswith("AUC"):
        return f"{v*100:.1f}%"
    return f"{v:.2f}"


def print_table(rows):
    col_w = max(len(HEADERS[c]) for c in COLS)
    method_w = max(len(r[0]) for r in rows)

    header = f"{'Method':<{method_w}}  " + "  ".join(
        f"{HEADERS[c]:>{col_w}}" for c in COLS
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for method, stats in rows:
        if stats is None:
            print(f"{method:<{method_w}}  (no results)")
            continue
        row = f"{method:<{method_w}}  " + "  ".join(
            f"{fmt(stats[c], c):>{col_w}}" for c in COLS
        )
        print(row)
    print(sep)


def save_csv(rows, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + COLS)
        for method, stats in rows:
            if stats is None:
                continue
            w.writerow([method] + [stats[c] for c in COLS])
    print(f"Saved CSV to {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt-dir",      required=True,
                   help="Directory with ground-truth JSON files")
    p.add_argument("--results-dir", required=True,
                   help="Parent directory; each sub-directory is one method")
    p.add_argument("--methods",     nargs="+",
                   help="Method sub-directory names to evaluate "
                        "(default: all sub-directories)")
    p.add_argument("--output",      default=None,
                   help="Optional CSV output path")
    args = p.parse_args()

    if args.methods:
        methods = args.methods
    else:
        methods = sorted(
            d for d in os.listdir(args.results_dir)
            if os.path.isdir(os.path.join(args.results_dir, d))
        )

    rows = []
    for m in methods:
        method_dir = os.path.join(args.results_dir, m)
        if not os.path.isdir(method_dir):
            print(f"WARNING: {method_dir} not found, skipping")
            rows.append((m, None))
            continue
        print(f"Evaluating {m} ...")
        stats = evaluate_method(method_dir, args.gt_dir)
        rows.append((m, stats))

    print_table(rows)

    if args.output:
        save_csv(rows, args.output)


if __name__ == "__main__":
    main()
