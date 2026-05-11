"""
Aggregate per-pair cloud_consistency.json files into a summary CSV.

Expected directory structure (same as compute_errors.py):
  <pairs_dir>/pair_rot_*/
      info.json               ← rotation_angle, translation_dist
  <results_dir>/pair_rot_*/
      cloud_consistency.json  ← output of compute_cloud_consistency.py

Usage:
  python aggregate_cloud_consistency.py \\
      --pairs-dir   /experiments/ADT/eval_full/test_pairs \\
      --results-dir /experiments/ADT/eval_full/results \\
      --output-csv  /experiments/ADT/eval_full/cloud_consistency_summary.csv \\
      --rotation-bins    0 10 20 30 40 50 \\
      --translation-bins 0.0 0.5 1.0 1.5 2.0
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

METHODS = [
    "GT",
    "Ours",
    "Ours+RANSAC",
    "MADPose",
    "MADPose+rect",
    "Fisheye-Emat",
    "Pinhole",
    "DUSt3R",
    "DUSt3R+rect",
]

METRICS = [
    ("chamfer_l1",      "ChamferL1"),
    ("rmse_symmetric",  "RMSE"),
    ("overlap_ratio",   "Overlap"),
]


def rot_bin_label(angle: float, bins: List[float]) -> str:
    for b in range(len(bins) - 1):
        if bins[b] <= angle < bins[b + 1]:
            return f"{int(bins[b]):02d}-{int(bins[b+1]):02d}"
    return "other"


def trans_bin_label(dist: float, bins: List[float]) -> str:
    for b in range(len(bins) - 1):
        lo, hi = bins[b], bins[b + 1]
        if lo <= dist < hi:
            return f"{lo:.1f}-{hi:.1f}"
    return "other"


def load_consistency(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def main(args):
    rotation_bins    = args.rotation_bins
    translation_bins = args.translation_bins

    pairs_dir   = Path(args.pairs_dir)
    results_dir = Path(args.results_dir)

    pair_dirs = sorted(
        p for p in pairs_dir.iterdir()
        if p.is_dir() and (p / "info.json").exists()
    )

    if not pair_dirs:
        raise FileNotFoundError(f"No pair directories found in {pairs_dir}")

    print(f"Found {len(pair_dirs)} pairs.")

    # rows[method] = list of dicts with bin labels + metric values
    rows: Dict[str, List[dict]] = defaultdict(list)

    n_loaded = 0
    n_missing = 0

    for pd in pair_dirs:
        info = json.loads((pd / "info.json").read_text())
        gt_rot   = info["rotation_angle"]
        gt_trans = info["translation_dist"]
        rb = rot_bin_label(gt_rot,   rotation_bins)
        tb = trans_bin_label(gt_trans, translation_bins)

        cc_path = results_dir / pd.name / "cloud_consistency.json"
        cc = load_consistency(cc_path)
        if cc is None:
            n_missing += 1
            continue

        n_loaded += 1
        for method in METHODS:
            entry = cc.get(method)
            if entry is None:
                continue
            row = {"pair": pd.name, "rot_bin": rb, "trans_bin": tb,
                   "gt_rot": gt_rot, "gt_trans": gt_trans}
            for key, _ in METRICS:
                row[key] = entry.get(key)
            rows[method].append(row)

    print(f"Loaded: {n_loaded}  Missing: {n_missing}")

    # ── Per-method, per-rotation-bin summary table (median) ──────────────────

    rot_labels = [
        f"{int(rotation_bins[b]):02d}-{int(rotation_bins[b+1]):02d}"
        for b in range(len(rotation_bins) - 1)
    ]

    for method in METHODS:
        method_rows = rows.get(method)
        if not method_rows:
            continue
        print(f"\n── {method} ──")
        for metric_key, metric_label in METRICS:
            print(f"  {metric_label} (median per rotation bin):")
            bin_vals = {r: [] for r in rot_labels}
            all_vals = []
            for row in method_rows:
                v = row.get(metric_key)
                if v is not None and row["rot_bin"] in bin_vals:
                    bin_vals[row["rot_bin"]].append(v)
                    all_vals.append(v)
            cells = []
            for rl in rot_labels:
                vals = bin_vals[rl]
                cells.append(f"{np.median(vals):.4f} (n={len(vals)})" if vals else "N/A")
            overall = f"{np.median(all_vals):.4f}" if all_vals else "N/A"
            header = "  | ".join(f"{rl:>10}" for rl in rot_labels) + f"  | {'all':>10}"
            values = "  | ".join(f"{c:>10}" for c in cells) + f"  | {overall:>10}"
            print(f"    {header}")
            print(f"    {values}")

    # ── Write flat CSV ────────────────────────────────────────────────────────

    if not args.output_csv:
        return 0

    fieldnames = ["method", "pair", "rot_bin", "trans_bin", "gt_rot", "gt_trans"]
    for key, _ in METRICS:
        fieldnames.append(key)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for method in METHODS:
            for row in rows.get(method, []):
                writer.writerow({"method": method, **row})

    print(f"\nSaved CSV: {out_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate cloud_consistency.json files into a summary CSV"
    )
    parser.add_argument("--pairs-dir",   required=True,
                        help="eval_full/test_pairs directory (contains pair_rot_*/info.json)")
    parser.add_argument("--results-dir", required=True,
                        help="eval_full/results directory (contains pair_rot_*/cloud_consistency.json)")
    parser.add_argument("--output-csv",  default=None,
                        help="Path to write the flat per-pair CSV summary")
    parser.add_argument("--rotation-bins",    nargs="+", type=float,
                        default=[0, 10, 20, 30, 40, 50])
    parser.add_argument("--translation-bins", nargs="+", type=float,
                        default=[0.0, 0.5, 1.0, 1.5, 2.0])
    raise SystemExit(main(parser.parse_args()))
