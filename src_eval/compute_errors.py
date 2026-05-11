"""
Aggregate rotation and translation errors across all evaluated pairs.

Expected directory structure:
  <results_root>/
    pair_rot_00-10_1/
      result_procrustes.json
      result_pinhole.json
      result_dust3r.json
      ...

Pair directories must also contain image0.txt / image1.txt (GT cam-to-world)
and info.json (with rotation_angle, translation_dist).

Usage:
  python compute_errors.py \\
      --pairs-dir  /experiments/ADT/test_pairs_full \\
      --results-dir /experiments/ADT/results_full \\
      --output-csv  /experiments/ADT/errors.csv
"""

import argparse
import json
import numpy as np
import csv
from pathlib import Path

# 90 CW image rotation in camera 3D coords: p_rot = R_cw90 @ p_orig
# Derived from: rotating image 90 CW maps pixel (u,v) -> (H-1-v, u),
# which corresponds to (X,Y,Z)_orig -> (-Y,X,Z) in camera frame
R_CW90 = np.array([[0, -1, 0],
                   [1,  0, 0],
                   [0,  0, 1]], dtype=float)

METHODS = [
    ('Ours',           'result_procrustes.json'),
    ('Ours+RANSAC',    'result_procrustes_ransac.json'),
    ('MADPose',        'result_madpose.json'),
    ('MADPose+rect',   'result_madpose_rect.json'),
    ('Fisheye-Emat',   'result_fisheye_emat.json'),
    ('Pinhole',        'result_pinhole.json'),
    ('DUSt3R',         'result_dust3r.json'),
    ('DUSt3R+rect',    'result_dust3r_rect.json'),
]


def geodesic_deg(R1, R2):
    dR = R1 @ R2.T
    cos = np.clip((np.trace(dR) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos)))


def angle_deg(R):
    cos = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos)))


def rotation_error(R_est, R_gt, gt_is_corrected=False):
    """Geodesic rotation error.

    If gt_is_corrected (loaded from result_gt.json), do a direct comparison.
    Otherwise try both raw and 90-CW-rotated GT frames and take the minimum.
    """
    if gt_is_corrected:
        return geodesic_deg(R_est, R_gt)
    R_gt_rot = R_CW90 @ R_gt @ R_CW90.T
    return min(geodesic_deg(R_est, R_gt), geodesic_deg(R_est, R_gt_rot))


def translation_error(t_est, t_gt, gt_is_corrected=False):
    """
    Angular error (degrees) between translation directions. Scale-invariant.

    If gt_is_corrected (loaded from result_gt.json), do a direct comparison.
    Otherwise tries both raw and 90-CW-rotated GT coordinate frame conventions.
    """
    norm_est = np.linalg.norm(t_est)
    norm_gt  = np.linalg.norm(t_gt)
    if norm_est < 1e-10 or norm_gt < 1e-10:
        return 180.0
    n_est = t_est / norm_est
    n_gt  = t_gt / norm_gt
    err = float(np.degrees(np.arccos(np.clip(np.dot(n_est, n_gt), -1, 1))))
    if gt_is_corrected:
        return err
    n_gt_rot = R_CW90 @ n_gt
    err_rot = float(np.degrees(np.arccos(np.clip(np.dot(n_est, n_gt_rot), -1, 1))))
    return min(err, err_rot)


def rot_bin_label(angle, bins):
    for b in range(len(bins) - 1):
        if bins[b] <= angle < bins[b + 1]:
            return f"{int(bins[b]):02d}-{int(bins[b+1]):02d}"
    return "other"


def trans_bin_label(dist, bins):
    for b in range(len(bins) - 1):
        lo = bins[b]
        hi = bins[b + 1]
        if lo <= dist < hi:
            hi_str = f"{hi:.1f}" if hi != float('inf') else "inf"
            return f"{lo:.1f}-{hi_str}"
    return "other"


def load_gt_pose(pair_dir, results_dir=None):
    """Returns (R_gt, t_gt) for relative pose cam0->cam1.

    Prefers result_gt.json in results_dir (already frame-corrected) if available;
    falls back to recomputing from raw cam-to-world pose files.
    """
    if results_dir is not None:
        gt_path = results_dir / pair_dir.name / 'result_gt.json'
        if gt_path.exists():
            try:
                d = json.loads(gt_path.read_text())
                R = d.get('R_est')
                t = d.get('t_est')
                if R is not None and t is not None:
                    return np.array(R), np.array(t).ravel()
            except Exception:
                pass
    T0 = np.loadtxt(pair_dir / 'image0.txt')
    T1 = np.loadtxt(pair_dir / 'image1.txt')
    rel = np.linalg.inv(T1) @ T0
    return rel[:3, :3], rel[:3, 3]


def load_est(path):
    """Returns (R_est, t_est) or (None, None) if unavailable."""
    if not path.exists():
        return None, None
    try:
        d = json.loads(path.read_text())
        R_val = d.get('R_est')
        t_val = d.get('t_est')
        R = np.array(R_val) if R_val is not None else None
        t = np.array(t_val).ravel() if t_val is not None else None
        return R, t
    except Exception:
        return None, None


def _print_bin_table(rows, error_key, metric, rotation_bins, translation_bins):
    """Print a rotation-bin × translation-bin error table."""
    agg = np.mean if metric == 'mean' else np.median
    n_rot   = len(rotation_bins) - 1
    n_trans = len(translation_bins) - 1
    t_labels = [f"{translation_bins[b]:.1f}-{translation_bins[b+1]:.1f}"
                for b in range(n_trans)]
    col_w = 10
    row_hdr = 'R \\ T'
    header = f"{row_hdr:>10}" + "".join(f"  {tl:>{col_w}}" for tl in t_labels) + f"  {'all':>{col_w}}"
    print(header)
    print('-' * len(header))

    all_errs = []
    for rb in range(n_rot):
        r_lo, r_hi = rotation_bins[rb], rotation_bins[rb + 1]
        r_label = f"{int(r_lo):02d}-{int(r_hi):02d}"
        row_rows = [r for r in rows if r['rot_bin'] == r_label]
        line = f"{r_label:>10}"
        row_errs = []
        for tb in range(n_trans):
            t_label = t_labels[tb]
            cell_rows = [r for r in row_rows if r['trans_bin'] == t_label]
            errs = [r[error_key] for r in cell_rows if r[error_key] is not None]
            if errs:
                line += f"  {agg(errs):>{col_w}.1f}"
                row_errs.extend(errs)
                all_errs.extend(errs)
            else:
                n_cell = len(cell_rows)
                line += f"  {'N/A' if n_cell == 0 else '-':>{col_w}}"
        if row_errs:
            line += f"  {agg(row_errs):>{col_w}.1f}"
        else:
            line += f"  {'N/A':>{col_w}}"
        line += f"  (n={len(row_rows)})"
        print(line)

    print('-' * len(header))
    foot = f"{'all':>10}"
    for tb in range(n_trans):
        t_label = t_labels[tb]
        errs = [r[error_key] for r in rows
                if r['trans_bin'] == t_label and r[error_key] is not None]
        foot += f"  {agg(errs):>{col_w}.1f}" if errs else f"  {'N/A':>{col_w}}"
    foot += f"  {agg(all_errs):>{col_w}.1f}" if all_errs else f"  {'N/A':>{col_w}}"
    foot += f"  (n={len([r for r in rows if r[error_key] is not None])})"
    print(foot)


def main(args):
    rotation_bins    = args.rotation_bins
    translation_bins = args.translation_bins

    pairs_dir   = Path(args.pairs_dir)
    results_dir = Path(args.results_dir)

    pair_dirs = sorted(p for p in pairs_dir.iterdir()
                       if p.is_dir() and (p / 'info.json').exists())

    if not pair_dirs:
        raise FileNotFoundError(f"No pair directories found in {pairs_dir}")

    print(f"Found {len(pair_dirs)} pairs.")

    rows = []
    for pd in pair_dirs:
        info = json.loads((pd / 'info.json').read_text())
        gt_rot   = info['rotation_angle']
        gt_trans = info['translation_dist']

        try:
            R_gt, t_gt = load_gt_pose(pd, results_dir)
        except Exception as e:
            print(f"  WARN: cannot load GT pose for {pd.name}: {e}")
            continue

        # True when GT was loaded from result_gt.json (frame-corrected)
        gt_corrected = (results_dir / pd.name / 'result_gt.json').exists()

        res_dir = results_dir / pd.name
        row = {
            'pair':      pd.name,
            'gt_rot':    round(gt_rot, 2),
            'gt_trans':  round(gt_trans, 3),
            'rot_bin':   rot_bin_label(gt_rot, rotation_bins),
            'trans_bin': trans_bin_label(gt_trans, translation_bins),
        }
        for method_name, fname in METHODS:
            R_est, t_est = load_est(res_dir / fname)
            if R_est is not None:
                r_err = rotation_error(R_est, R_gt, gt_corrected)
                row[f'{method_name}_R_est'] = round(angle_deg(R_est), 2)
                row[f'{method_name}_R_err'] = round(r_err, 2)
            else:
                row[f'{method_name}_R_est'] = None
                row[f'{method_name}_R_err'] = None
            if t_est is not None and R_est is not None:
                t_err = translation_error(t_est, t_gt, gt_corrected)
                row[f'{method_name}_t_err'] = round(t_err, 2)
            else:
                row[f'{method_name}_t_err'] = None
        rows.append(row)

    if not rows:
        print("No results found.")
        return

    # ── Per-pair table ────────────────────────────────────────────────────────
    header = f"{'Pair':<25} {'GT R':>5} {'GT t':>5}"
    for name, _ in METHODS:
        header += f" | {name:>12} {'R_err':>6} {'t_err':>6}"
    print("\n" + header)
    print('-' * len(header))
    for r in rows:
        line = f"{r['pair']:<25} {r['gt_rot']:>5.1f} {r['gt_trans']:>5.2f}"
        for name, _ in METHODS:
            r_est = r[f'{name}_R_est']
            r_err = r[f'{name}_R_err']
            t_err = r[f'{name}_t_err']
            if r_err is not None:
                t_str = f"{t_err:>6.1f}" if t_err is not None else f"{'N/A':>6}"
                line += f" | {r_est:>12.1f} {r_err:>6.1f} {t_str}"
            else:
                line += f" | {'N/A':>12} {'N/A':>6} {'N/A':>6}"
        print(line)

    # ── Rotation error tables (by rot bin × trans bin) ────────────────────────
    for metric in ('mean', 'median'):
        for method_name, _ in METHODS:
            print(f"\n--- {method_name}: {metric} rotation error (deg) ---")
            _print_bin_table(rows, f'{method_name}_R_err', metric,
                             rotation_bins, translation_bins)

    # ── Translation error tables (by rot bin × trans bin) ────────────────────
    for metric in ('mean', 'median'):
        for method_name, _ in METHODS:
            print(f"\n--- {method_name}: {metric} translation direction error (deg) ---")
            _print_bin_table(rows, f'{method_name}_t_err', metric,
                             rotation_bins, translation_bins)

    # ── Overall summary ───────────────────────────────────────────────────────
    print("\n--- Overall (rotation error) ---")
    for name, _ in METHODS:
        errs = [r[f'{name}_R_err'] for r in rows if r[f'{name}_R_err'] is not None]
        if errs:
            print(f"  {name:<14}  mean={np.mean(errs):5.1f}  "
                  f"median={np.median(errs):5.1f}  "
                  f"max={np.max(errs):5.1f}  n={len(errs)}")

    print("\n--- Overall (translation direction error) ---")
    for name, _ in METHODS:
        errs = [r[f'{name}_t_err'] for r in rows if r[f'{name}_t_err'] is not None]
        if errs:
            print(f"  {name:<14}  mean={np.mean(errs):5.1f}  "
                  f"median={np.median(errs):5.1f}  "
                  f"max={np.max(errs):5.1f}  n={len(errs)}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if args.output_csv:
        fieldnames = list(rows[0].keys())
        with open(args.output_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV saved to: {args.output_csv}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pairs-dir',   required=True,
                   help='Directory containing pair_rot_* subdirs with info.json and pose txts')
    p.add_argument('--results-dir', required=True,
                   help='Directory containing per-pair results (result_procrustes.json etc.)')
    p.add_argument('--output-csv',  default=None,
                   help='Optional path to save results as CSV')
    p.add_argument('--rotation-bins',    nargs='+', type=float,
                   default=[0, 10, 20, 30, 40, 50],
                   help='Rotation bin edges in degrees')
    p.add_argument('--translation-bins', nargs='+', type=float,
                   default=[0.0, 0.5, 1.0, 1.5, 2.0],
                   help='Translation bin edges in metres (KITTI example: 0 5 10 20 50)')
    main(p.parse_args())
