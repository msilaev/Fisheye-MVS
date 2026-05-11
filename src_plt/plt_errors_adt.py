"""
Plot rotation and translation error heatmaps from the metrics results JSON.

Usage:
  python src_plt/plt_errors_adt.py \\
      --sample-t-R-test-db-path /path/sample_t_R_pair_dataset.json \\
      --fig-path-rot    results/rot_error_adt.png \\
      --fig-path-transl results/transl_error_adt.png \\
      --csv-path        results/errors_adt.csv
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

TRANSLATION_BINS = [0.0, 0.5, 1.0, 1.5, 2.0]
ROTATION_BINS    = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]

FIG_WIDTH      = 3.3
FIG_HEIGHT     = 3.0
LABEL_FONTSIZE = 12
TITLE_FONTSIZE = 12
CBAR_FONTSIZE  = 12
TICK_FONTSIZE  = 12


def load_results(path):
    with open(path) as f:
        test_set = json.load(f)
    dR_gt, dt_gt, dR_err, dt_err = [], [], [], []
    for entry in test_set:
        if entry.get("rotation_error_est") is None:
            continue
        dR_gt.append(abs(entry["rotation_angle"]))
        dt_gt.append(abs(entry["translation_dist"]))
        dR_err.append(entry["rotation_error_est"])
        dt_err.append(entry["translation_error_est"])
    print(f"Loaded {len(dR_gt)} evaluated pairs from {path}")
    return (np.array(dR_gt),  np.array(dt_gt),
            np.array(dR_err), np.array(dt_err))


def compute_median_heatmaps(dR_gt, dt_gt, dR_err, dt_err):
    nR = len(ROTATION_BINS) - 1
    nT = len(TRANSLATION_BINS) - 1
    buckets_rot   = [[[] for _ in range(nT)] for _ in range(nR)]
    buckets_trans = [[[] for _ in range(nT)] for _ in range(nR)]

    for i in range(len(dR_err)):
        r = int(np.digitize(dR_gt[i], ROTATION_BINS)) - 1
        t = int(np.digitize(dt_gt[i], TRANSLATION_BINS)) - 1
        if 0 <= r < nR and 0 <= t < nT:
            buckets_rot[r][t].append(dR_err[i])
            buckets_trans[r][t].append(dt_err[i])

    med_rot   = np.full((nR, nT), np.nan)
    med_trans = np.full((nR, nT), np.nan)
    for r in range(nR):
        for t in range(nT):
            if buckets_rot[r][t]:
                med_rot[r, t]   = np.median(buckets_rot[r][t])
                med_trans[r, t] = np.median(buckets_trans[r][t])
    return med_rot, med_trans


def save_heatmap_figure(matrix, title, cbar_label, fig_path):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    im = ax.imshow(
        matrix.T, origin="lower", aspect="auto",
        extent=[ROTATION_BINS[0], ROTATION_BINS[-1],
                TRANSLATION_BINS[0], TRANSLATION_BINS[-1]]
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=CBAR_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.set_xlabel("|Rot. GT| (deg)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("|Trans. GT| (m)", fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    fig.tight_layout()
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved {fig_path}")


def save_csv(csv_path, med_rot, med_trans):
    rows = []
    for r in range(med_rot.shape[0]):
        for t in range(med_rot.shape[1]):
            rows.append({
                "rot_bin_min":        ROTATION_BINS[r],
                "rot_bin_max":        ROTATION_BINS[r + 1],
                "trans_bin_min":      TRANSLATION_BINS[t],
                "trans_bin_max":      TRANSLATION_BINS[t + 1],
                "median_rot_err_deg": med_rot[r, t],
                "median_trans_err_m": med_trans[r, t],
            })
    df = pd.DataFrame(rows)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    return df


def print_summary(med_rot, med_trans):
    valid_r = med_rot[~np.isnan(med_rot)]
    valid_t = med_trans[~np.isnan(med_trans)]
    print(f"\n--- Summary ---")
    print(f"Median rotation error:    {np.nanmedian(valid_r):.2f}°  (mean {np.nanmean(valid_r):.2f}°)")
    print(f"Median translation error: {np.nanmedian(valid_t):.4f}m (mean {np.nanmean(valid_t):.4f}m)")


def main(args):
    dR_gt, dt_gt, dR_err, dt_err = load_results(args.sample_t_R_test_db_path)
    med_rot, med_trans = compute_median_heatmaps(dR_gt, dt_gt, dR_err, dt_err)

    save_heatmap_figure(med_rot,   "Rotation error",    "Median rot. error (deg)", args.fig_path_rot)
    save_heatmap_figure(med_trans, "Translation error", "Median trans. error (m)", args.fig_path_transl)
    save_csv(args.csv_path, med_rot, med_trans)
    print_summary(med_rot, med_trans)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample-t-R-test-db-path", required=True)
    p.add_argument("--fig-path-rot",    required=True)
    p.add_argument("--fig-path-transl", required=True)
    p.add_argument("--csv-path",        required=True)
    main(p.parse_args())
