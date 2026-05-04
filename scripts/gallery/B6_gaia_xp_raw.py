"""B6: Raw Gaia XP coefficients before Ye+2024 correction.

What this shows:
- Rolling median and 16-84 percentile deviation of each Hermite coefficient
  amplitude across magnitude bands (G in [10,12], [12,14], [14,16], [16,18]).
- Two side-by-side panels (BP, RP).
- Each panel has 4 lines (one per mag bin) with shaded +/- 1sigma band.
- This reveals trends in coefficient amplitude with magnitude and order.

What it reads:
- data/processed/pipeline1_features_stream1_kiel.parquet
  (needs g_mag, bp_coef_norm_1..54, rp_coef_norm_1..54).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _common import apply_style, save_fig


def main() -> None:
    apply_style()

    parquet = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"{parquet} not found")

    # Read RAW Hermite coefficients (bp_coef_0..54, rp_coef_0..54) and divide
    # by c_0 per star — this matches the docstring framing of "raw with c_0
    # normalisation". The previously-used bp_coef_norm_* columns are
    # z-scored (mean 0, std 1) and contain exact zeros → log scale crash.
    cols = (
        ["g_mag"]
        + [f"bp_coef_{i}" for i in range(0, 55)]
        + [f"rp_coef_{i}" for i in range(0, 55)]
    )
    df_all = pd.read_parquet(parquet, columns=cols)

    mag_bins = [(10, 12), (12, 14), (14, 16), (16, 18)]
    bin_labels = ["G [10,12)", "G [12,14)", "G [14,16)", "G [16,18)"]
    colors_bin = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    linestyles = ["-", "--", ":", "-."]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    coef_idx = np.arange(1, 55)

    for band_idx, band in enumerate(["bp", "rp"]):
        ax = axes[band_idx]

        for bin_idx, (g_min, g_max) in enumerate(mag_bins):
            mask = (df_all["g_mag"] >= g_min) & (df_all["g_mag"] < g_max)
            if mask.sum() < 10:
                continue
            sub = df_all.loc[mask]
            c0_arr = sub[f"{band}_coef_0"].to_numpy(dtype=np.float64)
            coef_cols = [f"{band}_coef_{i}" for i in range(1, 55)]
            ci = sub[coef_cols].to_numpy(dtype=np.float64)
            # Drop rows with NaN c_0 or NaN c_i (Ye-failed stars).
            ok = np.isfinite(c0_arr) & (np.abs(c0_arr) > 0) & np.isfinite(ci).all(axis=1)
            if int(ok.sum()) < 10:
                continue
            ratio = np.abs(ci[ok] / c0_arr[ok, None])
            # Avoid exact zeros for the log scale (rare, but possible).
            ratio = np.where(ratio > 0, ratio, np.nan)
            median = np.nanmedian(ratio, axis=0)
            p16 = np.nanpercentile(ratio, 16, axis=0)
            p84 = np.nanpercentile(ratio, 84, axis=0)

            ax.fill_between(coef_idx, p16, p84, alpha=0.2, color=colors_bin[bin_idx])
            ax.plot(
                coef_idx, median,
                linestyle=linestyles[bin_idx], marker="o", markersize=3,
                color=colors_bin[bin_idx], alpha=0.85, linewidth=1.2,
                label=f"{bin_labels[bin_idx]} (n={int(ok.sum()):,})",
            )

        ax.set_xlabel("Hermite coefficient index (1-54)")
        ax.set_ylabel(r"$|c_i / c_0|$ (raw, c$_0$-normalised)")
        ax.set_yscale("log")
        ax.set_title(f"{band.upper()}: coefficient amplitude by magnitude bin")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2, which="both")

    fig.suptitle(
        "B6 — Stream 1: Raw Gaia XP Hermite coefficient amplitude trends\n"
        "(pre-Ye+2024, pre-normalisation; raw GaiaXPy output with per-band c_0 normalisation)",
        fontsize=11,
        fontweight="semibold",
    )
    save_fig(fig, REPO / "reports/gallery/B_preprocessing" / "B6_gaia_xp_raw", formats=("pdf", "png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot B6: raw Gaia XP coefficients.")
    args = parser.parse_args()
    main()
