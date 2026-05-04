"""B7: Ye+2024 NN flux correction — real Stream-1 flag distribution.

What this shows (real data only):
- Histogram of the Ye+2024 correction flag (0=OK, 1=minor, 2=major,
  3=catastrophic) on the Stream-1 (APOGEE × Gaia DR3) feature parquet.
- Per-flag G-magnitude distribution: how does Ye-failure rate vary with
  apparent brightness?
- Side panel: legend explaining the flag categories and convergence-residual
  thresholds Ye+2024 use.

What it reads:
- data/processed/pipeline1_features_stream1_kiel.parquet (columns: g_mag,
  ye2024_flag if present; xp_fit_flag_residual_high as fallback).

The Ye+2024 correction removes instrumental wavelength-dependent flux
systematics (NOT extinction, NOT colour). Per-star raw vs corrected XP
sampled-flux comparisons require the corrected_flux array (large list<float>),
which lives in data/interim/xp_sampled_corrected.parquet — see notebooks for
that view; this gallery script only summarises the flag distribution.
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


def _resolve_flag_column(df: pd.DataFrame) -> str:
    """The Stream-1 feature parquet may carry the Ye flag under different
    names depending on emit-script version. Pick the first one present."""
    for cand in ("ye2024_flag", "ye_flag", "xp_fit_flag_residual_high"):
        if cand in df.columns:
            return cand
    raise KeyError("no Ye-style flag column found in feature parquet")


def main() -> None:
    apply_style()

    parquet = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"{parquet} not found")

    import pyarrow.parquet as pq

    schema = pq.ParquetFile(parquet).schema_arrow
    schema_df = pd.DataFrame({c.name: pd.Series(dtype="float64") for c in schema}).head(0)
    flag_col = _resolve_flag_column(schema_df)
    df = pd.read_parquet(parquet, columns=["g_mag", flag_col])
    df = df.rename(columns={flag_col: "flag"})
    df = df[df["g_mag"].notna() & df["flag"].notna()].copy()
    df["flag"] = df["flag"].astype(int)

    fig = plt.figure(figsize=(13, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.6], wspace=0.35)
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_g = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[0, 2])
    ax_legend.axis("off")

    # Panel 1: flag count histogram (log y)
    counts = df["flag"].value_counts().sort_index()
    flag_labels = {0: "OK", 1: "minor", 2: "major", 3: "catastrophic"}
    flag_colors = {0: "#2ca02c", 1: "#ff7f0e", 2: "#d62728", 3: "#8b0000"}
    bar_x = list(counts.index)
    bar_y = counts.values
    bars = ax_hist.bar(
        bar_x,
        bar_y,
        color=[flag_colors.get(f, "#1f77b4") for f in bar_x],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
    )
    ax_hist.set_xticks(bar_x)
    ax_hist.set_xticklabels([flag_labels.get(f, str(f)) for f in bar_x])
    ax_hist.set_ylabel("number of stars (log)")
    ax_hist.set_yscale("log")
    ax_hist.set_title(f"Ye+2024 flag distribution (Stream 1, n={len(df):,})")
    for bar, count in zip(bars, bar_y):
        ax_hist.text(
            bar.get_x() + bar.get_width() / 2,
            count,
            f"{int(count):,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Panel 2: per-flag G-mag distribution
    bins_g = np.linspace(df["g_mag"].min(), df["g_mag"].max(), 40)
    for f in sorted(counts.index):
        sub = df.loc[df["flag"] == f, "g_mag"]
        ax_g.hist(
            sub,
            bins=bins_g,
            alpha=0.55,
            density=True,
            color=flag_colors.get(f, "#1f77b4"),
            label=f"{flag_labels.get(f, str(f))} (n={len(sub):,})",
        )
    ax_g.set_xlabel(r"$G$ [mag]")
    ax_g.set_ylabel("density")
    ax_g.set_title("Per-flag G distribution")
    ax_g.legend(fontsize=8, loc="best")
    ax_g.grid(True, alpha=0.25)

    # Panel 3: legend / explanation
    legend_text = (
        "Ye+2024 NN convergence residuals\n"
        "(relative flux RMS):\n"
        "  OK             <1.5%\n"
        "  minor       1.5 - 3% (usable)\n"
        "  major       3 - 5%   (suspect)\n"
        "  catastrophic > 5%   (rejected)\n"
        "\n"
        "Correction removes instrumental\n"
        "wavelength-dependent systematics\n"
        "(330 - 1050 nm, BP+RP)\n"
        "and is NOT an extinction correction."
    )
    ax_legend.text(
        0.0,
        1.0,
        legend_text,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.85),
    )

    fig.suptitle(
        "B7 — Stream 1: Ye+2024 NN flux correction flag overview (real data)",
        fontsize=11,
        fontweight="semibold",
    )
    save_fig(
        fig, REPO / "reports/gallery/B_preprocessing" / "B7_ye_correction", formats=("pdf", "png")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot B7: Ye+2024 flag overview.")
    args = parser.parse_args()
    main()
