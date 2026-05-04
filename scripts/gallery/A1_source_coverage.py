"""A1: Raw data coverage per stream (sky maps + magnitude distributions).

What this shows:
- Sky Mollweide projections (Galactic coords) for Streams 1, 2, 3 showing
  geometric coverage (sampled to 50k points per stream for clarity).
- G-magnitude histograms per stream showing the magnitude distribution
  up to the G ≤ 17 cap.
- Kiel diagrams (full pre-cut cohort in greyscale, in-cut subset in viridis)
  with the stream-specific selection box overlaid in red.

What it reads:
- Stream 1: data/processed/pipeline1_features_stream1_kiel.parquet
  (Kiel-bounded RGB pool, logg ∈ [1.0, 3.5], Teff ∈ [4000, 5500])
- Stream 2: data/processed/pipeline1_features_stream2.parquet (post-cut)
- Stream 3: data/processed/pipeline1_features_stream3.parquet (post-cut)
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

from _common import (
    PALETTE,
    apply_style,
    radec_to_galactic_mollweide,
    sample_index,
    save_fig,
    style_galactic_mollweide,
)

OUT = REPO / "reports/gallery/A_raw_data"

# Kiel-diagram view bounds (display only — cohort cuts removed 2026-04-29).
KIEL_PLOT_X = (7800, 2800)
KIEL_PLOT_Y = (5.5, -0.2)


def _kiel_panel(
    ax,
    teff,
    logg,
    *,
    title,
    color,
    n_actual: int | None = None,
):
    """Hexbin Kiel diagram of the actual stream cohort, no Kiel bounding box.

    Bounding boxes were dropped on 2026-04-29: the OOD flags
    (ood_mahalanobis_score, ood_disagreement_flag, regime_b_flag,
    mode_ambiguous_flag) handle evolutionary-stage outliers downstream.
    """
    m = np.isfinite(teff) & np.isfinite(logg)
    if m.sum() > 100:
        h = ax.hexbin(
            teff[m],
            logg[m],
            gridsize=80,
            mincnt=5,
            cmap="viridis",
            bins="log",
            extent=[2800, 7800, -0.2, 5.5],
            alpha=0.95,
        )
        plt.colorbar(h, ax=ax, label="log10 N")
    ax.set_xlim(*KIEL_PLOT_X)
    ax.set_ylim(*KIEL_PLOT_Y)
    ax.set_xlabel(r"$T_\mathrm{eff}$ [K]", fontsize=8)
    ax.set_ylabel(r"$\log g$ [dex]", fontsize=8)
    n = int(n_actual if n_actual is not None else m.sum())
    ax.set_title(f"{title}\nn = {n:,}", fontsize=9, color=color)


def main() -> None:
    apply_style()

    # Stream 1 = Kiel-bounded training pool (logg ∈ [1.0, 3.5],
    # Teff ∈ [4000, 5500] K). The full APOGEE × Gaia cohort lives at
    # pipeline1_features_stream1.parquet; the production training pool is the
    # Kiel-masked sibling.
    s1 = pd.read_parquet(
        REPO / "data/processed/pipeline1_features_stream1_kiel.parquet",
        columns=["source_id", "ra_deg", "dec_deg", "g_mag", "teff_apogee", "logg_apogee"],
    )
    s3 = pd.read_parquet(
        REPO / "data/processed/pipeline1_features_stream3.parquet",
        columns=["source_id", "ra_deg", "dec_deg", "g_mag", "teff_andrae", "logg_andrae"],
    )
    # Stream 2
    s2_path = REPO / "data/processed/pipeline1_features_stream2.parquet"
    if s2_path.exists():
        s2 = pd.read_parquet(
            s2_path,
            columns=["source_id", "ra_deg", "dec_deg", "g_mag", "teff_andrae", "logg_andrae"],
        )
        s2 = s2.rename(columns={"teff_andrae": "teff_gspphot", "logg_andrae": "logg_gspphot"})
    else:
        s2 = None

    # Pre-cut reference teff/logg arrays (stub for now; full FITS load omitted for speed)
    np.array([])
    np.array([])
    len(s1)
    np.array([])
    np.array([])
    len(s3)

    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.32, width_ratios=[1, 1, 1, 1.05])

    # Row 1: Sky Mollweides
    streams_sky = [
        (s1, "Stream 1 — APOGEE × Gaia (training)", PALETTE["apogee"]),
        (s2, "Stream 2 — Hon+21 TESS × Gaia (asteroseismic)", "#9467bd"),
        (s3, "Stream 3 — Andrae+23 × Gaia DR3 XP (inference)", PALETTE["andrae_volume"]),
    ]
    for col, (df, name, color) in enumerate(streams_sky):
        ax = fig.add_subplot(gs[0, col], projection="mollweide")
        if df is None:
            ax.text(0, 0, "Stream 2 not built", ha="center", va="center", fontsize=10)
            ax.set_title(name + "\n(pending)", fontsize=9)
            continue
        idx = sample_index(len(df), 50_000, rng)
        x, y = radec_to_galactic_mollweide(
            df["ra_deg"].iloc[idx].to_numpy(), df["dec_deg"].iloc[idx].to_numpy()
        )
        ax.scatter(x, y, s=0.4, alpha=0.35, color=color, rasterized=True)
        style_galactic_mollweide(ax)
        ax.set_title(f"{name}\nn = {len(df):,}", fontsize=9)

    # G-magnitude histogram
    ax_g = fig.add_subplot(gs[0, 3])
    bins = np.linspace(6, 18, 61)
    ax_g.hist(
        s1["g_mag"].dropna(),
        bins=bins,
        color=PALETTE["apogee"],
        alpha=0.55,
        label=f"S1 ({len(s1):,})",
    )
    if s2 is not None:
        ax_g.hist(
            s2["g_mag"].dropna(),
            bins=bins,
            color="#9467bd",
            alpha=0.55,
            label=f"S2 ({len(s2):,})",
        )
    ax_g.hist(
        s3["g_mag"].dropna(),
        bins=bins,
        color=PALETTE["andrae_volume"],
        alpha=0.55,
        label=f"S3 ({len(s3):,})",
    )
    ax_g.axvline(17.0, color="red", lw=0.9, ls="--", label="G = 17 cap")
    ax_g.set_xlabel(r"Gaia $G$ [mag]")
    ax_g.set_ylabel("count")
    ax_g.set_title("Magnitude distribution per stream")
    ax_g.legend(fontsize=7, loc="upper left", frameon=True, framealpha=0.95)

    # Row 2: Kiel diagrams (no bounding box — OOD flags handle evolutionary outliers)
    ax_k1 = fig.add_subplot(gs[1, 0])
    _kiel_panel(
        ax_k1,
        s1["teff_apogee"].to_numpy(),
        s1["logg_apogee"].to_numpy(),
        title="Stream 1 — APOGEE DR19 (Kiel-masked RGB)",
        color=PALETTE["apogee"],
        n_actual=len(s1),
    )
    ax_k1.add_patch(
        plt.Rectangle(
            (4000, 1.0),
            5500 - 4000,
            3.5 - 1.0,
            fill=False,
            edgecolor="red",
            linewidth=1.2,
            linestyle="--",
        )
    )

    ax_k2 = fig.add_subplot(gs[1, 1])
    if s2 is not None:
        _kiel_panel(
            ax_k2,
            s2["teff_gspphot"].to_numpy(),
            s2["logg_gspphot"].to_numpy(),
            title="Stream 2 — Gaia GSP-Phot Kiel",
            color="#9467bd",
            n_actual=len(s2),
        )
    else:
        ax_k2.set_axis_off()

    ax_k3 = fig.add_subplot(gs[1, 2])
    _kiel_panel(
        ax_k3,
        s3["teff_andrae"].to_numpy(),
        s3["logg_andrae"].to_numpy(),
        title="Stream 3 — Andrae+23",
        color=PALETTE["andrae_volume"],
        n_actual=len(s3),
    )

    # Summary text panel
    ax_cut = fig.add_subplot(gs[1, 3])
    ax_cut.set_axis_off()
    txt = (
        "Per-stream cohort\n"
        "──────────────────\n"
        "S1: APOGEE × Gaia\n"
        "    flag_bad == 0\n"
        "    SNR > 70\n"
        "    [M/H] in [-2.0, 0.5]\n"
        "    + Kiel mask:\n"
        "      logg [1.0, 3.5]\n"
        "      Teff [4000, 5500]\n\n"
        "S2: Hon+2021 × Gaia\n"
        "    has_xp_continuous\n"
        "    Ye+2024 OK\n\n"
        "S3: Andrae+2023 × Gaia\n"
        "    Ye+2024 OK\n"
        "    (no Kiel mask —\n"
        "     inference target)\n"
    )
    ax_cut.text(
        0.0,
        1.0,
        txt,
        transform=ax_cut.transAxes,
        fontsize=8,
        ha="left",
        va="top",
        family="monospace",
    )

    n1 = len(s1)
    n2 = len(s2) if s2 is not None else 0
    n3 = len(s3)
    fig.suptitle(
        f"A1 — Raw data coverage per stream. "
        f"Stream 1 = Kiel-masked training pool ({n1 // 1000}k, "
        f"logg ∈ [1.0, 3.5], Teff ∈ [4000, 5500] K).  "
        f"Stream 2 = asteroseismic ({n2 // 1000}k post-cut).  "
        f"Stream 3 = inference ({n3 // 1000}k post-cut).",
        fontsize=10,
        y=0.98,
    )
    save_fig(fig, OUT / "A1_source_coverage", tight=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot A1: Source coverage per stream.")
    args = parser.parse_args()
    main()
