"""Stage 00: source coverage of all three streams.

What the deploy did:
- Stream 1: ``data.ingest_stream1`` joined APOGEE DR19 (raw 1.10 M stars
  before any cut) × Gaia DR3 → 324 k after RGB+RC selection (training pool).
- Stream 2: ``data.ingest_stream2`` joined Hon+2021 TESS × TIC v8.2 ×
  Gaia DR3 → 158 k asteroseismic giants.
- Stream 3: ``data.ingest_stream3`` materialised the Andrae+2023 RGB cohort
  × Gaia DR3 XP → 614 k inference stars.

What we plot, 2 × 4 layout:
- Row 1 — sky Mollweide of Streams 1 / 2 / 3 (Galactic coords) + G-mag
  histogram with a G = 17 reference line (the Pipeline-1 invariant cap).
- Row 2 — Kiel-equivalent diagrams. The FULL pre-cut cohort is shown in
  greyscale; the in-cut subset (Teff ∈ [3500, 6500] K, log g ∈ [0, 3.8]
  dex) is overlaid in colour. A red dashed bounding box outlines the cut
  region. The text panel on the right explains the cut.

Why the cut: Pipeline 1 trains and predicts on RGB + RC giants only. The
log g < 3.8 dex cap is set by Mészáros+2025 Table 3 polynomial validity
(the [X/M] corrections were calibrated only for RGB giants and the
embedded RC subset). CLAUDE.md hard rule #5 additionally caps inference
at G ≤ 17 (XP-native regime).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
from _common import (apply_style, save_fig, radec_to_galactic_mollweide,
                     style_galactic_mollweide, sample_index, PALETTE)

OUT = REPO / "reports/gallery/00_source_coverage"
S1 = REPO / "data/processed/pipeline1_features_stream1.parquet"
S2_FEAT = REPO / "data/processed/pipeline1_features_stream2.parquet"
S2_RAW = REPO / "data/interim/stream2_tess_gaia.parquet"
S3 = REPO / "data/processed/pipeline1_features_stream3.parquet"

# Raw pre-cut cohorts.
S1_RAW_FITS = REPO / "data/raw/apogee_dr19/astraAllStarASPCAP-0.6.0.fits.gz"
S3_RAW = REPO / "data/raw/andrae2023/andrae2023_rgb.parquet"

# Per-stream cuts ACTUALLY APPLIED by the build scripts:
# - Stream 1 (build_pipeline1_features_stream1.py): RGB scope cut Teff ∈
#   [4000, 5500] K AND log g ∈ [1.0, 3.5] (tighter than Mészáros validity;
#   matches APOGEE [C/N]-age calibration scope).
# - Stream 2 (build_pipeline1_features_stream2.py): Mészáros+2025 validity
#   Teff ∈ [3500, 6500] K AND log g ∈ [0.0, 3.8].
# - Stream 3 (Andrae+2023 published cohort): RGB sample as defined by Andrae,
#   plus Mészáros+2025 validity at inference time.
S1_CUT_TEFF = (4000.0, 5500.0)
S1_CUT_LOGG = (1.0, 3.5)
S2_CUT_TEFF = (3500.0, 6500.0)
S2_CUT_LOGG = (0.0, 3.8)
S3_CUT_TEFF = (3500.0, 6500.0)
S3_CUT_LOGG = (0.0, 3.8)
KIEL_PLOT_X = (7800, 2800)  # x-axis range in plot (Teff decreasing rightward)
KIEL_PLOT_Y = (5.5, -0.2)   # y-axis range in plot (log g decreasing upward)


def _kiel_panel(ax, teff_full, logg_full, teff_in, logg_in, *, title, color,
                cut_teff: tuple[float, float], cut_logg: tuple[float, float],
                n_full_actual: int | None = None, n_in_actual: int | None = None):
    """Plot the full pre-cut Kiel in greyscale + the in-cut subset in color +
    the red-dashed selection box drawn at the STREAM-SPECIFIC cut (passed via
    cut_teff/cut_logg) — different cuts for S1 vs S2/S3.
    ``n_full_actual`` / ``n_in_actual`` are the real cohort sizes used in the
    legend; when omitted the plotted-array sizes are used (correct only when
    nothing was subsampled)."""
    m_full = np.isfinite(teff_full) & np.isfinite(logg_full)
    if m_full.sum() > 100:
        ax.hexbin(teff_full[m_full], logg_full[m_full],
                  gridsize=80, mincnt=5, cmap="Greys", bins="log",
                  extent=[2800, 7800, -0.2, 5.5], alpha=0.6, zorder=1)
    m_in = np.isfinite(teff_in) & np.isfinite(logg_in)
    if m_in.sum() > 100:
        h = ax.hexbin(teff_in[m_in], logg_in[m_in],
                      gridsize=80, mincnt=5, cmap="viridis", bins="log",
                      extent=[2800, 7800, -0.2, 5.5], alpha=0.95, zorder=2)
        plt.colorbar(h, ax=ax, label="log10 N (in-cut)")
    box = patches.Rectangle(
        (cut_teff[0], cut_logg[0]),
        cut_teff[1] - cut_teff[0], cut_logg[1] - cut_logg[0],
        linewidth=2.0, edgecolor="red", linestyle="--", facecolor="none",
        zorder=10,
    )
    ax.add_patch(box)
    ax.set_xlim(*KIEL_PLOT_X)
    ax.set_ylim(*KIEL_PLOT_Y)
    ax.set_xlabel("Teff (K)", fontsize=8)
    ax.set_ylabel(r"$\log g$ (dex)", fontsize=8)
    n_full = int(n_full_actual if n_full_actual is not None else m_full.sum())
    n_in = int(n_in_actual if n_in_actual is not None else m_in.sum())
    ax.set_title(f"{title}\nfull cohort: {n_full:,}  ·  in-cut: {n_in:,} "
                  f"({100*n_in/max(n_full,1):.2f}%)", fontsize=9, color=color)


def _add_box_label(ax, cut_teff: tuple[float, float], cut_logg: tuple[float, float]):
    """Annotate the cut box. Label sits just inside the upper-right corner of
    the box (low log g, high Teff) — sparse RGB region — so it stays legible."""
    teff_pad = 0.04 * (cut_teff[1] - cut_teff[0])
    logg_pad = 0.04 * (cut_logg[1] - cut_logg[0])
    label_x = cut_teff[1] - teff_pad
    label_y = cut_logg[0] + logg_pad
    ax.text(
        label_x, label_y,
        f"selection\nlog g ∈ [{cut_logg[0]:.1f}, {cut_logg[1]:.1f}]\n"
        f"Teff ∈ [{cut_teff[0]/1000:.1f}, {cut_teff[1]/1000:.1f}] kK",
        color="red", fontsize=7.5, ha="left", va="top", fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="red", lw=0.6, alpha=0.92, pad=2),
        zorder=11,
    )


def _load_raw_apogee_kiel(n_max: int = 200_000) -> tuple[np.ndarray, np.ndarray, int]:
    """Read raw APOGEE DR19 Teff/log g (zgr_*) — pre-cut, all luminosity classes.

    Returns ``(teff_subsampled, logg_subsampled, n_total_raw)`` where the third
    value is the count BEFORE subsampling so the % in the panel title is honest.
    """
    if not S1_RAW_FITS.exists():
        return np.array([]), np.array([]), 0
    from astropy.io import fits
    with fits.open(S1_RAW_FITS, memmap=True) as f:
        t = f[2].data
        teff = np.asarray(t["zgr_teff"], dtype=np.float64)
        logg = np.asarray(t["zgr_logg"], dtype=np.float64)
    n_total = len(teff)
    if n_total > n_max:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_total, n_max, replace=False)
        teff = teff[idx]; logg = logg[idx]
    return teff, logg, n_total


def main() -> None:
    apply_style()

    # ---- Load post-cut (in-cut) parquets ----
    s1 = pd.read_parquet(S1, columns=["source_id", "ra_deg", "dec_deg", "g_mag",
                                       "teff_apogee", "logg_apogee"])
    s3 = pd.read_parquet(S3, columns=["source_id", "ra_deg", "dec_deg", "g_mag",
                                       "teff_andrae", "logg_andrae"])
    # Stream 2 — prefer the post-cut features parquet (RGB+RC selected) when it
    # exists, falling back to the pre-cut interim Gaia parquet.
    if S2_FEAT.exists():
        s2 = pd.read_parquet(S2_FEAT, columns=["source_id", "ra_deg", "dec_deg",
                                                "g_mag", "teff_andrae", "logg_andrae"])
        s2 = s2.rename(columns={"teff_andrae": "teff_gspphot",
                                 "logg_andrae": "logg_gspphot"})
    elif S2_RAW.exists():
        s2 = pd.read_parquet(S2_RAW, columns=["source_id", "ra", "dec",
                                               "phot_g_mean_mag_corr",
                                               "teff_gspphot", "logg_gspphot"])
        s2 = s2.rename(columns={"ra": "ra_deg", "dec": "dec_deg",
                                 "phot_g_mean_mag_corr": "g_mag"})
    else:
        s2 = None

    rng = np.random.default_rng(0)

    # ---- Load raw pre-cut cohorts ----
    raw_apogee_teff, raw_apogee_logg, n_apogee_raw = _load_raw_apogee_kiel(n_max=300_000)
    if S3_RAW.exists():
        raw_andrae = pd.read_parquet(S3_RAW, columns=["teff", "logg"])
        raw_andrae_teff = raw_andrae.teff.to_numpy()
        raw_andrae_logg = raw_andrae.logg.to_numpy()
        n_andrae_raw = len(raw_andrae)
        if n_andrae_raw > 500_000:
            idx = rng.choice(n_andrae_raw, 500_000, replace=False)
            raw_andrae_teff = raw_andrae_teff[idx]
            raw_andrae_logg = raw_andrae_logg[idx]
    else:
        raw_andrae_teff = np.array([]); raw_andrae_logg = np.array([])
        n_andrae_raw = 0

    # Stream 2 has no pre-cut cohort distinct from itself (the Hon+21
    # source set IS already a giant-only sample); use the full Stream-2
    # interim parquet as the "pre-cut" reference.
    if s2 is not None:
        raw_s2_teff = s2.teff_gspphot.to_numpy()
        raw_s2_logg = s2.logg_gspphot.to_numpy()
    else:
        raw_s2_teff = np.array([]); raw_s2_logg = np.array([])

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 4, hspace=0.38, wspace=0.32,
                           width_ratios=[1, 1, 1, 1.05])

    # ---- Row 1: sky Mollweides + G-mag histogram ----
    streams_sky = [
        (s1, "Stream 1 — APOGEE × Gaia (training)", PALETTE["apogee"]),
        (s2, "Stream 2 — Hon+21 TESS × Gaia (asteroseismic)", "#9467bd"),
        (s3, "Stream 3 — Andrae+23 × Gaia DR3 XP (inference)", PALETTE["andrae_volume"]),
    ]
    for col, (df, name, color) in enumerate(streams_sky):
        ax = fig.add_subplot(gs[0, col], projection="mollweide")
        if df is None:
            ax.text(0, 0, "Stream 2 features not yet built", ha="center", va="center",
                    fontsize=10, transform=ax.transData)
            ax.set_title(name + "\n(pending pipeline)", fontsize=9)
            continue
        idx = sample_index(len(df), 50_000, rng)
        x, y = radec_to_galactic_mollweide(df["ra_deg"].iloc[idx].to_numpy(),
                                           df["dec_deg"].iloc[idx].to_numpy())
        ax.scatter(x, y, s=0.4, alpha=0.35, color=color, rasterized=True)
        style_galactic_mollweide(ax)
        ax.set_title(f"{name}\nn = {len(df):,}", fontsize=9)

    # G-magnitude histogram
    ax_g = fig.add_subplot(gs[0, 3])
    bins = np.linspace(6, 18, 61)
    ax_g.hist(s1["g_mag"].dropna(), bins=bins, color=PALETTE["apogee"], alpha=0.55,
               label=f"S1 ({len(s1):,})  max G={s1.g_mag.max():.1f}")
    if s2 is not None:
        ax_g.hist(s2["g_mag"].dropna(), bins=bins, color="#9467bd", alpha=0.55,
                   label=f"S2 ({len(s2):,})  max G={s2.g_mag.max():.1f}")
    ax_g.hist(s3["g_mag"].dropna(), bins=bins, color=PALETTE["andrae_volume"], alpha=0.55,
               label=f"S3 ({len(s3):,})  max G={s3.g_mag.max():.1f}")
    ax_g.axvline(17.0, color="red", lw=0.9, ls="--",
                  label="G = 17 (Pipeline-1 cap)")
    ax_g.set_xlabel("Gaia G (mag, corrected)")
    ax_g.set_ylabel("count")
    ax_g.set_title("Magnitude distribution per stream")
    ax_g.legend(fontsize=7, loc="upper left", frameon=True, framealpha=0.95,
                facecolor="white", edgecolor="0.4")

    # ---- Row 2: Kiel diagrams (each with its STREAM-SPECIFIC cut box) ----
    ax_k1 = fig.add_subplot(gs[1, 0])
    _kiel_panel(ax_k1,
                raw_apogee_teff, raw_apogee_logg,
                s1["teff_apogee"].to_numpy(), s1["logg_apogee"].to_numpy(),
                title="Stream 1 — APOGEE DR19 raw → cut",
                color=PALETTE["apogee"],
                cut_teff=S1_CUT_TEFF, cut_logg=S1_CUT_LOGG,
                n_full_actual=n_apogee_raw, n_in_actual=len(s1))
    _add_box_label(ax_k1, S1_CUT_TEFF, S1_CUT_LOGG)

    ax_k2 = fig.add_subplot(gs[1, 1])
    if s2 is not None:
        _kiel_panel(ax_k2,
                    raw_s2_teff, raw_s2_logg,
                    raw_s2_teff, raw_s2_logg,  # S2 not yet ML-cut; show same data
                    title="Stream 2 — Gaia GSP-Phot Kiel\n(pre-pipeline; full = in-cut shown)",
                    color="#9467bd",
                    cut_teff=S2_CUT_TEFF, cut_logg=S2_CUT_LOGG,
                    n_full_actual=len(s2), n_in_actual=len(s2))
        _add_box_label(ax_k2, S2_CUT_TEFF, S2_CUT_LOGG)
    else:
        ax_k2.set_axis_off()

    ax_k3 = fig.add_subplot(gs[1, 2])
    _kiel_panel(ax_k3,
                raw_andrae_teff, raw_andrae_logg,
                s3["teff_andrae"].to_numpy(), s3["logg_andrae"].to_numpy(),
                title="Stream 3 — Andrae+23 raw → cut",
                color=PALETTE["andrae_volume"],
                cut_teff=S3_CUT_TEFF, cut_logg=S3_CUT_LOGG,
                n_full_actual=n_andrae_raw, n_in_actual=len(s3))
    _add_box_label(ax_k3, S3_CUT_TEFF, S3_CUT_LOGG)

    # ---- Row 2, col 4: cut summary text panel ----
    ax_cut = fig.add_subplot(gs[1, 3]); ax_cut.set_axis_off()
    txt = (
        "Cuts ACTUALLY APPLIED (per build script)\n"
        "----------------------------------------\n"
        "S1 (APOGEE training):\n"
        "    Teff  ∈  [4000, 5500] K\n"
        "    log g ∈  [1.0, 3.5] dex\n"
        "    (RGB scope cut — tighter than Mészáros)\n\n"
        "S2 (TESS asteroseismic) and S3 (Andrae XP):\n"
        "    Teff  ∈  [3500, 6500] K\n"
        "    log g ∈  [0.0, 3.8] dex\n"
        "    (Mészáros+2025 polynomial validity)\n\n"
        "Why two different cuts\n"
        "----------------------\n"
        "S1 trains on a tight RGB scope so APOGEE\n"
        "[C/N]-age calibration assumptions hold; S2/S3\n"
        "infer over the broader Mészáros+2025 range\n"
        "where the [X/M] polynomial corrections apply.\n\n"
        "Per-panel Kiel axes\n"
        "-------------------\n"
        "  Greyscale  = pre-cut cohort (everything ingested).\n"
        "  Viridis    = in-cut subset (released to pipeline).\n"
        "  Red dashed = the STREAM-SPECIFIC cut window.\n\n"
        "Pre-cut sources\n"
        "---------------\n"
        "  S1: SDSS-V astraAllStarASPCAP DR19 (1.10 M stars).\n"
        "  S2: Hon+21 + Gaia GSP-Phot (no separate pre-cut).\n"
        "  S3: Andrae+23 published RGB sample (10.48 M stars).\n"
    )
    ax_cut.text(0.0, 1.0, txt, transform=ax_cut.transAxes, fontsize=8,
                 ha="left", va="top", family="monospace")

    fig.suptitle(
        "Stage 00 — source coverage of all three streams (full Kiel pre-cut + selection box).\n"
        "Stream 1 = training (324 k post-cut of 1.10 M raw).  "
        "Stream 2 = asteroseismic (72 k post-cut of 158 k Hon+21).  "
        "Stream 3 = inference (614 k post-cut of 10.48 M raw).",
        fontsize=10,
    )
    save_fig(fig, OUT / "source_coverage.png", tight=False)


if __name__ == "__main__":
    main()
