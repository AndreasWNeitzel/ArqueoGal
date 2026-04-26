"""Stage 26: Stream-2 inference summary.

Mirror of stages 19/20/21 but for the Stream-2 hybrid release. **All
predicted columns shown here are ``*_hybrid_pred`` (the catalog's
user-facing surface)** so this stage tells the same story as stage 20.
The raw ``*_pred`` regressor surface is intentionally not shown here:
the hybrid layer's role is to substitute the kNN-median for stars whose
σ exceeds the per-element prior-collapse threshold, and using the raw
regressor here would re-expose the prior-collapse spike that the hybrid
layer is designed to mask.

Inputs:
- ``release/D-Cat-b/hybrid_pipeline_run_stream2/predictions_with_features.parquet``
  (71,573 rows; full hybrid release driven by the Hon+21 TESS × Gaia DR3
  asteroseismic-giant cohort).

Layout 2 × 3:
  (0,0) Galactic sky-map of Stream 2 predictions, coloured by hybrid [M/H].
  (0,1) Hybrid Kiel diagram coloured by hybrid [M/H] (giant branch).
  (0,2) Per-element tier distribution: stacked bars Tier 1 / Tier 2 / Tier 3.
  (1,0) Hybrid [α/M] vs hybrid [M/H], coloured by hybrid Teff.
  (1,1) GSP-Phot Teff vs hybrid Teff (S2 has GSP-Phot as a comparison
        baseline; no true labels, but GSP-Phot is what Andrae+2023's RGB
        sample is built from).
  (1,2) Mahalanobis OOD score histogram with the joint-OOD flag overlaid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
from _common import (apply_style, save_fig, radec_to_galactic_mollweide,
                     style_galactic_mollweide, sample_index)

OUT = REPO / "reports/gallery/26_stream2_inference"
PARQUET = REPO / "release/D-Cat-b/hybrid_pipeline_run_stream2/predictions_with_features.parquet"


def main() -> None:
    apply_style()
    if not PARQUET.exists():
        print(f"[plot_26] missing {PARQUET}; skip")
        return

    # Use *_hybrid_pred (the catalog's user-facing surface) so this stage
    # tells the same story as plot 20. The raw *_pred columns include the
    # prior-collapse spike and σ-inflated values that the hybrid layer
    # substitutes with kNN-median; mixing the two across the gallery
    # produces the apparent Stream-2 inconsistency between stages 20 and 26.
    cols = [
        "source_id", "ra_deg", "dec_deg", "b_deg", "g_mag",
        "teff_hybrid_pred", "logg_hybrid_pred", "mh_hybrid_pred",
        "alpha_m_hybrid_pred", "mg_h_hybrid_pred",
        "teff_hybrid_sigma", "mh_hybrid_sigma",
        "teff_gspphot",
        "release_tier__teff", "release_tier__logg", "release_tier__mh",
        "release_tier__alpha_m", "release_tier__mg_h",
        "ood_mahalanobis_score", "ood_joint_flag",
        "mode_ambiguous_flag", "aux_missing_any",
    ]
    df = pd.read_parquet(PARQUET, columns=cols)
    n = len(df)

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30,
                           width_ratios=[1.05, 1, 1])

    rng = np.random.default_rng(0)

    # (0,0) Galactic sky map coloured by [M/H]
    ax = fig.add_subplot(gs[0, 0], projection="mollweide")
    idx = sample_index(n, 50_000, rng)
    sub = df.iloc[idx]
    x, y = radec_to_galactic_mollweide(sub.ra_deg.to_numpy(), sub.dec_deg.to_numpy())
    sc = ax.scatter(x, y, s=0.7, c=sub.mh_hybrid_pred.to_numpy(), cmap="viridis",
                     vmin=-1.5, vmax=0.5, alpha=0.6, rasterized=True)
    style_galactic_mollweide(ax)
    cb = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("hybrid [M/H] (dex)", fontsize=8)
    ax.set_title(f"Stream 2 Galactic sky map (n={n:,})\ncolour = hybrid [M/H]",
                  fontsize=9)

    # (0,1) Predicted Kiel coloured by [M/H]
    ax = fig.add_subplot(gs[0, 1])
    teff = df.teff_hybrid_pred.to_numpy()
    logg = df.logg_hybrid_pred.to_numpy()
    mh = df.mh_hybrid_pred.to_numpy()
    ok = np.isfinite(teff) & np.isfinite(logg) & np.isfinite(mh)
    sc = ax.scatter(teff[ok], logg[ok], s=0.5, c=mh[ok], cmap="viridis",
                     vmin=-1.5, vmax=0.5, alpha=0.4, rasterized=True)
    ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_xlim(7000, 3500); ax.set_ylim(4.5, -0.5)
    ax.set_xlabel("hybrid Teff (K)")
    ax.set_ylabel(r"hybrid $\log g$ (dex)")
    plt.colorbar(sc, ax=ax, label="hybrid [M/H] (dex)")
    ax.set_title(f"Hybrid Kiel of S2 (n={int(ok.sum()):,})", fontsize=9)

    # (0,2) Per-element tier bars
    ax = fig.add_subplot(gs[0, 2])
    elements = ("teff", "logg", "mh", "alpha_m", "mg_h")
    el_labels = {"teff": "Teff", "logg": "log g", "mh": "[M/H]",
                  "alpha_m": "[α/M]", "mg_h": "[Mg/H]"}
    tiers = [1, 2, 3]
    counts = []
    for e in elements:
        col = f"release_tier__{e}"
        cnts = [int((df[col] == t).sum()) for t in tiers]
        counts.append(cnts)
    x = np.arange(len(elements))
    w = 0.27
    t1 = [c[0] for c in counts]; t2 = [c[1] for c in counts]; t3 = [c[2] for c in counts]
    ax.bar(x - w, t1, w, color="#2ca02c", label="Tier 1")
    ax.bar(x, t2, w, color="#ff7f0e", label="Tier 2")
    ax.bar(x + w, t3, w, color="#d62728", label="Tier 3")
    ax.set_xticks(x); ax.set_xticklabels([el_labels[e] for e in elements])
    ax.set_ylabel("count")
    ax.set_title(f"Per-element tier distribution (n={n:,})", fontsize=9)
    ax.legend(fontsize=8, loc="upper right", frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="0.4")
    # Annotate: % Tier 1 per element
    for i, e in enumerate(elements):
        pct1 = 100 * t1[i] / n
        ax.text(i, t1[i] * 1.02, f"T1: {pct1:.1f}%", ha="center", va="bottom",
                fontsize=7, color="#2ca02c")

    # (1,0) Hybrid [α/M] vs hybrid [M/H] coloured by Teff
    ax = fig.add_subplot(gs[1, 0])
    a = df.alpha_m_hybrid_pred.to_numpy()
    ok = np.isfinite(mh) & np.isfinite(a) & np.isfinite(teff)
    h = ax.hexbin(mh[ok], a[ok], gridsize=70, mincnt=10, cmap="viridis",
                   bins="log", extent=[-2.0, 0.6, -0.05, 0.40])
    ax.set_xlim(-2.0, 0.6); ax.set_ylim(-0.05, 0.40)
    ax.set_xlabel("hybrid [M/H] (dex)")
    ax.set_ylabel(r"hybrid [$\alpha$/M] (dex)")
    plt.colorbar(h, ax=ax, label="log10 N")
    ax.set_title(r"Stream 2 hybrid [$\alpha$/M] vs [M/H]", fontsize=9)

    # (1,1) GSP-Phot Teff vs hybrid Teff
    ax = fig.add_subplot(gs[1, 1])
    tg = df.teff_gspphot.to_numpy()
    ok = np.isfinite(tg) & np.isfinite(teff)
    h = ax.hexbin(tg[ok], teff[ok], gridsize=70, mincnt=10, cmap="viridis",
                   bins="log", extent=[3500, 6500, 3500, 6500])
    ax.plot([3500, 6500], [3500, 6500], "r--", lw=1, alpha=0.7,
             label="1-to-1")
    ax.set_xlim(3500, 6500); ax.set_ylim(3500, 6500)
    ax.set_xlabel("GSP-Phot Teff (K)")
    ax.set_ylabel("hybrid Teff (K)")
    plt.colorbar(h, ax=ax, label="log10 N")
    diff = teff[ok] - tg[ok]
    bias = float(np.mean(diff))
    rms = float(np.sqrt(np.mean(diff**2)))
    ax.text(0.04, 0.96,
             f"bias = {bias:+.0f} K\nRMS = {rms:.0f} K\nn = {int(ok.sum()):,}",
             transform=ax.transAxes, fontsize=8, ha="left", va="top",
             bbox=dict(facecolor="white", edgecolor="0.4", alpha=0.9, pad=2))
    ax.legend(fontsize=8, loc="lower right", frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="0.4")
    ax.set_title("Hybrid Teff vs GSP-Phot baseline", fontsize=9)

    # (1,2) Mahalanobis OOD score histogram
    ax = fig.add_subplot(gs[1, 2])
    score = df.ood_mahalanobis_score.dropna().to_numpy()
    bins = np.linspace(0, np.percentile(score, 99.5), 80)
    ax.hist(score, bins=bins, color="#1f77b4", alpha=0.65,
             label=f"all ({len(score):,})")
    flagged = df.loc[df.ood_joint_flag.astype(bool), "ood_mahalanobis_score"].dropna()
    ax.hist(flagged, bins=bins, color="#d62728", alpha=0.75,
             label=f"OOD-flagged ({len(flagged):,})")
    ax.set_xlabel("Mahalanobis distance (108-D XP)")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    n_aux = int(df.aux_missing_any.astype(bool).sum())
    n_modea = int(df.mode_ambiguous_flag.astype(bool).sum())
    ax.set_title(
        f"OOD score; aux-missing={n_aux:,} ({100*n_aux/n:.1f}%, diagnostic-only in v5); "
        f"mode-ambig={n_modea:,} ({100*n_modea/n:.1f}%, gates α/M only in v5)",
        fontsize=7.5,
    )
    ax.legend(fontsize=8, loc="upper right", frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="0.4")

    fig.suptitle(
        "Stage 26 — Stream 2 hybrid inference summary "
        f"(n = {n:,} asteroseismic giants).  All predicted values are "
        r"$\mathit{hybrid}$ (regressor substituted by kNN-median where σ exceeds "
        "the per-element prior-collapse threshold; same surface as stage 20).  "
        "Stream 2 has GSP-Phot Teff as a comparison baseline (not training truth).",
        fontsize=10,
    )
    save_fig(fig, OUT / "stream2_inference_summary.png", tight=False)


if __name__ == "__main__":
    main()
