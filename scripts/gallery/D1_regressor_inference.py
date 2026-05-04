"""D1: Stream 1 holdout regressor inference (raw, pre-kNN-rescue).

Pred vs truth scatter plots for 5 selected elements (Teff, log g, [M/H],
[alpha/M], [Mg/H]) on Stream 1 holdout test set. Top row: pred vs truth coloured
by predicted sigma. Bottom row: residuals coloured by truth value.

Stream: Stream 1 (APOGEE x Gaia DR3, holdout test set).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

import pandas as pd

from _common import load_real_stream, save_fig

from arqueogal.utils.plotting import set_aa_style


def main(n_stars: int | None = None) -> None:
    set_aa_style()

    # Read full Stream 1: real features (incl. APOGEE truth) + real predictions
    # (incl. real per-label sigma). Don't downsample by default.
    cols = [
        "source_id",
        "teff_apogee", "logg_apogee", "mh_apogee", "alpha_m_apogee", "mg_h_apogee",
    ]
    data = load_real_stream(1, columns=cols)
    # Keep only rows with finite truth labels for at least the 5 elements used.
    data = data.dropna(subset=[
        "teff_apogee", "logg_apogee", "mh_apogee", "alpha_m_apogee", "mg_h_apogee",
        "teff_pred", "logg_pred", "mh_pred", "alpha_m_pred", "mg_h_pred",
        "teff_sigma", "logg_sigma", "mh_sigma", "alpha_m_sigma", "mg_h_sigma",
    ])

    if n_stars is not None and n_stars < len(data):
        data = data.sample(n=n_stars, random_state=42)

    # Use real per-label sigma columns from the predictions parquet — DO NOT
    # fabricate sigma from |pred - truth|.
    preds = {
        "Teff":      ("teff_apogee",    "teff_pred",    "teff_sigma"),
        "log g":     ("logg_apogee",    "logg_pred",    "logg_sigma"),
        "[M/H]":     ("mh_apogee",      "mh_pred",      "mh_sigma"),
        "[alpha/M]": ("alpha_m_apogee", "alpha_m_pred", "alpha_m_sigma"),
        "[Mg/H]":    ("mg_h_apogee",    "mg_h_pred",    "mg_h_sigma"),
    }
    preds = {k: {"y_true": data[t].values, "y_pred": data[p].values, "sigma": data[s].values}
             for k, (t, p, s) in preds.items()}

    elements = list(preds.keys())
    n_cols = 5
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6))

    # Hexbin density-mapped panels. Plain scatter on N=324k × 10 subplots
    # generates a per-marker path for every point even with rasterized=True,
    # which dominates the runtime. Hexbin reduces it to ~80×80 = 6400 cells
    # per panel and is two orders of magnitude faster while preserving the
    # pred-vs-truth structure (the cell colour shows the diagnostic statistic
    # — median σ on the top row, density on the bottom row).
    GRID = 80

    for i, elem in enumerate(elements):
        y_true = preds[elem]["y_true"]
        y_pred = preds[elem]["y_pred"]
        sigma = preds[elem]["sigma"]

        # Top row: pred vs truth, hex cells coloured by **median predicted σ**.
        ax = axes[0, i]
        hb = ax.hexbin(
            y_true, y_pred, C=sigma, reduce_C_function=np.median,
            gridsize=GRID, cmap="viridis", mincnt=1,
        )
        cbar = plt.colorbar(hb, ax=ax, pad=0.02)
        cbar.set_label(r"median $\sigma$ in cell", fontsize=7)

        # Diagonal line (perfect agreement)
        rng_min = float(min(np.min(y_true), np.min(y_pred)))
        rng_max = float(max(np.max(y_true), np.max(y_pred)))
        ax.plot([rng_min, rng_max], [rng_min, rng_max], "k--", lw=0.7, alpha=0.4)

        resid = y_pred - y_true
        rmse = float(np.sqrt(np.mean(resid**2)))
        bias = float(np.median(resid))

        ax.set_xlabel(f"{elem} truth", fontsize=8)
        ax.set_ylabel(f"{elem} pred", fontsize=8)
        ax.set_title(f"{elem}", fontsize=9, fontweight="semibold")
        ax.text(
            0.05, 0.95,
            f"RMSE={rmse:.2g}\nbias={bias:+.2g}",
            transform=ax.transAxes, fontsize=6.5, ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.88, pad=2),
        )
        ax.grid(True, alpha=0.25)

        # Bottom row: residual vs truth, hex cells coloured by **log10 density**.
        ax = axes[1, i]
        hb2 = ax.hexbin(
            y_true, resid, gridsize=GRID, cmap="plasma", mincnt=1, bins="log",
        )
        cbar2 = plt.colorbar(hb2, ax=ax, pad=0.02)
        cbar2.set_label(r"log$_{10}$ N", fontsize=7)
        ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.4,
                   label="zero-bias reference")

        # Bias of the residual distribution (median = robust estimator) and a
        # shaded ±1σ band around it. The σ here is the residual std — the
        # spread of the model's per-star errors at this label, not the
        # uncertainty on the bias estimator (that is much smaller, ≈σ/√n).
        bias = float(np.median(resid))
        sigma_std = float(np.std(resid))
        x_lo, x_hi = ax.get_xlim()
        ax.axhspan(bias - sigma_std, bias + sigma_std,
                   color="#ffffff", alpha=0.18, zorder=2,
                   label=fr"bias $\pm 1\sigma$  ({bias:+.2g}, $\sigma={sigma_std:.2g}$)")
        ax.axhline(bias, color="white", lw=1.4, ls="-", alpha=0.95,
                   zorder=3)
        ax.axhline(bias + sigma_std, color="white", lw=0.6, ls=":",
                   alpha=0.7, zorder=3)
        ax.axhline(bias - sigma_std, color="white", lw=0.6, ls=":",
                   alpha=0.7, zorder=3)
        ax.set_xlim(x_lo, x_hi)
        ax.legend(fontsize=6, loc="upper right", framealpha=0.85)

        ax.set_xlabel(f"{elem} truth", fontsize=8)
        ax.set_ylabel("residual (pred − truth)", fontsize=8)
        ax.set_title(f"{elem} residual", fontsize=9, fontweight="semibold")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"D1. Stream 1 (Kiel-bounded RGB pool, n={len(data):,}): "
        f"5-label regressor inference, raw pre-kNN-rescue. "
        r"Top row: pred vs truth, hex colour = median $\sigma$ in cell. "
        r"Bottom row: residual vs truth, hex colour = $\log_{10}$ density.",
        fontsize=9,
    )

    fig.tight_layout()

    out = REPO / "reports" / "gallery" / "D_predictions"
    out.mkdir(parents=True, exist_ok=True)
    save_fig(fig, out / "D1_regressor_inference", formats=("pdf", "png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-stars",
        type=int,
        default=None,
        help="Optional: downsample to N stars (default: use all)",
    )
    args = parser.parse_args()
    main(n_stars=args.n_stars)
