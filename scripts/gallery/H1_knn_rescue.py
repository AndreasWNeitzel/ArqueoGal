"""H1: kNN rescue mechanism — example from real Stream 2 data.

What this shows:
- One example star from Stream 2 selected by highest regressor sigma.
- For each of 5 elements (Teff, log g, [M/H], [alpha/M], [Mg/H]):
  - The regressor prediction (red dashed) and 1-sigma band (red shaded).
  - The kNN-neighbour summary statistics: median (green solid), p25 / p75
    (green dotted), and ± std band (green shaded).
- These are the only kNN quantities persisted on disk (the K=50 individual
  neighbour values are not stored). The plot compares the regressor's
  posterior to the empirical kNN summary so the reader can see where the
  hybrid composer prefers the kNN median over a high-sigma regressor.

Uses real Stream 2 kNN rescue data: one example star with high regressor sigma
selected from pipeline1_knn_rescue_stream2.parquet.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _common import apply_style, save_fig

OUT = REPO / "reports/gallery/H_hybrid_release"
ELEMENTS = ("teff", "logg", "mh", "alpha_m", "mg_h")
ELEMENT_LABELS = (
    "Teff (K)",
    r"$\log g$ (dex)",
    "[M/H] (dex)",
    r"[$\alpha$/M] (dex)",
    "[Mg/H] (dex)",
)


def main(argv: list[str] | None = None) -> int:
    apply_style()
    print("[H1] Loading real Stream 2 kNN rescue data")

    # Load Stream 2 predictions and kNN rescue data
    pred2 = pd.read_parquet(
        REPO / "data/processed/pipeline1_predictions_stream2.parquet",
        columns=[
            "source_id",
            "teff_pred",
            "logg_pred",
            "mh_pred",
            "alpha_m_pred",
            "mg_h_pred",
            "teff_sigma",
            "logg_sigma",
            "mh_sigma",
            "alpha_m_sigma",
            "mg_h_sigma",
        ],
    )
    knn2 = pd.read_parquet(REPO / "data/processed/pipeline1_knn_rescue_stream2.parquet")

    # Merge
    data = pred2.merge(knn2, on="source_id", how="inner")

    # Select one example with high regressor sigma (prior collapse candidate)
    # Use max sigma across elements as selection criterion
    data["max_sigma"] = data[
        ["teff_sigma", "logg_sigma", "mh_sigma", "alpha_m_sigma", "mg_h_sigma"]
    ].max(axis=1)
    holdout_idx = data["max_sigma"].idxmax()
    holdout = data.iloc[[holdout_idx]]

    print(f"[H1] Selected example star (source_id={holdout['source_id'].iloc[0]}) with max_sigma={holdout['max_sigma'].iloc[0]:.2f}")
    print("[H1] Rendering single-example kNN rescue mechanism")
    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    axes[-1].set_visible(False)

    for idx, (elem, elem_label) in enumerate(zip(ELEMENTS, ELEMENT_LABELS)):
        ax = axes[idx]

        knn_col = f"knn_{elem}"
        med_val = holdout[f"{knn_col}_med"].iloc[0]
        p25_val = holdout[f"{knn_col}_p25"].iloc[0]
        p75_val = holdout[f"{knn_col}_p75"].iloc[0]
        std_val = holdout[f"{knn_col}_std"].iloc[0]
        pred_val = holdout[f"{elem}_pred"].iloc[0]
        sigma_val = holdout[f"{elem}_sigma"].iloc[0]

        # Real kNN summary band: shaded ± 1 std around median (green) — these
        # are the on-disk summary statistics from K=50 neighbour values
        # computed at training time; per-neighbour values are NOT stored.
        ax.axvspan(med_val - std_val, med_val + std_val,
                   color="#2ca02c", alpha=0.18, label=f"kNN median ± std (std={std_val:.3f})")
        ax.axvline(med_val, color="#2ca02c", lw=2.0, ls="-",
                   label=f"kNN median: {med_val:.3f}")
        ax.axvline(p25_val, color="#2ca02c", lw=1.0, ls=":", alpha=0.7)
        ax.axvline(p75_val, color="#2ca02c", lw=1.0, ls=":", alpha=0.7,
                   label=f"kNN p25/p75: [{p25_val:.3f}, {p75_val:.3f}]")

        # Regressor prediction band (red): pred ± 1 sigma
        ax.axvspan(pred_val - sigma_val, pred_val + sigma_val,
                   color="#d62728", alpha=0.15, label=f"Regressor ± σ (σ={sigma_val:.3f})")
        ax.axvline(pred_val, color="#d62728", lw=2.0, ls="--",
                   label=f"Regressor: {pred_val:.3f}")

        x_lo = min(p25_val, pred_val - sigma_val) - 0.05 * abs(med_val - pred_val + 1e-3)
        x_hi = max(p75_val, pred_val + sigma_val) + 0.05 * abs(med_val - pred_val + 1e-3)
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel(elem_label)
        ax.set_yticks([])
        ax.set_title(f"{elem}: regressor vs kNN summary")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Stream 2 real example — kNN rescue mechanism (high-sigma prior-collapsed star). kNN rescue is ONLY invoked when regressor sigma exceeds prior-collapse threshold.",
        fontsize=10,
        fontweight="semibold",
    )
    fig.set_layout_engine("constrained")
    save_fig(fig, OUT / "H1_knn_rescue.pdf", formats=("pdf", "png"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
