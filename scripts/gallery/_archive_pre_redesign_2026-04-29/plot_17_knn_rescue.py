"""Stage 15: latent-kNN rescue on Stream 3.

What the deploy did: ``scripts/run_knn_rescue.py`` computed encoder
projections (z, L2-normalised) for every Stream-3 star, ran a GPU
brute-force cosine kNN against the Stream-1 training pool with K=50, and
emitted ``data/processed/pipeline1_knn_rescue.parquet`` with per-element
neighbour-label statistics (median, p25, p75, IQR, std) plus distances.

What we plot: top-1 neighbour distance distribution; kNN-IQR / regressor-σ
ratio per element (where the kNN tightens or widens the regressor); kNN
median vs regressor pred per element.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
from _common import apply_style, save_fig

OUT = REPO / "reports/gallery/17_knn_rescue"
KNN_S3 = REPO / "data/processed/pipeline1_knn_rescue.parquet"
PRED_S3 = REPO / "data/processed/pipeline1_predictions_stream3.parquet"
KNN_S2 = REPO / "data/processed/pipeline1_knn_rescue_stream2.parquet"
PRED_S2 = REPO / "data/processed/pipeline1_predictions_stream2.parquet"

ELEMENTS = ("teff", "logg", "mh", "alpha_m", "mg_h")


def _build(knn_path: Path, pred_path: Path) -> pd.DataFrame | None:
    if not knn_path.exists() or not pred_path.exists():
        return None
    knn = pd.read_parquet(knn_path)
    pred = pd.read_parquet(
        pred_path,
        columns=["source_id"] + [f"{e}_pred" for e in ELEMENTS] + [f"{e}_sigma" for e in ELEMENTS],
    )
    return knn.merge(pred, on="source_id")


def main() -> None:
    apply_style()
    df = _build(KNN_S3, PRED_S3)
    df_s2 = _build(KNN_S2, PRED_S2)
    if df is None:
        return

    # Layout: 5 rows × 5 cols.
    # Row 0: 4 separate distance histograms (S3 top-1, S2 top-1, S3 med, S2 med)
    # Row 1: Stream 3 σ comparison per element
    # Row 2: Stream 3 pred-vs-kNN per element
    # Row 3: Stream 2 σ comparison per element
    # Row 4: Stream 2 pred-vs-kNN per element
    fig, axes = plt.subplots(5, 5, figsize=(17, 17))

    # Top row: 4 separate panels — S3 top-1, S2 top-1, S3 median, S2 median.
    # Overlay was unreadable because S2 / S3 distance distributions are very
    # similar in shape and the line colours blended; separating them gives
    # each its own visible histogram.
    d3 = df["knn_top_distance"].to_numpy()
    md3 = df["knn_median_distance"].to_numpy()
    d2 = df_s2["knn_top_distance"].to_numpy() if df_s2 is not None else None
    md2 = df_s2["knn_median_distance"].to_numpy() if df_s2 is not None else None

    # Shared upper bin for top-1 (so S2 / S3 panels are directly comparable).
    upper_top = np.nanpercentile(d3, 99.5)
    if d2 is not None:
        upper_top = max(upper_top, np.nanpercentile(d2, 99.5))
    bins_top = np.linspace(0, upper_top, 51)

    upper_med = np.nanpercentile(md3, 99.5)
    if md2 is not None:
        upper_med = max(upper_med, np.nanpercentile(md2, 99.5))
    bins_med = np.linspace(0, upper_med, 51)

    def _hist_panel(ax, vals, bins, color, name, kind):
        ax.hist(vals, bins=bins, color=color, alpha=0.75, edgecolor=color, lw=0.6)
        med = float(np.nanmedian(vals))
        ax.axvline(med, color="k", lw=0.7, ls="--")
        ax.text(
            0.96,
            0.94,
            f"n = {len(vals):,}\nmedian = {med:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", edgecolor="0.4", alpha=0.92, pad=2),
        )
        ax.set_xlabel(f"{kind} cosine distance")
        ax.set_ylabel("count")
        ax.set_title(f"{name} — {kind}")

    _hist_panel(axes[0, 0], d3, bins_top, "#d62728", "Stream 3", "top-1")
    if d2 is not None:
        _hist_panel(axes[0, 1], d2, bins_top, "#9467bd", "Stream 2", "top-1")
    else:
        axes[0, 1].set_axis_off()
    _hist_panel(axes[0, 2], md3, bins_med, "#d62728", "Stream 3", "median K=50")
    if md2 is not None:
        _hist_panel(axes[0, 3], md2, bins_med, "#9467bd", "Stream 2", "median K=50")
    else:
        axes[0, 3].set_axis_off()
    axes[0, 4].set_axis_off()

    ranges = {
        "teff": (3500, 6500),
        "logg": (0.5, 4.0),
        "mh": (-2.5, 0.6),
        "alpha_m": (-0.2, 0.5),
        "mg_h": (-1.5, 0.5),
    }

    def _sigma_row(row, frame, name):
        for i, e in enumerate(ELEMENTS):
            ax = axes[row, i]
            knn_sigma = (frame[f"knn_{e}_iqr"] / 1.349).to_numpy()
            reg_sigma = frame[f"{e}_sigma"].to_numpy()
            m = np.isfinite(knn_sigma) & np.isfinite(reg_sigma) & (reg_sigma > 0)
            if m.sum() > 100:
                lo = float(np.nanmin([knn_sigma[m].min(), reg_sigma[m].min()]))
                hi = float(np.nanpercentile(np.concatenate([knn_sigma[m], reg_sigma[m]]), 99))
                h = ax.hexbin(
                    reg_sigma[m],
                    knn_sigma[m],
                    gridsize=45,
                    mincnt=10,
                    cmap="viridis",
                    bins="log",
                    extent=[lo, hi, lo, hi],
                )
                plt.colorbar(h, ax=ax, label="log10 N")
                ax.plot([lo, hi], [lo, hi], color="red", lw=0.6, ls="--")
                ax.set_xlabel(f"σ_{e} regressor")
                ax.set_ylabel(f"σ_{e} kNN IQR/1.349")
                ax.set_title(f"{name} — σ: {e}", fontsize=9)

    def _pred_row(row, frame, name):
        for i, e in enumerate(ELEMENTS):
            ax = axes[row, i]
            p = frame[f"{e}_pred"].to_numpy()
            knn_m = frame[f"knn_{e}_med"].to_numpy()
            m = np.isfinite(p) & np.isfinite(knn_m)
            if m.sum() > 100:
                lo, hi = ranges[e]
                h = ax.hexbin(
                    p[m],
                    knn_m[m],
                    gridsize=50,
                    mincnt=10,
                    cmap="viridis",
                    bins="log",
                    extent=[lo, hi, lo, hi],
                )
                plt.colorbar(h, ax=ax, label="log10 N")
                ax.plot([lo, hi], [lo, hi], color="red", lw=0.6, ls="--")
                ax.set_xlabel(f"{e} regressor pred")
                ax.set_ylabel(f"{e} kNN median")
                ax.set_title(f"{name} — pred vs kNN: {e}", fontsize=9)

    _sigma_row(1, df, "Stream 3")
    _pred_row(2, df, "Stream 3")
    if df_s2 is not None:
        _sigma_row(3, df_s2, "Stream 2")
        _pred_row(4, df_s2, "Stream 2")
    else:
        for c in range(5):
            axes[3, c].set_axis_off()
            axes[4, c].set_axis_off()

    fig.suptitle(
        "Latent-kNN rescue surface (K=50 cosine over Stream 1 training pool).\n"
        "Row 0: distance histograms (S3 + S2, separate panels). "
        "Rows 1-2: Stream 3 σ + pred-vs-kNN per element. "
        "Rows 3-4: Stream 2 σ + pred-vs-kNN per element.",
        fontsize=10,
    )
    fig.tight_layout()
    save_fig(fig, OUT / "knn_rescue.png", tight=False)


if __name__ == "__main__":
    main()
