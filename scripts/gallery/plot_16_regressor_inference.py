"""Stage 16: strong-contrastive-v2 regressor inference on S2 + S3.

Per-element σ histograms (with prior-collapse threshold marked) and pred vs G
hexbins for both Stream 2 and Stream 3 inference cohorts. Stream 1 is the
training pool, not an inference target.

Layout 4 × 5 (rows = stream × {sigma, pred-vs-G}, cols = 5 elements):
- Row 0: Stream 3 σ histograms
- Row 1: Stream 3 pred vs G
- Row 2: Stream 2 σ histograms
- Row 3: Stream 2 pred vs G
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

OUT = REPO / "reports/gallery/16_regressor_inference"
PRED_S2 = REPO / "data/processed/pipeline1_predictions_stream2.parquet"
FEAT_S2 = REPO / "data/processed/pipeline1_features_stream2.parquet"
PRED_S3 = REPO / "data/processed/pipeline1_predictions_stream3.parquet"
FEAT_S3 = REPO / "data/processed/pipeline1_features_stream3.parquet"

ELEMENTS = ("teff", "logg", "mh", "alpha_m", "mg_h")
THRESHOLDS = {"teff": 150.0, "logg": 0.30, "mh": 0.20, "alpha_m": 0.05, "mg_h": 0.20}


def _load_stream(pred_path: Path, feat_path: Path) -> pd.DataFrame | None:
    if not pred_path.exists() or not feat_path.exists():
        return None
    cols = ["source_id"] + [f"{e}_pred" for e in ELEMENTS] + [f"{e}_sigma" for e in ELEMENTS]
    pred = pd.read_parquet(pred_path, columns=cols)
    feat = pd.read_parquet(feat_path, columns=["source_id", "g_mag"])
    return pred.merge(feat, on="source_id", how="left")


def _sigma_panel(ax, sigma: np.ndarray, thr: float, name: str, color: str):
    lo, hi = np.nanpercentile(sigma, [0.5, 99.5])
    bins = np.linspace(lo, max(hi, thr * 1.5), 41)
    ax.hist(sigma, bins=bins, color=color, alpha=0.7)
    ax.axvline(thr, color="#d62728", lw=0.8, ls="--",
                label=f"thr {thr:g}")
    rate = (sigma > thr).mean() * 100
    ax.text(0.95, 0.95, f"σ-inf: {rate:.1f}%\nn={len(sigma):,}",
            ha="right", va="top", transform=ax.transAxes, fontsize=7,
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.92, pad=2))
    ax.set_xlabel(f"σ_{name}")
    ax.legend(fontsize=6, loc="upper left")


def _pred_panel(ax, g: np.ndarray, p: np.ndarray, name: str):
    m = np.isfinite(g) & np.isfinite(p)
    h = ax.hexbin(g[m], p[m], gridsize=50, mincnt=10, cmap="viridis", bins="log")
    plt.colorbar(h, ax=ax, label="log10 N")
    ax.set_xlabel("G (mag)")
    ax.set_ylabel(f"{name} pred")


def main() -> None:
    apply_style()

    s3 = _load_stream(PRED_S3, FEAT_S3)
    s2 = _load_stream(PRED_S2, FEAT_S2)
    streams = [(s3, "Stream 3", "#d62728"), (s2, "Stream 2", "#9467bd")]

    n_rows = sum(2 for d, _, _ in streams if d is not None)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, 5, figsize=(16, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    row = 0
    for df, name, color in streams:
        if df is None:
            continue
        for i, e in enumerate(ELEMENTS):
            _sigma_panel(axes[row, i], df[f"{e}_sigma"].dropna().to_numpy(),
                         THRESHOLDS[e], e, color)
            axes[row, i].set_title(f"{name} — {e}: σ", fontsize=9)
        row += 1
        for i, e in enumerate(ELEMENTS):
            _pred_panel(axes[row, i], df["g_mag"].to_numpy(),
                         df[f"{e}_pred"].to_numpy(), e)
            axes[row, i].set_title(f"{name} — {e}: pred vs G", fontsize=9)
        row += 1

    fig.suptitle(
        "Stage 16 — regressor inference (strong-contrastive-v2): "
        "σ histograms (with prior-collapse threshold) + pred vs G, per stream.",
        fontsize=10,
    )
    fig.tight_layout()
    save_fig(fig, OUT / "regressor_inference.png", tight=False)


if __name__ == "__main__":
    main()
