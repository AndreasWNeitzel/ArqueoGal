"""Stage 12: Pipeline-1 validation on Stream-1 held-out split.

Outputs:
  - reports/gallery/12_pipeline1_validation/residual_hist_per_label.png
  - reports/gallery/12_pipeline1_validation/true_vs_pred_per_label.png
  - reports/gallery/12_pipeline1_validation/residual_by_teff_logg.png
  - reports/gallery/12_pipeline1_validation/calibration_per_label.png

Reads `reports/pipeline1/run_a_v11/val_predictions.parquet`
(columns: {label}_{truth,pred,sigma,epi} × 5 labels).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).resolve().parent))
from _common import GALLERY, apply_style, save_fig  # noqa: E402

OUT = GALLERY / "12_pipeline1_validation"

LABELS = ["teff", "logg", "mh", "alpha_m", "mg_h"]
LABEL_TEX = {
    "teff": r"$T_{\rm eff}$  [K]",
    "logg": r"$\log g$",
    "mh": r"$[{\rm M}/{\rm H}]$",
    "alpha_m": r"$[\alpha/{\rm M}]$",
    "mg_h": r"$[{\rm Mg}/{\rm H}]$",
}


def _load() -> "pd.DataFrame":
    import pandas as pd  # noqa: F401

    return pq.read_table("reports/pipeline1/run_a_v11/val_predictions.parquet").to_pandas()


def residual_hist_per_label() -> None:
    df = _load()
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    for ax, lbl in zip(axes.flat, LABELS):
        res = df[f"{lbl}_pred"].to_numpy() - df[f"{lbl}_truth"].to_numpy()
        res = res[np.isfinite(res)]
        lo, hi = np.percentile(res, [0.5, 99.5])
        ax.hist(np.clip(res, lo, hi), bins=60, color="#1f77b4", edgecolor="#333", alpha=0.85)
        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.set_title(
            rf"{LABEL_TEX[lbl]}   $\mu$={res.mean():.3f}"
            rf"   $\sigma$={res.std():.3f}",
            fontsize=10,
        )
        ax.set_xlabel("pred - truth")
        if lbl == LABELS[0]:
            ax.set_ylabel("count")
    fig.suptitle(
        f"Pipeline-1 v1.1 validation residuals  —  n={len(df):,} val stars",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    save_fig(fig, OUT / "residual_hist_per_label.png")


def true_vs_pred_per_label() -> None:
    df = _load()
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    for ax, lbl in zip(axes.flat, LABELS):
        t = df[f"{lbl}_truth"].to_numpy()
        p = df[f"{lbl}_pred"].to_numpy()
        m = np.isfinite(t) & np.isfinite(p)
        lo = min(np.percentile(t[m], 0.5), np.percentile(p[m], 0.5))
        hi = max(np.percentile(t[m], 99.5), np.percentile(p[m], 99.5))
        hb = ax.hexbin(
            t[m], p[m], gridsize=70, cmap="viridis", bins="log", mincnt=1, extent=(lo, hi, lo, hi)
        )
        plt.colorbar(hb, ax=ax, shrink=0.85, pad=0.02, label="log N")
        ax.plot([lo, hi], [lo, hi], "r-", lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"truth  {LABEL_TEX[lbl]}")
        ax.set_ylabel(f"pred  {LABEL_TEX[lbl]}")
        rmse = np.sqrt(np.mean((p[m] - t[m]) ** 2))
        ax.set_title(f"{LABEL_TEX[lbl]}   RMSE={rmse:.3f}", fontsize=10)
    fig.suptitle(
        f"Pipeline-1 v1.1 truth vs predicted  —  5 labels, val n={len(df):,}",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    save_fig(fig, OUT / "true_vs_pred_per_label.png")


def residual_by_teff_logg() -> None:
    """2-D Kiel-cell residual heatmap per label."""
    df = _load()
    t = df["teff_truth"].to_numpy()
    g = df["logg_truth"].to_numpy()

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for ax, lbl in zip(axes.flat, LABELS):
        res = df[f"{lbl}_pred"].to_numpy() - df[f"{lbl}_truth"].to_numpy()
        # robust color limit
        absmax = np.nanpercentile(np.abs(res), 95)
        m = np.isfinite(t) & np.isfinite(g) & np.isfinite(res)
        sc = ax.hexbin(
            t[m],
            g[m],
            C=res[m],
            reduce_C_function=np.mean,
            gridsize=35,
            cmap="coolwarm",
            vmin=-absmax,
            vmax=+absmax,
            mincnt=20,
        )
        plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02, label=f"mean  (pred - truth)")
        ax.set_xlim(5500, 3800)  # Kiel convention
        ax.set_ylim(3.8, 0.5)
        ax.set_xlabel(r"$T_{\rm eff}^{\rm truth}$ [K]")
        ax.set_ylabel(r"$\log g^{\rm truth}$")
        ax.set_title(f"bias  {LABEL_TEX[lbl]}", fontsize=10)
    fig.suptitle(
        r"Pipeline-1 v1.1 — residual mean per Kiel cell "
        r"($T_{\rm eff}^{\rm truth} \times \log g^{\rm truth}$)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    save_fig(fig, OUT / "residual_by_teff_logg.png")


def calibration_per_label() -> None:
    """Per-label reliability: predicted σ bins vs empirical |residual| std."""
    df = _load()
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    for ax, lbl in zip(axes.flat, LABELS):
        sigma = df[f"{lbl}_sigma"].to_numpy()
        res = df[f"{lbl}_pred"].to_numpy() - df[f"{lbl}_truth"].to_numpy()
        m = np.isfinite(sigma) & np.isfinite(res) & (sigma > 0)
        sigma, res = sigma[m], res[m]
        # bin by predicted sigma (percentile-based)
        edges = np.percentile(sigma, np.linspace(0, 100, 11))
        centers, empirical = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (sigma >= lo) & (sigma < hi)
            if mask.sum() > 50:
                centers.append(np.median(sigma[mask]))
                empirical.append(np.std(res[mask]))
        centers = np.array(centers)
        empirical = np.array(empirical)
        if len(centers) > 1:
            lim_lo = min(centers.min(), empirical.min()) * 0.8
            lim_hi = max(centers.max(), empirical.max()) * 1.2
            ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "r--", lw=0.8, label="ideal")
            ax.plot(centers, empirical, "o-", color="#1f77b4", lw=1.4, ms=5)
            ax.set_xlim(lim_lo, lim_hi)
            ax.set_ylim(lim_lo, lim_hi)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(rf"predicted $\sigma$  ({LABEL_TEX[lbl]})")
        ax.set_ylabel(r"empirical $\sigma_{\rm resid}$")
        ax.set_title(LABEL_TEX[lbl], fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle(
        "Pipeline-1 v1.1 reliability per label (log-log) — perfect on the y=x line",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    save_fig(fig, OUT / "calibration_per_label.png")


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    residual_hist_per_label()
    true_vs_pred_per_label()
    residual_by_teff_logg()
    calibration_per_label()


if __name__ == "__main__":
    main()
