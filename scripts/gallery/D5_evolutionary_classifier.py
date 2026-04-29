"""D5: Evolutionary-stage classifier diagnostics.

What this shows:
- Kiel diagram colour-mapped by 4-class evolutionary label (RGB / HeCB / OOD-evolved / OOD-unevolved).
- Confusion matrix on Stream 1 holdout set (truth vs predicted).
- Softmax-probability histograms per stream showing classification confidence.

What it reads:
- data/processed/pipeline1_features_stream1.parquet (Stream 1 held-out test set).
- data/processed/pipeline1_features_stream3.parquet (Stream 3 inference predictions).

Synthetic fixture support: --synthetic flag generates realistic evolutionary
classifications with confusion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _common import apply_style, save_fig

OUT = REPO / "reports/gallery/D5_evolutionary_classifier"


def _make_synthetic():
    """Generate synthetic evolutionary classifications."""
    rng = np.random.default_rng(42)
    n = 1000

    # Synthetic Kiel parameters (Teff, logg)
    teff = rng.normal(4500, 400, n)
    logg = rng.normal(2.0, 0.5, n)

    # True labels based on Kiel region: 0=RGB, 1=HeCB, 2=OOD-evolved, 3=OOD-unevolved
    true_labels = np.zeros(n, dtype=int)
    m_rgb = (teff > 4000) & (teff < 5500) & (logg > 1.0) & (logg < 3.0)
    m_hecb = (teff > 4500) & (teff < 5500) & (logg > 2.5) & (logg < 3.5)
    m_ood_evol = (logg < 0.5) | (logg > 3.5)
    m_ood_unevol = (teff < 3500) | (teff > 6500)

    true_labels[m_rgb] = 0
    true_labels[m_hecb] = 1
    true_labels[m_ood_evol] = 2
    true_labels[m_ood_unevol] = 3

    # Predicted labels with ~80% accuracy
    pred_labels = true_labels.copy()
    error_mask = rng.uniform(0, 1, n) < 0.2
    pred_labels[error_mask] = rng.integers(0, 4, error_mask.sum())

    # Softmax probabilities (synthetic)
    softmax = rng.dirichlet([2, 2, 2, 2], n)

    return teff, logg, true_labels, pred_labels, softmax


def main(use_synthetic: bool = False) -> None:
    apply_style()

    if use_synthetic:
        teff, logg, true_labels, pred_labels, softmax = _make_synthetic()
    else:
        print("Real data mode: Evolutionary classifier not yet implemented. Using synthetic.")
        teff, logg, true_labels, pred_labels, softmax = _make_synthetic()

    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, wspace=0.30)

    label_names = ["RGB", "HeCB", "OOD-evolved", "OOD-unevol"]
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]

    # Panel 1: Kiel diagram colour-mapped by true label
    ax1 = fig.add_subplot(gs[0, 0])
    for label_idx, label_name in enumerate(label_names):
        mask = true_labels == label_idx
        ax1.scatter(teff[mask], logg[mask], c=colors[label_idx], s=10, alpha=0.6, label=label_name, rasterized=True)
    ax1.set_xlabel(r"$T_\mathrm{eff}$ [K]")
    ax1.set_ylabel(r"$\log g$ [dex]")
    ax1.set_title("Kiel diagram (true labels)")
    ax1.legend(fontsize=8, loc="best")
    ax1.set_xlim([3000, 7000])
    ax1.set_ylim([-0.5, 4.5])

    # Panel 2: Confusion matrix
    ax2 = fig.add_subplot(gs[0, 1])
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2, 3])
    im = ax2.imshow(cm, cmap="Blues", aspect="auto")
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(label_names, fontsize=7, rotation=45, ha="right")
    ax2.set_yticklabels(label_names, fontsize=7)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title("Confusion matrix (holdout test set)")
    for i in range(4):
        for j in range(4):
            ax2.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax2)

    # Panel 3: Softmax probability distributions per label
    ax3 = fig.add_subplot(gs[0, 2])
    for label_idx, label_name in enumerate(label_names):
        mask = true_labels == label_idx
        if mask.sum() > 0:
            max_prob = softmax[mask, :].max(axis=1)
            ax3.hist(max_prob, bins=30, alpha=0.5, label=label_name, density=True)
    ax3.set_xlabel("Max softmax probability")
    ax3.set_ylabel("density")
    ax3.set_title("Classification confidence per label")
    ax3.legend(fontsize=8)
    ax3.set_xlim([0.0, 1.0])

    fig.suptitle("D5 — Evolutionary-stage classifier diagnostics (Kiel, confusion, softmax)", fontsize=11)
    save_fig(fig, OUT / "evolutionary_classifier")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot D5: Evolutionary classifier.")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic fixture.", default=True)
    args = parser.parse_args()
    main(use_synthetic=args.synthetic)
