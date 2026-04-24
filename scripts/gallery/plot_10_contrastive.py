"""Stage 10: contrastive pretraining — loss curve.

Outputs:
  - reports/gallery/10_contrastive_pretraining/pretrain_loss_curve.png

The halfway-UMAP figures already exist under reports/pipeline1/halfway/ and
are linked from the stage README directly, so this script only produces
the single new loss-curve figure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from _common import GALLERY, apply_style, save_fig  # noqa: E402

OUT = GALLERY / "10_contrastive_pretraining"


def _find_history() -> Path | None:
    for p in (
        Path("reports/pipeline1/run_a/contrastive_history.json"),
        Path("reports/pipeline1/run_a_v11/contrastive_history.json"),
    ):
        if p.exists():
            return p
    return None


def _extract_curve(h: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (epoch, train, val) curves from a history JSON in either shape."""
    hist = h.get("history", h)
    if isinstance(hist, list):
        epochs = np.array([r.get("epoch", i) for i, r in enumerate(hist)])
        train = np.array([r.get("train_loss", r.get("loss", np.nan)) for r in hist])
        val = np.array([r.get("val_loss", np.nan) for r in hist])
    elif isinstance(hist, dict):
        n = max(len(v) for v in hist.values() if isinstance(v, list))
        epochs = np.arange(n)
        train = np.array(hist.get("train_loss", hist.get("loss", [np.nan] * n)))
        val = np.array(hist.get("val_loss", [np.nan] * n))
    else:
        epochs = np.array([])
        train = np.array([])
        val = np.array([])
    return epochs, train, val


def pretrain_loss_curve() -> None:
    path = _find_history()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if path is None:
        ax.text(
            0.5,
            0.5,
            "contrastive history not found",
            ha="center",
            va="center",
            fontsize=12,
            color="#d62728",
            transform=ax.transAxes,
        )
        save_fig(fig, OUT / "pretrain_loss_curve.png")
        return

    with open(path) as f:
        h = json.load(f)
    epochs, tr, va = _extract_curve(h)

    ax.plot(epochs, tr, "o-", color="#1f77b4", lw=1.5, ms=4, label="train")
    ax.plot(epochs, va, "s--", color="#d62728", lw=1.5, ms=4, label="val")
    best = h.get("best_epoch")
    best_val = h.get("best_val_loss")
    if best is not None and best_val is not None:
        ax.axvline(
            best, color="#333", lw=0.8, ls=":", label=f"best epoch {best} (val={best_val:.4f})"
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("contrastive SupCon loss")
    ax.set_title(
        "Run-A contrastive pretraining  —  loss vs epoch", fontsize=12, fontweight="semibold"
    )
    ax.legend()
    save_fig(fig, OUT / "pretrain_loss_curve.png")


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    pretrain_loss_curve()


if __name__ == "__main__":
    main()
