"""Stage 11: supervised fine-tune + ensemble member spread.

Outputs:
  - reports/gallery/11_supervised_training/finetune_loss_curve.png
  - reports/gallery/11_supervised_training/ensemble_member_spread.png
  - reports/gallery/11_supervised_training/finetune_lr_schedule.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from _common import GALLERY, apply_style, save_fig  # noqa: E402

OUT = GALLERY / "11_supervised_training"


def _load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _curve(h: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hist = h.get("history", h)
    if isinstance(hist, list):
        epochs = np.array([r.get("epoch", i) for i, r in enumerate(hist)])
        train = np.array([r.get("train_loss", r.get("loss", np.nan)) for r in hist])
        val = np.array([r.get("val_loss", np.nan) for r in hist])
        lr = np.array([r.get("lr", r.get("learning_rate", np.nan)) for r in hist])
    else:
        n = max(len(v) for v in hist.values() if isinstance(v, list))
        epochs = np.arange(n)
        train = np.array(hist.get("train_loss", hist.get("loss", [np.nan] * n)))
        val = np.array(hist.get("val_loss", [np.nan] * n))
        lr = np.array(hist.get("lr", hist.get("learning_rate", [np.nan] * n)))
    return epochs, train, val, lr


def finetune_loss_curve() -> None:
    # Prefer 5-label history if present
    candidates = [
        Path("reports/pipeline1/run_a/finetune_history_5label.json"),
        Path("reports/pipeline1/run_a_v11/finetune_history.json"),
        Path("reports/pipeline1/run_a/finetune_history.json"),
    ]
    path = next((p for p in candidates if p.exists()), None)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    if path is None:
        ax.text(
            0.5,
            0.5,
            "finetune history not found",
            ha="center",
            va="center",
            fontsize=12,
            color="#d62728",
            transform=ax.transAxes,
        )
        save_fig(fig, OUT / "finetune_loss_curve.png")
        return

    h = _load_json(path)
    epochs, tr, va, _ = _curve(h)
    ax.plot(epochs, tr, "o-", color="#1f77b4", lw=1.5, ms=4, label="train")
    ax.plot(epochs, va, "s--", color="#d62728", lw=1.5, ms=4, label="val")
    best = h.get("best_epoch")
    best_val = h.get("best_val_loss")
    if best is not None and best_val is not None:
        ax.axvline(
            best, color="#333", lw=0.8, ls=":", label=f"best epoch {best} (val={best_val:.4f})"
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"beta-NLL loss ($\beta=0.5$)")
    ax.set_title(
        f"5-label supervised fine-tune  —  {path.parent.name}/{path.name}",
        fontsize=11,
        fontweight="semibold",
    )
    ax.legend()
    save_fig(fig, OUT / "finetune_loss_curve.png")


def ensemble_member_spread() -> None:
    """Plot per-member val loss curves from ensemble_history.json."""
    candidates = [
        Path("reports/pipeline1/run_a_v11/ensemble_history.json"),
        Path("reports/pipeline1/run_a/ensemble_history.json"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    if path is None:
        ax.text(
            0.5,
            0.5,
            "ensemble history not found",
            ha="center",
            va="center",
            fontsize=12,
            color="#d62728",
            transform=ax.transAxes,
        )
        save_fig(fig, OUT / "ensemble_member_spread.png")
        return
    h = _load_json(path)
    members = h.get("members", [])
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(members))))
    any_plotted = False
    for i, m in enumerate(members):
        hist = m.get("history", m.get("finetune_history", []))
        if isinstance(hist, list) and len(hist):
            eps = np.array([r.get("epoch", j) for j, r in enumerate(hist)])
            va = np.array([r.get("val_loss", np.nan) for r in hist])
            ax.plot(eps, va, "-", color=colors[i % 10], lw=1.2, label=f"seed {m.get('seed', i)}")
            any_plotted = True
    if h.get("val_loss_mean") is not None:
        ax.axhline(
            h["val_loss_mean"],
            color="k",
            lw=0.7,
            ls="--",
            label=f"mean best val = {h['val_loss_mean']:.4f} "
            f"(spread={h.get('val_loss_spread', 0):.4f})",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("val loss")
    ax.set_title(
        f"5-label ensemble — per-member val-loss curves  ({len(members)} members)",
        fontsize=11,
        fontweight="semibold",
    )
    if any_plotted:
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    save_fig(fig, OUT / "ensemble_member_spread.png")


def lr_schedule() -> None:
    candidates = [
        Path("reports/pipeline1/run_a/finetune_history_5label.json"),
        Path("reports/pipeline1/run_a/finetune_history.json"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if path is None:
        ax.text(0.5, 0.5, "no history", ha="center", va="center")
        save_fig(fig, OUT / "finetune_lr_schedule.png")
        return
    h = _load_json(path)
    epochs, _, _, lr = _curve(h)
    if np.all(np.isnan(lr)):
        ax.text(
            0.5,
            0.5,
            "lr not logged in history",
            ha="center",
            va="center",
            fontsize=12,
            color="#555",
            transform=ax.transAxes,
        )
    else:
        ax.plot(epochs, lr, "o-", color="#2ca02c", lw=1.5, ms=4)
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_title("OneCycleLR learning-rate schedule", fontsize=11, fontweight="semibold")
    save_fig(fig, OUT / "finetune_lr_schedule.png")


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    finetune_loss_curve()
    ensemble_member_spread()
    lr_schedule()


if __name__ == "__main__":
    main()
