"""Stage 13: strong-contrastive-v2 joint-loss training curves.

What the deploy did: ``scripts/run_ensemble.py`` with the convergence-tuned
defaults (SupCon=0.3, β-NLL=1.0, Barlow=0.8, τ_init=0.15, single seed) trained
the canonical ensemble at
``models/main/xp_abundances/20260425_6b96c06_cd1cbb9_ensemble_5label/``.
Training history was persisted to
``reports/pipeline1/long_train_2026-04-25/ensemble_history.json`` (30 epochs,
patience=5, last-epoch best — converged on the noise plateau).

What we plot: per-epoch loss components (total, SupCon, β-NLL, Barlow);
training-stability diagnostics (grad-norm + τ).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
from _common import apply_style, save_fig

OUT = REPO / "reports/gallery/13_training"
HIST = REPO / "reports/pipeline1/long_train_2026-04-25/ensemble_history.json"


def main() -> None:
    apply_style()
    if not HIST.exists():
        return
    payload = json.loads(HIST.read_text())
    member = payload["members"][0]
    history = member["history"]

    epoch = np.asarray([row["epoch"] for row in history])

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.8))
    components = (
        ("loss", "total"),
        ("supcon", "SupCon"),
        ("nll", r"$\beta$-NLL"),
        ("barlow", "Barlow"),
    )
    for ax, (key, label) in zip(axes.flatten()[:4], components):
        train = np.asarray([row[f"train_{key}"] for row in history])
        val = np.asarray([row[f"val_{key}"] for row in history])
        ax.plot(epoch, train, "o-", color="#1f77b4", label="train", ms=4)
        ax.plot(epoch, val, "o-", color="#d62728", label="val", ms=4)
        ax.axvline(member["best_epoch"], color="0.5", lw=0.6, ls="--")
        ax.set_title(label)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(fontsize=8)
        # Annotate train/val gap at best_epoch and overall direction
        be = member["best_epoch"]
        gap_be = float(val[be] - train[be])
        delta_train = float(train[-1] - train[0])
        delta_val = float(val[-1] - val[0])
        gen_status = (
            "→ overfitting"
            if (gap_be > 0 and delta_train < delta_val)
            else ("→ generalising" if abs(gap_be) < 0.02 * abs(train[be] + 1e-9) else "→ tracking")
        )
        txt = (
            f"val-train @best = {gap_be:+.3f}\n"
            f"Δ train = {delta_train:+.3f}, Δ val = {delta_val:+.3f}\n"
            f"{gen_status}"
        )
        ax.text(
            0.02,
            0.02,
            txt,
            transform=ax.transAxes,
            fontsize=6.5,
            ha="left",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.92, pad=2),
        )

    # Grad norms
    ax = axes[1, 1]
    gmax = np.asarray([row["train_grad_norm_max"] for row in history])
    gmean = np.asarray([row["train_grad_norm_mean"] for row in history])
    ax.plot(epoch, gmax, "o-", color="#1f77b4", label="max", ms=4)
    ax.plot(epoch, gmean, "o-", color="#ff7f0e", label="mean", ms=4)
    ax.axhline(500, color="#d62728", lw=0.6, ls="--", label="abort threshold")
    ax.axvline(member["best_epoch"], color="0.5", lw=0.6, ls="--")
    ax.set_yscale("log")
    ax.set_title("Train grad norms")
    ax.set_xlabel("epoch")
    ax.set_ylabel("|g|")
    ax.legend(fontsize=8)

    # τ
    ax = axes[1, 2]
    tau_train = np.asarray([row["train_tau"] for row in history])
    tau_val = np.asarray([row["val_tau"] for row in history])
    ax.plot(epoch, tau_train, "o-", color="#1f77b4", label="train τ", ms=4)
    ax.plot(epoch, tau_val, "o-", color="#d62728", label="val τ", ms=4)
    ax.axvline(member["best_epoch"], color="0.5", lw=0.6, ls="--")
    ax.set_title("SupCon temperature τ")
    ax.set_xlabel("epoch")
    ax.set_ylabel("τ")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"strong-contrastive-v2 training (30 epochs, patience 5, single seed): "
        f"SupCon=0.3 + β-NLL=1.0 + Barlow=0.8, τ_init=0.15  "
        f"(best_val_loss={member['best_val_loss']:.3f} @ epoch {member['best_epoch']} of {len(history)})\n"
        f"All three components plateau over the last 5 epochs (val_supcon ~3.584, "
        f"val_nll ~-0.302, val_barlow ~0.143) — converged on the noise floor.",
        fontsize=10,
    )
    save_fig(fig, OUT / "training_curves.png")


if __name__ == "__main__":
    main()
