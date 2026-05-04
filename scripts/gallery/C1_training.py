"""C1: Single-seed ensemble training history (real per-epoch metrics).

What this shows:
- Per-epoch joint loss + each loss component (β-NLL, SupCon, Barlow, ARI),
  train and validation for the single production seed.
- Best-val epoch marker.
- Gradient-norm and tau (SupCon temperature) trajectories.

What it reads:
- Pretrain history: models/main/xp_abundances/20260429_1d71682_8870bbf/
  xp_abundances_main_contrastive_seed0_best.pt
- Fine-tune history (canonical):
  models/main/xp_abundances/20260429_1d71682_3790caf_ensemble_5label/
  member_seed0/xp_abundances_main_ensemble_5label_seed0_best.pt
  Each checkpoint stores ``training_metrics["history"]`` per-epoch.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _common import apply_style, save_fig

ENSEMBLE = REPO / "models/main/xp_abundances/20260503_1d71682_2ae55d3_finetune_5label"
PRETRAIN_CKPT = (
    REPO
    / "models/main/xp_abundances/20260503_1d71682_9eae588"
    / "xp_abundances_main_contrastive_seed0_best.pt"
)
OUT = REPO / "reports/gallery/C_training"
SEED_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def _load_history(run_dir: Path) -> dict[int, dict]:
    """Load per-seed training history from a run directory.

    Two structures supported:
      (a) member_seed{N}/ subdirs each holding a *_best.pt — the
          historical 5-member ensemble layout.
      (b) flat layout — a single *_best.pt directly under run_dir,
          treated as seed 0. The single-seed cadence-chain runs use
          this.
    """
    out: dict[int, dict] = {}
    flat_best = sorted(run_dir.glob("*_best.pt"))
    if flat_best:
        ck = torch.load(flat_best[0], map_location="cpu", weights_only=False)
        m = ck.get("training_metrics", {})
        out[0] = {
            "history": m.get("history", []),
            "best_val_loss": float(m.get("best_val_loss", float("nan"))),
            "best_epoch": int(m.get("best_epoch", -1)),
        }
        return out
    for sd in sorted(run_dir.iterdir()):
        if not (sd.is_dir() and sd.name.startswith("member_seed")):
            continue
        seed = int(sd.name.split("seed")[-1])
        for f in sorted(sd.iterdir()):
            if f.suffix == ".pt":
                ck = torch.load(f, map_location="cpu", weights_only=False)
                m = ck.get("training_metrics", {})
                out[seed] = {
                    "history": m.get("history", []),
                    "best_val_loss": float(m.get("best_val_loss", float("nan"))),
                    "best_epoch": int(m.get("best_epoch", -1)),
                }
                break
    return out


def main() -> None:
    apply_style()
    if not ENSEMBLE.exists():
        raise SystemExit(f"missing ensemble dir: {ENSEMBLE}")

    runs = _load_history(ENSEMBLE)
    if not runs:
        raise SystemExit("no checkpoints with training_metrics found")
    print(f"[C1] loaded {len(runs)} seeds")

    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Helper: extract per-epoch arrays from history list of dicts.
    def arr(history, key):
        return np.array([h[key] for h in history], dtype=float)

    components = [
        ("loss",   "Total joint loss"),
        ("nll",    r"$\beta$-NLL"),
        ("supcon", "SupCon"),
        ("barlow", "Barlow Twins"),
        ("ari",    "soft-ARI ([α/M] contamination)"),
    ]

    for idx, (key, title) in enumerate(components):
        ax = axes.flatten()[idx]
        for sid, color in zip(sorted(runs.keys()), SEED_COLORS):
            h = runs[sid]["history"]
            if not h:
                continue
            ep = arr(h, "epoch")
            tr = arr(h, f"train_{key}")
            va = arr(h, f"val_{key}")
            ax.plot(ep, tr, "-", color=color, lw=1.0, alpha=0.65,
                    label=f"seed {sid} train" if idx == 0 else None)
            ax.plot(ep, va, "--", color=color, lw=1.2, alpha=0.85,
                    label=f"seed {sid} val" if idx == 0 else None)
            best_ep = runs[sid]["best_epoch"]
            ax.axvline(best_ep, color=color, lw=0.5, ls=":", alpha=0.4)
        ax.set_xlabel("epoch", fontsize=9)
        ax.set_ylabel(f"{title} loss", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(fontsize=6, loc="upper right", ncol=2)

    # Panel 6: gradient-norm trajectories
    ax = axes[1, 1]
    for sid, color in zip(sorted(runs.keys()), SEED_COLORS):
        h = runs[sid]["history"]
        if not h:
            continue
        ep = arr(h, "epoch")
        ax.plot(ep, arr(h, "train_grad_norm_max"), "-", color=color, lw=1.0,
                alpha=0.7, label=f"seed {sid}")
    ax.axhline(4000, color="black", lw=0.7, ls="--", alpha=0.6, label="abort threshold (4000)")
    ax.set_xlabel("epoch", fontsize=9)
    ax.set_ylabel(r"$\max |\nabla|$", fontsize=9)
    ax.set_title("Gradient norm (per-batch max)", fontsize=9)
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.25)

    # Panel 7: SupCon temperature tau
    ax = axes[1, 2]
    for sid, color in zip(sorted(runs.keys()), SEED_COLORS):
        h = runs[sid]["history"]
        if not h:
            continue
        ep = arr(h, "epoch")
        ax.plot(ep, arr(h, "train_tau"), "-", color=color, lw=1.0, alpha=0.75)
    ax.set_xlabel("epoch", fontsize=9)
    ax.set_ylabel(r"$\tau$", fontsize=9)
    ax.set_title("SupCon temperature", fontsize=9)
    ax.grid(True, alpha=0.25)

    # Panel 8: per-seed best-val summary table
    ax = axes[1, 3]
    ax.set_axis_off()
    rows = ["Per-seed best validation loss", "─" * 32]
    for sid in sorted(runs.keys()):
        rows.append(f"seed {sid}: {runs[sid]['best_val_loss']:.4f} at epoch {runs[sid]['best_epoch']}")
    rows.append("")
    best_vals = [runs[s]["best_val_loss"] for s in runs if runs[s]["history"]]
    if best_vals:
        rows.append(f"Ensemble mean: {np.mean(best_vals):.4f}")
        rows.append(f"Ensemble spread (max−min): {max(best_vals) - min(best_vals):.4f}")
    ax.text(0.0, 1.0, "\n".join(rows), transform=ax.transAxes,
            fontsize=9, ha="left", va="top", family="monospace")

    # Pull the actual loss weights from the run's _best.pt config_yaml so the
    # suptitle reflects what the model was *actually* trained with — no more
    # stale "(SupCon=1.0 + β-NLL=1.0 + Barlow=0.5 + ARI=0.1)" hard-coding
    # disagreeing with the run's real recipe.
    flat_best = sorted(ENSEMBLE.glob("*_best.pt"))
    weights_str = "loss-weights unavailable"
    n_epochs_run = "?"
    if flat_best:
        ck = torch.load(flat_best[0], map_location="cpu", weights_only=False)
        cy = ck.get("config_yaml", "")
        # config_yaml is a dataclass-asdict YAML string; pull the weights via regex.
        import re
        def _get(k):
            m = re.search(rf"\b{k}:\s*([-\d.eE]+)", cy)
            return float(m.group(1)) if m else None
        wb = _get("beta_nll")
        ws = _get("supcon")
        wba = _get("barlow")
        wa = _get("ari")
        parts = []
        if ws is not None: parts.append(f"SupCon={ws:g}")
        if wb is not None: parts.append(rf"$\beta$-NLL={wb:g}")
        if wba is not None: parts.append(f"Barlow={wba:g}")
        if wa is not None: parts.append(f"ARI={wa:g}")
        if parts:
            weights_str = "joint " + " + ".join(parts)
        m = ck.get("training_metrics", {})
        n_epochs_run = (max(h["epoch"] for h in m.get("history", [])) + 1
                        if m.get("history") else "?")

    fig.suptitle(
        "C1. Stream 1 single-seed fine-tune history "
        f"({n_epochs_run} epochs, BS=2048, Kiel-bounded RGB pool: "
        r"$\log g \in [1.0, 3.5]$, $T_{\rm eff} \in [4000, 5500]$ K, 5-label)."
        "\n"
        f"{weights_str}.  Real per-epoch metrics.",
        fontsize=10, fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, OUT / "C1_training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot C1: real ensemble training history.")
    args = parser.parse_args()
    main()
