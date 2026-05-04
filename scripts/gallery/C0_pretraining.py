"""C0: Contrastive-pretraining loss history per epoch.

Pretraining stage of the two-stage (or joint) Pipeline-1 recipe — the
encoder learns a label-aware contrastive embedding under SupCon-soft, with
optional Barlow-Twins decorrelation. β-NLL is *not* active here; the
supervised head only joins at the fine-tune stage (see C1).

Reads ``training_metrics["history"]`` from the run's ``_best.pt``. The run
location is hard-coded to the latest cadence-chain pretrain output; update
``PRETRAIN_DIR`` when retraining.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _common import apply_style, save_fig  # noqa: E402

PRETRAIN_DIR = REPO / "models/main/xp_abundances/20260503_1d71682_9eae588"
OUT = REPO / "reports/gallery/C_training"
SEED_COLOR = "#1f77b4"


def _load_history(run_dir: Path) -> dict:
    bests = sorted(run_dir.glob("*_best.pt"))
    if not bests:
        raise SystemExit(f"no _best.pt in {run_dir}")
    ck = torch.load(bests[0], map_location="cpu", weights_only=False)
    m = ck.get("training_metrics", {})
    return {
        "history": m.get("history", []),
        "best_val_loss": float(m.get("best_val_loss", float("nan"))),
        "best_epoch": int(m.get("best_epoch", -1)),
        "config_yaml": ck.get("config_yaml", ""),
    }


def _weights_from_yaml(cy: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ("supcon", "beta_nll", "barlow", "ari"):
        mre = re.search(rf"\b{k}:\s*([-\d.eE]+)", cy)
        if mre:
            out[k] = float(mre.group(1))
    return out


def main() -> None:
    apply_style()
    if not PRETRAIN_DIR.exists():
        raise SystemExit(f"missing pretrain dir: {PRETRAIN_DIR}")

    run = _load_history(PRETRAIN_DIR)
    h = run["history"]
    if not h:
        raise SystemExit("no per-epoch history found in _best.pt")
    print(f"[C0] {len(h)} epochs of pretrain history")

    OUT.mkdir(parents=True, exist_ok=True)

    def arr(key: str) -> np.ndarray:
        return np.array([row[key] for row in h], dtype=float)

    weights = _weights_from_yaml(run["config_yaml"])
    # Active components in pretrain: anything with a non-zero weight. SupCon
    # is always shown (it's the load-bearing loss); β-NLL is always omitted
    # (deliberately disabled at pretrain).
    components: list[tuple[str, str, str]] = [
        ("supcon", "SupCon", "soft-positive InfoNCE"),
    ]
    if weights.get("barlow", 0.0) > 0:
        components.append(("barlow", "Barlow Twins", "decorrelation"))
    if weights.get("ari", 0.0) > 0:
        components.append(("ari", "soft-ARI", "α-bimodality"))
    components.append(("loss", "Total weighted loss", "all components"))

    n_panels = len(components)
    fig, axes = plt.subplots(2, max(n_panels, 2),
                              figsize=(4.0 * max(n_panels, 2), 8))
    axes = axes if n_panels > 1 else axes.reshape(2, -1)

    for j, (key, name, sub) in enumerate(components):
        ax = axes[0, j]
        ep = arr("epoch")
        try:
            tr = arr(f"train_{key}")
            va = arr(f"val_{key}")
        except (KeyError, ValueError):
            ax.text(0.5, 0.5, f"no '{key}' history", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray")
            ax.set_title(f"{name}\n({sub})", fontsize=10)
            continue
        ax.plot(ep, tr, "-", color=SEED_COLOR, lw=1.0, alpha=0.65,
                label="train")
        ax.plot(ep, va, "--", color=SEED_COLOR, lw=1.4, alpha=0.95,
                label="val")
        ax.axvline(run["best_epoch"], color="black", lw=0.6, ls=":",
                    alpha=0.5, label=f"best ep {run['best_epoch']}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(f"{name} loss")
        ax.set_title(f"{name}\n({sub})", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

    # Bottom row, panel 1: SupCon temperature τ.
    ax = axes[1, 0]
    try:
        ep = arr("epoch")
        tau = arr("train_tau")
        ax.plot(ep, tau, "-", color=SEED_COLOR, lw=1.2)
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"$\tau$")
        ax.set_title(r"SupCon temperature $\tau$ (learned)")
        ax.grid(True, alpha=0.25)
    except (KeyError, ValueError):
        ax.text(0.5, 0.5, "no τ history", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")

    # Bottom row, panel 2: gradient norms.
    ax = axes[1, 1]
    try:
        ep = arr("epoch")
        ax.plot(ep, arr("train_grad_norm_max"), "-", color="#d62728",
                lw=1.0, alpha=0.85, label="max")
        ax.plot(ep, arr("train_grad_norm_mean"), "-", color="#2ca02c",
                lw=1.0, alpha=0.85, label="mean")
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"$|\nabla|$")
        ax.set_title("Gradient norm (per-batch)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    except (KeyError, ValueError):
        ax.text(0.5, 0.5, "no grad-norm history", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")

    # Hide remaining unused bottom-row panels.
    for j in range(2, n_panels):
        axes[1, j].set_axis_off()

    weights_str = " + ".join(
        f"{k}={v:g}" for k, v in weights.items() if v > 0
    ) or "no weights parsed"
    fig.suptitle(
        "C0. Stream 1 contrastive-pretraining history "
        f"({len(h)} epochs, encoder + projection head, "
        r"5-label kernel)."
        "\n"
        f"weights: {weights_str}.  Real per-epoch metrics.",
        fontsize=10, fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_fig(fig, OUT / "C0_pretraining")


if __name__ == "__main__":
    main()
