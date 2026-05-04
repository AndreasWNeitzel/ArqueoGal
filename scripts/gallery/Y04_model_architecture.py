"""Y04: Model architecture schematic.

Single-figure diagram of the MLP regressor: 140-D input,
hidden layers, ten-head output (5 means + 5 sigmas), β-NLL loss.
Cartoon perceptron-stack rather than literal weight matrices.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _presentation import PALETTE, apply_style, headline, save  # noqa: E402


def _column(ax, x, n_visible, n_total, color, label):
    """Vertical stack of `n_visible` circles with a `n_total`-cell label below."""
    y_top, y_bot = 4.4, 1.5
    ys = np.linspace(y_top, y_bot, n_visible)
    for y in ys:
        c = mpatches.Circle((x, y), 0.16, facecolor=color, edgecolor="white", linewidth=1.2)
        ax.add_patch(c)
    if n_total > n_visible:
        ax.text(x, (y_top + y_bot) / 2, "⋮", ha="center", va="center", fontsize=22, color=color)
    ax.text(
        x, 1.1, label, ha="center", va="top", fontsize=11, color=PALETTE["ink"], fontweight="bold"
    )


def _connect(ax, x0, x1, n0, n1):
    y_top, y_bot = 4.4, 1.5
    y0 = np.linspace(y_top, y_bot, n0)
    y1 = np.linspace(y_top, y_bot, n1)
    for a in y0:
        for b in y1:
            ax.plot(
                [x0 + 0.16, x1 - 0.16], [a, b], color=PALETTE["mist"], lw=0.4, alpha=0.55, zorder=0
            )


def main() -> int:
    apply_style()
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0.0, 5.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Layer x-coords: input, h1, h2, h3, output_mu, output_sigma.
    xs = [1.6, 4.4, 7.0, 9.6, 12.6]
    visible = [7, 6, 6, 6, 5]
    totals = [140, 256, 256, 256, 5]
    labels = ["INPUT\n140-D", "Hidden\n256", "Hidden\n256", "Hidden\n256", "OUTPUT\n5 means + 5 σ"]
    colors = [
        PALETTE["navy_light"],
        PALETTE["navy"],
        PALETTE["navy"],
        PALETTE["navy"],
        PALETTE["accent"],
    ]

    # Connection lines BEHIND the circles.
    for i in range(len(xs) - 1):
        _connect(ax, xs[i], xs[i + 1], visible[i], visible[i + 1])

    for x, nv, nt, c, lab in zip(xs, visible, totals, colors, labels):
        _column(ax, x, nv, nt, c, lab)

    # Output split markers.
    out_y = np.linspace(4.4, 1.5, 5)
    out_labels = ["Teff", "log g", "[M/H]", "[α/M]", "[Mg/H]"]
    for y, name in zip(out_y, out_labels):
        ax.text(
            xs[-1] + 0.55,
            y,
            f"{name}  μ, σ",
            ha="left",
            va="center",
            fontsize=11.5,
            color=PALETTE["ink"],
            fontweight="bold",
        )

    # Loss / training annotation strip — well below the layer labels at y=1.1.
    ax.text(
        7.0,
        0.35,
        r"Loss: per-element $\beta$-NLL ($\beta=0$, plain Gaussian NLL).  "
        r"Optimiser: AdamW.  Training data: Stream 1 train split (70%, seed=0).",
        ha="center",
        va="center",
        fontsize=11,
        fontstyle="italic",
        color=PALETTE["ash"],
        bbox=dict(boxstyle="round,pad=0.4", facecolor=PALETTE["paper"], edgecolor=PALETTE["mist"]),
    )

    headline(
        fig,
        "Model — single MLP, ten output heads",
        "5 mean labels + 5 per-star σ heads.  Identical architecture for all three streams.",
        top=0.86,
    )
    save(fig, "Y04_model_architecture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
