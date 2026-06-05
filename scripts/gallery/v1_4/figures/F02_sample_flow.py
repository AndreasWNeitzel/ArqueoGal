"""F02: sample-flow cascade + log-y volume bar (slide 3)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from arqueogal.style import CHROME, OKABE_ITO, apply_style, save  # noqa: E402

STAGES = [
    ("Gaia DR3 XP",   219_197_643, "BP/RP coefficients"),
    ("S1 raw join",   326_724,     "x APOGEE DR19"),
    ("S1 dedup",      292_948,     "one row / source_id"),
    ("Holdout",       87_882,      "val + test, seed = 0"),
    ("Tier 1",        82_548,      "ood_joint = 0 + label-Mahal cut"),
]


def main() -> int:
    apply_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(13.0, 4.5),
        gridspec_kw=dict(width_ratios=[3.0, 1.0]),
        layout="constrained",
    )

    ax = axes[0]
    ax.set_xlim(0, 13.6); ax.set_ylim(0, 1.0)
    ax.set_aspect("auto"); ax.axis("off")

    n = len(STAGES)
    box_w = 2.45; box_h = 0.46
    centres = [0.5 + box_w / 2 + i * (13.6 - 1.0 - box_w) / (n - 1)
               for i in range(n)]
    y_box = 0.52

    for i, ((label, count, sub), x) in enumerate(zip(STAGES, centres)):
        rect = FancyBboxPatch(
            (x - box_w / 2, y_box - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor="white", edgecolor=OKABE_ITO["blue"], linewidth=1.0,
        )
        ax.add_patch(rect)
        ax.text(x, y_box + 0.06, f"{count:,}",
                ha="center", va="center", fontsize=15,
                color=OKABE_ITO["blue"])
        ax.text(x, y_box - 0.10, label,
                ha="center", va="center", fontsize=10,
                color=CHROME["body"])
        ax.text(x, y_box - box_h / 2 - 0.12, sub,
                ha="center", va="center", fontsize=8.5,
                color=CHROME["muted"], style="italic")
        if i < n - 1:
            x_next = centres[i + 1]
            arrow_lo = x + box_w / 2 + 0.04
            arrow_hi = x_next - box_w / 2 - 0.04
            ax.annotate(
                "", xy=(arrow_hi, y_box), xytext=(arrow_lo, y_box),
                arrowprops=dict(arrowstyle="->", color=CHROME["muted"],
                                lw=1.5, shrinkA=2, shrinkB=2,
                                mutation_scale=14),
            )
            survival = STAGES[i + 1][1] / count
            ax.text(
                (arrow_lo + arrow_hi) / 2, y_box + box_h / 2 + 0.20,
                f"keeps {survival * 100:.2f}%",
                ha="center", va="bottom", fontsize=9,
                color=CHROME["muted"],
            )

    ax = axes[1]
    counts = [s[1] for s in STAGES]
    labels = [s[0] for s in STAGES]
    ypos = np.arange(n)[::-1]
    bars = ax.barh(ypos, counts, color=OKABE_ITO["blue"],
                    edgecolor="white", linewidth=0.8)
    for y, c in zip(ypos, counts):
        ax.text(c * 1.10, y, f"{c:,}", va="center", ha="left",
                fontsize=9, color=CHROME["body"])
    ax.set_xscale("log")
    ax.set_xlim(1e4, 1e10)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel(r"sample size (log)")
    ax.set_title(r"cohort sizes")
    ax.grid(False)
    # Hide top + right spines AND their ticks (keep left + bottom only).
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(top=False, right=False, which="both")
    ax.minorticks_off()

    save(fig, "F02_sample_flow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
