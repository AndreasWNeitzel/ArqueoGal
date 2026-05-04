"""Y13: The XP-abundance catalog landscape — where ArqueoGal sits.

Two-panel summary of the published Gaia DR3 XP → stellar-label catalogs
(Andrae+2023 GSP-Phot release, AspGap = Li+2024, Guiglion+2024 hybrid CNN,
Zhang+2023 XP-NN, Fallows+Sanders 2024, Andrae+2023 XGBoost-on-XP) compared
to ArqueoGal v1.

  Panel A: catalog size (Gaia DR3 sources covered)  — log axis
  Panel B: methodological scoreboard
              calibrated σ ?    individual-element coverage ?
              information audit ?  source code public ?

Numbers and references hard-coded from research_brief §2.1; not read from
the catalog parquets at runtime.
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


# (name, n_stars_million, n_labels, calibrated_sigma, info_audit, code_public, year)
CATALOGS = [
    ("Andrae+2023\nGSP-Phot",       470, 4,  False, False, False, 2023),
    ("Zhang+2023\nXP-NN",           220, 5,  False, False, True,  2023),
    ("Andrae+2023\nXGBoost-on-XP",  175, 2,  False, False, True,  2023),
    ("Fallows+Sanders\n2024",       175, 1,  False, False, True,  2024),
    ("AspGap\n(Li+2024)",            23, 17, False, False, True,  2024),
    ("Guiglion+2024\nhybrid CNN",     0.886, 5, False, False, False, 2024),
    ("ArqueoGal v1\n(this work)",     0.614, 5, True,  True,  True,  2026),
]


def _panel_a(ax):
    names = [c[0] for c in CATALOGS]
    n_stars = [c[1] for c in CATALOGS]
    is_us = ["ArqueoGal" in n for n in names]
    colors = [PALETTE["accent"] if u else PALETTE["navy_light"] for u in is_us]

    y = np.arange(len(names))
    bars = ax.barh(y, n_stars, color=colors, edgecolor="white", linewidth=1.4,
                   height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.3, 1500)
    ax.set_xlabel("number of stars (millions)")
    ax.set_title("Catalog size",
                 color=PALETTE["navy"], fontsize=15)
    for bar, val in zip(bars, n_stars):
        if val < 1:
            label = f"{val*1000:.0f}k"
        elif val < 10:
            label = f"{val:.1f}M"
        else:
            label = f"{val:.0f}M"
        ax.text(val * 1.18, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=11, fontweight="bold",
                color=PALETTE["ink"])
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)


def _panel_b(ax):
    names = [c[0] for c in CATALOGS]
    flags = np.array([
        [c[3], c[4], c[5]] for c in CATALOGS
    ], dtype=bool)
    label_axis = ["calibrated σ", "info-content audit", "open source"]

    is_us = np.array(["ArqueoGal" in n for n in names])
    n = len(names)

    ax.set_xlim(-0.5, len(label_axis) - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks(range(len(label_axis)))
    ax.set_xticklabels(label_axis, fontsize=12, rotation=20, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()
    ax.set_title("Methodological scoreboard",
                 color=PALETTE["navy"], fontsize=15)
    ax.grid(False)

    for i in range(n):
        # Row stripe for ArqueoGal.
        if is_us[i]:
            ax.add_patch(mpatches.Rectangle(
                (-0.5, i - 0.5), len(label_axis), 1.0,
                facecolor=PALETTE["paper"], edgecolor="none", zorder=0,
            ))
        for j in range(len(label_axis)):
            ok = flags[i, j]
            color = PALETTE["tier1"] if ok else PALETTE["tier3"]
            sym = "✓" if ok else "✗"
            ax.scatter(j, i, s=460, c=color, edgecolor="white", linewidth=1.6,
                       zorder=2)
            ax.text(j, i, sym, ha="center", va="center", color="white",
                    fontsize=15, fontweight="bold", zorder=3)
    ax.tick_params(left=False, bottom=False)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)


def main() -> int:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(20, 8),
                             gridspec_kw={"width_ratios": [1.0, 1.0]})
    plt.subplots_adjust(wspace=0.55)
    _panel_a(axes[0])
    _panel_b(axes[1])

    headline(
        fig,
        "Where ArqueoGal sits in the XP landscape",
        "Smaller than the AspGap / Andrae+23 catalogs by design — "
        "depth-of-treatment over reach. Calibrated σ + audit are the differentiators.",
        top=0.80,
    )
    save(fig, "Y13_landscape")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
