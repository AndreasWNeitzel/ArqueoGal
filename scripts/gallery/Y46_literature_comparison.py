"""Y46: head-to-head with the XP literature, RMSE per label.

Per-label grouped bar of paper-reported RMSE on a common label set
(Teff, log g, [M/H], [alpha/M] where available). The compared works
are XP-trained pipelines:

  - ArqueoGal v1.1 (this work),  Tier-1 holdout (8.8e4 stars).
  - Andrae+ 2023, XGBoost on XP coefficients, Gaia DR3 RGB.
  - Zhang+ 2023, data-driven XP forward model, ~2.2e8 stars.
  - Khalatyan+ 2024, CNN on XP, ~1.7e8 stars.

These are paper-quoted numbers, not re-derived; sources are stamped in
the script and printed in the figure caption. APOGEE DR19 internal
precision (Tier-1 floor) is plotted as a dashed line per panel.

Caveat: the four works do NOT share an identical hold-out, the disc
selection cuts differ (Andrae+23 is RGB-only, Zhang+23 has no log-g
floor), and APOGEE DR19 is the truth label for ArqueoGal but a
calibration target for the others. The bars compare order-of-
magnitude precision, not a like-for-like benchmark.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _presentation import OKABE_ITO, PALETTE, apply_style, headline, save  # noqa: E402

# (label, latex, unit, APOGEE floor, dict {paper: rmse}). NaN means the
# paper does not publish a value for this label.
_TABLE = [
    ("teff",    r"$T_\mathrm{eff}$",  "K",   80.0, {
        "ArqueoGal v1.1":  42.4,
        "Andrae+ 2023":    91.0,
        "Zhang+ 2023":     110.0,
        "Khalatyan+ 2024": 105.0,
    }),
    ("logg",    r"$\log g$",          "dex", 0.05, {
        "ArqueoGal v1.1":  0.117,
        "Andrae+ 2023":    0.13,
        "Zhang+ 2023":     0.18,
        "Khalatyan+ 2024": 0.16,
    }),
    ("mh",      r"[M/H]",             "dex", 0.04, {
        "ArqueoGal v1.1":  0.069,
        "Andrae+ 2023":    0.10,
        "Zhang+ 2023":     0.12,
        "Khalatyan+ 2024": 0.11,
    }),
    ("alpha_m", r"[$\alpha$/M]",      "dex", 0.03, {
        "ArqueoGal v1.1":  0.039,
        "Andrae+ 2023":    np.nan,
        "Zhang+ 2023":     0.07,
        "Khalatyan+ 2024": 0.06,
    }),
]

PAPERS = ["ArqueoGal v1.1", "Andrae+ 2023", "Zhang+ 2023", "Khalatyan+ 2024"]
PAPER_COLORS = {
    "ArqueoGal v1.1":  OKABE_ITO[0],   # blue
    "Andrae+ 2023":    OKABE_ITO[2],   # green
    "Zhang+ 2023":     OKABE_ITO[3],   # red-purple
    "Khalatyan+ 2024": OKABE_ITO[4],   # orange
}

_TITLE_KW = dict(fontsize=12, fontweight="regular", color=PALETTE["ink"], pad=6)


def main() -> int:
    apply_style()

    fig, axes = plt.subplots(1, len(_TABLE), figsize=(15.5, 4.6),
                              sharey=False)
    width = 0.6
    n_papers = len(PAPERS)
    x_pos = np.arange(n_papers)

    for ax, (_key, tex, unit, floor, vals) in zip(axes, _TABLE):
        rmse_vals = np.asarray([vals.get(p, np.nan) for p in PAPERS])
        colors = [PAPER_COLORS[p] for p in PAPERS]
        bars = ax.bar(x_pos, rmse_vals, width=width,
                       color=colors, edgecolor="white", linewidth=1.0)
        for b, v in zip(bars, rmse_vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                        f"{v:.3g}", ha="center", va="bottom",
                        fontsize=10, color=PALETTE["ink"])
            else:
                ax.text(b.get_x() + b.get_width() / 2, 0.02,
                        "n/a", ha="center", va="bottom",
                        fontsize=10, color=PALETTE["ash"])
        ax.axhline(floor, color="#000000", lw=1.0, ls="--", alpha=0.85,
                   label=f"APOGEE floor = {floor:g} {unit}")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(PAPERS, rotation=22, ha="right", fontsize=9)
        ax.set_ylabel(f"RMSE ({unit})")
        ymax = float(np.nanmax(rmse_vals)) * 1.30
        ax.set_ylim(0, max(ymax, floor * 1.5))
        ax.set_title(tex, **_TITLE_KW)
        ax.grid(True, axis="y", alpha=0.20)
        ax.legend(loc="upper left", fontsize=8.5, frameon=False)

    fig.subplots_adjust(left=0.05, right=0.985, top=0.78,
                        bottom=0.20, wspace=0.30)
    headline(
        fig,
        "Head-to-head with the XP literature",
        "Paper-quoted RMSE; sources are not on a matched holdout, so "
        "this is order-of-magnitude. APOGEE DR19 internal precision is "
        "the dashed reference per panel.",
        top=0.78,
    )
    save(fig, "Y46_literature_comparison")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
