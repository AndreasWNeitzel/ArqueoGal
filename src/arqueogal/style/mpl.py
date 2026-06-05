"""Canonical matplotlib rcParams for ArqueoGal v1.4 talk figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from cycler import cycler

OKABE_CYCLE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
]


def _register_inter() -> None:
    """Register Inter (and Inter-Italic) if a copy is bundled in the repo."""
    repo = Path(__file__).resolve().parents[3]
    inter_dir = repo / "assets" / "fonts" / "Inter"
    if not inter_dir.exists():
        return
    for f in inter_dir.glob("*.ttf"):
        try:
            fm.fontManager.addfont(str(f))
        except Exception:
            pass


def apply_style() -> None:
    """v1.4 canonical rcParams.

    Per the v1.4 brief: Inter (with DejaVu Sans fallback), 12-pt body,
    13-pt panel-title with weight 600, 11-pt ticks, 0.8-pt spines, no
    suptitle (the slide carries the title), 300-DPI savefig with
    pad_inches=0.10 transparent.
    """
    _register_inter()
    plt.rcParams.update({
        "axes.prop_cycle": cycler(color=OKABE_CYCLE),
        "image.cmap": "viridis",

        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.10,
        "savefig.transparent": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "DejaVu Sans", "Arial"],
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.titleweight": "regular",
        "axes.titlepad": 8,
        "axes.labelsize": 12,
        "axes.labelweight": 400,
        "axes.labelpad": 6,
        "axes.labelcolor": "#2B2D42",
        "axes.titlecolor": "#2B2D42",
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "regular",

        "axes.linewidth": 0.8,
        "axes.edgecolor": "#2B2D42",
        "xtick.color": "#2B2D42",
        "ytick.color": "#2B2D42",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,

        "axes.grid": True,
        "grid.color": "#D0D3DC",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.6,

        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.5,
        "legend.columnspacing": 1.2,

        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "scatter.marker": "o",

        "figure.constrained_layout.use": True,
        "figure.autolayout": False,
    })
