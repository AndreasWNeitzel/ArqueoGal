"""Shared style and helpers for the Y_presentation gallery.

Per docs/STYLE_GUIDE.md: Okabe-Ito categorical palette, viridis sequential,
white background slide chrome.  16:9 slide-ready PNG output at 300 DPI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports" / "gallery" / "Y_presentation"


# Okabe-Ito palette per STYLE_GUIDE.md.
OKABE_ITO = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
]

# Slide chrome + tier semantic mapping. Tier colours pulled from Okabe-Ito
# slots (green=ok, vermilion=caution, black-on-white as the strongest
# rejection signal — but we keep the legacy 'd62728'-style red here for
# tier3 because the tier semantic chain T1-T2-T3 reads better with three
# distinct hues than green/vermilion/black).
PALETTE: dict[str, str] = {
    "ink": "#2B2D42",
    "subink": "#5C6378",
    "mist": "#D0D3DC",
    "paper": "#F8F9FB",
    "white": "#FFFFFF",
    "title": OKABE_ITO[0],  # title accent = blue
    "accent": OKABE_ITO[1],  # highlight accent = vermilion
    # Tier mapping (H5/H6/H7/H8/H10).
    "tier1": OKABE_ITO[2],  # green = science-grade
    "tier2": OKABE_ITO[1],  # vermilion = label-Mahalanobis caution
    "tier3": OKABE_ITO[3],  # red-purple = XP-Mahalanobis hard reject
    # Comparison rows (F6 narrative).
    "ours": OKABE_ITO[0],
    "gspspec": OKABE_ITO[3],
    "apogee": OKABE_ITO[2],
    # Legacy aliases used by older Y scripts.
    "navy": OKABE_ITO[0],
    "navy_light": "#56B4E9",
    "ash": "#5C6378",
    "accent_light": "#FFB14E",
}


def apply_style() -> None:
    """Set matplotlib rcParams to slide-grade defaults per STYLE_GUIDE.md."""
    mpl.rcParams.update(
        {
            "axes.prop_cycle": cycler(color=OKABE_ITO),
            "image.cmap": "viridis",
            "text.usetex": False,
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "it",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 14.0,
            "axes.titlesize": 18.0,
            "axes.titleweight": "bold",
            "axes.titlepad": 12.0,
            "axes.labelsize": 14.0,
            "axes.labelweight": "regular",
            "axes.labelcolor": PALETTE["ink"],
            "axes.edgecolor": PALETTE["ink"],
            "axes.linewidth": 1.2,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": PALETTE["mist"],
            "grid.alpha": 0.45,
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "legend.frameon": False,
            "legend.fontsize": 12.0,
            "legend.title_fontsize": 12.0,
            "lines.linewidth": 1.8,
            "lines.markersize": 6.0,
            "figure.dpi": 110,
            "savefig.dpi": 220,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.unicode_minus": False,
        }
    )


def stamp(fig: plt.Figure, text: str = "ArqueoGal | Pipeline 1") -> None:
    """No-op as of 2026-05-03 — slide-attribution stamp was removed per
    user feedback (showed up everywhere, added no value).  The function
    signature is kept so existing callers don't break.
    """
    return None


def headline(
    fig: plt.Figure,
    title: str,
    subtitle: str | None = None,
    *,
    top: float = 0.86,
) -> None:
    """Big headline title spanning the figure, optional subtitle below.

    `top` sets the upper edge of the axes region (subplots_adjust top), so the
    headline never collides with panel titles. Tune lower (e.g. 0.80) when
    the figure has tall panel titles, higher (e.g. 0.92) for schematic figures
    with no axes.
    """
    fig.subplots_adjust(top=top)
    fig.suptitle(
        title,
        fontsize=22,
        fontweight="bold",
        color=PALETTE["ink"],
        x=0.02,
        y=0.985,
        ha="left",
        va="top",
    )
    if subtitle:
        fig.text(
            0.02,
            0.945,
            subtitle,
            ha="left",
            va="top",
            fontsize=14,
            color=PALETTE["ash"],
        )


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure to OUT/{name}.png with a stamp.

    PDF emission was frozen 2026-05-03; gallery is review-only at this stage.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    stamp(fig)
    path = OUT / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white", format="png")
    print(f"[Y] wrote {path.relative_to(REPO)}")
    plt.close(fig)
