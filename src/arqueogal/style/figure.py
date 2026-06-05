"""Figure-level helpers shared across every v1.4 figure script."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyBboxPatch  # noqa: F401  (re-exports)

from arqueogal.style.palette import CHROME, TIER

REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO / "reports" / "gallery" / "v1_4" / "figs"


def save(fig, name: str, *, out_dir: Path | None = None) -> tuple[Path, Path]:
    """Save ``fig`` as both PDF (vector) and PNG (raster) at v1.4 spec."""
    out = Path(out_dir) if out_dir is not None else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    pdf_path = out / f"{name}.pdf"
    png_path = out / f"{name}.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight",
                pad_inches=0.10, transparent=True)
    fig.savefig(png_path, dpi=220, bbox_inches="tight",
                pad_inches=0.10, transparent=True)
    plt.close(fig)
    return pdf_path, png_path


def hexbin_density(ax, x, y, *, gridsize: int = 50, mincnt: int = 4,
                   cmap: str = "viridis", lognorm: bool = True,
                   vmin=None, vmax=None, extent=None, alpha: float = 1.0):
    """Standard density panel. log-norm + mincnt-gated by default."""
    norm = LogNorm(vmin=vmin, vmax=vmax) if lognorm else None
    return ax.hexbin(
        x, y, gridsize=gridsize, mincnt=mincnt, cmap=cmap,
        norm=norm, linewidths=0, extent=extent, alpha=alpha,
    )


def median_per_cell(ax, x, y, c, *, gridsize: int = 40, mincnt: int = 4,
                    cmap: str = "viridis", vmin=None, vmax=None,
                    extent=None):
    """Bin (x, y) cells, colour by median(c). Used for kinematic and
    Galactic geometry plots where we want a chemistry surface."""
    return ax.hexbin(
        x, y, C=c, reduce_C_function=np.nanmedian,
        gridsize=gridsize, mincnt=mincnt, cmap=cmap,
        vmin=vmin, vmax=vmax, linewidths=0, extent=extent,
    )


def annotate_corner(ax, text: str, *,
                    loc: Literal["upper left", "upper right",
                                  "lower left", "lower right"] = "upper left",
                    color: str | None = None,
                    fontsize: float = 10.0,
                    bbox_alpha: float = 0.85) -> None:
    """Numeric / RMSE corner annotation with a subtle white backing."""
    color = color or CHROME["body"]
    pad = 0.025
    ha, va, x, y = {
        "upper left":  ("left",  "top",    pad, 1 - pad),
        "upper right": ("right", "top",    1 - pad, 1 - pad),
        "lower left":  ("left",  "bottom", pad, pad),
        "lower right": ("right", "bottom", 1 - pad, pad),
    }[loc]
    ax.text(
        x, y, text, transform=ax.transAxes,
        ha=ha, va=va, fontsize=fontsize, color=color,
        bbox=dict(facecolor="white", edgecolor="none",
                   alpha=bbox_alpha, pad=2.0),
    )


def reference_line(ax, kind: str, **kwargs) -> None:
    """Draw a canonical reference line per the v1.4 conventions.

    kind in {'1to1', 'apogee_floor', 'p99_xp', 'p99_label',
    'solar', 'sun_position', 'zero'}.
    """
    if kind == "1to1":
        lo = max(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = min(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color="#000000",
                lw=0.8, ls="--", alpha=0.85, zorder=3, **kwargs)
    elif kind == "apogee_floor":
        x = kwargs.pop("x", None)
        if x is not None:
            ax.axvline(x, color="#000000", lw=1.0, ls="--",
                       alpha=0.7, zorder=3, **kwargs)
        else:
            ax.axhline(kwargs.pop("y", 1.0), color="#000000",
                       lw=1.0, ls="--", alpha=0.7, zorder=3, **kwargs)
    elif kind == "p99_xp":
        ax.axvline(0.99, color=TIER["T3"], lw=1.4, ls="--",
                   alpha=0.85, zorder=3, **kwargs)
    elif kind == "p99_label":
        ax.axhline(0.99, color=TIER["T2"], lw=1.4, ls="--",
                   alpha=0.85, zorder=3, **kwargs)
    elif kind == "zero":
        ax.axvline(0.0, color="#000000", lw=0.6, ls="-",
                   alpha=0.6, zorder=3, **kwargs)
    elif kind == "sun_position":
        x = kwargs.pop("x", -8.0); y = kwargs.pop("y", 0.0)
        ax.plot([x], [y], marker="*", markerfacecolor="white",
                markeredgecolor="#000000", markeredgewidth=0.8,
                markersize=10, zorder=5, **kwargs)
    elif kind == "solar":
        ax.axhline(0.0, color="#000000", lw=0.6, ls=":",
                   alpha=0.5, zorder=3, **kwargs)
    else:
        raise ValueError(f"unknown reference-line kind: {kind!r}")


def colorbar(ax, mappable, label: str, *, fraction: float = 0.045,
             pad: float = 0.02, **kwargs):
    """Standardised colorbar with the v1.4 typography."""
    cb = plt.colorbar(mappable, ax=ax, fraction=fraction, pad=pad, **kwargs)
    cb.set_label(label, fontsize=10, color=CHROME["body"])
    cb.ax.tick_params(labelsize=10, color=CHROME["body"])
    cb.outline.set_edgecolor(CHROME["body"])
    cb.outline.set_linewidth(0.8)
    return cb


def fit_aspect_axes(ax) -> None:
    """Lock to data aspect (1:1 in data units) for chemistry / Kiel panels."""
    ax.set_aspect("equal", adjustable="box")


def percentile_vlim(values, lo: float = 1.0, hi: float = 99.0):
    """Return (vmin, vmax) at the requested data percentiles."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))
