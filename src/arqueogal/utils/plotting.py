"""Publication-quality plotting — utils/DESIGN.md + A&A format.

Infrastructure inherited from ``TESS_ML/plot_config.py`` with the
improvements listed in the DESIGN:

- A&A single/double column widths,
- LaTeX detection with graceful ``usetex=False`` fallback,
- colourblind-safe palette option,
- helpers: hexbin, density_2d, residual_panel, coverage_curve, save_figure.

matplotlib is imported inside the helpers so ``from arqueogal.utils.plotting
import *`` doesn't pull in matplotlib when a caller only wants e.g.
:func:`set_aa_style`. This matters on the HPC node where headless plotting
is expected.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

# A&A style figure widths in inches (A&A provides widths in mm).
AA_SINGLE_COLUMN_IN: float = 8.8 / 2.54
AA_DOUBLE_COLUMN_IN: float = 18.3 / 2.54

# Colorblind-safe 8-colour palette (Wong 2011, Nature Methods 8:441).
WONG_PALETTE: tuple[str, ...] = (
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
)


def _latex_available() -> bool:
    """True if a system LaTeX binary is on PATH."""
    return shutil.which("latex") is not None


def set_aa_style(
    *,
    usetex: bool | None = None,
    colorblind: bool = True,
    font_size: float = 9.0,
) -> None:
    """Apply A&A-style rcParams.

    Parameters
    ----------
    usetex
        Force LaTeX rendering on/off. ``None`` (default) auto-detects:
        enable if ``latex`` is on PATH, warn + disable otherwise.
    colorblind
        Use Wong 2011 colorblind-safe palette.
    font_size
        Base font size in pt (A&A body text ≈ 9 pt).
    """
    import warnings

    import matplotlib as mpl

    if usetex is None:
        usetex = _latex_available()
        if not usetex:
            warnings.warn(
                "set_aa_style: system LaTeX not found — falling back to "
                "mathtext. Install texlive to enable LaTeX rendering.",
                stacklevel=2,
            )

    rc: dict[str, Any] = {
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "legend.fontsize": font_size - 1,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "legend.frameon": False,
        "text.usetex": bool(usetex),
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
    if colorblind:
        rc["axes.prop_cycle"] = mpl.cycler(color=list(WONG_PALETTE))
    mpl.rcParams.update(rc)


def hexbin_with_colorbar(  # noqa: PLR0913 — matplotlib-style plotting kwargs
    ax: Any,
    x: Any,
    y: Any,
    *,
    gridsize: int = 60,
    cmap: str = "viridis",
    bins: str | None = "log",
    mincnt: int = 1,
    label: str = "counts",
) -> Any:
    """Hexbin density plot + colorbar — preferred over raw scatter for N > 5 k."""
    import matplotlib.pyplot as plt

    hb = ax.hexbin(x, y, gridsize=gridsize, cmap=cmap, bins=bins, mincnt=mincnt)
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label(label)
    return hb


def density_2d(  # noqa: PLR0913 — matplotlib-style plotting kwargs
    ax: Any,
    x: Any,
    y: Any,
    *,
    bins: int | tuple[int, int] = 80,
    cmap: str = "viridis",
    normalize: bool = True,
) -> Any:
    """2-D histogram with optional row-sum normalisation."""
    import numpy as np

    H, xe, ye = np.histogram2d(x, y, bins=bins)
    if normalize:
        total = H.sum()
        if total > 0:
            H = H / total
    im = ax.imshow(
        H.T,
        origin="lower",
        extent=(xe[0], xe[-1], ye[0], ye[-1]),
        aspect="auto",
        cmap=cmap,
    )
    return im


def residual_panel(
    ax: Any,
    y_true: Any,
    y_pred: Any,
    *,
    label: str = "",
    sigma: Any | None = None,
) -> None:
    """Plot (y_pred - y_true) vs y_true with optional ±1σ band."""
    import numpy as np

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    resid = y_pred - y_true
    ax.axhline(0.0, color="0.5", lw=0.6, ls="--")
    ax.scatter(y_true, resid, s=2, alpha=0.3, rasterized=True)
    if sigma is not None:
        sigma = np.asarray(sigma)
        order = np.argsort(y_true)
        ax.fill_between(
            y_true[order],
            -sigma[order],
            sigma[order],
            alpha=0.2,
            color="tab:blue",
            label="±1σ",
        )
    if label:
        ax.set_ylabel(f"{label} residual")


def coverage_curve(
    ax: Any,
    sigma: Any,
    error: Any,
    *,
    n_sigmas: Any = None,
    label: str | None = None,
) -> None:
    """Plot empirical P(|ε| < nσ) vs nσ against the Gaussian expectation.

    A perfectly calibrated uncertainty σ gives the CDF of the folded
    normal on ``n_sigmas``: ``erf(n / sqrt(2))``.
    """
    import numpy as np
    from scipy.special import erf

    sigma = np.asarray(sigma, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    if n_sigmas is None:
        n_sigmas = np.linspace(0.0, 3.0, 31)
    n_sigmas = np.asarray(n_sigmas, dtype=np.float64)

    # Avoid div-by-zero: drop σ ≤ 0.
    mask = sigma > 0
    z = np.abs(error[mask]) / sigma[mask]
    empirical = np.array([float(np.mean(z < n)) for n in n_sigmas])
    expected = erf(n_sigmas / np.sqrt(2.0))

    ax.plot(n_sigmas, empirical, label=label or "empirical")
    ax.plot(n_sigmas, expected, "--", color="0.5", label="Gaussian expected")
    ax.set_xlabel(r"$n\sigma$")
    ax.set_ylabel(r"$P(|\varepsilon| < n\sigma)$")
    ax.set_xlim(n_sigmas.min(), n_sigmas.max())
    ax.set_ylim(0, 1)


def save_figure(
    fig: Any,
    path: str | Path,
    *,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int | None = None,
) -> list[Path]:
    """Save ``fig`` to ``path.<ext>`` for each ``ext`` in ``formats``.

    Writes atomically (temp → rename). Returns the list of paths written.
    """
    path = Path(path).with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for fmt in formats:
        target = path.with_suffix(f".{fmt}")
        tmp = target.with_suffix(target.suffix + ".tmp")
        kwargs: dict[str, Any] = {}
        if dpi is not None:
            kwargs["dpi"] = dpi
        fig.savefig(tmp, format=fmt, **kwargs)
        tmp.replace(target)
        out.append(target)
    return out


__all__ = [
    "AA_DOUBLE_COLUMN_IN",
    "AA_SINGLE_COLUMN_IN",
    "WONG_PALETTE",
    "coverage_curve",
    "density_2d",
    "hexbin_with_colorbar",
    "residual_panel",
    "save_figure",
    "set_aa_style",
]
