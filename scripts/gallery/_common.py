"""Shared helpers for gallery plotting scripts."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_EXTERNAL = ROOT / "data" / "external"
GALLERY = ROOT / "reports" / "gallery"


def apply_style() -> None:
    mpl.rcParams.update({
        "text.usetex": False,
        # Use matplotlib's default mathtext (italic-serif) so $T_{\rm eff}$,
        # $\log g$, $\in$ and friends render as proper math. "regular" was
        # making math look like plain text.
        "mathtext.fontset": "dejavuserif",
        "mathtext.default": "it",
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.family": "DejaVu Sans",
        "font.size": 10.0,
        "axes.unicode_minus": False,
        "axes.labelsize": 10.0,
        "axes.titlesize": 11.0,
        "axes.titleweight": "semibold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 9.0,
        "figure.autolayout": False,
    })


PALETTE = {
    "apogee": "#1f77b4",
    "tess": "#2ca02c",
    "andrae_volume": "#d62728",
    "andrae_uniform": "#9467bd",
    "v1": "#555555",
    "v11": "#d62728",
    "v12": "#1f77b4",
    "edenhofer": "#1f77b4",
    "lallement": "#2ca02c",
    "sfd": "#ff7f0e",
    "nbhd": "#d62728",
    "ok": "#2ca02c",
    "bad": "#d62728",
    "neutral": "#7f7f7f",
}


def radec_to_galactic(ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert ICRS (ra, dec) in degrees to Galactic (l, b) in degrees."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    c = SkyCoord(ra=np.asarray(ra_deg, dtype=float) * u.deg,
                 dec=np.asarray(dec_deg, dtype=float) * u.deg, frame="icrs")
    g = c.galactic
    return g.l.degree, g.b.degree


def galactic_mollweide(l_deg: np.ndarray, b_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wrap (l, b) into Mollweide x, y with RIGHT-TO-LEFT Galactic-longitude convention.

    Standard convention in Galactic-coord sky maps: l=0 at centre, l increasing to the
    LEFT (so the Galactic plane "rolls" the correct way). Matplotlib's mollweide is
    inherently left-to-right, so we negate longitude after wrapping to [-180, 180).
    """
    lon = np.asarray(l_deg, dtype=float).copy()
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    return -np.deg2rad(lon), np.deg2rad(np.asarray(b_deg, dtype=float))


def radec_to_galactic_mollweide(ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    l, b = radec_to_galactic(ra_deg, dec_deg)
    return galactic_mollweide(l, b)


def style_galactic_mollweide(ax: plt.Axes) -> None:
    """Configure a Mollweide axes for Galactic-coord sky maps."""
    ax.grid(True, linewidth=0.4, alpha=0.35, color="#888888")
    # Labels at every 60° — the NEGATION in galactic_mollweide means we must
    # invert the numeric labels so they read correctly (l increasing right-to-left).
    xticks_rad = np.deg2rad([-120, -60, 0, 60, 120])
    xticks_lbl = ["120°", "60°", "0°", "300°", "240°"]
    ax.set_xticks(xticks_rad)
    ax.set_xticklabels(xticks_lbl, fontsize=8, color="#555")
    yticks_rad = np.deg2rad([-60, -30, 0, 30, 60])
    ax.set_yticks(yticks_rad)
    ax.set_yticklabels([f"{d}°" for d in [-60, -30, 0, 30, 60]], fontsize=8, color="#555")


# kept for back-compat; new callers should prefer radec_to_galactic_mollweide
def ra_dec_to_mollweide(ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(ra_deg, dtype=float).copy()
    lon[lon > 180.0] -= 360.0
    return -np.deg2rad(lon), np.deg2rad(np.asarray(dec_deg, dtype=float))


def galactic_to_mollweide(l_deg: np.ndarray, b_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return galactic_mollweide(l_deg, b_deg)


def add_mollweide_grid(ax: plt.Axes) -> None:
    style_galactic_mollweide(ax)


def sample_index(n_rows: int, n_target: int, rng: np.random.Generator | None = None) -> np.ndarray:
    if n_target >= n_rows:
        return np.arange(n_rows)
    rng = rng or np.random.default_rng(42)
    return rng.choice(n_rows, size=n_target, replace=False)


def save_fig(fig: plt.Figure, path: Path, *, tight: bool = True) -> None:
    # Re-assert usetex=False: some third-party imports (gaiaxpy, dust map helpers)
    # can flip this back on globally. Guards the gallery against their side-effects.
    mpl.rcParams["text.usetex"] = False
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[gallery] wrote {path.relative_to(ROOT)}")


def caption_box(ax: plt.Axes, text: str, *, loc: str = "lower right") -> None:
    ax.text(
        0.98 if "right" in loc else 0.02,
        0.03 if "lower" in loc else 0.97,
        text,
        transform=ax.transAxes,
        ha="right" if "right" in loc else "left",
        va="bottom" if "lower" in loc else "top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.85),
    )
