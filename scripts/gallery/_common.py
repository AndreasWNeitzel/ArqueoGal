"""Shared helpers for gallery plotting scripts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import pandas as pd  # noqa: F401  (used only in quoted type hints)

ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_EXTERNAL = ROOT / "data" / "external"
GALLERY = ROOT / "reports" / "gallery"


from cycler import cycler

# Okabe-Ito categorical palette per docs/STYLE_GUIDE.md.
OKABE_ITO = [
    "#0072B2",  # 0 blue        — primary series, default
    "#D55E00",  # 1 vermillion  — secondary / contrast
    "#009E73",  # 2 green       — model / fit
    "#CC79A7",  # 3 red-purple  — tertiary
    "#E69F00",  # 4 orange      — quaternary
    "#56B4E9",  # 5 sky blue    — companion
    "#F0E442",  # 6 yellow      — needs edge color on white
    "#000000",  # 7 black       — reference, observed data, 1:1 line
]


def apply_style() -> None:
    """Apply the project STYLE_GUIDE.md (Okabe-Ito + Wong 2011 CVD-safe).

    Updates rcParams to the slide-grade defaults documented in
    docs/STYLE_GUIDE.md. Call once at the top of every plotting script.
    """
    mpl.rcParams.update(
        {
            # Color cycle + default cmap.
            "axes.prop_cycle": cycler(color=OKABE_ITO),
            "image.cmap": "viridis",
            # Disable usetex (mathtext only — defends against gaiaxpy etc. flipping it on).
            "text.usetex": False,
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "it",
            # Geometry.
            "figure.figsize": (8, 5),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "savefig.transparent": False,  # gallery PNGs render on white; keep opaque
            # Typography.
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 12.0,
            "axes.titlesize": 14.0,
            "axes.titleweight": "semibold",
            "axes.labelsize": 12.0,
            "legend.fontsize": 11.0,
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
            "axes.unicode_minus": False,
            # Axes chrome.
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#2B2D42",
            "axes.labelcolor": "#2B2D42",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.color": "#2B2D42",
            "ytick.color": "#2B2D42",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            # Grid.
            "axes.grid": True,
            "grid.color": "#D0D3DC",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.6,
            # Legend / lines / markers.
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "lines.markersize": 6.0,
            "scatter.marker": "o",
            "figure.autolayout": False,
        }
    )


# Palette aliases. Older callers used semantic names ("ok", "bad", etc.)
# which are kept as aliases over the Okabe-Ito base so existing scripts
# don't break, but new code should prefer OKABE_ITO[i] directly.
PALETTE = {
    # Okabe-Ito by index.
    "ok_blue": OKABE_ITO[0],
    "ok_vermilion": OKABE_ITO[1],
    "ok_green": OKABE_ITO[2],
    "ok_purple": OKABE_ITO[3],
    "ok_orange": OKABE_ITO[4],
    "ok_sky": OKABE_ITO[5],
    "ok_yellow": OKABE_ITO[6],
    "ok_black": OKABE_ITO[7],
    # Legacy semantic aliases — remap to Okabe-Ito.
    "apogee": OKABE_ITO[0],
    "tess": OKABE_ITO[2],
    "andrae_volume": OKABE_ITO[1],
    "andrae_uniform": OKABE_ITO[3],
    "v1": "#5C6378",
    "v11": OKABE_ITO[1],
    "v12": OKABE_ITO[0],
    "edenhofer": OKABE_ITO[0],
    "lallement": OKABE_ITO[2],
    "sfd": OKABE_ITO[1],
    "nbhd": OKABE_ITO[3],
    "ok": OKABE_ITO[2],
    "bad": OKABE_ITO[1],
    "neutral": "#5C6378",
}


def radec_to_galactic(ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert ICRS (ra, dec) in degrees to Galactic (l, b) in degrees."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    c = SkyCoord(
        ra=np.asarray(ra_deg, dtype=float) * u.deg,
        dec=np.asarray(dec_deg, dtype=float) * u.deg,
        frame="icrs",
    )
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


def radec_to_galactic_mollweide(
    ra_deg: np.ndarray, dec_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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


def save_fig(
    fig: plt.Figure,
    path: Path,
    *,
    tight: bool = True,
    formats: tuple[str, ...] = ("png",),
) -> None:
    """Save a figure to disk as PNG only.

    PDF generation was frozen 2026-05-03 — gallery is review-only at this
    stage and slide-import PDFs add disk + render cost without value yet.
    The ``formats`` arg is honoured but PDF / SVG / EPS are silently
    dropped here so callers that still pass ``("pdf", "png")`` keep
    working without producing PDFs.

    Parameters
    ----------
    fig : plt.Figure
        The figure to save.
    path : Path or str
        Output file path (without extension; will be suffixed by format).
    tight : bool, optional
        If True, apply tight_layout() only if no constrained layout is active.
    formats : tuple of str, optional
        Format extensions requested. Non-PNG entries are dropped — see above.
    """
    mpl.rcParams["text.usetex"] = False
    path = Path(path) if not isinstance(path, Path) else path
    path_base = path.with_suffix("")
    path_base.parent.mkdir(parents=True, exist_ok=True)

    if tight and fig.get_layout_engine() is None:
        fig.tight_layout()

    formats = tuple(f for f in formats if f.lower() == "png") or ("png",)
    written = []
    for fmt in formats:
        target = path_base.with_suffix(f".{fmt}")
        fig.savefig(target, bbox_inches="tight", facecolor="white", format=fmt)
        written.append(target)
    plt.close(fig)
    for p in written:
        print(f"[gallery] wrote {p.relative_to(ROOT)}")


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


def load_real_stream(stream_id: int, columns: list[str] | None = None) -> "pd.DataFrame":  # type: ignore[name-defined]
    """Load real Stream {1,2,3} features joined with v1 ensemble predictions.

    Returns a DataFrame with one row per star, with all feature columns plus
    the per-label prediction columns (teff_pred, ..., teff_sigma, ...,
    cov_*, *_epistemic_var, ood_*_flag, regime_b_flag, mode_ambiguous_flag,
    selection_prob, *_missing_flag) joined on source_id.

    Parameters
    ----------
    stream_id : {1, 2, 3}
        Stream selector.
    columns : list of str, optional
        Restrict the feature parquet read to these columns (source_id is always
        added). Saves I/O on large parquets. None = read all feature columns.
    """
    import pandas as pd  # local import to keep top-level light

    # Stream 1 = Kiel-bounded RGB training pool (logg ∈ [1.0, 3.5],
    # Teff ∈ [4000, 5500] K). Streams 2 and 3 keep their full inference cohorts
    # (no Kiel mask — the bbox is a training-time decision).
    if stream_id == 1:
        feat = ROOT / "data/processed/pipeline1_features_stream1_kiel.parquet"
    else:
        feat = ROOT / f"data/processed/pipeline1_features_stream{stream_id}.parquet"
    pred = ROOT / f"data/processed/pipeline1_predictions_stream{stream_id}.parquet"
    if not feat.exists():
        raise FileNotFoundError(f"missing features parquet: {feat}")
    if not pred.exists():
        raise FileNotFoundError(
            f"missing predictions parquet: {pred} — "
            f"run scripts/run_pipeline1_inference.py "
            f"--input-parquet data/processed/pipeline1_features_stream{stream_id}.parquet "
            f"--output-parquet {pred}"
        )
    if columns is not None:
        cols = list(dict.fromkeys(["source_id", *columns]))
        df_feat = pd.read_parquet(feat, columns=cols)
    else:
        df_feat = pd.read_parquet(feat)
    df_pred = pd.read_parquet(pred)
    # Stream 1 features can have multiple rows per source_id (per-visit APOGEE
    # records joined onto a single Gaia source). The predictions parquet has
    # one row per source_id (one inference per star). Without dedup the inner
    # merge inflates row counts. Keep one feature row per source_id (first
    # occurrence — APOGEE per-visit averaging is upstream's responsibility).
    n_before = len(df_feat)
    df_feat = df_feat.drop_duplicates(subset="source_id", keep="first")
    if len(df_feat) != n_before:
        # Quiet info — useful for debugging but not noisy on Stream 2/3.
        pass
    out = df_feat.merge(df_pred, on="source_id", how="inner", suffixes=("", "_pred_dup"))
    return out


def _deprecated_synthetic_no_op(*_a, **_kw):
    raise NotImplementedError(
        "Synthetic gallery fixtures have been removed (2026-04-29). "
        "Use load_real_stream(stream_id, columns=...) to read real data."
    )


def make_holdout_fixture(*args, **kwargs):  # noqa: D401
    """REMOVED 2026-04-29 — was a synthetic-stars helper. Use load_real_stream(1)."""
    _deprecated_synthetic_no_op()


def load_stream1_holdout() -> "pd.DataFrame":  # type: ignore[name-defined]
    """Stream 1 (APOGEE × Gaia) joined with v1 predictions, restricted to the
    test-split rows used as the canonical holdout for D2/D5/E* truth-vs-pred plots.

    Truth labels live on the feature parquet under the *_apogee suffix:
    teff_apogee, logg_apogee, mh_apogee, alpha_m_apogee, mg_h_apogee.
    Quality cuts (Teff in [4000, 5500] K, log g in [1.0, 3.5]) are already
    applied at feature-build time per docs/decisions/ ADRs.
    """
    cols = [
        "source_id",
        "teff_apogee",
        "logg_apogee",
        "mh_apogee",
        "alpha_m_apogee",
        "mg_h_apogee",
        "ra_deg",
        "dec_deg",
        "b_deg",
        "g_mag",
        "bp_rp",
        "parallax",
        "parallax_error",
    ]
    return load_real_stream(1, columns=cols)
