"""Selection-function scorers for the ArqueoGal D-Cat-b release.

Two components are tabulated from Stream 1 (APOGEE × Gaia XP) and exposed
per-star at Stream-3 release time:

1. **Ye+2024 ``NO_SYNTH_PHOT`` retention** — ``score_selection_prob`` (v1).
   The probability a star at given ``(|b|, G)`` is retained (flag == 0) by
   the Ye+2024 NN flux correction. 5×5 grid. See ``selection_function_v1.md``.

2. **IR (2MASS/AllWISE) photometric completeness** — ``score_ir_completeness``
   (v1.1 introduction). The probability a star at given ``(|b|, G, Teff, log g)``
   has finite, non-zero J/H/K/W1/W2 in our Stream-1 training basis. 5×5×3×2
   grid with an always-dense 5×5 ``(|b|, G)`` marginal fallback for sparse
   4-D cells and for call sites where Teff / log g are not yet available.
   See ``ir_completeness_v1.md``.

``score_compound_selection_prob`` composes both into a single release-ready
``p_compound``:

.. code-block:: text

    p_compound = p_ye_retained · p_ir_complete · p_parallax · p_extinction

``p_parallax`` and ``p_extinction`` are 0/1 gates in v1.1 — they take a
boolean data-availability flag and emit 1.0 or 0.0. Smooth per-star
parallax / extinction-availability probabilities are earmarked for v1.2.

See ``docs/data_acquisition.md`` §6.6 for the scientific context and
``reports/selection_function/`` for artefact narratives.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Shared range clamps — identical to v1 so the two components compose cleanly.
SELECTION_PROB_FLOOR: Final[float] = 0.01
SELECTION_PROB_CEIL: Final[float] = 1.0
# Joint floor = product of the two component floors; keeps inverse weights
# finite at the plane-faint corner even under composition.
COMPOUND_PROB_FLOOR: Final[float] = SELECTION_PROB_FLOOR * SELECTION_PROB_FLOOR

_REPORTS: Final[Path] = Path(__file__).resolve().parents[3] / "reports" / "selection_function"

DEFAULT_ARTIFACT_PATH: Final[Path] = _REPORTS / "selection_function_v1.parquet"
DEFAULT_IR_ARTIFACT_PATH: Final[Path] = _REPORTS / "ir_completeness_v1.parquet"
DEFAULT_COMPOUND_ARTIFACT_PATH: Final[Path] = _REPORTS / "selection_function_v1.1.parquet"


def _default_artifact_path() -> Path:
    """Return the repo-local path to the v1 Ye-retention artefact.

    Kept as a function (not a module constant used directly) so tests can
    monkey-patch the filesystem layout without having to poke a frozen
    constant.
    """
    return DEFAULT_ARTIFACT_PATH


def _default_ir_artifact_path() -> Path:
    """Return the repo-local path to the v1 IR-completeness artefact."""
    return DEFAULT_IR_ARTIFACT_PATH


# ---------------------------------------------------------------------------
# v1 Ye-retention grid (unchanged, kept for backwards compatibility)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _load_grid(artifact_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the v1 Ye-retention 2-D grid artefact into numeric arrays.

    Returns
    -------
    b_edges : np.ndarray
        ``|b|`` bin edges in degrees, length ``n_b + 1``, monotone increasing,
        starting at 0.
    g_edges : np.ndarray
        ``G`` bin edges in magnitudes, length ``n_g + 1``, monotone increasing.
    selection_prob : np.ndarray
        Shape ``(n_b, n_g)``, float64, values in ``[SELECTION_PROB_FLOOR, 1.0]``.
    """
    df = pd.read_parquet(artifact_path)
    required = {"b_lo", "b_hi", "g_lo", "g_hi", "selection_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"selection-function artefact missing columns: {sorted(missing)}")

    b_edges_raw = np.concatenate([df["b_lo"].to_numpy(), df["b_hi"].to_numpy()])
    g_edges_raw = np.concatenate([df["g_lo"].to_numpy(), df["g_hi"].to_numpy()])
    b_edges = np.array(sorted(set(b_edges_raw.tolist())), dtype=np.float64)
    g_edges = np.array(sorted(set(g_edges_raw.tolist())), dtype=np.float64)

    n_b = len(b_edges) - 1
    n_g = len(g_edges) - 1
    prob = np.full((n_b, n_g), np.nan, dtype=np.float64)
    for row in df.itertuples(index=False):
        ib = int(np.searchsorted(b_edges, row.b_lo, side="right") - 1)
        ig = int(np.searchsorted(g_edges, row.g_lo, side="right") - 1)
        prob[ib, ig] = float(row.selection_prob)

    if np.isnan(prob).any():
        raise ValueError("selection-function artefact has holes — re-build required")
    return b_edges, g_edges, prob


def score_selection_prob(
    b_deg: np.ndarray | pd.Series,
    g_mag: np.ndarray | pd.Series,
    *,
    artifact_path: Path | str | None = None,
) -> np.ndarray:
    """Score the Ye+2024 ``NO_SYNTH_PHOT`` retention probability per star.

    Looks up the per-cell retention probability from the v1 2-D grid on
    ``(|b|, G)``. Out-of-range inputs are clamped to the nearest edge (we
    treat the grid as defined for all plausible Pipeline 1 / Stream 3
    coordinates and refuse to invent probabilities outside it).

    Parameters
    ----------
    b_deg : array-like of float
        Galactic latitude in degrees (can be signed; ``|b|`` is taken
        internally).
    g_mag : array-like of float
        Gaia DR3 ``G`` magnitude.
    artifact_path : Path or str, optional
        Path to the v1 Parquet artefact. Defaults to
        :data:`DEFAULT_ARTIFACT_PATH`.

    Returns
    -------
    np.ndarray
        Same length as inputs; dtype ``float64``; values in
        ``[SELECTION_PROB_FLOOR, SELECTION_PROB_CEIL]``.

    Notes
    -----
    To *inverse-weight* a downstream likelihood, multiply the per-star
    likelihood by ``1 / selection_prob``. The floor keeps this weight
    finite but deliberately large inside the Galactic plane at faint
    magnitudes, which is the scientifically honest behaviour.
    """
    path = Path(artifact_path) if artifact_path is not None else _default_artifact_path()
    b_edges, g_edges, prob = _load_grid(str(path))

    b_arr = np.asarray(b_deg, dtype=np.float64).ravel()
    g_arr = np.asarray(g_mag, dtype=np.float64).ravel()
    if b_arr.shape != g_arr.shape:
        raise ValueError(
            f"b_deg and g_mag must have the same length; got {b_arr.shape} vs {g_arr.shape}"
        )

    abs_b = np.abs(b_arr)
    # Clamp so searchsorted never returns an index outside [1, n_edges-1].
    abs_b_clamped = np.clip(abs_b, b_edges[0], np.nextafter(b_edges[-1], -np.inf))
    g_clamped = np.clip(g_arr, g_edges[0], np.nextafter(g_edges[-1], -np.inf))

    ib = np.searchsorted(b_edges, abs_b_clamped, side="right") - 1
    ig = np.searchsorted(g_edges, g_clamped, side="right") - 1
    ib = np.clip(ib, 0, prob.shape[0] - 1)
    ig = np.clip(ig, 0, prob.shape[1] - 1)

    out = prob[ib, ig]
    return np.clip(out, SELECTION_PROB_FLOOR, SELECTION_PROB_CEIL)


# ---------------------------------------------------------------------------
# v1.1 IR-completeness grid
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _load_ir_grid(
    artifact_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the IR-completeness v1 artefact.

    Returns a tuple:

    - ``b_edges, g_edges, t_edges, l_edges`` : grid edges in their native units.
    - ``prob_4d`` : float64, shape ``(n_b, n_g, n_t, n_l)``. Cells without a
      dense 4-D measurement (sparse or empty) are filled with ``NaN``; the
      caller must fall back to ``prob_bg`` for those.
    - ``prob_bg`` : float64, shape ``(n_b, n_g)``. The always-dense |b|×G
      marginal. Used when Teff/log g is unavailable or when the 4-D cell is
      sparse.
    """
    df = pd.read_parquet(artifact_path)
    required = {"b_lo", "b_hi", "g_lo", "g_hi", "p_ir_complete", "grid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"IR-completeness artefact missing columns: {sorted(missing)}")

    bg = df[df["grid"] == "bg"].copy()
    fourd = df[df["grid"] == "4d"].copy()
    if len(bg) == 0:
        raise ValueError("IR-completeness artefact missing 'bg' (|b|×G marginal) rows")
    if len(fourd) == 0:
        raise ValueError("IR-completeness artefact missing '4d' rows")

    # Recover edges from the bg marginal (|b|, G) and from the 4-D rows (teff, logg).
    b_edges = np.array(
        sorted(set(bg["b_lo"].tolist()) | set(bg["b_hi"].tolist())), dtype=np.float64
    )
    g_edges = np.array(
        sorted(set(bg["g_lo"].tolist()) | set(bg["g_hi"].tolist())), dtype=np.float64
    )
    t_edges = np.array(
        sorted(set(fourd["teff_lo"].tolist()) | set(fourd["teff_hi"].tolist())),
        dtype=np.float64,
    )
    l_edges = np.array(
        sorted(set(fourd["logg_lo"].tolist()) | set(fourd["logg_hi"].tolist())),
        dtype=np.float64,
    )

    n_b = len(b_edges) - 1
    n_g = len(g_edges) - 1
    n_t = len(t_edges) - 1
    n_l = len(l_edges) - 1

    # bg marginal — must be fully populated.
    prob_bg = np.full((n_b, n_g), np.nan, dtype=np.float64)
    for row in bg.itertuples(index=False):
        ib = int(np.searchsorted(b_edges, row.b_lo, side="right") - 1)
        ig = int(np.searchsorted(g_edges, row.g_lo, side="right") - 1)
        prob_bg[ib, ig] = float(row.p_ir_complete)
    if np.isnan(prob_bg).any():
        raise ValueError("IR-completeness |b|×G marginal has holes — rebuild required")

    # 4-D grid — only dense cells populate; sparse / empty stay NaN and
    # score_ir_completeness will fall back to prob_bg for them.
    prob_4d = np.full((n_b, n_g, n_t, n_l), np.nan, dtype=np.float64)
    for row in fourd.itertuples(index=False):
        if not bool(row.dense):
            continue
        ib = int(np.searchsorted(b_edges, row.b_lo, side="right") - 1)
        ig = int(np.searchsorted(g_edges, row.g_lo, side="right") - 1)
        it = int(np.searchsorted(t_edges, row.teff_lo, side="right") - 1)
        il = int(np.searchsorted(l_edges, row.logg_lo, side="right") - 1)
        prob_4d[ib, ig, it, il] = float(row.p_ir_complete)

    return b_edges, g_edges, t_edges, l_edges, prob_4d, prob_bg


def score_ir_completeness(
    b_deg: np.ndarray | pd.Series | float,
    g_mag: np.ndarray | pd.Series | float,
    teff: np.ndarray | pd.Series | float,
    logg: np.ndarray | pd.Series | float,
    *,
    artifact_path: Path | str | None = None,
) -> np.ndarray:
    """Score P(IR-complete | |b|, G, Teff, log g) per star.

    IR-complete = 2MASS J/H/K AND AllWISE W1/W2 all finite and non-zero in
    the Stream-1 training basis; the zero-sentinel is excluded because
    downstream inference uses ``nan_to_num(0.0)`` on missing IR rows.

    Parameters
    ----------
    b_deg : array-like of float
        Galactic latitude in degrees (signed; ``|b|`` is taken internally).
    g_mag : array-like of float
        Gaia DR3 ``G`` magnitude.
    teff : array-like of float
        Effective temperature in K. ``NaN`` allowed per element — those
        elements receive the |b|×G marginal value.
    logg : array-like of float
        Surface gravity in dex. ``NaN`` allowed per element — as above.
    artifact_path : Path or str, optional
        Override the default IR-completeness Parquet artefact.

    Returns
    -------
    np.ndarray
        Same length as inputs; dtype ``float64``; values in
        ``[SELECTION_PROB_FLOOR, SELECTION_PROB_CEIL]``.

    Notes
    -----
    Per-row fallback rules:

    - If ``teff`` or ``logg`` is NaN → use the |b|×G marginal directly.
    - If the (|b|, G, Teff, log g) 4-D cell is sparse (empty at build time
      or had too few stars to support a stable rate) → fall back to the
      |b|×G marginal.
    - Out-of-range inputs are clamped to the nearest edge (no extrapolation).
    """
    path = Path(artifact_path) if artifact_path is not None else _default_ir_artifact_path()
    b_edges, g_edges, t_edges, l_edges, prob_4d, prob_bg = _load_ir_grid(str(path))

    b_arr = np.asarray(b_deg, dtype=np.float64).ravel()
    g_arr = np.asarray(g_mag, dtype=np.float64).ravel()
    t_arr = np.asarray(teff, dtype=np.float64).ravel()
    l_arr = np.asarray(logg, dtype=np.float64).ravel()

    shapes = {b_arr.shape, g_arr.shape, t_arr.shape, l_arr.shape}
    if len(shapes) > 1:
        raise ValueError(
            f"b_deg, g_mag, teff, logg must have the same length; got {shapes}"
        )

    abs_b = np.abs(b_arr)
    abs_b_clamped = np.clip(abs_b, b_edges[0], np.nextafter(b_edges[-1], -np.inf))
    g_clamped = np.clip(g_arr, g_edges[0], np.nextafter(g_edges[-1], -np.inf))
    ib = np.clip(np.searchsorted(b_edges, abs_b_clamped, side="right") - 1, 0, prob_bg.shape[0] - 1)
    ig = np.clip(np.searchsorted(g_edges, g_clamped, side="right") - 1, 0, prob_bg.shape[1] - 1)

    # Baseline is the |b|×G marginal.
    out = prob_bg[ib, ig].copy()

    # Attempt the 4-D lookup for rows with finite Teff AND log g.
    has_tl = np.isfinite(t_arr) & np.isfinite(l_arr)
    if has_tl.any():
        t_clamped = np.clip(t_arr[has_tl], t_edges[0], np.nextafter(t_edges[-1], -np.inf))
        l_clamped = np.clip(l_arr[has_tl], l_edges[0], np.nextafter(l_edges[-1], -np.inf))
        it = np.clip(
            np.searchsorted(t_edges, t_clamped, side="right") - 1, 0, prob_4d.shape[2] - 1,
        )
        il = np.clip(
            np.searchsorted(l_edges, l_clamped, side="right") - 1, 0, prob_4d.shape[3] - 1,
        )
        candidate = prob_4d[ib[has_tl], ig[has_tl], it, il]
        # Sparse / empty 4-D cells stay NaN in the grid — keep the marginal
        # baseline there (already in ``out``); everywhere the 4-D cell is
        # dense, override with its value.
        dense_mask = np.isfinite(candidate)
        idx = np.flatnonzero(has_tl)[dense_mask]
        out[idx] = candidate[dense_mask]

    return np.clip(out, SELECTION_PROB_FLOOR, SELECTION_PROB_CEIL)


# ---------------------------------------------------------------------------
# v1.1 compound scorer
# ---------------------------------------------------------------------------


class _CompoundResult(TypedDict):
    """Per-star compound-selection bundle."""

    p_ye_retained: float
    p_ir_complete: float
    p_compound: float
    components: dict[str, float]


def _gate(present: bool) -> float:
    """0/1 data-availability gate (v1.1 parallax / extinction stub)."""
    return 1.0 if bool(present) else 0.0


def score_compound_selection_prob(
    b_deg: float,
    g_mag: float,
    teff: float,
    logg: float,
    *,
    parallax_over_error: float | bool | None = None,
    av_missing: bool = False,
    ye_artifact_path: Path | str | None = None,
    ir_artifact_path: Path | str | None = None,
) -> _CompoundResult:
    """Compound per-star selection probability for D-Cat-b release.

    Composes four factors multiplicatively:

    .. code-block:: text

        p_compound = p_ye_retained · p_ir_complete · p_parallax · p_extinction

    Parameters
    ----------
    b_deg, g_mag, teff, logg : float
        Scalar per-star inputs. ``teff``/``logg`` may be ``NaN`` — in that
        case the IR component falls back to the |b|×G marginal.
    parallax_over_error : float or bool or None, optional
        Data-availability flag for the parallax input. v1.1 is a 0/1 gate:
        ``None`` or ``False`` → 0.0; any other (numeric or ``True``) →
        1.0. v1.2 will replace this with a smooth function of the parallax
        S/N. The downstream-visible interpretation here is simply
        "is parallax available for this star?"
    av_missing : bool, optional
        If ``True``, the extinction component is 0.0 (star unusable for
        analyses that need Av). If ``False`` (default), the extinction
        component is 1.0.
    ye_artifact_path, ir_artifact_path : Path or str, optional
        Override the default v1 / v1.1 Parquet artefact paths.

    Returns
    -------
    dict
        ``{'p_ye_retained', 'p_ir_complete', 'p_compound', 'components'}``
        where ``components`` is a dict ``{'ye', 'ir', 'parallax',
        'extinction'}``. All values are plain Python floats in
        ``[COMPOUND_PROB_FLOOR, SELECTION_PROB_CEIL]``.

    Notes
    -----
    - The compound floor is
      ``SELECTION_PROB_FLOOR * SELECTION_PROB_FLOOR = 1e-4``, not zero:
      inverse-weighting at the plane-faint corner remains finite when both
      components hit their floors. (Pure-zero extinction / parallax gates
      will still zero ``p_compound`` when set — those are release-gating,
      not floor-eligible.)
    - Intended for per-star use inside a vectorised caller. For bulk
      scoring, call the component functions directly on arrays.
    """
    p_ye_arr = score_selection_prob(
        np.asarray([b_deg]), np.asarray([g_mag]), artifact_path=ye_artifact_path,
    )
    p_ir_arr = score_ir_completeness(
        np.asarray([b_deg]), np.asarray([g_mag]),
        np.asarray([teff]), np.asarray([logg]),
        artifact_path=ir_artifact_path,
    )
    p_ye = float(p_ye_arr[0])
    p_ir = float(p_ir_arr[0])

    has_parallax = parallax_over_error is not None and parallax_over_error is not False
    p_parallax = _gate(has_parallax)
    p_extinction = _gate(not av_missing)

    product = p_ye * p_ir * p_parallax * p_extinction
    # Only apply the positive floor if all gates are non-zero; a zero gate
    # is release-blocking and must propagate as 0.0 (not 1e-4).
    if p_parallax == 0.0 or p_extinction == 0.0:
        p_compound = 0.0
    else:
        p_compound = float(np.clip(product, COMPOUND_PROB_FLOOR, SELECTION_PROB_CEIL))

    return {
        "p_ye_retained": p_ye,
        "p_ir_complete": p_ir,
        "p_compound": p_compound,
        "components": {
            "ye": p_ye,
            "ir": p_ir,
            "parallax": p_parallax,
            "extinction": p_extinction,
        },
    }


__all__ = [
    "COMPOUND_PROB_FLOOR",
    "DEFAULT_ARTIFACT_PATH",
    "DEFAULT_COMPOUND_ARTIFACT_PATH",
    "DEFAULT_IR_ARTIFACT_PATH",
    "SELECTION_PROB_CEIL",
    "SELECTION_PROB_FLOOR",
    "score_compound_selection_prob",
    "score_ir_completeness",
    "score_selection_prob",
]
