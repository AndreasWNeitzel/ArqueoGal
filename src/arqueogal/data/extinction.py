"""Interstellar extinction & reddening corrections for Pipeline-1 features.

This module is the canonical home for *explicit dereddening* of the broadband
auxiliary photometry (2MASS J/H/K, AllWISE W1/W2) consumed by the
xp_abundances model. The dereddening recipe is fixed at v1 frozen-stats time
so train and inference apply byte-identical transforms — see the
``ExtinctionLaw`` dataclass + ``apply_extinction_corrections`` entry point.

Decision (2026-04-29, after the literature review): adopt the **hybrid
dereddening recipe** documented in
``docs/protocols/extinction_correction.md`` (referenced from the
methods-paper §5):

1. Broadband JHK + W1W2 are explicitly de-reddened using fixed
   :data:`YUAN2013_AV_RATIOS` and a per-star A_V drawn from the
   Edenhofer+2024 / Lallement+2022 / SFD+neighbourhood dust-map fusion.
2. The 55-coefficient BP/RP Hermite basis is **not** re-dereddened in this
   module. Ye+2024 instrumental flux corrections (mandatory upstream in
   :mod:`arqueogal.data.gaia_xp`) already absorb wavelength-dependent
   systematics; applying CCM89 to the Hermite coefficients in addition is
   double-counting and unsupported by any published recipe at the
   coefficient level. A_V is retained as an *auxiliary* feature alongside
   the dereddened photometry so the encoder still sees the residual XP
   extinction signal.
3. The extinction law is fixed at Cardelli, Clayton & Mathis 1989 (CCM89,
   ApJ 345, 245) with ``R_V = 3.1`` — the operational consensus across
   AspGap (Li+2024), SHBoost (Khalatyan+2024), Ye+2024, and Hattori+2024.
   Per-star R_V fitting is *not* implemented (Schlafly+2016 σ_R_V ≈ 0.18
   propagates to ≤ 0.02 dex on [Fe/H] for E(B-V) < 0.5, well below the
   methods-paper precision target).

References
----------
Yuan, Liu & Xiang 2013, MNRAS 430, 2188:
    A_J/A_V = 0.276, A_H/A_V = 0.176, A_Ks/A_V = 0.112,
    A_W1/A_V = 0.063, A_W2/A_V = 0.050.
Cardelli, Clayton & Mathis 1989, ApJ 345, 245: CCM89 R_V = 3.1.
Edenhofer+2024 (arXiv:2308.01295) + Lallement+2022 (A&A 664, A9) + SFD
    (1998 ApJ 500, 525): the dust-map fusion that produces ``av_los``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --- Frozen extinction-law constants -----------------------------------------

# Yuan+2013 A_λ / A_V ratios, locked at v1. Drift is forbidden post-frozen-
# stats; if a future revision changes any of these, the basis fingerprint
# must be re-derived and the change documented in ADR + methods paper §5.
YUAN2013_AV_RATIOS: Final[Mapping[str, float]] = MappingProxyType(
    {
        "j_mag": 0.276,
        "h_mag": 0.176,
        "k_mag": 0.112,
        "w1_mag": 0.063,
        "w2_mag": 0.050,
    }
)
"""Broadband A_λ / A_V ratios (Yuan, Liu & Xiang 2013, MNRAS 430, 2188).

Empirically derived from ~700 k stars; canonical for the disc-giant regime in
the 2025-2026 community (Wang & Chen 2019, ApJ 877, 116, gives matching
values within ≤ 0.01). Frozen at v1; mutation rejected at runtime via
``MappingProxyType``.
"""


_DEREDDENED_SUFFIX: Final[str] = "_dered"


@dataclass(frozen=True, slots=True)
class ExtinctionLaw:
    """The frozen v1 extinction-law contract.

    Attributes
    ----------
    name
        Human-readable tag for sidecar / methods-paper citation
        (``"CCM89, R_V=3.1, Yuan+2013 IR ratios"``).
    r_v
        Total-to-selective extinction ratio. Fixed at 3.1 for v1.
    av_ratios
        Per-broadband ``A_λ / A_V`` ratios. Read-only; default is
        :data:`YUAN2013_AV_RATIOS`.
    """

    name: str = "CCM89_RV3.1_Yuan2013"
    r_v: float = 3.1
    av_ratios: Mapping[str, float] = field(default_factory=lambda: YUAN2013_AV_RATIOS)

    def fingerprint(self) -> dict[str, object]:
        """Stable JSON-able snapshot for sidecar provenance."""
        return {
            "name": self.name,
            "r_v": float(self.r_v),
            "av_ratios": {k: float(v) for k, v in self.av_ratios.items()},
        }


DEFAULT_EXTINCTION_LAW: Final[ExtinctionLaw] = ExtinctionLaw()


# --- A_V quality flags --------------------------------------------------------

# Categorical encoding of which dust map produced ``av_los`` for a given star.
# Mirrors ``av_los_source`` in ``CATALOG_SCHEMA.md``: 0=Edenhofer, 1=Lallement,
# 2=SFD, -1=missing (added 3=neighborhood-median fallback when dust maps fail).
AV_SOURCE_CODES: Final[Mapping[str, int]] = MappingProxyType(
    {
        "edenhofer": 0,
        "lallement": 1,
        "sfd": 2,
        "neighborhood_median": 3,
        "missing": -1,
    }
)
AV_SOURCE_NAMES: Final[Mapping[int, str]] = MappingProxyType(
    {v: k for k, v in AV_SOURCE_CODES.items()}
)


# --- Public API ---------------------------------------------------------------


def select_av(  # noqa: PLR0913 — orthogonal dust-map columns
    df: pd.DataFrame,
    *,
    av_edenhofer_col: str = "av_edenhofer",
    av_lallement_col: str = "av_lallement",
    av_sfd_col: str = "av_sfd",
    av_nbhd_col: str = "av_nbhd_median",
    distance_col: str = "r_med_photogeo",
    edenhofer_d_max_kpc: float = 1.25,
    lallement_d_max_kpc: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a per-star A_V from the dust-map fusion.

    Implements the layered priority documented in
    ``CATALOG_SCHEMA.md`` "Distance" section: Edenhofer+2024 inside
    ``edenhofer_d_max_kpc``, Lallement+2022 between Edenhofer and
    ``lallement_d_max_kpc``, and SFD+neighbourhood-median beyond. The result
    is a dense float array with NaN where every layer fails, plus an
    ``int8`` source-code array compatible with ``AV_SOURCE_CODES``.

    Parameters
    ----------
    df
        Frame carrying the per-map A_V columns and a distance estimate
        (Bailer-Jones photogeometric, kpc).
    edenhofer_d_max_kpc
        Maximum distance at which Edenhofer is preferred. Default 1.25 kpc
        matches the Edenhofer+2024 published completeness limit.
    lallement_d_max_kpc
        Maximum distance at which Lallement is preferred. Beyond this the
        SFD column total is used (with the neighbourhood median as a final
        fallback).

    Returns
    -------
    av : np.ndarray
        Per-star A_V in mag. NaN where every layer is missing.
    source_code : np.ndarray
        Per-star ``int8`` source code (see :data:`AV_SOURCE_CODES`).
    """
    n = len(df)
    av = np.full(n, np.nan, dtype=np.float64)
    src = np.full(n, AV_SOURCE_CODES["missing"], dtype=np.int8)

    distance = (
        df[distance_col].to_numpy(dtype=np.float64, copy=False)
        if distance_col in df.columns
        else np.full(n, np.nan, dtype=np.float64)
    )

    def _take(col: str, mask: np.ndarray, code_key: str) -> None:
        if col not in df.columns:
            return
        values = df[col].to_numpy(dtype=np.float64, copy=False)
        usable = mask & np.isfinite(values) & np.isnan(av)
        if usable.any():
            av[usable] = values[usable]
            src[usable] = AV_SOURCE_CODES[code_key]

    near = np.isfinite(distance) & (distance <= edenhofer_d_max_kpc)
    mid = (
        np.isfinite(distance) & (distance > edenhofer_d_max_kpc) & (distance <= lallement_d_max_kpc)
    )
    far = np.isfinite(distance) & (distance > lallement_d_max_kpc)
    no_distance = ~np.isfinite(distance)

    # Preferred layer per distance regime.
    _take(av_edenhofer_col, near, "edenhofer")
    _take(av_lallement_col, mid, "lallement")
    _take(av_sfd_col, far, "sfd")

    # Cross-fill: when a regime's preferred map has NaN, walk down the
    # priority tree (Edenhofer → Lallement → SFD → neighbourhood-median)
    # rather than leaving the star with no Av.
    _take(av_edenhofer_col, np.isnan(av), "edenhofer")
    _take(av_lallement_col, np.isnan(av), "lallement")
    _take(av_sfd_col, np.isnan(av), "sfd")
    _take(av_nbhd_col, np.isnan(av), "neighborhood_median")
    # Distance-less stars: try the maps in priority order.
    _take(av_edenhofer_col, no_distance & np.isnan(av), "edenhofer")
    _take(av_lallement_col, no_distance & np.isnan(av), "lallement")
    _take(av_sfd_col, no_distance & np.isnan(av), "sfd")

    return av, src


def assign_av_quality(
    av: np.ndarray,
    source_code: np.ndarray,
    *,
    parallax_over_error: np.ndarray | None = None,
    parallax_snr_floor: float = 5.0,
    av_neighbourhood_std: np.ndarray | None = None,
    nbhd_std_high_mag: float = 0.5,
) -> dict[str, np.ndarray]:
    """Compute the trust flags consumers can filter on.

    Three boolean flags per star (DESIGN.md "Distance trust-flag" §):

    - ``av_is_neighborhood_fallback``: True iff the value came from the
      neighbourhood-median fallback (no per-sightline 3D dust map fired).
    - ``av_distance_prior_dominated``: True iff the per-star parallax SNR
      is below ``parallax_snr_floor``, so the underlying Bailer-Jones
      distance is prior-dominated and the dust-map A_V inherits that
      uncertainty.
    - ``av_neighbourhood_high_dispersion``: True iff the neighbourhood
      A_V dispersion exceeds ``nbhd_std_high_mag``. This is the
      "patchy-extinction" indicator — dust-map A_V is less reliable in
      such regions.
    """
    n = av.shape[0]
    nbhd = source_code == AV_SOURCE_CODES["neighborhood_median"]
    if parallax_over_error is None:
        prior_dominated = np.zeros(n, dtype=bool)
    else:
        prior_dominated = np.where(
            np.isfinite(parallax_over_error),
            parallax_over_error < parallax_snr_floor,
            False,
        )
    if av_neighbourhood_std is None:
        high_disp = np.zeros(n, dtype=bool)
    else:
        high_disp = np.where(
            np.isfinite(av_neighbourhood_std),
            av_neighbourhood_std > nbhd_std_high_mag,
            False,
        )
    return {
        "av_is_neighborhood_fallback": nbhd.astype(bool),
        "av_distance_prior_dominated": prior_dominated.astype(bool),
        "av_neighbourhood_high_dispersion": high_disp.astype(bool),
    }


def deredden_broadband(
    df: pd.DataFrame,
    av: np.ndarray,
    *,
    law: ExtinctionLaw = DEFAULT_EXTINCTION_LAW,
    suffix: str = _DEREDDENED_SUFFIX,
    bands: tuple[str, ...] = ("j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag"),
    inplace: bool = False,
) -> pd.DataFrame:
    """Apply ``mag_dered = mag - A_V * (A_λ / A_V)`` to the listed broadbands.

    Stars with a missing or non-finite ``av`` keep ``NaN`` in the dereddened
    column. Stars with non-finite raw magnitudes likewise propagate NaN.
    This is the train/inference invariant: the same transform is applied
    to the same columns at both stages, with the same Yuan+2013 ratios and
    the same fused A_V column.

    Returns a new frame (or the input frame, if ``inplace=True``) extended
    with the suffixed dereddened columns. The raw broadband columns are
    *retained* — the model consumes the dereddened columns, but the raw
    columns stay so external consumers and selection-function diagnostics
    can read them.
    """
    out = df if inplace else df.copy()
    av_arr = np.asarray(av, dtype=np.float64)
    if av_arr.shape != (len(out),):
        raise ValueError(f"av must be 1-D with length {len(out)}; got shape {av_arr.shape}")
    for band in bands:
        target_col = f"{band}{suffix}"
        if band not in out.columns:
            logger.info("deredden_broadband: skipping %s (column absent)", band)
            out[target_col] = np.full(len(out), np.nan, dtype=np.float64)
            continue
        ratio = law.av_ratios.get(band)
        if ratio is None:
            raise KeyError(f"ExtinctionLaw {law.name!r} has no A_lambda/A_V ratio for {band}")
        raw = out[band].to_numpy(dtype=np.float64, copy=False)
        with np.errstate(invalid="ignore"):
            dered = raw - av_arr * float(ratio)
        out[target_col] = dered
    return out


def dereddened_column_names(
    bands: tuple[str, ...] = ("j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag"),
    *,
    suffix: str = _DEREDDENED_SUFFIX,
) -> tuple[str, ...]:
    """Convenience: column names produced by :func:`deredden_broadband`.

    Used by the FeatureLayout / DEFAULT_AUX_COLS contract so callers do
    not hand-write ``"j_mag_dered"`` strings.
    """
    return tuple(f"{b}{suffix}" for b in bands)


def apply_extinction_corrections(
    df: pd.DataFrame,
    *,
    law: ExtinctionLaw = DEFAULT_EXTINCTION_LAW,
    distance_col: str = "r_med_photogeo",
    parallax_over_error_col: str | None = "parallax_over_error",
    av_nbhd_std_col: str | None = "av_nbhd_std",
    inplace: bool = False,
) -> pd.DataFrame:
    """End-to-end dereddening: select A_V → emit trust flags → deredden bands.

    The single entry point for callers (Stream 1 ingestion, Stream 2
    enrichment, Stream 3 inference). Produces (in addition to the raw
    columns the input already carries):

    - ``av_los`` (float64): fused per-star A_V used for dereddening.
    - ``av_los_source`` (int8): which dust-map layer fired (see
      :data:`AV_SOURCE_CODES`).
    - ``av_is_neighborhood_fallback`` (bool).
    - ``av_distance_prior_dominated`` (bool).
    - ``av_neighbourhood_high_dispersion`` (bool).
    - ``j_mag_dered``, ``h_mag_dered``, ``k_mag_dered``, ``w1_mag_dered``,
      ``w2_mag_dered`` (float64).

    The frozen-stats fingerprint (``frozen_stats.py``) covers the
    z-scoring of the dereddened columns; the *contents* of this dereddening
    layer (Yuan+2013 ratios, R_V=3.1) live in :data:`DEFAULT_EXTINCTION_LAW`
    and are surfaced to the provenance sidecar via :meth:`ExtinctionLaw.fingerprint`.
    """
    out = df if inplace else df.copy()

    av, src = select_av(out, distance_col=distance_col)
    out["av_los"] = av.astype(np.float64)
    out["av_los_source"] = src.astype(np.int8)

    parallax_over_error: np.ndarray | None = None
    if parallax_over_error_col is not None and parallax_over_error_col in out.columns:
        parallax_over_error = out[parallax_over_error_col].to_numpy(dtype=np.float64, copy=False)
    av_nbhd_std: np.ndarray | None = None
    if av_nbhd_std_col is not None and av_nbhd_std_col in out.columns:
        av_nbhd_std = out[av_nbhd_std_col].to_numpy(dtype=np.float64, copy=False)
    flags = assign_av_quality(
        av,
        src,
        parallax_over_error=parallax_over_error,
        av_neighbourhood_std=av_nbhd_std,
    )
    for name, arr in flags.items():
        out[name] = arr

    out = deredden_broadband(out, av, law=law, inplace=True)

    return out


__all__ = [
    "AV_SOURCE_CODES",
    "AV_SOURCE_NAMES",
    "DEFAULT_EXTINCTION_LAW",
    "ExtinctionLaw",
    "YUAN2013_AV_RATIOS",
    "apply_extinction_corrections",
    "assign_av_quality",
    "deredden_broadband",
    "dereddened_column_names",
    "select_av",
]
