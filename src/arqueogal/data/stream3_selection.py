"""Stream 3 sub-sampling — §5.3 of data_acquisition.md.

Two sampler flavours live here, both operating on an already-loaded DataFrame
(the Andrae+2023 Zenodo downloader lives elsewhere):

- :func:`stratified_subsample` — (Teff, log g, [M/H], G)-stratified draw used
  by the Pipeline-1 audit set. Essential to avoid the Gaia solar-disc bias
  that would under-sample the halo and metal-poor tails where interesting
  population structure lives.
- :func:`volume_limited_subsample` — natural-density uniform random draw from
  stars with ``distance < distance_cut_kpc``. Used for downstream
  density-based clustering (Starfold, separate repo), which requires the
  natural density profile that stratification deliberately breaks.

Both samplers are seeded (reproducible) and preserve the caller's schema.
Provenance metadata is returned so ingestion scripts can write the standard
``*.provenance.json`` sidecar.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_BINS_TEFF: Final[np.ndarray] = np.linspace(4000.0, 5500.0, 7)
"""§5.3: 7 edges → 6 bins across the RGB Teff range."""

DEFAULT_BINS_LOGG: Final[np.ndarray] = np.linspace(1.0, 3.5, 6)
"""§5.3: 6 edges → 5 bins across the giant log g range."""

DEFAULT_BINS_MH: Final[np.ndarray] = np.linspace(-2.0, 0.5, 6)
"""§5.3: 6 edges → 5 bins across [M/H]."""

DEFAULT_BINS_G: Final[np.ndarray] = np.linspace(7.0, 16.0, 10)
"""§5.3: 10 edges → 9 bins across G mag."""

DEFAULT_PER_CELL: Final[int] = 600
"""§5.3: 600 stars/cell × 2520 cells ≈ 1.5 M stars. Cells with fewer stars
contribute all they have."""


@dataclass(frozen=True, slots=True)
class StratificationResult:
    """Output of :func:`stratified_subsample`.

    ``sample`` is the selected subset DataFrame. ``cell_counts`` is a tidy
    per-cell accounting frame (bin indices, n_available, n_selected) that
    feeds provenance logging.
    """

    sample: pd.DataFrame
    cell_counts: pd.DataFrame
    bins_teff: np.ndarray
    bins_logg: np.ndarray
    bins_mh: np.ndarray
    bins_g: np.ndarray
    per_cell: int
    rng_seed: int
    columns: tuple[str, str, str, str] = field(
        default=("teff", "logg", "mh", "g_mag"),
    )

    def to_provenance(self) -> dict:
        """Serialisable summary for the Parquet sidecar."""
        return {
            "method": "stream3_stratified_subsample",
            "bins_teff": self.bins_teff.tolist(),
            "bins_logg": self.bins_logg.tolist(),
            "bins_mh": self.bins_mh.tolist(),
            "bins_g": self.bins_g.tolist(),
            "per_cell": self.per_cell,
            "rng_seed": self.rng_seed,
            "n_available": int(self.cell_counts["n_available"].sum()),
            "n_selected": int(self.cell_counts["n_selected"].sum()),
            "n_nonempty_cells": int((self.cell_counts["n_available"] > 0).sum()),
            "stratification_columns": list(self.columns),
        }


def stratified_subsample(  # noqa: PLR0913 — keyword-only tuning knobs with safe defaults
    df: pd.DataFrame,
    *,
    teff_col: str = "teff_xgboost",
    logg_col: str = "logg_xgboost",
    mh_col: str = "mh_xgboost",
    g_col: str = "phot_g_mean_mag",
    bins_teff: np.ndarray = DEFAULT_BINS_TEFF,
    bins_logg: np.ndarray = DEFAULT_BINS_LOGG,
    bins_mh: np.ndarray = DEFAULT_BINS_MH,
    bins_g: np.ndarray = DEFAULT_BINS_G,
    per_cell: int = DEFAULT_PER_CELL,
    rng_seed: int = 0,
) -> StratificationResult:
    """Draw a stratified sub-sample per §5.3.

    Parameters
    ----------
    df
        Input star catalogue (typically Andrae+2023 vetted RGB). Must
        contain the four stratification columns named by ``*_col`` kwargs.
    teff_col, logg_col, mh_col, g_col
        Column names for the four stratification axes.
    bins_teff, bins_logg, bins_mh, bins_g
        Bin *edges* — internally passed to :func:`numpy.digitize`. Default
        bins match §5.3.
    per_cell
        Maximum rows per non-empty cell. Cells with fewer stars contribute
        all of them.
    rng_seed
        Seed for :class:`numpy.random.Generator`. Fixed for reproducibility.

    Returns
    -------
    StratificationResult
        Selected sample + cell-level accounting + reproducibility metadata.
    """
    required = {teff_col, logg_col, mh_col, g_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"stratified_subsample requires columns: {sorted(missing)}")
    if per_cell <= 0:
        raise ValueError(f"per_cell must be positive, got {per_cell}")

    if df.empty:
        return _empty_result(
            bins_teff,
            bins_logg,
            bins_mh,
            bins_g,
            per_cell,
            rng_seed,
            (teff_col, logg_col, mh_col, g_col),
        )

    teff = df[teff_col].to_numpy(dtype=float)
    logg = df[logg_col].to_numpy(dtype=float)
    mh = df[mh_col].to_numpy(dtype=float)
    g_mag = df[g_col].to_numpy(dtype=float)

    # Stars outside [first_edge, last_edge] are excluded — their bin index
    # lands at 0 or len(bins) and is flagged.
    i_teff = np.digitize(teff, bins_teff) - 1
    i_logg = np.digitize(logg, bins_logg) - 1
    i_mh = np.digitize(mh, bins_mh) - 1
    i_g = np.digitize(g_mag, bins_g) - 1

    in_range = (
        (i_teff >= 0)
        & (i_teff < len(bins_teff) - 1)
        & (i_logg >= 0)
        & (i_logg < len(bins_logg) - 1)
        & (i_mh >= 0)
        & (i_mh < len(bins_mh) - 1)
        & (i_g >= 0)
        & (i_g < len(bins_g) - 1)
    )
    finite = np.isfinite(teff) & np.isfinite(logg) & np.isfinite(mh) & np.isfinite(g_mag)
    valid = in_range & finite

    n_excluded = (~valid).sum()
    if n_excluded:
        logger.info(
            "stratified_subsample: %d/%d rows excluded (NaN or outside bin range)",
            int(n_excluded),
            len(df),
        )

    # Build a string key per row so we can group without cross products.
    cell_df = pd.DataFrame(
        {
            "row_idx": np.arange(len(df)),
            "i_teff": i_teff,
            "i_logg": i_logg,
            "i_mh": i_mh,
            "i_g": i_g,
            "valid": valid,
        }
    )
    valid_cells = cell_df.loc[valid]

    rng = np.random.default_rng(rng_seed)
    selected_idx: list[int] = []
    counts_records: list[dict] = []

    for (it, il, im, ig), group in valid_cells.groupby(
        ["i_teff", "i_logg", "i_mh", "i_g"], sort=True
    ):
        n_available = len(group)
        n_selected = min(n_available, per_cell)
        if n_available <= per_cell:
            picks = group["row_idx"].to_numpy()
        else:
            picks = rng.choice(group["row_idx"].to_numpy(), size=per_cell, replace=False)
        selected_idx.extend(picks.tolist())
        counts_records.append(
            {
                "i_teff": int(it),
                "i_logg": int(il),
                "i_mh": int(im),
                "i_g": int(ig),
                "n_available": int(n_available),
                "n_selected": int(n_selected),
            }
        )

    if not counts_records:
        counts_df = pd.DataFrame(
            columns=["i_teff", "i_logg", "i_mh", "i_g", "n_available", "n_selected"]
        )
    else:
        counts_df = pd.DataFrame(counts_records)

    # Shuffle the final selection so concatenation order doesn't leak bin
    # structure to downstream consumers.
    if selected_idx:
        selected_idx = rng.permutation(selected_idx).tolist()
    sample = df.iloc[selected_idx].reset_index(drop=True)
    logger.info(
        "stratified_subsample: drew %d stars from %d non-empty cells (of %d possible)",
        len(sample),
        len(counts_df),
        (len(bins_teff) - 1) * (len(bins_logg) - 1) * (len(bins_mh) - 1) * (len(bins_g) - 1),
    )

    return StratificationResult(
        sample=sample,
        cell_counts=counts_df,
        bins_teff=np.asarray(bins_teff),
        bins_logg=np.asarray(bins_logg),
        bins_mh=np.asarray(bins_mh),
        bins_g=np.asarray(bins_g),
        per_cell=per_cell,
        rng_seed=rng_seed,
        columns=(teff_col, logg_col, mh_col, g_col),
    )


def _empty_result(  # noqa: PLR0913 — internal helper, matches StratificationResult shape
    bins_teff: np.ndarray,
    bins_logg: np.ndarray,
    bins_mh: np.ndarray,
    bins_g: np.ndarray,
    per_cell: int,
    rng_seed: int,
    columns: tuple[str, str, str, str],
) -> StratificationResult:
    return StratificationResult(
        sample=pd.DataFrame(),
        cell_counts=pd.DataFrame(
            columns=["i_teff", "i_logg", "i_mh", "i_g", "n_available", "n_selected"]
        ),
        bins_teff=np.asarray(bins_teff),
        bins_logg=np.asarray(bins_logg),
        bins_mh=np.asarray(bins_mh),
        bins_g=np.asarray(bins_g),
        per_cell=per_cell,
        rng_seed=rng_seed,
        columns=columns,
    )


# -----------------------------------------------------------------------------
# Volume-limited sampler — natural-density draw for downstream clustering
# (Option C)
# -----------------------------------------------------------------------------
#
# Rationale. Density-based clustering (now performed downstream in the
# separate Starfold repository, not in this repo) learns cluster structure in
# natural density space, so stratification — which deliberately flattens the
# density distribution — is the wrong sampling scheme for that use case.
# Instead we take a volume-limited cut: keep every star with distance < d_cut,
# then (if the below-cut pool is still too large) draw a uniform random
# subsample from it.
#
# Distance cut default: 2.5 kpc. Rationale:
#   - data_acquisition.md §8.2 / §8.4 primary-Av strategy treats d < 1.25 kpc
#     (Edenhofer+2024) and 1.25–3 kpc (Lallement+2022 cross-check + neighborhood
#     median) as the two reliable regimes; beyond 3 kpc, Av relies on SFD +
#     GSP-Phot median only.
#   - 2.5 kpc sits inside the Lallement+2022 coverage, keeps Av uncertainty
#     dominated by line-of-sight maps rather than SFD saturation, and is a
#     common volume-limited cut for RGB+RC in the population-classification
#     literature (comparable to Dodd+2023's local sample).
#   - Neitzel+2025 used ~2 kpc on the TESS-CVZ side; 2.5 kpc extends that
#     radius without entering the regime where distance uncertainties blow
#     up the action-space scatter.
# The caller can override via the ``distance_cut_kpc`` argument; the value is
# returned in :class:`VolumeLimitedResult.to_provenance` for the sidecar.


DEFAULT_DISTANCE_CUT_KPC: Final[float] = 2.5
"""Default volume-limit in kiloparsecs for :func:`volume_limited_subsample`.
See module docstring for the rationale (data_acquisition.md §8.2 + §8.4)."""


@dataclass(frozen=True, slots=True)
class VolumeLimitedResult:
    """Output of :func:`volume_limited_subsample`.

    ``sample`` is the selected subset DataFrame. ``n_below_cut`` counts the
    pool size before the optional uniform downsample to ``n_target``.
    """

    sample: pd.DataFrame
    distance_col: str
    distance_cut_kpc: float
    n_target: int
    n_input: int
    n_below_cut: int
    n_selected: int
    rng_seed: int

    def to_provenance(self) -> dict:
        """Serialisable summary for the Parquet sidecar."""
        return {
            "method": "stream3_volume_limited_subsample",
            "distance_col": self.distance_col,
            "distance_cut_kpc": self.distance_cut_kpc,
            "n_target": self.n_target,
            "n_input": self.n_input,
            "n_below_cut": self.n_below_cut,
            "n_selected": self.n_selected,
            "rng_seed": self.rng_seed,
        }


def volume_limited_subsample(
    catalogue: pd.DataFrame,
    *,
    distance_col: str,
    distance_cut_kpc: float = DEFAULT_DISTANCE_CUT_KPC,
    n_target: int,
    seed: int = 0,
) -> VolumeLimitedResult:
    """Draw a volume-limited subsample for downstream density-based clustering.

    Selects stars with ``distance < distance_cut_kpc``. If the below-cut pool
    exceeds ``n_target``, a uniform random subsample of size ``n_target`` is
    taken from it — preserving natural density, which density-based
    clustering (e.g. HDBSCAN in Starfold) needs. If the pool is smaller than
    ``n_target``, the full pool is returned with a logged warning.

    Parameters
    ----------
    catalogue
        Input star catalogue. Must carry a distance column whose values are
        in **kiloparsecs**. Per data_acquisition.md §7.3, this is typically
        ``r_med_photogeo / 1000`` (Bailer-Jones+2021 ``r_med_photogeo`` is in
        parsecs). The caller is responsible for attaching and unit-converting
        the distance column — this function does not re-fetch distances.
    distance_col
        Column name holding the distance in kpc. Required — forcing the
        caller to name it makes pc-vs-kpc unit confusion a visible choice.
    distance_cut_kpc
        Keep stars with ``distance < distance_cut_kpc``. Default
        :data:`DEFAULT_DISTANCE_CUT_KPC` (2.5 kpc).
    n_target
        Target number of stars in the output. The realised size is
        ``min(n_target, n_below_cut)``.
    seed
        Random seed for the uniform downsample. Fixed for reproducibility.

    Returns
    -------
    VolumeLimitedResult
        Subsampled DataFrame (same schema as input) + accounting metadata.

    Notes
    -----
    Rows with NaN in ``distance_col`` are treated as above-cut and excluded.
    This is intentional: without a distance we cannot place the star in the
    volume, and density-based clustering on unknown-distance stars would
    smear the density estimate.
    """
    if distance_col not in catalogue.columns:
        raise KeyError(
            f"volume_limited_subsample requires column {distance_col!r}; "
            f"available: {sorted(catalogue.columns)}"
        )
    if distance_cut_kpc <= 0:
        raise ValueError(f"distance_cut_kpc must be positive, got {distance_cut_kpc}")
    if n_target <= 0:
        raise ValueError(f"n_target must be positive, got {n_target}")

    distances = catalogue[distance_col].to_numpy(dtype=float)
    n_input = len(catalogue)
    # np.less with NaN returns False → NaN rows are dropped from the mask.
    below_cut = np.less(distances, distance_cut_kpc)
    n_below_cut = int(below_cut.sum())

    below_pool = catalogue.loc[below_cut]

    if n_below_cut <= n_target:
        if n_below_cut < n_target:
            logger.warning(
                "volume_limited_subsample: below-cut pool (%d) smaller than "
                "n_target (%d); returning full pool",
                n_below_cut,
                n_target,
            )
        sample = below_pool.reset_index(drop=True)
    else:
        rng = np.random.default_rng(seed)
        picks = rng.choice(n_below_cut, size=n_target, replace=False)
        picks.sort()
        sample = below_pool.iloc[picks].reset_index(drop=True)

    logger.info(
        "volume_limited_subsample: d < %.3f kpc on column %r: "
        "%d/%d below cut, %d selected (seed=%d)",
        distance_cut_kpc,
        distance_col,
        n_below_cut,
        n_input,
        len(sample),
        seed,
    )

    return VolumeLimitedResult(
        sample=sample,
        distance_col=distance_col,
        distance_cut_kpc=float(distance_cut_kpc),
        n_target=int(n_target),
        n_input=int(n_input),
        n_below_cut=n_below_cut,
        n_selected=int(len(sample)),
        rng_seed=int(seed),
    )


__all__ = [
    "DEFAULT_BINS_G",
    "DEFAULT_BINS_LOGG",
    "DEFAULT_BINS_MH",
    "DEFAULT_BINS_TEFF",
    "DEFAULT_DISTANCE_CUT_KPC",
    "DEFAULT_PER_CELL",
    "StratificationResult",
    "VolumeLimitedResult",
    "stratified_subsample",
    "volume_limited_subsample",
]
