"""Stream 3 stratified sub-sampling — §5.3 of data_acquisition.md.

Draws a (Teff, logg, [M/H], G)-stratified 1.5 M-star subset from the
Andrae+2023 vetted RGB catalogue (Zenodo 7945154). The stratification is
essential for Pipeline 2 diagnostics — an unstratified Gaia draw is
dominated by disc stars at solar metallicity and under-samples the halo
and metal-poor tails where interesting population structure lives (§5.3).

The downloader (Andrae+2023 Zenodo) lives elsewhere; this module operates
on an already-loaded DataFrame.

Algorithm:

1. Assign each star to a 4D bin using ``np.digitize`` on the four axes.
2. For each non-empty bin, take all stars if the bin has ≤ ``per_cell``,
   else draw a uniform random subsample of ``per_cell`` stars.
3. Concatenate and shuffle (reproducibly via ``rng_seed``).

Both the stratification parameters and the seed are part of the returned
:class:`StratificationResult` for provenance logging.
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
            bins_teff, bins_logg, bins_mh, bins_g, per_cell, rng_seed,
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
        (i_teff >= 0) & (i_teff < len(bins_teff) - 1)
        & (i_logg >= 0) & (i_logg < len(bins_logg) - 1)
        & (i_mh >= 0) & (i_mh < len(bins_mh) - 1)
        & (i_g >= 0) & (i_g < len(bins_g) - 1)
    )
    finite = np.isfinite(teff) & np.isfinite(logg) & np.isfinite(mh) & np.isfinite(g_mag)
    valid = in_range & finite

    n_excluded = (~valid).sum()
    if n_excluded:
        logger.info(
            "stratified_subsample: %d/%d rows excluded (NaN or outside bin range)",
            int(n_excluded), len(df),
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
                "i_teff": int(it), "i_logg": int(il), "i_mh": int(im), "i_g": int(ig),
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
        "stratified_subsample: drew %d stars from %d non-empty cells "
        "(of %d possible)",
        len(sample), len(counts_df),
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
    bins_teff: np.ndarray, bins_logg: np.ndarray,
    bins_mh: np.ndarray, bins_g: np.ndarray,
    per_cell: int, rng_seed: int,
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


__all__ = [
    "DEFAULT_BINS_G",
    "DEFAULT_BINS_LOGG",
    "DEFAULT_BINS_MH",
    "DEFAULT_BINS_TEFF",
    "DEFAULT_PER_CELL",
    "StratificationResult",
    "stratified_subsample",
]
