"""Cross-matches — §4.3 and §11.1 of data_acquisition.md.

Two operations live here:

1. **DR2 → DR3 via ``gaiadr3.dr2_neighbourhood``** — the official Gaia
   cross-match table. Used by Stream 2 (TESS/Hon+2021 ν_max stars carry a
   DR2 ID from the TESS Input Catalog; we need DR3 to query XP and
   astrometry). Also used in any legacy DR2-indexed catalogue lookup.

2. **Filtering + tie-breaking** — ``dr2_neighbourhood`` is many-to-many
   (data_acquisition.md §14.6). We apply

   - ``angular_distance < 300 mas``,
   - ``|magnitude_difference| < 0.1``,
   - brightest-DR3 tie-breaker per DR2 source (prefer the DR3 match with
     the **smallest** ``magnitude_difference``; a DR3 star brighter than
     the DR2 source has negative ``magnitude_difference`` — "brightest" is
     ambiguous without a tie-breaker rule, so we define it as smallest
     magnitude offset to the input, which mathematically matches picking
     the DR3 source closest in brightness to the original DR2 detection).

The TAP-fetch step is delegated to :func:`arqueogal.data.tap.batched_fetch_df`;
filtering and tie-breaking are pure pandas so they're fully offline-testable.

TIC → Gaia is a two-step: TIC v8.2 publishes a DR2 ID in the ``GAIA`` column,
so the DR2-to-DR3 path above is the Stream-2 glue. Pulling TIC rows from
VizieR is a separate concern handled by :mod:`arqueogal.data.tess_hon2021`.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd
from pyvo.dal.tap import TAPService

from arqueogal.data.tap import BATCH_PLACEHOLDER, DEFAULT_ASYNC_TIMEOUT_SEC, batched_fetch_df

logger = logging.getLogger(__name__)

DEFAULT_ANGULAR_DISTANCE_MAS: Final[float] = 300.0
"""§4.3 cut. The dr2_neighbourhood field is in mas."""

DEFAULT_MAG_DIFF_LIMIT: Final[float] = 0.1
"""§4.3 cut. ``magnitude_difference = G_DR3 − G_DR2``."""

DR2_NEIGHBOURHOOD_ADQL = f"""
SELECT
    nbh.dr2_source_id,
    nbh.dr3_source_id,
    nbh.angular_distance,
    nbh.magnitude_difference
FROM gaiadr3.dr2_neighbourhood AS nbh
WHERE nbh.dr2_source_id IN ({BATCH_PLACEHOLDER})
"""
"""Parameterised over a comma-joined batch of DR2 source ids."""


def fetch_dr2_neighbourhood(  # noqa: PLR0913 — keyword-only tuning knobs with safe defaults
    service: TAPService,
    dr2_source_ids: list[int] | np.ndarray,
    *,
    batch_size: int = 5_000,
    checkpoint_dir=None,  # noqa: ANN001 — Path | None but None default mutable-free
    mode: str = "async",
    timeout_sec: int = DEFAULT_ASYNC_TIMEOUT_SEC,
) -> pd.DataFrame:
    """Fetch raw ``gaiadr3.dr2_neighbourhood`` rows for the given DR2 ids.

    Delegates to :func:`batched_fetch_df`. Output has columns:
    ``dr2_source_id``, ``dr3_source_id``, ``angular_distance`` (mas),
    ``magnitude_difference`` (G_DR3 − G_DR2, mag).

    Raw output is many-to-many — pass through :func:`resolve_dr2_to_dr3`
    to apply §4.3 cuts and pick a unique DR3 per DR2.
    """
    return batched_fetch_df(
        service,
        dr2_source_ids,
        DR2_NEIGHBOURHOOD_ADQL,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix="dr2_nbh",
        mode=mode,
        timeout_sec=timeout_sec,
    )


def resolve_dr2_to_dr3(
    neighbourhood_df: pd.DataFrame,
    *,
    max_angular_distance_mas: float = DEFAULT_ANGULAR_DISTANCE_MAS,
    max_mag_diff: float = DEFAULT_MAG_DIFF_LIMIT,
) -> pd.DataFrame:
    """Apply §4.3 cuts and tie-break to a raw dr2_neighbourhood table.

    Keeps only rows with ``angular_distance < max_angular_distance_mas`` and
    ``|magnitude_difference| < max_mag_diff``. Where a DR2 source still has
    multiple surviving DR3 candidates (close-binary splits), the one with
    the smallest ``|magnitude_difference|`` is retained — per §14.6, never
    pick the first arbitrarily. Multiplicity is logged at INFO.

    Parameters
    ----------
    neighbourhood_df
        Output of :func:`fetch_dr2_neighbourhood`. Must carry
        ``dr2_source_id``, ``dr3_source_id``, ``angular_distance``,
        ``magnitude_difference``.

    Returns
    -------
    pd.DataFrame
        One row per DR2 source that passed the cuts. Columns:
        ``dr2_source_id``, ``dr3_source_id``, ``angular_distance``,
        ``magnitude_difference``, ``n_candidates`` (count of DR3 matches
        that passed the cuts for this DR2 — ``>1`` flags an ambiguity
        resolved by the tie-breaker).
    """
    required = {"dr2_source_id", "dr3_source_id", "angular_distance", "magnitude_difference"}
    missing = required - set(neighbourhood_df.columns)
    if missing:
        raise KeyError(f"resolve_dr2_to_dr3 requires columns: {sorted(missing)}")
    if max_angular_distance_mas <= 0:
        raise ValueError(
            f"max_angular_distance_mas must be positive, got {max_angular_distance_mas}"
        )
    if max_mag_diff <= 0:
        raise ValueError(f"max_mag_diff must be positive, got {max_mag_diff}")

    df = neighbourhood_df.copy()
    n_raw = len(df)
    mask_ang = df["angular_distance"] < max_angular_distance_mas
    mask_mag = df["magnitude_difference"].abs() < max_mag_diff
    df = df.loc[mask_ang & mask_mag].copy()
    logger.info(
        "resolve_dr2_to_dr3: %d/%d rows passed cuts (angular_distance<%s mas, |Δmag|<%s)",
        len(df), n_raw, max_angular_distance_mas, max_mag_diff,
    )

    if df.empty:
        return pd.DataFrame(
            columns=[
                "dr2_source_id", "dr3_source_id",
                "angular_distance", "magnitude_difference", "n_candidates",
            ]
        )

    # Count candidates per DR2 id *before* tie-breaking so downstream sees
    # how ambiguous each match was.
    candidate_counts = df.groupby("dr2_source_id").size().rename("n_candidates")
    if (candidate_counts > 1).any():
        logger.info(
            "resolve_dr2_to_dr3: %d DR2 ids have >1 DR3 candidate; tie-breaking by |Δmag|",
            int((candidate_counts > 1).sum()),
        )

    # Tie-break: smallest |magnitude_difference| wins.
    df["_abs_dmag"] = df["magnitude_difference"].abs()
    df = df.sort_values(["dr2_source_id", "_abs_dmag"], ascending=[True, True])
    df = df.drop_duplicates(subset="dr2_source_id", keep="first")
    df = df.drop(columns="_abs_dmag").reset_index(drop=True)
    df = df.merge(candidate_counts, on="dr2_source_id", how="left")
    return df[
        [
            "dr2_source_id", "dr3_source_id",
            "angular_distance", "magnitude_difference", "n_candidates",
        ]
    ]


def crossmatch_dr2_to_dr3(  # noqa: PLR0913 — keyword-only tuning knobs with safe defaults
    service: TAPService,
    dr2_source_ids: list[int] | np.ndarray,
    *,
    batch_size: int = 5_000,
    checkpoint_dir=None,  # noqa: ANN001
    max_angular_distance_mas: float = DEFAULT_ANGULAR_DISTANCE_MAS,
    max_mag_diff: float = DEFAULT_MAG_DIFF_LIMIT,
    mode: str = "async",
    timeout_sec: int = DEFAULT_ASYNC_TIMEOUT_SEC,
) -> pd.DataFrame:
    """End-to-end DR2→DR3 cross-match: fetch + filter + tie-break.

    Convenience wrapper that chains :func:`fetch_dr2_neighbourhood` and
    :func:`resolve_dr2_to_dr3` with the default §4.3 cuts.
    """
    raw = fetch_dr2_neighbourhood(
        service, dr2_source_ids,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
        mode=mode,
        timeout_sec=timeout_sec,
    )
    return resolve_dr2_to_dr3(
        raw,
        max_angular_distance_mas=max_angular_distance_mas,
        max_mag_diff=max_mag_diff,
    )


__all__ = [
    "DEFAULT_ANGULAR_DISTANCE_MAS",
    "DEFAULT_MAG_DIFF_LIMIT",
    "DR2_NEIGHBOURHOOD_ADQL",
    "crossmatch_dr2_to_dr3",
    "fetch_dr2_neighbourhood",
    "resolve_dr2_to_dr3",
]
