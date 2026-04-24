"""Deduplicate Stream 1 training rows on Gaia DR3 ``source_id``.

Motivation
----------
The APOGEE DR19 Astra ASPCAP summary (``astraAllStarASPCAP-0.6.0.fits.gz``)
carries multiple ASPCAP rows per physical star when the same spectrum went
through more than one Astra task (``task_pk``). For the Stream 1 post-cut
pool (N = 354 890) this hits 27 920 Gaia source_ids, producing 31 106 extra
rows — a 9.6% row-weight inflation toward heavily-observed fields (bulge,
open clusters).

Why this matters for Pipeline 1
-------------------------------
- Random train/val/test splits would leak the *same* star across splits.
  Reliability diagrams on the validation fold would look calibrated while
  Stream 3 inference would be miscalibrated in production.
- Upweighting heavily-observed fields biases the loss toward bulge/cluster
  Teff-[Fe/H]-logg regions, degrading performance in the halo/hot/low-[Fe/H]
  tails — exactly the regions Pipeline 1 needs to be good at for D-Cat-b.

Empirical facts on the 2026-04 Stream 1
---------------------------------------
- Within-duplicate label scatter is far below APOGEE precision:

  ============ ================ =================
  label        within-star σ    APOGEE precision
  ============ ================ =================
  Teff          7.6 K            50–70 K
  [Fe/H]        0.010 dex        0.02–0.04 dex
  [Mg/Fe]       0.014 dex        0.02–0.04 dex
  ============ ================ =================

  i.e. the duplicates are *not* independent label realisations — they are
  re-runs of the ASPCAP fit on the same (or very similar) spectrum that
  converge to effectively identical answers.
- ``v_astra`` is constant at ``0.6.0`` across the entire post-cut pool.
  There is no DR17-vs-DR19 pipeline-variant mix, so a single-level sort on
  SNR is the correct dedup key. If a future DR ships mixed ``v_astra``,
  upgrade the dedup key to ``(v_astra desc, snr desc)``.

Usage
-----
.. code-block:: python

    from arqueogal.data.dedup import dedup_by_source_id

    df, stats = dedup_by_source_id(stream1_features)
    # stats["rows_in"], stats["rows_out"], stats["unique_source_ids"],
    # stats["n_duplicate_stars"], stats["max_duplicates_per_star"] ...

See ``tests/data/test_dedup.py`` for the invariants the function guarantees.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_SORT_COLUMN = "snr"
"""Dedup key: highest SNR wins. Validated 2026-04 on Stream 1: ``v_astra``
is constant, so no pipeline-variant tiebreak is needed."""


@dataclass(frozen=True, slots=True)
class DedupStats:
    """Summary returned alongside a deduplicated DataFrame."""

    rows_in: int
    rows_out: int
    unique_source_ids: int
    n_duplicate_stars: int
    """Number of distinct source_ids that had >1 row on input."""
    max_duplicates_per_star: int
    sort_column: str
    sort_ascending: bool
    source_id_column: str
    histogram: dict[int, int] = field(default_factory=dict)
    """Map of duplicate-count → number-of-stars-with-that-count. Key 1 = stars
    that appeared once; key 2 = stars that appeared twice; etc."""

    def to_dict(self) -> dict:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "unique_source_ids": self.unique_source_ids,
            "n_duplicate_stars": self.n_duplicate_stars,
            "max_duplicates_per_star": self.max_duplicates_per_star,
            "sort_column": self.sort_column,
            "sort_ascending": self.sort_ascending,
            "source_id_column": self.source_id_column,
            "duplicate_histogram": self.histogram,
        }


def dedup_by_source_id(
    df: pd.DataFrame,
    *,
    sort_by: str = DEFAULT_SORT_COLUMN,
    ascending: bool = False,
    source_id_col: str = "source_id",
) -> tuple[pd.DataFrame, DedupStats]:
    """Return one row per ``source_id``, keeping the highest-``sort_by`` row.

    Parameters
    ----------
    df : pd.DataFrame
        Input training-pool frame. Must contain ``source_id_col`` and
        ``sort_by`` columns.
    sort_by : str, default "snr"
        Column to sort on when choosing which row to keep per star.
    ascending : bool, default False
        If ``False`` (default), the row with the *largest* ``sort_by`` value
        is kept. Set ``True`` if a smaller value is "better" for your key.
    source_id_col : str, default "source_id"
        Name of the group-identity column.

    Returns
    -------
    deduped : pd.DataFrame
        One row per unique ``source_id_col`` value. Index is reset.
    stats : DedupStats
        Row counts, histogram of duplicate multiplicities, and the exact sort
        key used — suitable to embed verbatim in a provenance sidecar.

    Raises
    ------
    KeyError
        If ``source_id_col`` or ``sort_by`` is absent.
    ValueError
        If ``sort_by`` has any NaN values among rows that would otherwise be
        chosen as canonical — the dedup choice becomes undefined.

    Notes
    -----
    The output preserves *input* row ordering after the dedup: the kept row
    per star is the first encountered under the sort. ``index`` is reset.
    """
    if source_id_col not in df.columns:
        raise KeyError(f"column {source_id_col!r} not in DataFrame")
    if sort_by not in df.columns:
        raise KeyError(f"sort-by column {sort_by!r} not in DataFrame")

    rows_in = len(df)
    counts = df[source_id_col].value_counts()
    n_unique = int(counts.size)
    n_dup_stars = int((counts > 1).sum())
    max_dup = int(counts.max()) if n_unique > 0 else 0

    # If there are no duplicates, skip the sort entirely.
    if n_dup_stars == 0:
        logger.info(
            "dedup_by_source_id: no duplicates in %d rows (unique %d)",
            rows_in,
            n_unique,
        )
        stats = DedupStats(
            rows_in=rows_in,
            rows_out=rows_in,
            unique_source_ids=n_unique,
            n_duplicate_stars=0,
            max_duplicates_per_star=max_dup,
            sort_column=sort_by,
            sort_ascending=ascending,
            source_id_column=source_id_col,
            histogram={1: n_unique} if n_unique else {},
        )
        return df.reset_index(drop=True), stats

    # Reject NaN in the sort column — the "best" row is undefined.
    sort_nan = df[sort_by].isna()
    if sort_nan.any():
        n_nan = int(sort_nan.sum())
        affected_sids = df.loc[sort_nan, source_id_col].nunique()
        raise ValueError(
            f"cannot deduplicate: {n_nan} row(s) across {affected_sids} "
            f"source_ids have NaN in sort column {sort_by!r}"
        )

    sorted_df = df.sort_values(sort_by, ascending=ascending, kind="mergesort")
    deduped = sorted_df.drop_duplicates(subset=source_id_col, keep="first").reset_index(drop=True)

    histogram = {int(k): int(v) for k, v in counts.value_counts().sort_index().items()}
    stats = DedupStats(
        rows_in=rows_in,
        rows_out=len(deduped),
        unique_source_ids=n_unique,
        n_duplicate_stars=n_dup_stars,
        max_duplicates_per_star=max_dup,
        sort_column=sort_by,
        sort_ascending=ascending,
        source_id_column=source_id_col,
        histogram=histogram,
    )
    logger.info(
        "dedup_by_source_id: %d → %d rows (%d unique, %d dup-stars, max=%d)",
        stats.rows_in,
        stats.rows_out,
        stats.unique_source_ids,
        stats.n_duplicate_stars,
        stats.max_duplicates_per_star,
    )
    return deduped, stats


__all__ = [
    "DEFAULT_SORT_COLUMN",
    "DedupStats",
    "dedup_by_source_id",
]
