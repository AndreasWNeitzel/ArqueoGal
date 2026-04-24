"""Unit tests for :mod:`arqueogal.data.dedup`.

Invariants verified:

1. No-dup input is returned unchanged in row count, and the index is reset.
2. Duplicates collapse to one row per ``source_id``.
3. The kept row per star has the maximum ``sort_by`` value (default: SNR).
4. ``ascending=True`` keeps the *minimum* instead.
5. Returned stats match the observed row/star counts and histogram.
6. Missing columns raise KeyError with a useful message.
7. NaN in the sort column raises ValueError rather than silently picking
   a NaN-ranked row.
8. Non-``snr`` sort key works (verifying the function is not hard-coded).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arqueogal.data.dedup import (
    DEFAULT_SORT_COLUMN,
    DedupStats,
    dedup_by_source_id,
)


def _make_apogee_like(
    source_ids: list[int],
    snrs: list[float],
    extras: dict[str, list] | None = None,
) -> pd.DataFrame:
    data: dict[str, list] = {"source_id": source_ids, "snr": snrs}
    if extras:
        data.update(extras)
    return pd.DataFrame(data)


def test_no_duplicates_returns_input_unchanged() -> None:
    df = _make_apogee_like([1, 2, 3, 4], [100.0, 200.0, 150.0, 180.0])
    out, stats = dedup_by_source_id(df)
    assert len(out) == 4
    assert set(out["source_id"]) == {1, 2, 3, 4}
    assert stats.rows_in == 4
    assert stats.rows_out == 4
    assert stats.n_duplicate_stars == 0
    assert stats.max_duplicates_per_star == 1
    # Index is reset
    assert list(out.index) == [0, 1, 2, 3]


def test_duplicates_collapse_to_unique_source_ids() -> None:
    df = _make_apogee_like(
        source_ids=[1, 1, 2, 2, 2, 3],
        snrs=[100.0, 500.0, 50.0, 60.0, 55.0, 300.0],
    )
    out, stats = dedup_by_source_id(df)
    assert len(out) == 3
    assert sorted(out["source_id"].tolist()) == [1, 2, 3]
    assert stats.rows_in == 6
    assert stats.rows_out == 3
    assert stats.unique_source_ids == 3
    assert stats.n_duplicate_stars == 2  # source_ids 1 and 2 had >1 row
    assert stats.max_duplicates_per_star == 3


def test_highest_snr_row_is_kept_per_star() -> None:
    df = _make_apogee_like(
        source_ids=[1, 1, 1, 2, 2],
        snrs=[100.0, 999.0, 50.0, 200.0, 150.0],
        extras={"payload": ["a", "b", "c", "d", "e"]},
    )
    out, _ = dedup_by_source_id(df)
    kept_1 = out.loc[out["source_id"] == 1, "payload"].iloc[0]
    kept_2 = out.loc[out["source_id"] == 2, "payload"].iloc[0]
    assert kept_1 == "b", "source_id=1 kept row must be the snr=999 row"
    assert kept_2 == "d", "source_id=2 kept row must be the snr=200 row"


def test_ascending_true_keeps_lowest_value() -> None:
    df = _make_apogee_like(
        source_ids=[1, 1, 2, 2],
        snrs=[100.0, 50.0, 300.0, 400.0],
        extras={"payload": ["hi1", "lo1", "lo2", "hi2"]},
    )
    out, stats = dedup_by_source_id(df, ascending=True)
    assert out.loc[out["source_id"] == 1, "payload"].iloc[0] == "lo1"
    assert out.loc[out["source_id"] == 2, "payload"].iloc[0] == "lo2"
    assert stats.sort_ascending is True


def test_stats_histogram_matches_observed_multiplicities() -> None:
    # Multiplicities: source_id 1 → 3x, 2 → 2x, 3 → 1x, 4 → 1x
    df = _make_apogee_like(
        source_ids=[1, 1, 1, 2, 2, 3, 4],
        snrs=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
    )
    _, stats = dedup_by_source_id(df)
    # The histogram reported is "number of stars with K rows" for each K.
    # 2 stars have a single row (source_ids 3, 4),
    # 1 star has two rows (source_id 2),
    # 1 star has three rows (source_id 1).
    assert stats.histogram == {1: 2, 2: 1, 3: 1}
    assert stats.max_duplicates_per_star == 3


def test_missing_source_id_column_raises() -> None:
    df = pd.DataFrame({"snr": [1.0, 2.0]})
    with pytest.raises(KeyError, match="source_id"):
        dedup_by_source_id(df)


def test_missing_sort_column_raises() -> None:
    df = pd.DataFrame({"source_id": [1, 2]})
    with pytest.raises(KeyError, match="snr"):
        dedup_by_source_id(df)


def test_nan_in_sort_column_raises() -> None:
    df = _make_apogee_like(
        source_ids=[1, 1, 2],
        snrs=[np.nan, 100.0, 200.0],
    )
    with pytest.raises(ValueError, match="NaN"):
        dedup_by_source_id(df)


def test_non_snr_sort_key_works() -> None:
    # Use an alternate sort key (e.g. an ASPCAP_CHI2 ascending=True scenario)
    df = pd.DataFrame(
        {
            "source_id": [1, 1, 2],
            "snr": [10.0, 20.0, 30.0],
            "aspcap_chi2": [5.0, 2.0, 3.0],  # lower is better
        }
    )
    out, stats = dedup_by_source_id(df, sort_by="aspcap_chi2", ascending=True)
    assert stats.sort_column == "aspcap_chi2"
    assert stats.sort_ascending is True
    # For source_id=1 the chi2=2.0 row (snr=20) must be kept
    assert out.loc[out["source_id"] == 1, "snr"].iloc[0] == 20.0


def test_stats_to_dict_is_json_friendly() -> None:
    df = _make_apogee_like([1, 1, 2], [10.0, 20.0, 30.0])
    _, stats = dedup_by_source_id(df)
    d = stats.to_dict()
    # All values must be primitive types so Provenance sidecar can JSON-encode.
    import json

    json.dumps(d)  # will raise if not serialisable
    assert d["sort_column"] == DEFAULT_SORT_COLUMN
    assert d["rows_out"] == 2


def test_index_is_reset_after_dedup() -> None:
    df = _make_apogee_like([1, 1, 2], [10.0, 20.0, 30.0])
    df.index = [100, 200, 300]
    out, _ = dedup_by_source_id(df)
    assert list(out.index) == [0, 1]


def test_dedupstats_is_frozen() -> None:
    stats = DedupStats(
        rows_in=1,
        rows_out=1,
        unique_source_ids=1,
        n_duplicate_stars=0,
        max_duplicates_per_star=1,
        sort_column="snr",
        sort_ascending=False,
        source_id_column="source_id",
    )
    with pytest.raises((AttributeError, TypeError)):
        stats.rows_in = 99  # type: ignore[misc]
