"""Offline tests for arqueogal.data.crossmatch — §4.3 DR2→DR3 resolution.

TAP is mocked via the :mod:`arqueogal.data.tap` hooks (same strategy as
test_distances.py). The tie-breaking logic is pure pandas and is covered
by inputs constructed directly.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from astropy.table import Table
from pyvo.dal.tap import TAPService

from arqueogal.data import tap as tap_mod
from arqueogal.data.crossmatch import (
    DEFAULT_ANGULAR_DISTANCE_MAS,
    DEFAULT_MAG_DIFF_LIMIT,
    DR2_NEIGHBOURHOOD_ADQL,
    crossmatch_dr2_to_dr3,
    fetch_dr2_neighbourhood,
    resolve_dr2_to_dr3,
)


def _extract_ids(adql: str) -> list[int]:
    match = re.search(r"IN \(([^)]+)\)", adql)
    assert match is not None
    return [int(x) for x in match.group(1).split(",")]


def _fake_nbh_table(rows: list[tuple[int, int, float, float]]) -> Table:
    """Build a pyvo-like result table from (dr2, dr3, ang_dist, mag_diff) tuples."""
    if not rows:
        return Table(
            {
                "dr2_source_id": np.array([], dtype=np.int64),
                "dr3_source_id": np.array([], dtype=np.int64),
                "angular_distance": np.array([], dtype=float),
                "magnitude_difference": np.array([], dtype=float),
            }
        )
    dr2, dr3, ang, dmag = zip(*rows, strict=True)
    return Table(
        {
            "dr2_source_id": list(dr2),
            "dr3_source_id": list(dr3),
            "angular_distance": list(ang),
            "magnitude_difference": list(dmag),
        }
    )


# ---- constants & ADQL shape --------------------------------------------------


def test_query_targets_dr2_neighbourhood_table() -> None:
    assert "gaiadr3.dr2_neighbourhood" in DR2_NEIGHBOURHOOD_ADQL
    assert "angular_distance" in DR2_NEIGHBOURHOOD_ADQL
    assert "magnitude_difference" in DR2_NEIGHBOURHOOD_ADQL


def test_default_cuts_match_section_4_3() -> None:
    assert DEFAULT_ANGULAR_DISTANCE_MAS == 300.0
    assert DEFAULT_MAG_DIFF_LIMIT == 0.1


# ---- fetch_dr2_neighbourhood -------------------------------------------------


def test_fetch_dr2_neighbourhood_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[int]] = []

    def fake_async(_svc, adql, **_kw):  # noqa: ANN001
        ids = _extract_ids(adql)
        calls.append(ids)
        # One 1:1 match per input DR2 id.
        return _fake_nbh_table([(d, d * 10, 50.0, 0.01) for d in ids])

    monkeypatch.setattr(tap_mod, "run_async", fake_async)
    monkeypatch.setattr(tap_mod, "run_sync", lambda *_a, **_kw: pytest.fail("sync should not fire"))

    service = MagicMock(spec=TAPService)
    out = fetch_dr2_neighbourhood(service, [1, 2, 3, 4, 5], batch_size=2)
    assert calls == [[1, 2], [3, 4], [5]]
    assert len(out) == 5


def test_fetch_dr2_neighbourhood_empty_input() -> None:
    service = MagicMock(spec=TAPService)
    assert fetch_dr2_neighbourhood(service, []).empty


# ---- resolve_dr2_to_dr3: cuts ------------------------------------------------


def test_resolve_applies_angular_distance_cut() -> None:
    df = pd.DataFrame(
        {
            "dr2_source_id": [1, 2],
            "dr3_source_id": [10, 20],
            "angular_distance": [50.0, 500.0],  # second exceeds 300 mas
            "magnitude_difference": [0.01, 0.01],
        }
    )
    out = resolve_dr2_to_dr3(df)
    assert list(out["dr2_source_id"]) == [1]


def test_resolve_applies_magnitude_difference_cut() -> None:
    df = pd.DataFrame(
        {
            "dr2_source_id": [1, 2, 3],
            "dr3_source_id": [10, 20, 30],
            "angular_distance": [50.0, 50.0, 50.0],
            "magnitude_difference": [0.05, -0.05, 0.2],  # third fails |Δmag| < 0.1
        }
    )
    out = resolve_dr2_to_dr3(df)
    assert list(out["dr2_source_id"]) == [1, 2]


def test_resolve_negative_mag_difference_still_accepted() -> None:
    """magnitude_difference<0 means DR3 brighter than DR2 — still valid."""
    df = pd.DataFrame(
        {
            "dr2_source_id": [1],
            "dr3_source_id": [10],
            "angular_distance": [50.0],
            "magnitude_difference": [-0.05],
        }
    )
    out = resolve_dr2_to_dr3(df)
    assert len(out) == 1
    assert out["magnitude_difference"].iloc[0] == -0.05


# ---- resolve_dr2_to_dr3: tie-breaking ----------------------------------------


def test_resolve_tie_breaks_by_smallest_abs_magnitude_difference() -> None:
    """Same DR2 → two DR3 candidates; keep the one with smaller |Δmag|."""
    df = pd.DataFrame(
        {
            "dr2_source_id": [1, 1],
            "dr3_source_id": [10, 11],
            "angular_distance": [100.0, 100.0],
            "magnitude_difference": [0.08, 0.02],  # second is closer in brightness
        }
    )
    out = resolve_dr2_to_dr3(df)
    assert len(out) == 1
    assert out["dr3_source_id"].iloc[0] == 11
    assert out["n_candidates"].iloc[0] == 2


def test_resolve_tie_breaks_symmetrically_on_negative_deltas() -> None:
    df = pd.DataFrame(
        {
            "dr2_source_id": [1, 1],
            "dr3_source_id": [10, 11],
            "angular_distance": [50.0, 50.0],
            "magnitude_difference": [-0.01, 0.05],  # |−0.01| < |0.05|
        }
    )
    out = resolve_dr2_to_dr3(df)
    assert out["dr3_source_id"].iloc[0] == 10


def test_resolve_reports_n_candidates_correctly() -> None:
    """n_candidates counts the surviving-cut matches per DR2 id."""
    df = pd.DataFrame(
        {
            "dr2_source_id": [1, 1, 1, 2, 2, 3],
            "dr3_source_id": [10, 11, 12, 20, 21, 30],
            "angular_distance": [50.0, 60.0, 70.0, 50.0, 80.0, 50.0],
            "magnitude_difference": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        }
    )
    out = resolve_dr2_to_dr3(df).set_index("dr2_source_id")
    assert out.loc[1, "n_candidates"] == 3
    assert out.loc[2, "n_candidates"] == 2
    assert out.loc[3, "n_candidates"] == 1


def test_resolve_n_candidates_excludes_cut_rows() -> None:
    """Candidates that failed the cuts don't inflate n_candidates."""
    df = pd.DataFrame(
        {
            "dr2_source_id": [1, 1],
            "dr3_source_id": [10, 11],
            "angular_distance": [50.0, 500.0],  # second cut
            "magnitude_difference": [0.01, 0.01],
        }
    )
    out = resolve_dr2_to_dr3(df)
    assert out["n_candidates"].iloc[0] == 1


# ---- resolve_dr2_to_dr3: edge cases -----------------------------------------


def test_resolve_empty_input() -> None:
    df = pd.DataFrame(
        {
            "dr2_source_id": np.array([], dtype=np.int64),
            "dr3_source_id": np.array([], dtype=np.int64),
            "angular_distance": np.array([], dtype=float),
            "magnitude_difference": np.array([], dtype=float),
        }
    )
    out = resolve_dr2_to_dr3(df)
    assert out.empty
    assert "n_candidates" in out.columns


def test_resolve_all_cut() -> None:
    df = pd.DataFrame(
        {
            "dr2_source_id": [1, 2],
            "dr3_source_id": [10, 20],
            "angular_distance": [1000.0, 2000.0],
            "magnitude_difference": [0.01, 0.01],
        }
    )
    out = resolve_dr2_to_dr3(df)
    assert out.empty


def test_resolve_missing_column_raises() -> None:
    df = pd.DataFrame({"dr2_source_id": [1], "dr3_source_id": [10]})
    with pytest.raises(KeyError, match="resolve_dr2_to_dr3 requires columns"):
        resolve_dr2_to_dr3(df)


def test_resolve_rejects_nonpositive_thresholds() -> None:
    df = _fake_nbh_table([(1, 10, 50.0, 0.01)]).to_pandas()
    with pytest.raises(ValueError, match="max_angular_distance_mas"):
        resolve_dr2_to_dr3(df, max_angular_distance_mas=0.0)
    with pytest.raises(ValueError, match="max_mag_diff"):
        resolve_dr2_to_dr3(df, max_mag_diff=0.0)


def test_resolve_respects_custom_thresholds() -> None:
    """Tightening the cuts drops rows that would pass with defaults."""
    df = pd.DataFrame(
        {
            "dr2_source_id": [1, 2],
            "dr3_source_id": [10, 20],
            "angular_distance": [250.0, 50.0],
            "magnitude_difference": [0.05, 0.08],
        }
    )
    # Default: both pass.
    out_default = resolve_dr2_to_dr3(df)
    assert len(out_default) == 2

    # Tighter angular cut drops DR2=1.
    out_tight = resolve_dr2_to_dr3(df, max_angular_distance_mas=100.0)
    assert list(out_tight["dr2_source_id"]) == [2]


# ---- end-to-end --------------------------------------------------------------


def test_crossmatch_dr2_to_dr3_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_async(_svc, adql, **_kw):  # noqa: ANN001
        ids = _extract_ids(adql)
        rows = []
        for d in ids:
            if d == 1:
                # Clean 1:1
                rows.append((1, 101, 30.0, 0.01))
            elif d == 2:
                # Two candidates, both pass cuts — tie-break to 202.
                rows.append((2, 201, 50.0, 0.08))
                rows.append((2, 202, 50.0, 0.02))
            elif d == 3:
                # Fails angular cut.
                rows.append((3, 301, 500.0, 0.01))
        return _fake_nbh_table(rows)

    monkeypatch.setattr(tap_mod, "run_async", fake_async)
    service = MagicMock(spec=TAPService)

    out = crossmatch_dr2_to_dr3(service, [1, 2, 3], batch_size=10)
    # 3 was cut; 1 and 2 remain.
    assert sorted(out["dr2_source_id"].tolist()) == [1, 2]
    two = out[out["dr2_source_id"] == 2].iloc[0]
    assert two["dr3_source_id"] == 202
    assert two["n_candidates"] == 2
    one = out[out["dr2_source_id"] == 1].iloc[0]
    assert one["n_candidates"] == 1
