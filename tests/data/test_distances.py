"""Offline tests for arqueogal.data.distances — §7.

TAP is mocked: ``run_async`` / ``run_sync`` in the distances module are
monkeypatched to return synthesised astropy Tables. No network I/O.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from astropy.table import Table
from pyvo.dal.tap import TAPService

from arqueogal.data import tap as tap_mod
from arqueogal.data.distances import (
    BAILERJONES_ADQL,
    DIST_CONFLICT_LOG_THRESHOLD,
    STARHORSE2_SAMPLE_TABLES,
    fetch_bailerjones,
    fetch_starhorse2,
    merge_distances,
)

# ---- fixtures ----------------------------------------------------------------


def _extract_ids(adql: str) -> list[int]:
    match = re.search(r"IN \(([^)]+)\)", adql)
    assert match is not None
    return [int(x) for x in match.group(1).split(",")]


def _fake_bj_table(source_ids: list[int]) -> Table:
    n = len(source_ids)
    return Table(
        {
            "source_id": source_ids,
            "r_med_geo": np.linspace(100.0, 500.0, n),
            "r_lo_geo": np.linspace(80.0, 400.0, n),
            "r_hi_geo": np.linspace(130.0, 600.0, n),
            "r_med_photogeo": np.linspace(110.0, 520.0, n),
            "r_lo_photogeo": np.linspace(90.0, 420.0, n),
            "r_hi_photogeo": np.linspace(140.0, 620.0, n),
            "flag": ["ok"] * n,
        }
    )


def _fake_sh2_table(source_ids: list[int]) -> Table:
    n = len(source_ids)
    return Table(
        {
            "source_id": source_ids,
            "dist16": np.linspace(100.0, 500.0, n),
            "dist50": np.linspace(120.0, 520.0, n),
            "dist84": np.linspace(140.0, 540.0, n),
            "av16": np.zeros(n),
            "av50": np.full(n, 0.1),
            "av84": np.full(n, 0.2),
            "mass16": np.full(n, 0.8),
            "mass50": np.full(n, 1.0),
            "mass84": np.full(n, 1.2),
            "age16": np.full(n, 5.0),
            "age50": np.full(n, 8.0),
            "age84": np.full(n, 12.0),
            "starhorse_outputflag": ["00000"] * n,
            "starhorse_ageflag": [1] * n,
        }
    )


# ---- fetch_bailerjones -------------------------------------------------------


def test_bailerjones_query_targets_correct_table() -> None:
    assert "gedr3dist.main" in BAILERJONES_ADQL
    assert "r_med_photogeo" in BAILERJONES_ADQL


def test_fetch_bailerjones_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[int]] = []

    def fake_async(_svc, adql, **_kw):  # noqa: ANN001
        ids = _extract_ids(adql)
        calls.append(ids)
        return _fake_bj_table(ids)

    monkeypatch.setattr(tap_mod, "run_async", fake_async)
    monkeypatch.setattr(tap_mod, "run_sync", lambda *_a, **_kw: pytest.fail("sync should not fire"))

    service = MagicMock(spec=TAPService)
    out = fetch_bailerjones(service, list(range(1, 8)), batch_size=3)
    assert calls == [[1, 2, 3], [4, 5, 6], [7]]
    assert len(out) == 7


def test_fetch_bailerjones_empty_input() -> None:
    service = MagicMock(spec=TAPService)
    assert fetch_bailerjones(service, []).empty


def test_fetch_bailerjones_checkpoint_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ckpt = tmp_path / "bj"
    ckpt.mkdir()
    _fake_bj_table([10, 11]).to_pandas().to_parquet(ckpt / "bj_batch_0000.parquet", index=False)

    def no_network(*_a, **_kw):
        raise AssertionError("should not hit TAP")

    monkeypatch.setattr(tap_mod, "run_async", no_network)
    service = MagicMock(spec=TAPService)
    out = fetch_bailerjones(service, [10, 11], batch_size=5, checkpoint_dir=ckpt)
    assert list(out["source_id"]) == [10, 11]


# ---- fetch_starhorse2 --------------------------------------------------------


def test_fetch_starhorse2_unknown_sample_raises() -> None:
    service = MagicMock(spec=TAPService)
    with pytest.raises(ValueError, match="unknown StarHorse2 sample"):
        fetch_starhorse2(service, [1], sample="not-a-real-sample")


def test_fetch_starhorse2_apogee_uses_v2_table(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_adql: list[str] = []

    def fake_async(_svc, adql, **_kw):  # noqa: ANN001
        seen_adql.append(adql)
        return _fake_sh2_table(_extract_ids(adql))

    monkeypatch.setattr(tap_mod, "run_async", fake_async)
    service = MagicMock(spec=TAPService)
    fetch_starhorse2(service, [1, 2, 3], sample="apogee_dr17", batch_size=10)

    assert seen_adql
    assert "aqueiroz2023_apogee_dr17_v2" in seen_adql[0]
    # Guard against accidental v1 use.
    assert "_v1" not in seen_adql[0]


def test_fetch_starhorse2_gaia_rvs_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_async(_svc, adql, **_kw):  # noqa: ANN001
        return _fake_sh2_table(_extract_ids(adql))

    monkeypatch.setattr(tap_mod, "run_async", fake_async)
    service = MagicMock(spec=TAPService)
    df = fetch_starhorse2(service, [1, 2], sample="gaia_rvs")
    assert len(df) == 2
    assert STARHORSE2_SAMPLE_TABLES["gaia_rvs"].endswith("_v2")


# ---- merge_distances ---------------------------------------------------------


def test_merge_distances_without_sh2() -> None:
    bj = _fake_bj_table([1, 2, 3]).to_pandas()
    merged = merge_distances(bj)
    assert "dist_primary_pc" in merged.columns
    assert "dist_sigma_sym_pc" in merged.columns
    # dist_sigma_sym_pc = (r_hi - r_lo) / 2
    expected = (bj["r_hi_photogeo"] - bj["r_lo_photogeo"]) / 2.0
    assert np.allclose(merged["dist_sigma_sym_pc"], expected)
    # No SH2 → no conflict column.
    assert "dist_conflict" not in merged.columns


def test_merge_distances_with_sh2_no_conflict() -> None:
    ids = [1, 2, 3]
    bj = _fake_bj_table(ids).to_pandas()
    sh2 = _fake_sh2_table(ids).to_pandas()
    merged = merge_distances(bj, sh2)
    # Distances are close (∼110–520 vs ∼120–520) — no factor-of-2 disagreement.
    assert not merged["dist_conflict"].any()


def test_merge_distances_flags_factor_of_2_disagreement() -> None:
    bj = _fake_bj_table([1]).to_pandas()
    # Set r_med_photogeo to 1000, SH2 dist50 to 100 → log10(10) = 1.0 > 0.3
    bj["r_med_photogeo"] = 1000.0
    bj["r_lo_photogeo"] = 900.0
    bj["r_hi_photogeo"] = 1100.0
    sh2 = _fake_sh2_table([1]).to_pandas()
    sh2["dist50"] = 100.0

    merged = merge_distances(bj, sh2)
    assert merged["dist_conflict"].iloc[0]


def test_merge_distances_conflict_threshold_is_exactly_factor_of_2() -> None:
    """|log10(2)| ≈ 0.301 > 0.3 → flagged; log10(1.9) ≈ 0.279 → not flagged."""
    bj = _fake_bj_table([1, 2]).to_pandas()
    bj.loc[0, ["r_med_photogeo", "r_lo_photogeo", "r_hi_photogeo"]] = [200.0, 180.0, 220.0]
    bj.loc[1, ["r_med_photogeo", "r_lo_photogeo", "r_hi_photogeo"]] = [190.0, 180.0, 200.0]

    sh2 = _fake_sh2_table([1, 2]).to_pandas()
    sh2.loc[0, "dist50"] = 100.0  # ratio 2.0 → |log10| ≈ 0.301
    sh2.loc[1, "dist50"] = 100.0  # ratio 1.9 → |log10| ≈ 0.279

    merged = merge_distances(bj, sh2)
    assert merged.loc[0, "dist_conflict"]
    assert not merged.loc[1, "dist_conflict"]


def test_merge_distances_missing_sh2_star_is_not_conflict() -> None:
    """A star only in BJ (SH2 dist is NaN after left-join) must not flag."""
    ids_bj = [1, 2, 3]
    ids_sh2 = [1, 2]  # star 3 missing from SH2
    bj = _fake_bj_table(ids_bj).to_pandas()
    sh2 = _fake_sh2_table(ids_sh2).to_pandas()

    merged = merge_distances(bj, sh2)
    assert len(merged) == 3
    assert not merged.loc[merged["source_id"] == 3, "dist_conflict"].iloc[0]


def test_merge_distances_requires_expected_columns() -> None:
    bj_bad = pd.DataFrame({"source_id": [1]})  # no r_med_photogeo
    with pytest.raises(KeyError, match="r_med_photogeo"):
        merge_distances(bj_bad)


def test_merge_distances_sh2_missing_dist50_raises() -> None:
    bj = _fake_bj_table([1]).to_pandas()
    sh2 = pd.DataFrame({"source_id": [1]})  # no dist50
    with pytest.raises(KeyError, match="dist50"):
        merge_distances(bj, sh2)


def test_dist_conflict_threshold_constant() -> None:
    assert DIST_CONFLICT_LOG_THRESHOLD == 0.3
