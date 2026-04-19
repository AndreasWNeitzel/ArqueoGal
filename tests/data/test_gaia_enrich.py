"""Offline tests for arqueogal.data.gaia_enrich.

``run_sync`` / ``run_async`` are monkeypatched with a fake that returns a
synthesized astropy Table shaped like the §3.6 enrichment result, so no real
TAP service is contacted.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from astropy.table import Table
from pyvo.dal.tap import TAPService

from arqueogal.data import tap as tap_mod
from arqueogal.data.gaia_enrich import (
    ENRICHMENT_ADQL,
    enrich_source_ids,
)
from arqueogal.data.tap import BATCH_PLACEHOLDER


def _fake_result(source_ids: list[int]) -> Table:
    """Build a Table shaped like a single-batch §3.6 query result."""
    return Table(
        {
            "source_id": source_ids,
            "ra": [float(i) for i in source_ids],
            "dec": [0.5 * i for i in source_ids],
            "phot_g_mean_mag": [14.0 + (i % 3) for i in source_ids],
        }
    )


def _extract_batch_ids(adql: str) -> list[int]:
    """Parse the ``IN (1,2,3)`` list substituted for ``__batch__``."""
    match = re.search(r"IN \(([^)]+)\)", adql)
    assert match is not None, f"no IN (...) clause found in\n{adql}"
    return [int(x.strip()) for x in match.group(1).split(",") if x.strip()]


@pytest.fixture
def capture_sync(monkeypatch: pytest.MonkeyPatch):
    """Replace run_sync with a capture function. run_async is poisoned."""
    calls: list[list[int]] = []

    def fake_sync(_service: object, adql: str, **_kw: object) -> Table:
        ids = _extract_batch_ids(adql)
        calls.append(ids)
        return _fake_result(ids)

    def poison(*_a: object, **_kw: object) -> Table:
        raise AssertionError("run_async should not be used in sync-mode tests")

    monkeypatch.setattr(tap_mod, "run_sync", fake_sync)
    monkeypatch.setattr(tap_mod, "run_async", poison)
    return calls


@pytest.fixture
def capture_async(monkeypatch: pytest.MonkeyPatch):
    calls: list[list[int]] = []

    def fake_async(_service: object, adql: str, **_kw: object) -> Table:
        ids = _extract_batch_ids(adql)
        calls.append(ids)
        return _fake_result(ids)

    def poison(*_a: object, **_kw: object) -> Table:
        raise AssertionError("run_sync should not be used in async-mode tests")

    monkeypatch.setattr(tap_mod, "run_async", fake_async)
    monkeypatch.setattr(tap_mod, "run_sync", poison)
    return calls


def test_enrich_empty_short_circuits_without_query(monkeypatch: pytest.MonkeyPatch) -> None:
    def poison(*_a: object, **_kw: object) -> Table:
        raise AssertionError("no query should be made for empty input")

    monkeypatch.setattr(tap_mod, "run_sync", poison)
    monkeypatch.setattr(tap_mod, "run_async", poison)

    service = MagicMock(spec=TAPService)
    out = enrich_source_ids(service, [])
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_enrich_placeholder_validation() -> None:
    service = MagicMock(spec=TAPService)
    with pytest.raises(ValueError, match="placeholder"):
        enrich_source_ids(service, [1, 2], adql="SELECT * FROM x")


def test_enrich_batch_size_validation() -> None:
    service = MagicMock(spec=TAPService)
    with pytest.raises(ValueError, match="batch_size"):
        enrich_source_ids(service, [1], batch_size=0, mode="sync")


def test_enrich_batches_and_concats(capture_sync) -> None:
    service = MagicMock(spec=TAPService)
    ids = list(range(1, 8))  # 7 ids → 3 batches of 3
    df = enrich_source_ids(service, ids, batch_size=3, mode="sync")

    assert capture_sync == [[1, 2, 3], [4, 5, 6], [7]]
    assert list(df["source_id"]) == ids
    assert len(df) == 7


def test_enrich_async_above_threshold(capture_async) -> None:
    service = MagicMock(spec=TAPService)
    df = enrich_source_ids(service, [42, 43], batch_size=10_000, mode="auto")
    assert len(capture_async) == 1
    assert list(df["source_id"]) == [42, 43]


def test_enrich_auto_mode_at_threshold_picks_sync(capture_sync) -> None:
    # batch_size exactly at threshold → sync per tap.py auto rule (> threshold → async).
    from arqueogal.data.tap import SYNC_ROW_THRESHOLD

    service = MagicMock(spec=TAPService)
    enrich_source_ids(service, [1], batch_size=SYNC_ROW_THRESHOLD, mode="auto")
    assert len(capture_sync) == 1


def test_enrich_writes_checkpoint_parquet(capture_sync, tmp_path: Path) -> None:
    service = MagicMock(spec=TAPService)
    ckpt = tmp_path / "enrich_batches"
    df = enrich_source_ids(
        service, [1, 2, 3, 4, 5], batch_size=2, mode="sync", checkpoint_dir=ckpt
    )

    files = sorted(ckpt.glob("batch_*.parquet"))
    expected = ["batch_0000.parquet", "batch_0001.parquet", "batch_0002.parquet"]
    assert [f.name for f in files] == expected
    # No stale .part files after successful writes.
    assert list(ckpt.glob("*.part")) == []
    reloaded = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    pd.testing.assert_frame_equal(reloaded, df)


def test_enrich_reuses_checkpoint_without_requery(
    capture_sync, tmp_path: Path
) -> None:
    service = MagicMock(spec=TAPService)
    ckpt = tmp_path / "enrich_batches"
    ckpt.mkdir()
    # Pre-populate batch 0 with a distinct value so we can detect reuse.
    pd.DataFrame({"source_id": [999], "ra": [0.0], "dec": [0.0]}).to_parquet(
        ckpt / "batch_0000.parquet", index=False
    )

    df = enrich_source_ids(
        service, [1, 2, 3, 4], batch_size=2, mode="sync", checkpoint_dir=ckpt
    )

    # batch 0 was reused → only batch 1 hit the fake TAP.
    assert capture_sync == [[3, 4]]
    # Row from the pre-populated checkpoint appears in the result.
    assert 999 in set(df["source_id"])
    assert set(df["source_id"]) == {999, 3, 4}


def test_enrichment_adql_has_exactly_one_placeholder() -> None:
    assert ENRICHMENT_ADQL.count(BATCH_PLACEHOLDER) == 1


def test_enrichment_adql_references_expected_tables() -> None:
    assert "gaiadr3.gaia_source" in ENRICHMENT_ADQL
    assert "gaiadr3.astrophysical_parameters" in ENRICHMENT_ADQL
    assert "LEFT JOIN" in ENRICHMENT_ADQL
