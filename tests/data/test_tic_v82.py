"""Offline tests for arqueogal.data.tic_v82 — §4.3 Step 1 VizieR lookup.

TAP traffic is monkeypatched at the :mod:`arqueogal.data.tap` module level.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from pyvo.dal.tap import TAPService

from arqueogal.data import tap as tap_mod
from arqueogal.data.tic_v82 import (
    TIC_V82_ADQL,
    TIC_V82_COLUMNS,
    TIC_V82_VIZIER_TABLE,
    fetch_tic_v82,
)


def _fake_tic_row_table(tic_ids: list[int]) -> pd.DataFrame:
    n = len(tic_ids)
    return pd.DataFrame(
        {
            "TIC": np.asarray(tic_ids, dtype=np.int64),
            "GAIA": np.asarray([t * 10 for t in tic_ids], dtype=np.int64),
            "RAJ2000": np.linspace(0.0, 359.0, n),
            "DEJ2000": np.linspace(-30.0, 60.0, n),
            "Tmag": np.linspace(8.0, 14.0, n),
            "plx": np.linspace(0.5, 4.0, n),
        }
    )


# ---- constants / ADQL --------------------------------------------------------


def test_vizier_table_is_iv_39_tic82() -> None:
    assert "IV/39/tic82" in TIC_V82_VIZIER_TABLE


def test_columns_match_section_4_3() -> None:
    for col in ("TIC", "GAIA", "RAJ2000", "DEJ2000", "Tmag", "plx"):
        assert col in TIC_V82_COLUMNS


def test_adql_uses_tic_as_in_filter() -> None:
    assert "TIC IN (__batch__)" in TIC_V82_ADQL
    for col in TIC_V82_COLUMNS:
        assert col in TIC_V82_ADQL


# ---- fetch_tic_v82 -----------------------------------------------------------


def test_fetch_batches_and_concatenates(monkeypatch) -> None:
    """Two batches of 3 TICs each → 6 rows, one run_async call per batch."""
    seen_batches: list[list[int]] = []

    def fake_run_async(_svc, adql, **_kw):  # noqa: ANN001
        # Pull the in-list — batched_fetch_df substitutes __batch__ with
        # comma-joined ids before handing the ADQL to run_async.
        after = adql.split("TIC IN (")[1].split(")")[0]
        chunk = [int(x) for x in after.split(",")]
        seen_batches.append(chunk)
        from astropy.table import Table

        return Table.from_pandas(_fake_tic_row_table(chunk))

    monkeypatch.setattr(tap_mod, "run_async", fake_run_async)
    service = MagicMock(spec=TAPService)

    tic_ids = [11, 22, 33, 44, 55, 66]
    out = fetch_tic_v82(service, tic_ids, batch_size=3)

    assert len(out) == 6
    assert seen_batches == [[11, 22, 33], [44, 55, 66]]
    for col in TIC_V82_COLUMNS:
        assert col in out.columns


def test_fetch_empty_input_returns_empty_frame(monkeypatch) -> None:
    def fake_run_async(*_a, **_kw):
        raise AssertionError("should not be called for empty input")

    monkeypatch.setattr(tap_mod, "run_async", fake_run_async)
    service = MagicMock(spec=TAPService)
    out = fetch_tic_v82(service, [])
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_fetch_writes_checkpoints(monkeypatch, tmp_path) -> None:
    """Checkpoint prefix is ``tic_v82`` per module design."""

    def fake_run_async(_svc, adql, **_kw):  # noqa: ANN001
        after = adql.split("TIC IN (")[1].split(")")[0]
        chunk = [int(x) for x in after.split(",")]
        from astropy.table import Table

        return Table.from_pandas(_fake_tic_row_table(chunk))

    monkeypatch.setattr(tap_mod, "run_async", fake_run_async)
    service = MagicMock(spec=TAPService)

    fetch_tic_v82(service, [1, 2, 3, 4], batch_size=2, checkpoint_dir=tmp_path)
    files = sorted(p.name for p in tmp_path.glob("tic_v82_*.parquet"))
    assert files == ["tic_v82_0000.parquet", "tic_v82_0001.parquet"]
