"""Offline tests for arqueogal.data.tess_hon2021 — §4.1 VizieR fetch.

TAP is mocked via :mod:`arqueogal.data.tap`. No network I/O.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from astropy.table import Table
from pyvo.dal.tap import TAPService

from arqueogal.data import tap as tap_mod
from arqueogal.data.tess_hon2021 import (
    HON2021_COLUMNS,
    HON2021_DEFAULT_PROB_THRESHOLD,
    HON2021_VIZIER_TABLE,
    build_hon2021_adql,
    fetch_hon2021,
)


def _fake_hon2021_table(n: int) -> Table:
    return Table(
        {
            "TIC": np.arange(1, n + 1, dtype=np.int64),
            "RAJ2000": np.linspace(0.0, 359.0, n),
            "DEJ2000": np.linspace(-30.0, 60.0, n),
            "numax": np.linspace(30.0, 200.0, n),
            "e_numax": np.full(n, 1.5),
            "Teff": np.linspace(4500.0, 5100.0, n),
            "R": np.linspace(5.0, 15.0, n),
            "Prob": np.linspace(0.96, 0.99, n),
        }
    )


# ---- constants ---------------------------------------------------------------


def test_default_threshold_matches_section_4_1() -> None:
    assert HON2021_DEFAULT_PROB_THRESHOLD == 0.95


def test_table_is_vizier_j_apj_919_131() -> None:
    assert "J/ApJ/919/131" in HON2021_VIZIER_TABLE


def test_column_set_matches_section_4_1() -> None:
    for col in ("TIC", "RAJ2000", "DEJ2000", "numax", "e_numax", "Teff", "R", "Prob"):
        assert col in HON2021_COLUMNS


# ---- build_hon2021_adql ------------------------------------------------------


def test_adql_contains_prob_cut() -> None:
    adql = build_hon2021_adql(prob_threshold=0.97)
    assert "Prob > 0.97" in adql
    assert HON2021_VIZIER_TABLE in adql
    for col in HON2021_COLUMNS:
        assert col in adql


def test_adql_default_threshold_is_0_95() -> None:
    adql = build_hon2021_adql()
    assert "Prob > 0.95" in adql


def test_adql_rejects_out_of_range_threshold() -> None:
    with pytest.raises(ValueError, match="prob_threshold"):
        build_hon2021_adql(prob_threshold=-0.1)
    with pytest.raises(ValueError, match="prob_threshold"):
        build_hon2021_adql(prob_threshold=1.5)


# ---- fetch_hon2021 -----------------------------------------------------------


def test_fetch_hon2021_default_mode_is_async(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"async": 0, "sync": 0}

    def fake_async(_svc, adql, **_kw):  # noqa: ANN001
        assert "Prob > 0.95" in adql
        called["async"] += 1
        return _fake_hon2021_table(5)

    monkeypatch.setattr(tap_mod, "run_async", fake_async)
    monkeypatch.setattr(
        tap_mod, "run_sync", lambda *_a, **_kw: called.update({"sync": called["sync"] + 1})
        or _fake_hon2021_table(0),
    )
    service = MagicMock(spec=TAPService)

    df = fetch_hon2021(service)
    assert called == {"async": 1, "sync": 0}
    assert len(df) == 5
    for col in HON2021_COLUMNS:
        assert col in df.columns


def test_fetch_hon2021_sync_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_sync(_svc, adql):  # noqa: ANN001
        assert "Prob > 0.95" in adql
        return _fake_hon2021_table(3)

    monkeypatch.setattr(tap_mod, "run_sync", fake_sync)
    monkeypatch.setattr(
        tap_mod, "run_async", lambda *_a, **_kw: pytest.fail("async should not fire")
    )
    service = MagicMock(spec=TAPService)

    df = fetch_hon2021(service, mode="sync")
    assert len(df) == 3


def test_fetch_hon2021_custom_threshold_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_async(_svc, adql, **_kw):  # noqa: ANN001
        seen.append(adql)
        return _fake_hon2021_table(1)

    monkeypatch.setattr(tap_mod, "run_async", fake_async)
    service = MagicMock(spec=TAPService)

    fetch_hon2021(service, prob_threshold=0.99)
    assert any("Prob > 0.99" in q for q in seen)


def test_fetch_hon2021_rejects_unknown_mode() -> None:
    service = MagicMock(spec=TAPService)
    with pytest.raises(ValueError, match="mode must be"):
        fetch_hon2021(service, mode="turbo")


def test_fetch_hon2021_returns_empty_frame_when_tap_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tap_mod, "run_async", lambda *_a, **_kw: _fake_hon2021_table(0)
    )
    service = MagicMock(spec=TAPService)
    df = fetch_hon2021(service)
    assert isinstance(df, pd.DataFrame)
    assert df.empty
