"""Offline tests for arqueogal.data.tap.

Network smoke tests (live TAP queries) are deliberately omitted here; add them
when credentials and connectivity are set up. These tests only verify local
behaviour: service construction, batching, placeholder validation.
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

import arqueogal.data.tap as tap_mod
from arqueogal.data.credentials import (
    AIP_TOKEN_ENV_VAR,
    Credentials,
    ServiceCredentials,
)
from arqueogal.data.tap import (
    AIP_TAP_URL,
    ESA_TAP_URL,
    GAVO_TAP_URL,
    VIZIER_TAP_URL,
    aip_service,
    batched_fetch_df,
    batched_in_query,
    esa_service,
    gavo_service,
    vizier_service,
)


@pytest.fixture(autouse=True)
def _clean_aip_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure GAIA_AIP_TOKEN never leaks in from the developer's shell."""
    monkeypatch.delenv(AIP_TOKEN_ENV_VAR, raising=False)


def test_gavo_service_no_auth() -> None:
    service = gavo_service()
    assert isinstance(service, TAPService)
    assert service.baseurl == GAVO_TAP_URL


def test_vizier_service_no_auth() -> None:
    service = vizier_service()
    assert service.baseurl == VIZIER_TAP_URL


def test_esa_service_no_credentials() -> None:
    service = esa_service(credentials=Credentials())
    assert service.baseurl == ESA_TAP_URL


def test_esa_service_with_credentials() -> None:
    creds = Credentials(esa=ServiceCredentials(user="u", password="p"))
    service = esa_service(credentials=creds)
    assert service.baseurl == ESA_TAP_URL


def test_aip_service_with_credentials() -> None:
    creds = Credentials(aip=ServiceCredentials(user="u", password="p"))
    service = aip_service(credentials=creds)
    assert service.baseurl == AIP_TAP_URL


def test_aip_service_missing_credentials_raises() -> None:
    with pytest.raises(RuntimeError, match="AIP credentials missing"):
        aip_service(credentials=Credentials())


def test_aip_service_falls_back_to_env_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(AIP_TOKEN_ENV_VAR, "supersecret-aip-token")
    service = aip_service(credentials=Credentials())
    assert service.baseurl == AIP_TAP_URL
    assert service._session.headers["Authorization"] == "Token supersecret-aip-token"


def test_aip_service_yaml_wins_over_env_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(AIP_TOKEN_ENV_VAR, "ignored-if-yaml-present")
    creds = Credentials(aip=ServiceCredentials(user="u", password="p"))
    service = aip_service(credentials=creds)
    # YAML path produces a pyvo AuthSession, NOT a plain requests.Session with a
    # bearer header — the token fallback should not fire.
    assert (
        not hasattr(service._session, "headers")
        or service._session.headers.get("Authorization") != "Token ignored-if-yaml-present"
    )


def test_aip_service_blank_env_token_not_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(AIP_TOKEN_ENV_VAR, "   ")
    with pytest.raises(RuntimeError, match="AIP credentials missing"):
        aip_service(credentials=Credentials())


def test_aip_service_survives_missing_yaml_when_env_token_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # Point the YAML loader at a file that does not exist; env token should
    # still drive the service.
    monkeypatch.setenv("ARQUEOGAL_CREDENTIALS_PATH", str(tmp_path / "missing.yaml"))
    monkeypatch.setenv(AIP_TOKEN_ENV_VAR, "tok")
    service = aip_service()
    assert service._session.headers["Authorization"] == "Token tok"


def test_batched_in_query_requires_placeholder() -> None:
    service = MagicMock(spec=TAPService)
    with pytest.raises(ValueError, match="placeholder"):
        list(batched_in_query(service, "SELECT * FROM foo", [1, 2, 3]))


def test_batched_in_query_rejects_multiple_placeholders() -> None:
    service = MagicMock(spec=TAPService)
    template = "SELECT * FROM foo WHERE a IN (__batch__) OR b IN (__batch__)"
    with pytest.raises(ValueError, match="placeholder"):
        list(batched_in_query(service, template, [1]))


def test_batched_in_query_rejects_zero_batch() -> None:
    service = MagicMock(spec=TAPService)
    with pytest.raises(ValueError, match="batch_size"):
        list(batched_in_query(service, "WHERE id IN (__batch__)", [1], batch_size=0))


def test_batched_in_query_substitutes_ids_and_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise batching logic by replacing run_sync with a capture function."""
    import arqueogal.data.tap as tap_mod

    calls: list[str] = []

    def fake_run_sync(_service: object, adql: str, **_kw: object):  # noqa: ANN001 — stub
        calls.append(adql)
        return f"table-for:{adql}"

    monkeypatch.setattr(tap_mod, "run_sync", fake_run_sync)

    service = MagicMock(spec=TAPService)
    template = "SELECT id FROM tbl WHERE id IN (__batch__)"
    results = list(batched_in_query(service, template, range(1, 8), batch_size=3, mode="sync"))

    assert calls == [
        "SELECT id FROM tbl WHERE id IN (1,2,3)",
        "SELECT id FROM tbl WHERE id IN (4,5,6)",
        "SELECT id FROM tbl WHERE id IN (7)",
    ]
    assert len(results) == 3


def test_batched_in_query_auto_mode_picks_async_above_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import arqueogal.data.tap as tap_mod

    sync_calls: list[str] = []
    async_calls: list[str] = []

    def fake_sync(_s, adql, **_kw):  # noqa: ANN001
        sync_calls.append(adql)

    def fake_async(_s, adql, **_kw):  # noqa: ANN001
        async_calls.append(adql)

    monkeypatch.setattr(tap_mod, "run_sync", fake_sync)
    monkeypatch.setattr(tap_mod, "run_async", fake_async)

    service = MagicMock(spec=TAPService)
    template = "WHERE id IN (__batch__)"

    # batch_size at threshold → sync
    list(batched_in_query(service, template, [1], batch_size=5_000, mode="auto"))
    assert len(sync_calls) == 1 and not async_calls

    # batch_size above threshold → async
    list(batched_in_query(service, template, [1], batch_size=10_000, mode="auto"))
    assert len(async_calls) == 1


def test_batched_in_query_empty_input_yields_nothing() -> None:
    service = MagicMock(spec=TAPService)
    assert list(batched_in_query(service, "WHERE id IN (__batch__)", [])) == []


def test_batched_in_query_non_int_raises() -> None:
    service = MagicMock(spec=TAPService)
    with pytest.raises((TypeError, ValueError)):
        list(batched_in_query(service, "WHERE id IN (__batch__)", ["not-an-int"]))


# -----------------------------------------------------------------------------
# batched_fetch_df
# -----------------------------------------------------------------------------


def _extract_ids(adql: str) -> list[int]:
    match = re.search(r"IN \(([^)]+)\)", adql)
    assert match is not None
    return [int(x) for x in match.group(1).split(",")]


def _fake_table(source_ids: list[int]) -> Table:
    return Table({"source_id": source_ids, "value": np.arange(len(source_ids))})


def test_batched_fetch_df_requires_placeholder() -> None:
    service = MagicMock(spec=TAPService)
    with pytest.raises(ValueError, match="placeholder"):
        batched_fetch_df(service, [1, 2], "SELECT * FROM foo", batch_size=10)


def test_batched_fetch_df_rejects_multiple_placeholders() -> None:
    service = MagicMock(spec=TAPService)
    template = "SELECT * FROM foo WHERE a IN (__batch__) OR b IN (__batch__)"
    with pytest.raises(ValueError, match="placeholder"):
        batched_fetch_df(service, [1], template, batch_size=10)


def test_batched_fetch_df_rejects_zero_batch() -> None:
    service = MagicMock(spec=TAPService)
    with pytest.raises(ValueError, match="batch_size"):
        batched_fetch_df(service, [1], "WHERE id IN (__batch__)", batch_size=0)


def test_batched_fetch_df_empty_input_returns_empty_frame() -> None:
    service = MagicMock(spec=TAPService)
    out = batched_fetch_df(service, [], "WHERE id IN (__batch__)", batch_size=10)
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_batched_fetch_df_batches_and_concatenates(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[int]] = []

    def fake_sync(_svc, adql, **_kw):  # noqa: ANN001
        ids = _extract_ids(adql)
        calls.append(ids)
        return _fake_table(ids)

    monkeypatch.setattr(tap_mod, "run_sync", fake_sync)
    monkeypatch.setattr(
        tap_mod, "run_async", lambda *_a, **_kw: pytest.fail("async should not fire")
    )

    service = MagicMock(spec=TAPService)
    out = batched_fetch_df(
        service,
        range(1, 8),
        "SELECT * FROM t WHERE source_id IN (__batch__)",
        batch_size=3,
        mode="sync",
    )

    assert calls == [[1, 2, 3], [4, 5, 6], [7]]
    assert list(out["source_id"]) == [1, 2, 3, 4, 5, 6, 7]


def test_batched_fetch_df_auto_mode_picks_sync_at_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_calls: list[str] = []
    async_calls: list[str] = []

    def fake_sync(_s, adql, **_kw):  # noqa: ANN001
        sync_calls.append(adql)
        return _fake_table(_extract_ids(adql))

    def fake_async(_s, adql, **_kw):  # noqa: ANN001
        async_calls.append(adql)
        return _fake_table(_extract_ids(adql))

    monkeypatch.setattr(tap_mod, "run_sync", fake_sync)
    monkeypatch.setattr(tap_mod, "run_async", fake_async)

    service = MagicMock(spec=TAPService)
    template = "WHERE source_id IN (__batch__)"

    batched_fetch_df(service, [1], template, batch_size=5_000, mode="auto")
    assert len(sync_calls) == 1 and not async_calls

    batched_fetch_df(service, [1], template, batch_size=10_000, mode="auto")
    assert len(async_calls) == 1


def test_batched_fetch_df_async_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    async_calls: list[str] = []

    def fake_async(_s, adql, **_kw):  # noqa: ANN001
        async_calls.append(adql)
        return _fake_table(_extract_ids(adql))

    monkeypatch.setattr(tap_mod, "run_async", fake_async)
    monkeypatch.setattr(tap_mod, "run_sync", lambda *_a, **_kw: pytest.fail("sync should not fire"))

    service = MagicMock(spec=TAPService)
    batched_fetch_df(service, [1, 2], "WHERE id IN (__batch__)", batch_size=5)
    assert len(async_calls) == 1


def test_batched_fetch_df_writes_checkpoints_atomically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        tap_mod, "run_sync", lambda _s, adql, **_kw: _fake_table(_extract_ids(adql))
    )

    service = MagicMock(spec=TAPService)
    out = batched_fetch_df(
        service,
        [1, 2, 3, 4, 5],
        "WHERE id IN (__batch__)",
        batch_size=2,
        mode="sync",
        checkpoint_dir=tmp_path,
        checkpoint_prefix="xx",
    )

    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["xx_0000.parquet", "xx_0001.parquet", "xx_0002.parquet"]
    assert not list(tmp_path.glob("*.part"))  # atomic — no leftover temp files
    assert len(out) == 5


def test_batched_fetch_df_reuses_checkpoint_on_rerun(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pd.DataFrame({"source_id": [10, 11], "value": [0, 1]}).to_parquet(
        tmp_path / "xx_0000.parquet", index=False
    )

    def no_network(*_a, **_kw):
        raise AssertionError("should not hit TAP when checkpoint exists")

    monkeypatch.setattr(tap_mod, "run_async", no_network)
    monkeypatch.setattr(tap_mod, "run_sync", no_network)

    service = MagicMock(spec=TAPService)
    out = batched_fetch_df(
        service,
        [10, 11],
        "WHERE id IN (__batch__)",
        batch_size=5,
        checkpoint_dir=tmp_path,
        checkpoint_prefix="xx",
    )
    assert list(out["source_id"]) == [10, 11]


def test_batched_fetch_df_checkpoint_prefix_distinguishes_kinds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two different query kinds must coexist in one checkpoint dir."""
    monkeypatch.setattr(
        tap_mod, "run_sync", lambda _s, adql, **_kw: _fake_table(_extract_ids(adql))
    )

    service = MagicMock(spec=TAPService)
    batched_fetch_df(
        service,
        [1],
        "WHERE id IN (__batch__)",
        batch_size=5,
        mode="sync",
        checkpoint_dir=tmp_path,
        checkpoint_prefix="alpha",
    )
    batched_fetch_df(
        service,
        [2],
        "WHERE id IN (__batch__)",
        batch_size=5,
        mode="sync",
        checkpoint_dir=tmp_path,
        checkpoint_prefix="beta",
    )

    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["alpha_0000.parquet", "beta_0000.parquet"]


def test_batched_fetch_df_rejects_non_castable_ids() -> None:
    service = MagicMock(spec=TAPService)
    with pytest.raises((TypeError, ValueError)):
        batched_fetch_df(service, ["not-an-int"], "WHERE id IN (__batch__)", batch_size=10)
