"""Offline tests for arqueogal.data.ir_photometry.

TAP is mocked by monkeypatching :func:`arqueogal.data.tap.batched_upload_fetch_df`
— the ir_photometry module imports it via ``from arqueogal.data.tap import
batched_upload_fetch_df`` so we patch the symbol on the ir_photometry module
itself, not on tap.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pyvo.dal.tap import TAPService

from arqueogal.data import ir_photometry as ir_mod
from arqueogal.data.ir_photometry import (
    ALLWISE_ADQL_UPLOAD,
    ALLWISE_SCHEMA,
    TMASS_ADQL_UPLOAD,
    TMASS_SCHEMA,
    assemble_ir_photometry,
    crossmatch_2mass,
    crossmatch_allwise,
)

# ---- helpers ----------------------------------------------------------------


def _fake_tmass_row(source_id: int, *, match: bool = True) -> dict:
    if match:
        return {
            "source_id": int(source_id),
            "j_mag": 12.1 + 0.001 * source_id,
            "e_j_mag": 0.03,
            "h_mag": 11.5 + 0.001 * source_id,
            "e_h_mag": 0.03,
            "k_mag": 11.2 + 0.001 * source_id,
            "e_k_mag": 0.03,
            "tmass_source_id": f"2MASS-J{source_id:010d}",
            "tmass_angular_distance": 50.0 + (source_id % 5),
            "tmass_xm_quality_flag": 0,
        }
    return {
        "source_id": int(source_id),
        "j_mag": np.nan,
        "e_j_mag": np.nan,
        "h_mag": np.nan,
        "e_h_mag": np.nan,
        "k_mag": np.nan,
        "e_k_mag": np.nan,
        "tmass_source_id": None,
        "tmass_angular_distance": np.nan,
        "tmass_xm_quality_flag": None,
    }


def _fake_allwise_row(source_id: int, *, match: bool = True) -> dict:
    if match:
        return {
            "source_id": int(source_id),
            "w1_mag": 10.4 + 0.001 * source_id,
            "e_w1_mag": 0.02,
            "w2_mag": 10.3 + 0.001 * source_id,
            "e_w2_mag": 0.02,
            "allwise_source_id": f"J{source_id:010d}",
            "allwise_angular_distance": 60.0 + (source_id % 7),
            "allwise_xm_quality_flag": 0,
        }
    return {
        "source_id": int(source_id),
        "w1_mag": np.nan,
        "e_w1_mag": np.nan,
        "w2_mag": np.nan,
        "e_w2_mag": np.nan,
        "allwise_source_id": None,
        "allwise_angular_distance": np.nan,
        "allwise_xm_quality_flag": None,
    }


def _make_fake_upload_fetch(
    row_builder,
    *,
    missing_ids: set[int] | None = None,
    record: list | None = None,
    chunk_records: list | None = None,
):
    """Factory for a stand-in batched_upload_fetch_df that honours chunking.

    The returned callable chunks ``source_ids`` in ``batch_size`` slices,
    records each slice into ``chunk_records`` (if provided), and returns a
    single concatenated pandas frame. Stars whose id is in ``missing_ids``
    come back as all-NaN rows (simulating the LEFT-JOIN-no-match path).
    """
    missing = missing_ids or set()

    def fake(  # noqa: PLR0913 — must mirror batched_upload_fetch_df's signature
        _service,
        source_ids,
        adql_template,
        *,
        upload_name="ids",  # noqa: ARG001
        batch_size=10_000,
        checkpoint_dir=None,  # noqa: ARG001
        checkpoint_prefix="batch",  # noqa: ARG001
        timeout_sec=None,  # noqa: ARG001
        queue=None,  # noqa: ARG001
        runid=None,  # noqa: ARG001
    ):
        ids = list(source_ids)
        if record is not None:
            record.append(
                {
                    "batch_size": batch_size,
                    "n_ids": len(ids),
                    "adql": adql_template,
                }
            )
        frames = []
        for i in range(0, len(ids), batch_size):
            chunk = ids[i : i + batch_size]
            if chunk_records is not None:
                chunk_records.append(list(chunk))
            rows = [
                row_builder(sid, match=(sid not in missing))
                for sid in chunk
            ]
            frames.append(pd.DataFrame(rows))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    return fake


# ---- ADQL shape / schema tests ---------------------------------------------


def test_tmass_adql_shape() -> None:
    # 2-join via bn.original_ext_source_id (skips the broken-on-ESA
    # gaiadr3.tmass_psc_xsc_join middleman).
    assert "tap_upload.ids" in TMASS_ADQL_UPLOAD
    assert "gaiadr3.tmass_psc_xsc_best_neighbour" in TMASS_ADQL_UPLOAD
    assert "gaiadr1.tmass_original_valid" in TMASS_ADQL_UPLOAD
    assert "bn.original_ext_source_id" in TMASS_ADQL_UPLOAD
    assert "LEFT JOIN" in TMASS_ADQL_UPLOAD
    # Ensures we don't accidentally bake an IN (__batch__) into the upload path.
    assert "__batch__" not in TMASS_ADQL_UPLOAD


def test_allwise_adql_shape() -> None:
    assert "tap_upload.ids" in ALLWISE_ADQL_UPLOAD
    assert "gaiadr3.allwise_best_neighbour" in ALLWISE_ADQL_UPLOAD
    assert "gaiadr1.allwise_original_valid" in ALLWISE_ADQL_UPLOAD
    assert "LEFT JOIN" in ALLWISE_ADQL_UPLOAD
    assert "__batch__" not in ALLWISE_ADQL_UPLOAD
    # Regression guard: ESA's TAP backend deterministically fails with
    # `PooledConnection has already been closed` on the allwise_oid
    # equi-join. The working AllWISE join key is `designation =
    # original_ext_source_id` (parallels the 2MASS path). Reproduced
    # 3/3 on 2026-04-19 after the pre-fix blind-retry run.
    assert "bn.original_ext_source_id" in ALLWISE_ADQL_UPLOAD
    assert "a.allwise_oid = bn.allwise_oid" not in ALLWISE_ADQL_UPLOAD


def test_schemas_declared() -> None:
    # Contract with the downstream feature matrix.
    assert TMASS_SCHEMA == (
        "source_id",
        "j_mag",
        "e_j_mag",
        "h_mag",
        "e_h_mag",
        "k_mag",
        "e_k_mag",
        "tmass_source_id",
        "tmass_angular_distance",
        "tmass_xm_quality_flag",
    )
    assert ALLWISE_SCHEMA == (
        "source_id",
        "w1_mag",
        "e_w1_mag",
        "w2_mag",
        "e_w2_mag",
        "allwise_source_id",
        "allwise_angular_distance",
        "allwise_xm_quality_flag",
    )


# ---- empty-input short-circuits --------------------------------------------


def test_crossmatch_2mass_empty_input(monkeypatch: pytest.MonkeyPatch) -> None:
    def poison(*_a, **_kw):
        raise AssertionError("no TAP call should be made for empty input")

    monkeypatch.setattr(ir_mod, "batched_upload_fetch_df", poison)
    svc = MagicMock(spec=TAPService)
    out = crossmatch_2mass(svc, [])
    assert isinstance(out, pl.DataFrame)
    assert out.height == 0
    assert out.columns == list(TMASS_SCHEMA)


def test_crossmatch_allwise_empty_input(monkeypatch: pytest.MonkeyPatch) -> None:
    def poison(*_a, **_kw):
        raise AssertionError("no TAP call should be made for empty input")

    monkeypatch.setattr(ir_mod, "batched_upload_fetch_df", poison)
    svc = MagicMock(spec=TAPService)
    out = crossmatch_allwise(svc, [])
    assert isinstance(out, pl.DataFrame)
    assert out.height == 0
    assert out.columns == list(ALLWISE_SCHEMA)


# ---- chunking ---------------------------------------------------------------


def test_tmass_chunks_at_requested_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks: list[list[int]] = []
    fake = _make_fake_upload_fetch(_fake_tmass_row, chunk_records=chunks)
    monkeypatch.setattr(ir_mod, "batched_upload_fetch_df", fake)

    svc = MagicMock(spec=TAPService)
    ids = list(range(1, 26))  # 25 ids → 3 chunks of 10, 10, 5
    out = crossmatch_2mass(svc, ids, batch_size=10)

    assert chunks == [list(range(1, 11)), list(range(11, 21)), list(range(21, 26))]
    assert out.height == 25
    assert set(out["source_id"].to_list()) == set(ids)


def test_tmass_default_batch_size_is_10k(monkeypatch: pytest.MonkeyPatch) -> None:
    records: list[dict] = []
    fake = _make_fake_upload_fetch(_fake_tmass_row, record=records)
    monkeypatch.setattr(ir_mod, "batched_upload_fetch_df", fake)
    svc = MagicMock(spec=TAPService)
    crossmatch_2mass(svc, [1, 2, 3])
    assert records[0]["batch_size"] == 10_000


def test_allwise_default_batch_size_is_10k(monkeypatch: pytest.MonkeyPatch) -> None:
    records: list[dict] = []
    fake = _make_fake_upload_fetch(_fake_allwise_row, record=records)
    monkeypatch.setattr(ir_mod, "batched_upload_fetch_df", fake)
    svc = MagicMock(spec=TAPService)
    crossmatch_allwise(svc, [1, 2, 3])
    assert records[0]["batch_size"] == 10_000


# ---- schema / dtypes --------------------------------------------------------


def test_tmass_output_schema_and_dtypes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _make_fake_upload_fetch(_fake_tmass_row)
    monkeypatch.setattr(ir_mod, "batched_upload_fetch_df", fake)
    svc = MagicMock(spec=TAPService)
    out = crossmatch_2mass(svc, [100, 101, 102], batch_size=10)

    assert out.columns == list(TMASS_SCHEMA)
    assert out.schema["source_id"] == pl.Int64
    assert out.schema["j_mag"] == pl.Float32
    assert out.schema["e_j_mag"] == pl.Float32
    assert out.schema["h_mag"] == pl.Float32
    assert out.schema["k_mag"] == pl.Float32
    assert out.schema["tmass_angular_distance"] == pl.Float32
    assert out.schema["tmass_source_id"] == pl.Utf8
    assert out.schema["tmass_xm_quality_flag"] == pl.Int8


def test_allwise_output_schema_and_dtypes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _make_fake_upload_fetch(_fake_allwise_row)
    monkeypatch.setattr(ir_mod, "batched_upload_fetch_df", fake)
    svc = MagicMock(spec=TAPService)
    out = crossmatch_allwise(svc, [100, 101, 102], batch_size=10)

    assert out.columns == list(ALLWISE_SCHEMA)
    assert out.schema["source_id"] == pl.Int64
    assert out.schema["w1_mag"] == pl.Float32
    assert out.schema["w2_mag"] == pl.Float32
    assert out.schema["allwise_angular_distance"] == pl.Float32
    assert out.schema["allwise_source_id"] == pl.Utf8
    assert out.schema["allwise_xm_quality_flag"] == pl.Int8


# ---- missing-counterpart handling -------------------------------------------


def test_tmass_missing_counterpart_kept_with_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate: id=102 has no 2MASS match. LEFT JOIN returns a row with NaN
    # magnitudes, which the module must preserve rather than drop.
    fake = _make_fake_upload_fetch(_fake_tmass_row, missing_ids={102})
    monkeypatch.setattr(ir_mod, "batched_upload_fetch_df", fake)
    svc = MagicMock(spec=TAPService)
    out = crossmatch_2mass(svc, [100, 101, 102, 103], batch_size=10)

    assert out.height == 4
    by_sid = {row["source_id"]: row for row in out.iter_rows(named=True)}
    # polars normalises NaN → None on the magnitude columns.
    assert by_sid[102]["j_mag"] is None
    assert by_sid[102]["h_mag"] is None
    assert by_sid[102]["k_mag"] is None
    assert by_sid[102]["tmass_source_id"] is None
    # Matched stars keep their values.
    assert by_sid[100]["j_mag"] == pytest.approx(12.1 + 0.001 * 100, abs=1e-4)


def test_tmass_drops_rows_silently_resurfaced_as_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate a TAP backend that silently drops unmatched LEFT JOIN rows
    # (some ADQL planners have been observed to do this under upload
    # joins). The finaliser's anchor-merge must resurface them as NaN.
    def fake(_service, source_ids, _adql, **_kw):
        ids = list(source_ids)
        # Return matches only for ids in {10, 11}; drop 12 entirely.
        kept = [sid for sid in ids if sid in {10, 11}]
        rows = [_fake_tmass_row(sid, match=True) for sid in kept]
        return pd.DataFrame(rows)

    monkeypatch.setattr(ir_mod, "batched_upload_fetch_df", fake)
    svc = MagicMock(spec=TAPService)
    out = crossmatch_2mass(svc, [10, 11, 12])
    assert out.height == 3
    # id=12 should have materialised as null (polars normalises NaN → None).
    by_sid = {row["source_id"]: row for row in out.iter_rows(named=True)}
    assert set(by_sid) == {10, 11, 12}
    assert by_sid[12]["j_mag"] is None


def test_allwise_missing_counterpart_kept_with_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _make_fake_upload_fetch(_fake_allwise_row, missing_ids={201})
    monkeypatch.setattr(ir_mod, "batched_upload_fetch_df", fake)
    svc = MagicMock(spec=TAPService)
    out = crossmatch_allwise(svc, [200, 201, 202], batch_size=10)

    assert out.height == 3
    by_sid = {row["source_id"]: row for row in out.iter_rows(named=True)}
    assert by_sid[201]["w1_mag"] is None
    assert by_sid[201]["w2_mag"] is None
    assert by_sid[201]["allwise_source_id"] is None


# ---- assemble_ir_photometry flag logic --------------------------------------


def test_assemble_ir_all_five_present_flag_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake(_svc, ids, adql, **_kw):  # noqa: ARG001
        ids = list(ids)
        if "tmass" in adql:
            rows = [_fake_tmass_row(sid, match=True) for sid in ids]
        else:
            rows = [_fake_allwise_row(sid, match=True) for sid in ids]
        return pd.DataFrame(rows)

    monkeypatch.setattr(ir_mod, "batched_upload_fetch_df", fake)
    svc = MagicMock(spec=TAPService)
    out = assemble_ir_photometry(svc, [1, 2, 3])
    assert "ir_missing_flag" in out.columns
    # All 3 stars have both catalogues → flag must be False everywhere.
    assert out["ir_missing_flag"].to_list() == [False, False, False]


def test_assemble_ir_any_missing_flag_true(monkeypatch: pytest.MonkeyPatch) -> None:
    # Star 2: missing 2MASS only. Star 3: missing AllWISE only. Star 4: both
    # present. All three should come out with ir_missing_flag accordingly:
    # True, True, False.
    def fake(_svc, ids, adql, **_kw):  # noqa: ARG001
        ids = list(ids)
        if "tmass" in adql:
            rows = [_fake_tmass_row(sid, match=(sid != 2)) for sid in ids]
        else:
            rows = [_fake_allwise_row(sid, match=(sid != 3)) for sid in ids]
        return pd.DataFrame(rows)

    monkeypatch.setattr(ir_mod, "batched_upload_fetch_df", fake)
    svc = MagicMock(spec=TAPService)
    out = assemble_ir_photometry(svc, [2, 3, 4])
    flags = {row["source_id"]: row["ir_missing_flag"] for row in out.iter_rows(named=True)}
    assert flags[2] is True
    assert flags[3] is True
    assert flags[4] is False


def test_assemble_ir_all_missing_flag_true(monkeypatch: pytest.MonkeyPatch) -> None:
    # Neither 2MASS nor AllWISE has a match.
    def fake(_svc, ids, adql, **_kw):  # noqa: ARG001
        ids = list(ids)
        if "tmass" in adql:
            rows = [_fake_tmass_row(sid, match=False) for sid in ids]
        else:
            rows = [_fake_allwise_row(sid, match=False) for sid in ids]
        return pd.DataFrame(rows)

    monkeypatch.setattr(ir_mod, "batched_upload_fetch_df", fake)
    svc = MagicMock(spec=TAPService)
    out = assemble_ir_photometry(svc, [7, 8])
    assert out["ir_missing_flag"].to_list() == [True, True]


def test_assemble_ir_combined_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(_svc, ids, adql, **_kw):  # noqa: ARG001
        ids = list(ids)
        if "tmass" in adql:
            return pd.DataFrame([_fake_tmass_row(sid, match=True) for sid in ids])
        return pd.DataFrame([_fake_allwise_row(sid, match=True) for sid in ids])

    monkeypatch.setattr(ir_mod, "batched_upload_fetch_df", fake)
    svc = MagicMock(spec=TAPService)
    out = assemble_ir_photometry(svc, [100, 101])
    expected_cols = set(TMASS_SCHEMA) | set(ALLWISE_SCHEMA) | {"ir_missing_flag"}
    assert set(out.columns) == expected_cols
    assert out.schema["ir_missing_flag"] == pl.Boolean
    assert out.height == 2


def test_assemble_ir_checkpoint_subdirs_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed = []

    def fake(_svc, ids, adql, **kw):  # noqa: ARG001
        observed.append(
            {
                "ckpt": kw.get("checkpoint_dir"),
                "prefix": kw.get("checkpoint_prefix"),
                "is_tmass": "tmass" in adql,
            }
        )
        ids = list(ids)
        if "tmass" in adql:
            return pd.DataFrame([_fake_tmass_row(sid, match=True) for sid in ids])
        return pd.DataFrame([_fake_allwise_row(sid, match=True) for sid in ids])

    monkeypatch.setattr(ir_mod, "batched_upload_fetch_df", fake)
    svc = MagicMock(spec=TAPService)
    ckpt = tmp_path / "ckpt"
    assemble_ir_photometry(svc, [1, 2, 3], checkpoint_dir=ckpt)

    tmass_obs = next(o for o in observed if o["is_tmass"])
    allwise_obs = next(o for o in observed if not o["is_tmass"])
    assert Path(tmass_obs["ckpt"]) == ckpt / "tmass"
    assert Path(allwise_obs["ckpt"]) == ckpt / "allwise"
