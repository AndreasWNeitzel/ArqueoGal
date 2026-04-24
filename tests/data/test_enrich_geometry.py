"""Offline tests for arqueogal.data.enrich_geometry — §11 Level 4.

BJ21 / SH2 TAP fetches are monkeypatched; dust-map queries are synthetic
callables so we don't need the real ``dustmaps`` on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pyvo.dal.tap import TAPService

from arqueogal.data import enrich_geometry as mod


def _input_df(source_ids: list[int], *, with_ag: bool = True) -> pd.DataFrame:
    n = len(source_ids)
    cols: dict[str, np.ndarray] = {
        "source_id": np.asarray(source_ids, dtype=np.int64),
        "ra": np.linspace(10.0, 20.0, n),
        "dec": np.linspace(-5.0, 5.0, n),
    }
    if with_ag:
        cols["ag_gspphot"] = np.linspace(0.1, 0.5, n, dtype=np.float32)
    return pd.DataFrame(cols)


def _bj_df(source_ids: np.ndarray, *, dist_pc: np.ndarray | None = None) -> pd.DataFrame:
    n = len(source_ids)
    if dist_pc is None:
        # Spread the distances across the three §8.2 bins: near (<1.25 kpc),
        # mid (1.25–3 kpc), far (≥3 kpc).
        dist_pc = np.array([500.0, 2000.0, 4000.0, 900.0][:n], dtype=float)
        if n > 4:
            dist_pc = np.concatenate([dist_pc, np.full(n - 4, 1500.0)])
    return pd.DataFrame(
        {
            "source_id": np.asarray(source_ids, dtype=np.int64),
            "r_med_photogeo": dist_pc,
            "r_lo_photogeo": dist_pc * 0.9,
            "r_hi_photogeo": dist_pc * 1.1,
            "r_med_geo": dist_pc * 1.01,
            "r_lo_geo": dist_pc * 0.89,
            "r_hi_geo": dist_pc * 1.11,
            "flag": np.zeros(n, dtype=np.int32),
        }
    )


def _sh2_df(source_ids: np.ndarray) -> pd.DataFrame:
    n = len(source_ids)
    return pd.DataFrame(
        {
            "source_id": np.asarray(source_ids, dtype=np.int64),
            "dist16": np.full(n, 400.0),
            "dist50": np.full(n, 500.0),
            "dist84": np.full(n, 600.0),
            "av16": np.full(n, 0.05),
            "av50": np.full(n, 0.1),
            "av84": np.full(n, 0.2),
            "mass16": np.full(n, 0.9),
            "mass50": np.full(n, 1.0),
            "mass84": np.full(n, 1.1),
            "age16": np.full(n, 8.0),
            "age50": np.full(n, 9.0),
            "age84": np.full(n, 10.0),
            "starhorse_outputflag": np.zeros(n, dtype=np.int32),
            "starhorse_ageflag": np.zeros(n, dtype=np.int32),
        }
    )


def _near_query(coords) -> np.ndarray:  # noqa: ANN001 — SkyCoord duck
    return np.full(len(coords), 0.15, dtype=float)


def _mid_query(coords) -> np.ndarray:  # noqa: ANN001
    return np.full(len(coords), 0.30, dtype=float)


def _far_query(coords) -> np.ndarray:  # noqa: ANN001
    # SFD returns E(B-V); compose_av multiplies by SFD_TO_AV_COEFF.
    return np.full(len(coords), 0.10, dtype=float)


@pytest.fixture
def patched_tap(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_bj(service, source_ids, **kwargs):  # noqa: ANN001
        captured["bj_service"] = service
        ids = np.asarray(list(source_ids), dtype=np.int64)
        captured["bj_ids"] = ids.tolist()
        captured["bj_kwargs"] = kwargs
        return _bj_df(ids)

    def fake_sh2(service, source_ids, **kwargs):  # noqa: ANN001
        captured["sh2_service"] = service
        ids = np.asarray(list(source_ids), dtype=np.int64)
        captured["sh2_ids"] = ids.tolist()
        captured["sh2_kwargs"] = kwargs
        return _sh2_df(ids)

    monkeypatch.setattr(mod, "fetch_bailerjones", fake_bj)
    monkeypatch.setattr(mod, "fetch_starhorse2", fake_sh2)
    return captured


def test_happy_path_adds_distance_and_av_columns(
    tmp_path: Path, patched_tap: dict[str, object]
) -> None:
    gavo = MagicMock(spec=TAPService)
    df = _input_df([1, 2, 3, 4])
    out_path = tmp_path / "stream_geom.parquet"
    returned = mod.enrich_geometry(
        df,
        output_path=out_path,
        gavo=gavo,
        dust_queries=(_near_query, _mid_query, _far_query),
    )
    assert returned == out_path
    result = pd.read_parquet(out_path)
    for col in (
        "dist_primary_pc",
        "dist_sigma_sym_pc",
        "av_los",
        "av_los_source",
        "av_nbhd_median",
        "av_nbhd_std",
        "av_nbhd_n_neighbors",
    ):
        assert col in result.columns
    assert len(result) == 4


def test_av_source_codes_route_by_distance(tmp_path: Path, patched_tap: dict[str, object]) -> None:
    """Distances 500 / 2000 / 4000 / 900 → sources 0 / 1 / 2 / 0."""
    gavo = MagicMock(spec=TAPService)
    df = _input_df([1, 2, 3, 4])
    out_path = tmp_path / "stream_geom.parquet"
    mod.enrich_geometry(
        df,
        output_path=out_path,
        gavo=gavo,
        dust_queries=(_near_query, _mid_query, _far_query),
    )
    result = pd.read_parquet(out_path).sort_values("source_id").reset_index(drop=True)
    assert list(result["av_los_source"]) == [0, 1, 2, 0]
    # Near/mid return A_V directly; far (SFD) multiplies by 2.742.
    near_vals = result.loc[result["av_los_source"] == 0, "av_los"].to_numpy()
    mid_vals = result.loc[result["av_los_source"] == 1, "av_los"].to_numpy()
    far_vals = result.loc[result["av_los_source"] == 2, "av_los"].to_numpy()
    assert np.allclose(near_vals, 0.15, atol=1e-5)
    assert np.allclose(mid_vals, 0.30, atol=1e-5)
    assert np.allclose(far_vals, 0.10 * 2.742, atol=1e-4)


def test_missing_required_columns_raises(tmp_path: Path, patched_tap: dict[str, object]) -> None:
    gavo = MagicMock(spec=TAPService)
    df = pd.DataFrame({"source_id": [1, 2], "ra": [0.0, 1.0]})  # dec missing
    with pytest.raises(KeyError, match="dec"):
        mod.enrich_geometry(
            df,
            output_path=tmp_path / "x.parquet",
            gavo=gavo,
            dust_queries=(_near_query, _mid_query, _far_query),
        )


def test_starhorse2_requires_aip(tmp_path: Path, patched_tap: dict[str, object]) -> None:
    gavo = MagicMock(spec=TAPService)
    df = _input_df([1, 2])
    with pytest.raises(ValueError, match="aip"):
        mod.enrich_geometry(
            df,
            output_path=tmp_path / "x.parquet",
            gavo=gavo,
            include_starhorse2=True,
            dust_queries=(_near_query, _mid_query, _far_query),
        )


def test_starhorse2_flow_adds_sh2_columns_and_tap_source(
    tmp_path: Path, patched_tap: dict[str, object]
) -> None:
    gavo = MagicMock(spec=TAPService)
    aip = MagicMock(spec=TAPService)
    df = _input_df([1, 2, 3, 4])
    out_path = tmp_path / "stream_geom.parquet"
    mod.enrich_geometry(
        df,
        output_path=out_path,
        gavo=gavo,
        aip=aip,
        include_starhorse2=True,
        dust_queries=(_near_query, _mid_query, _far_query),
    )
    result = pd.read_parquet(out_path)
    assert "dist_sh2_pc" in result.columns
    assert "dist_conflict" in result.columns

    meta = json.loads(out_path.with_suffix("").with_suffix(".provenance.json").read_text())
    tap_names = [s["name"] for s in meta["sources"] if s["kind"] == "tap"]
    assert len(tap_names) == 2
    joined = " | ".join(tap_names).lower()
    assert "bailer-jones" in joined
    assert "starhorse2" in joined


def test_provenance_single_tap_source_without_sh2(
    tmp_path: Path, patched_tap: dict[str, object]
) -> None:
    gavo = MagicMock(spec=TAPService)
    df = _input_df([1, 2])
    out_path = tmp_path / "stream_geom.parquet"
    mod.enrich_geometry(
        df,
        output_path=out_path,
        gavo=gavo,
        dust_queries=(_near_query, _mid_query, _far_query),
    )
    meta = json.loads(out_path.with_suffix("").with_suffix(".provenance.json").read_text())
    tap_kinds = [s for s in meta["sources"] if s["kind"] == "tap"]
    assert len(tap_kinds) == 1
    assert "Bailer-Jones" in tap_kinds[0]["name"]


def test_skip_nbhd_when_ag_column_absent(tmp_path: Path, patched_tap: dict[str, object]) -> None:
    gavo = MagicMock(spec=TAPService)
    df = _input_df([1, 2, 3], with_ag=False)
    out_path = tmp_path / "stream_geom.parquet"
    mod.enrich_geometry(
        df,
        output_path=out_path,
        gavo=gavo,
        dust_queries=(_near_query, _mid_query, _far_query),
    )
    result = pd.read_parquet(out_path)
    assert "av_los" in result.columns  # §8.2 still runs
    assert "av_nbhd_median" not in result.columns
    assert "av_nbhd_std" not in result.columns
    assert "av_nbhd_n_neighbors" not in result.columns


def test_custom_ag_column_name_honored(tmp_path: Path, patched_tap: dict[str, object]) -> None:
    gavo = MagicMock(spec=TAPService)
    df = _input_df([1, 2, 3], with_ag=False)
    df["my_av"] = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    out_path = tmp_path / "stream_geom.parquet"
    mod.enrich_geometry(
        df,
        output_path=out_path,
        gavo=gavo,
        dust_queries=(_near_query, _mid_query, _far_query),
        ag_gspphot_col="my_av",
    )
    result = pd.read_parquet(out_path)
    assert "av_nbhd_median" in result.columns


def test_atomic_write_no_part_file_left(tmp_path: Path, patched_tap: dict[str, object]) -> None:
    gavo = MagicMock(spec=TAPService)
    df = _input_df([1, 2])
    out_path = tmp_path / "stream_geom.parquet"
    mod.enrich_geometry(
        df,
        output_path=out_path,
        gavo=gavo,
        dust_queries=(_near_query, _mid_query, _far_query),
    )
    leftover = list(tmp_path.glob("*.part"))
    assert not leftover


def test_provenance_row_counts_and_extras(tmp_path: Path, patched_tap: dict[str, object]) -> None:
    gavo = MagicMock(spec=TAPService)
    aip = MagicMock(spec=TAPService)
    df = _input_df([1, 2, 3, 4])
    out_path = tmp_path / "stream_geom.parquet"
    mod.enrich_geometry(
        df,
        output_path=out_path,
        gavo=gavo,
        aip=aip,
        include_starhorse2=True,
        dust_queries=(_near_query, _mid_query, _far_query),
    )
    meta = json.loads(out_path.with_suffix("").with_suffix(".provenance.json").read_text())
    assert meta["row_count_before"] == 4
    assert meta["row_count_after"] == 4
    extra = meta["extra"]
    assert extra["bj_rows"] == 4
    assert extra["sh2_rows"] == 4
    assert extra["sh2_sample"] == "apogee_dr17"


def test_batch_sizes_propagate_to_fetchers(tmp_path: Path, patched_tap: dict[str, object]) -> None:
    gavo = MagicMock(spec=TAPService)
    aip = MagicMock(spec=TAPService)
    df = _input_df([1, 2, 3])
    out_path = tmp_path / "stream_geom.parquet"
    mod.enrich_geometry(
        df,
        output_path=out_path,
        gavo=gavo,
        aip=aip,
        include_starhorse2=True,
        bj_batch_size=2500,
        sh2_batch_size=4000,
        dust_queries=(_near_query, _mid_query, _far_query),
    )
    assert patched_tap["bj_kwargs"]["batch_size"] == 2500
    assert patched_tap["sh2_kwargs"]["batch_size"] == 4000


def test_checkpoint_dirs_propagate(tmp_path: Path, patched_tap: dict[str, object]) -> None:
    gavo = MagicMock(spec=TAPService)
    aip = MagicMock(spec=TAPService)
    df = _input_df([1, 2])
    out_path = tmp_path / "stream_geom.parquet"
    bj_cp = tmp_path / "cp_bj"
    sh2_cp = tmp_path / "cp_sh2"
    mod.enrich_geometry(
        df,
        output_path=out_path,
        gavo=gavo,
        aip=aip,
        include_starhorse2=True,
        bj_checkpoint_dir=bj_cp,
        sh2_checkpoint_dir=sh2_cp,
        dust_queries=(_near_query, _mid_query, _far_query),
    )
    assert patched_tap["bj_kwargs"]["checkpoint_dir"] == bj_cp
    assert patched_tap["sh2_kwargs"]["checkpoint_dir"] == sh2_cp


def test_output_parent_is_created(tmp_path: Path, patched_tap: dict[str, object]) -> None:
    gavo = MagicMock(spec=TAPService)
    df = _input_df([1, 2])
    out_path = tmp_path / "nested" / "deep" / "geom.parquet"
    mod.enrich_geometry(
        df,
        output_path=out_path,
        gavo=gavo,
        dust_queries=(_near_query, _mid_query, _far_query),
    )
    assert out_path.is_file()
