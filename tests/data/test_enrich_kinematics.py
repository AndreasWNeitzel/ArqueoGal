"""Offline tests for arqueogal.data.enrich_kinematics — §11 Level 5.

``compute_actions`` is monkeypatched on the orchestrator module namespace
so galpy is never imported during the test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arqueogal.data import enrich_kinematics as mod
from arqueogal.data.kinematics import OUTPUT_COLS, KinematicsConfig


def _input_df(source_ids: list[int]) -> pd.DataFrame:
    n = len(source_ids)
    return pd.DataFrame(
        {
            "source_id": np.asarray(source_ids, dtype=np.int64),
            "ra": np.linspace(10.0, 20.0, n),
            "dec": np.linspace(-5.0, 5.0, n),
            "r_med_photogeo": np.linspace(500.0, 2000.0, n),
            "pmra": np.linspace(-2.0, 2.0, n),
            "pmdec": np.linspace(-1.0, 1.0, n),
            "radial_velocity": np.linspace(-30.0, 30.0, n),
            "phot_g_mean_mag": np.full(n, 13.0),  # extra column, must be preserved
        }
    )


def _fake_actions(source_ids: np.ndarray) -> pd.DataFrame:
    n = len(source_ids)
    out = {"source_id": np.asarray(source_ids, dtype=np.int64)}
    for col in OUTPUT_COLS[1:]:
        out[col] = np.arange(n, dtype=np.float64) + 0.5
    return pd.DataFrame(out)


@pytest.fixture
def patched_actions(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_compute_actions(df, *, config=None):  # noqa: ANN001
        captured["df_cols"] = list(df.columns)
        captured["df_len"] = len(df)
        captured["config"] = config
        return _fake_actions(df["source_id"].to_numpy())

    monkeypatch.setattr(mod, "compute_actions", fake_compute_actions)
    return captured


def test_happy_path_writes_parquet_and_sidecar(
    tmp_path: Path, patched_actions: dict[str, object]
) -> None:
    out_path = tmp_path / "stream_kin.parquet"
    df = _input_df([1, 2, 3])
    returned = mod.enrich_kinematics_stream(df, output_path=out_path)
    assert returned == out_path
    assert out_path.is_file()
    sidecar = out_path.with_suffix("").with_suffix(".provenance.json")
    assert sidecar.is_file()


def test_action_columns_added(tmp_path: Path, patched_actions: dict[str, object]) -> None:
    out_path = tmp_path / "stream_kin.parquet"
    df = _input_df([1, 2, 3])
    mod.enrich_kinematics_stream(df, output_path=out_path)
    result = pd.read_parquet(out_path)
    for col in OUTPUT_COLS[1:]:
        assert col in result.columns
    # Original columns preserved.
    assert "phot_g_mean_mag" in result.columns
    assert len(result) == 3


def test_missing_required_columns_raises(
    tmp_path: Path, patched_actions: dict[str, object]
) -> None:
    df = _input_df([1, 2]).drop(columns=["pmra"])
    with pytest.raises(KeyError, match="pmra"):
        mod.enrich_kinematics_stream(df, output_path=tmp_path / "x.parquet")


def test_left_join_preserves_unsolved_rows(
    tmp_path: Path, patched_actions: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """compute_actions can drop NaN rows; output must still have all input rows."""

    def drop_first(df, *, config=None):  # noqa: ANN001
        return _fake_actions(df["source_id"].to_numpy()[1:])

    monkeypatch.setattr(mod, "compute_actions", drop_first)
    out_path = tmp_path / "stream_kin.parquet"
    df = _input_df([10, 20, 30])
    mod.enrich_kinematics_stream(df, output_path=out_path)
    result = pd.read_parquet(out_path).sort_values("source_id").reset_index(drop=True)
    assert len(result) == 3
    # First row unsolved → J_R should be NaN; others finite.
    assert np.isnan(result.loc[0, "J_R_kpc_kms"])
    assert np.isfinite(result.loc[1, "J_R_kpc_kms"])
    assert np.isfinite(result.loc[2, "J_R_kpc_kms"])


def test_config_forwarded_to_compute_actions(
    tmp_path: Path, patched_actions: dict[str, object]
) -> None:
    cfg = KinematicsConfig(potential="mwpotential2014", staeckel_delta=0.40)
    df = _input_df([1, 2])
    mod.enrich_kinematics_stream(df, output_path=tmp_path / "k.parquet", config=cfg)
    assert patched_actions["config"] is cfg


def test_provenance_records_config_and_corrections(
    tmp_path: Path, patched_actions: dict[str, object]
) -> None:
    out_path = tmp_path / "stream_kin.parquet"
    cfg = KinematicsConfig(potential="mcmillan17", staeckel_delta=0.45)
    mod.enrich_kinematics_stream(_input_df([1, 2, 3]), output_path=out_path, config=cfg)
    meta = json.loads(out_path.with_suffix("").with_suffix(".provenance.json").read_text())
    assert any("mcmillan17" in c for c in meta["corrections"])
    assert any("Staeckel delta=0.45" in c for c in meta["corrections"])
    stored_cfg = meta["extra"]["kinematics_config"]
    assert stored_cfg["potential"] == "mcmillan17"
    assert stored_cfg["staeckel_delta"] == 0.45
    assert stored_cfg["ro_kpc"] == cfg.ro_kpc


def test_provenance_row_counts(tmp_path: Path, patched_actions: dict[str, object]) -> None:
    out_path = tmp_path / "stream_kin.parquet"
    mod.enrich_kinematics_stream(_input_df([1, 2, 3, 4]), output_path=out_path)
    meta = json.loads(out_path.with_suffix("").with_suffix(".provenance.json").read_text())
    assert meta["row_count_before"] == 4
    assert meta["row_count_after"] == 4
    assert meta["extra"]["rows_solved"] == 4
    assert meta["extra"]["rows_unsolved"] == 0


def test_provenance_has_no_tap_or_http_sources(
    tmp_path: Path, patched_actions: dict[str, object]
) -> None:
    out_path = tmp_path / "stream_kin.parquet"
    mod.enrich_kinematics_stream(_input_df([1, 2]), output_path=out_path)
    meta = json.loads(out_path.with_suffix("").with_suffix(".provenance.json").read_text())
    assert meta["sources"] == []


def test_atomic_write_no_part_file_left(tmp_path: Path, patched_actions: dict[str, object]) -> None:
    out_path = tmp_path / "stream_kin.parquet"
    mod.enrich_kinematics_stream(_input_df([1, 2]), output_path=out_path)
    leftover = list(tmp_path.glob("*.part"))
    assert not leftover


def test_output_parent_is_created(tmp_path: Path, patched_actions: dict[str, object]) -> None:
    out_path = tmp_path / "nested" / "deep" / "kin.parquet"
    mod.enrich_kinematics_stream(_input_df([1, 2]), output_path=out_path)
    assert out_path.is_file()


def test_action_columns_in_provenance_extra(
    tmp_path: Path, patched_actions: dict[str, object]
) -> None:
    out_path = tmp_path / "stream_kin.parquet"
    mod.enrich_kinematics_stream(_input_df([1, 2]), output_path=out_path)
    meta = json.loads(out_path.with_suffix("").with_suffix(".provenance.json").read_text())
    assert "J_R_kpc_kms" in meta["extra"]["action_columns"]
    assert "source_id" not in meta["extra"]["action_columns"]
