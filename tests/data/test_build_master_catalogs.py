"""Offline tests for arqueogal.data.build_master_catalogs — §11 Level 6."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arqueogal.data.build_master_catalogs import (
    build_pipeline1_inference,
    build_pipeline1_training,
)
from arqueogal.data.master_schema import (
    PIPELINE1_INFERENCE_SCHEMA,
    PIPELINE1_TRAINING_SCHEMA,
    XP_ARRAY_COLS,
    XP_N_COEFFS,
    XP_SCALAR_COLS,
    MasterSchema,
    SchemaError,
)


def _fabricate_stream(schema: MasterSchema, source_ids: np.ndarray) -> pd.DataFrame:
    """Build a stream frame that has every non-XP required column for ``schema``.

    XP columns are intentionally omitted — those come from the xp side of
    the join.
    """
    xp_cols = set(XP_ARRAY_COLS) | set(XP_SCALAR_COLS)
    n = len(source_ids)
    cols: dict[str, np.ndarray | list] = {}
    for col in schema.required:
        if col in xp_cols or col == "source_id":
            continue
        if col in {"apogee_id"}:
            cols[col] = np.array([f"APO-{i}" for i in source_ids], dtype=object)
        elif col in {"sdss_id"}:
            cols[col] = np.arange(n, dtype=np.int64) + 1000
        elif col == "av_los_source":
            cols[col] = np.zeros(n, dtype=np.int8)
        elif col == "flag_bad":
            cols[col] = np.zeros(n, dtype=np.int32)
        else:
            cols[col] = np.linspace(0.1, 1.0, n, dtype=np.float32)
    return pd.DataFrame({"source_id": source_ids, **cols})


def _fabricate_xp(source_ids: np.ndarray) -> pd.DataFrame:
    n = len(source_ids)
    arr = [np.linspace(0.0, 1.0, XP_N_COEFFS, dtype=np.float32).tolist() for _ in range(n)]
    return pd.DataFrame(
        {
            "source_id": source_ids,
            "bp_coeffs_norm": arr,
            "rp_coeffs_norm": arr,
            "bp_coeff_errs_norm": arr,
            "rp_coeff_errs_norm": arr,
            "bp_c0_z": np.linspace(-1.0, 1.0, n, dtype=np.float32),
            "rp_c0_z": np.linspace(-1.0, 1.0, n, dtype=np.float32),
        }
    )


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


@pytest.fixture
def training_inputs(tmp_path: Path):
    ids = np.arange(1, 6, dtype=np.int64)
    stream_path = tmp_path / "stream1_geom.parquet"
    xp_path = tmp_path / "xp_coeffs.parquet"
    _write_parquet(stream_path, _fabricate_stream(PIPELINE1_TRAINING_SCHEMA, ids))
    _write_parquet(xp_path, _fabricate_xp(ids))
    return stream_path, xp_path


@pytest.fixture
def inference_inputs(tmp_path: Path):
    ids = np.arange(1, 6, dtype=np.int64)
    stream_path = tmp_path / "stream3_geom.parquet"
    xp_path = tmp_path / "xp_coeffs.parquet"
    _write_parquet(stream_path, _fabricate_stream(PIPELINE1_INFERENCE_SCHEMA, ids))
    _write_parquet(xp_path, _fabricate_xp(ids))
    return stream_path, xp_path


def test_training_happy_path(tmp_path: Path, training_inputs) -> None:
    stream_path, xp_path = training_inputs
    out_path = tmp_path / "processed" / "pipeline1_training.parquet"
    returned = build_pipeline1_training(
        stream_path,
        xp_path,
        output_path=out_path,
    )
    assert returned == out_path
    assert out_path.is_file()
    sidecar = out_path.with_suffix("").with_suffix(".provenance.json")
    assert sidecar.is_file()


def test_training_schema_is_satisfied(tmp_path: Path, training_inputs) -> None:
    stream_path, xp_path = training_inputs
    out_path = tmp_path / "pipeline1_training.parquet"
    build_pipeline1_training(stream_path, xp_path, output_path=out_path)
    df = pd.read_parquet(out_path)
    # Raises if schema violated.
    PIPELINE1_TRAINING_SCHEMA.validate(df)


def test_inference_happy_path(tmp_path: Path, inference_inputs) -> None:
    stream_path, xp_path = inference_inputs
    out_path = tmp_path / "pipeline1_inference.parquet"
    build_pipeline1_inference(stream_path, xp_path, output_path=out_path)
    df = pd.read_parquet(out_path)
    PIPELINE1_INFERENCE_SCHEMA.validate(df)


def test_inner_join_drops_missing_xp_rows(tmp_path: Path) -> None:
    ids = np.arange(1, 6, dtype=np.int64)
    stream_path = tmp_path / "stream1_geom.parquet"
    xp_path = tmp_path / "xp_coeffs.parquet"
    _write_parquet(stream_path, _fabricate_stream(PIPELINE1_TRAINING_SCHEMA, ids))
    # XP missing source_ids 4 and 5.
    _write_parquet(xp_path, _fabricate_xp(ids[:3]))

    out_path = tmp_path / "training.parquet"
    build_pipeline1_training(stream_path, xp_path, output_path=out_path)
    df = pd.read_parquet(out_path)
    assert set(df["source_id"]) == {1, 2, 3}


def test_missing_schema_column_raises_schemaerror(tmp_path: Path) -> None:
    ids = np.arange(1, 4, dtype=np.int64)
    stream_path = tmp_path / "stream1_geom.parquet"
    xp_path = tmp_path / "xp_coeffs.parquet"
    bad = _fabricate_stream(PIPELINE1_TRAINING_SCHEMA, ids).drop(columns=["teff_apogee"])
    _write_parquet(stream_path, bad)
    _write_parquet(xp_path, _fabricate_xp(ids))

    with pytest.raises(SchemaError, match="teff_apogee"):
        build_pipeline1_training(
            stream_path,
            xp_path,
            output_path=tmp_path / "x.parquet",
        )


def test_row_counts_and_drops_in_provenance(tmp_path: Path) -> None:
    ids = np.arange(1, 11, dtype=np.int64)
    stream_path = tmp_path / "stream1_geom.parquet"
    xp_path = tmp_path / "xp_coeffs.parquet"
    _write_parquet(stream_path, _fabricate_stream(PIPELINE1_TRAINING_SCHEMA, ids))
    _write_parquet(xp_path, _fabricate_xp(ids[:7]))

    out_path = tmp_path / "training.parquet"
    build_pipeline1_training(stream_path, xp_path, output_path=out_path)
    meta = json.loads(out_path.with_suffix("").with_suffix(".provenance.json").read_text())
    assert meta["row_count_before"] == 10
    assert meta["row_count_after"] == 7
    extra = meta["extra"]
    assert extra["stream_rows"] == 10
    assert extra["xp_rows"] == 7
    assert extra["merged_rows"] == 7
    assert extra["rows_dropped_by_join"] == 3
    assert extra["schema_name"] == "pipeline1_training"


def test_both_inputs_listed_in_provenance_as_local(tmp_path: Path, training_inputs) -> None:
    stream_path, xp_path = training_inputs
    out_path = tmp_path / "training.parquet"
    build_pipeline1_training(stream_path, xp_path, output_path=out_path)
    meta = json.loads(out_path.with_suffix("").with_suffix(".provenance.json").read_text())
    local_sources = [s for s in meta["sources"] if s["kind"] == "local"]
    assert len(local_sources) == 2
    paths = {s["path"] for s in local_sources}
    assert str(stream_path) in paths
    assert str(xp_path) in paths


def test_array_length_check_can_fail(tmp_path: Path) -> None:
    """check_array_lengths=True should catch short XP rows."""
    ids = np.arange(1, 4, dtype=np.int64)
    stream_path = tmp_path / "stream1_geom.parquet"
    xp_path = tmp_path / "xp_coeffs.parquet"
    _write_parquet(stream_path, _fabricate_stream(PIPELINE1_TRAINING_SCHEMA, ids))

    bad_xp = _fabricate_xp(ids)
    # Stomp the first XP row with a length-3 array.
    bad_xp.at[0, "bp_coeffs_norm"] = [0.0, 0.1, 0.2]
    _write_parquet(xp_path, bad_xp)

    with pytest.raises(SchemaError, match="bp_coeffs_norm"):
        build_pipeline1_training(
            stream_path,
            xp_path,
            output_path=tmp_path / "x.parquet",
            check_array_lengths=True,
        )


def test_atomic_write_no_part_left(tmp_path: Path, training_inputs) -> None:
    stream_path, xp_path = training_inputs
    out_path = tmp_path / "processed" / "training.parquet"
    build_pipeline1_training(stream_path, xp_path, output_path=out_path)
    leftover = list(out_path.parent.glob("*.part"))
    assert not leftover


def test_inference_notes_mentions_andrae(tmp_path: Path, inference_inputs) -> None:
    stream_path, xp_path = inference_inputs
    out_path = tmp_path / "pipeline1_inference.parquet"
    build_pipeline1_inference(stream_path, xp_path, output_path=out_path)
    meta = json.loads(out_path.with_suffix("").with_suffix(".provenance.json").read_text())
    assert "Andrae" in meta["notes"]
    assert meta["extra"]["schema_name"] == "pipeline1_inference"
