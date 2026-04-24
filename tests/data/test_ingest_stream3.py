"""Offline tests for arqueogal.data.ingest_stream3.

All network-touching stages are monkeypatched; stratified_subsample is
exercised with its real default bins on a synthetic in-range frame.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pyvo.dal.tap import TAPService

from arqueogal.data import ingest_stream3 as mod
from arqueogal.data.stream3_selection import (
    DEFAULT_BINS_G,
    DEFAULT_BINS_LOGG,
    DEFAULT_BINS_MH,
    DEFAULT_BINS_TEFF,
)


def _andrae_df(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "source_id": np.arange(1, n + 1, dtype=np.int64),
            "teff_xgboost": rng.uniform(DEFAULT_BINS_TEFF[0], DEFAULT_BINS_TEFF[-1], n),
            "logg_xgboost": rng.uniform(DEFAULT_BINS_LOGG[0], DEFAULT_BINS_LOGG[-1], n),
            "mh_xgboost": rng.uniform(DEFAULT_BINS_MH[0], DEFAULT_BINS_MH[-1], n),
            "phot_g_mean_mag": rng.uniform(DEFAULT_BINS_G[0], DEFAULT_BINS_G[-1], n),
        }
    )


def _gaia_df(source_ids: np.ndarray) -> pd.DataFrame:
    n = len(source_ids)
    return pd.DataFrame(
        {
            "source_id": np.asarray(source_ids, dtype=np.int64),
            "ra": np.linspace(0.0, 10.0, n),
            "dec": np.linspace(-5.0, 5.0, n),
            "parallax": np.full(n, 1.0),
            "phot_g_mean_mag": np.full(n, 12.5),
        }
    )


@pytest.fixture
def patched_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured: dict[str, object] = {}
    # Orchestrator SHA-256s the Andrae FITS for provenance, so the path must
    # exist on disk even though load_andrae2023 itself is monkeypatched.
    (tmp_path / "andrae.fits").write_bytes(b"")

    def fake_load_andrae(path, **_kw):  # noqa: ANN001
        captured["load_path"] = Path(path)
        return _andrae_df(300)

    def fake_enrich(service, source_ids, **kwargs):  # noqa: ANN001
        captured["enrich_service"] = service
        ids = np.asarray(list(source_ids), dtype=np.int64)
        captured["enrich_source_ids"] = ids.tolist()
        captured["enrich_kwargs"] = kwargs
        # Drop the last id to simulate a missing Gaia row.
        return _gaia_df(ids[:-1])

    def fake_zpt(df, **_kw):
        captured["zpt_input_len"] = len(df)
        out = df.copy()
        out["parallax_corr"] = out["parallax"] - 0.01
        return out

    def fake_g_mag(df):
        captured["g_mag_input_len"] = len(df)
        out = df.copy()
        out["phot_g_mean_mag_corr"] = out["phot_g_mean_mag"] - 0.002
        return out

    monkeypatch.setattr(mod, "load_andrae2023", fake_load_andrae)
    monkeypatch.setattr(mod, "enrich_source_ids", fake_enrich)
    monkeypatch.setattr(mod, "apply_parallax_zpt", fake_zpt)
    monkeypatch.setattr(mod, "apply_g_mag_correction", fake_g_mag)
    return captured


def test_happy_path_writes_parquet_and_sidecar(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    svc = MagicMock(spec=TAPService)
    out = mod.ingest_stream3(
        tmp_path,
        andrae_fits=tmp_path / "andrae.fits",
        service=svc,
        per_cell=1,
    )
    assert out == tmp_path / "interim" / "stream3_gaia_rgbrc.parquet"
    assert out.is_file()
    assert (tmp_path / "interim" / "stream3_gaia_rgbrc.provenance.json").is_file()


def test_enrich_called_with_stratified_source_ids(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    svc = MagicMock(spec=TAPService)
    mod.ingest_stream3(
        tmp_path,
        andrae_fits=tmp_path / "andrae.fits",
        service=svc,
        per_cell=1,
    )
    # Subset of the 300 synthetic Andrae sources.
    sent = patched_pipeline["enrich_source_ids"]
    assert 0 < len(sent) <= 300
    assert set(sent).issubset(set(range(1, 301)))


def test_inner_merge_drops_missing_gaia_row(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    svc = MagicMock(spec=TAPService)
    out = mod.ingest_stream3(
        tmp_path,
        andrae_fits=tmp_path / "andrae.fits",
        service=svc,
        per_cell=1,
    )
    df = pd.read_parquet(out)
    # fake_enrich drops the last source_id; merged must have exactly (sent − 1).
    n_sent = len(patched_pipeline["enrich_source_ids"])
    assert len(df) == n_sent - 1


def test_corrections_fire_after_merge(tmp_path: Path, patched_pipeline: dict[str, object]) -> None:
    svc = MagicMock(spec=TAPService)
    out = mod.ingest_stream3(
        tmp_path,
        andrae_fits=tmp_path / "andrae.fits",
        service=svc,
        per_cell=1,
    )
    df = pd.read_parquet(out)
    assert "parallax_corr" in df.columns
    assert "phot_g_mean_mag_corr" in df.columns
    assert patched_pipeline["zpt_input_len"] == len(df)


def test_per_cell_and_seed_recorded_in_provenance(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    svc = MagicMock(spec=TAPService)
    mod.ingest_stream3(
        tmp_path,
        andrae_fits=tmp_path / "andrae.fits",
        service=svc,
        per_cell=3,
        rng_seed=42,
    )
    meta = json.loads((tmp_path / "interim" / "stream3_gaia_rgbrc.provenance.json").read_text())
    assert "stratified_subsample per_cell=3" in meta["cuts_applied"]
    assert "stratified_subsample rng_seed=42" in meta["cuts_applied"]
    strat = meta["extra"]["stratification"]
    assert strat["per_cell"] == 3
    assert strat["rng_seed"] == 42


def test_provenance_has_local_and_tap_sources(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    svc = MagicMock(spec=TAPService)
    mod.ingest_stream3(
        tmp_path,
        andrae_fits=tmp_path / "andrae.fits",
        service=svc,
        per_cell=1,
    )
    meta = json.loads((tmp_path / "interim" / "stream3_gaia_rgbrc.provenance.json").read_text())
    kinds = {s["kind"] for s in meta["sources"]}
    assert {"local", "tap"} <= kinds
    local = next(s for s in meta["sources"] if s["kind"] == "local")
    assert "Andrae+2023" in local["name"]
    assert "7945154" in local["name"]


def test_provenance_row_counts_thread_correctly(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    svc = MagicMock(spec=TAPService)
    mod.ingest_stream3(
        tmp_path,
        andrae_fits=tmp_path / "andrae.fits",
        service=svc,
        per_cell=1,
    )
    meta = json.loads((tmp_path / "interim" / "stream3_gaia_rgbrc.provenance.json").read_text())
    extra = meta["extra"]
    assert meta["row_count_before"] == 300
    assert extra["andrae_rows_loaded"] == 300
    assert extra["stratified_selected"] >= 1
    assert extra["merged"] == extra["stratified_selected"] - 1
    assert meta["row_count_after"] == extra["merged"]


def test_custom_batch_size_propagates(tmp_path: Path, patched_pipeline: dict[str, object]) -> None:
    svc = MagicMock(spec=TAPService)
    mod.ingest_stream3(
        tmp_path,
        andrae_fits=tmp_path / "andrae.fits",
        service=svc,
        per_cell=1,
        enrich_batch_size=250,
    )
    assert patched_pipeline["enrich_kwargs"]["batch_size"] == 250


def test_atomic_write_no_part_file_left(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    svc = MagicMock(spec=TAPService)
    mod.ingest_stream3(
        tmp_path,
        andrae_fits=tmp_path / "andrae.fits",
        service=svc,
        per_cell=1,
    )
    leftover = list((tmp_path / "interim").glob("*.part"))
    assert not leftover


def test_checkpoint_dir_recorded(tmp_path: Path, patched_pipeline: dict[str, object]) -> None:
    svc = MagicMock(spec=TAPService)
    mod.ingest_stream3(
        tmp_path,
        andrae_fits=tmp_path / "andrae.fits",
        service=svc,
        per_cell=1,
    )
    meta = json.loads((tmp_path / "interim" / "stream3_gaia_rgbrc.provenance.json").read_text())
    expected = tmp_path / "interim" / "enrich_batches" / "stream3"
    assert meta["extra"]["enrich_checkpoint_dir"] == str(expected)
    assert patched_pipeline["enrich_kwargs"]["checkpoint_dir"] == expected
