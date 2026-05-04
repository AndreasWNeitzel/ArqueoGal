"""Offline tests for arqueogal.data.ingest_stream1.

All network-touching and unimplemented stages (HTTPS download, TAP enrichment,
Mészáros+2025 corrections, Lindegren zpt, Riello+2021 G-mag) are monkeypatched
to pass-through so the orchestration plumbing is exercised end-to-end without
real I/O or the optional ``gaiadr3-zeropoint`` package.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pyvo.dal.tap import TAPService

from arqueogal.data import ingest_stream1 as mod
from arqueogal.data.apogee_dr19 import QualityCuts
from arqueogal.data.downloads import DownloadInfo


def _apogee_df(source_ids: list[int]) -> pd.DataFrame:
    """Realistic-enough APOGEE post-cut DataFrame; all rows pass §3.3 cuts."""
    n = len(source_ids)
    return pd.DataFrame(
        {
            "source_id": np.array(source_ids, dtype=np.int64),
            "apogee_id": [f"2M{i:016d}" for i in source_ids],
            "teff": np.full(n, 4800.0, dtype=np.float32),
            "logg": np.full(n, 2.5, dtype=np.float32),
            "m_h_atm": np.full(n, -0.1, dtype=np.float32),
            "snr": np.full(n, 120.0, dtype=np.float32),
            "flag_bad": np.zeros(n, dtype=np.int32),
            "c_fe": np.full(n, 0.05, dtype=np.float32),
            "n_fe": np.full(n, 0.10, dtype=np.float32),
            "e_c_fe": np.full(n, 0.02, dtype=np.float32),
            "e_n_fe": np.full(n, 0.03, dtype=np.float32),
        }
    )


def _gaia_df(source_ids: list[int]) -> pd.DataFrame:
    """Realistic-enough Gaia enrichment DataFrame — one row per source_id."""
    n = len(source_ids)
    return pd.DataFrame(
        {
            "source_id": np.array(source_ids, dtype=np.int64),
            "ra": np.linspace(10.0, 20.0, n, dtype=np.float64),
            "dec": np.linspace(-5.0, 5.0, n, dtype=np.float64),
            "parallax": np.full(n, 1.2, dtype=np.float64),
            "phot_g_mean_mag": np.full(n, 13.5, dtype=np.float32),
            "astrometric_params_solved": np.full(n, 31, dtype=np.int32),
        }
    )


@pytest.fixture
def patched_pipeline(monkeypatch: pytest.MonkeyPatch):
    """Pass-through patches for every stage that touches network / stubs.

    The patches capture the inputs they saw so individual tests can assert on
    orchestration ordering and the value threading between stages.
    """
    captured: dict[str, object] = {}

    def fake_download(url, dest, *, progress=True, **kwargs):  # noqa: ANN001
        captured["download_url"] = url
        captured["download_dest"] = Path(dest)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"fits-placeholder")
        return DownloadInfo(url=url, dest=Path(dest), size_bytes=16, sha256="deadbeef")

    def fake_load_dr19(path, **_kwargs):  # noqa: ANN001
        captured["load_path"] = Path(path)
        # 5 APOGEE rows — the enrich stage will return a 4-id subset so the
        # inner-merge drops the missing one.
        return _apogee_df([100, 200, 300, 400, 500])

    def fake_meszaros(df):
        captured["meszaros_input_len"] = len(df)
        out = df.copy()
        out["meszaros_applied"] = True
        return out

    def fake_enrich(service, source_ids, **kwargs):  # noqa: ANN001
        captured["enrich_service"] = service
        captured["enrich_source_ids"] = list(source_ids)
        captured["enrich_kwargs"] = kwargs
        # Gaia returns only 4 of the 5 APOGEE sources — simulates missing XM.
        return _gaia_df([100, 200, 300, 400])

    def fake_parallax_zpt(df, **_kwargs):
        captured["zpt_input_len"] = len(df)
        out = df.copy()
        out["parallax_corr"] = out["parallax"] - 0.01
        out["parallax_zpt"] = 0.01
        return out

    def fake_g_mag(df):
        captured["g_mag_input_len"] = len(df)
        out = df.copy()
        out["phot_g_mean_mag_corr"] = out["phot_g_mean_mag"] - 0.002
        return out

    monkeypatch.setattr(mod, "download", fake_download)
    monkeypatch.setattr(mod, "load_dr19", fake_load_dr19)
    monkeypatch.setattr(mod, "apply_meszaros2025_corrections", fake_meszaros)
    monkeypatch.setattr(mod, "enrich_source_ids", fake_enrich)

    # Post-2026-04-29: parallax-zpt + G-mag corrections moved into the
    # unified preprocessing pipeline. Patch at the new home and short-
    # circuit apply_pipeline1_preprocessing so the test stays focused on
    # Stream-1 orchestration logic.
    from arqueogal.data import preprocessing as preproc_mod

    monkeypatch.setattr(preproc_mod, "apply_parallax_zpt", fake_parallax_zpt)
    monkeypatch.setattr(preproc_mod, "apply_g_mag_correction", fake_g_mag)

    def fake_preprocessing(df, **kwargs):
        out = fake_parallax_zpt(df)
        out = fake_g_mag(out)
        return out

    monkeypatch.setattr(preproc_mod, "apply_pipeline1_preprocessing", fake_preprocessing)
    monkeypatch.setattr(mod, "apply_pipeline1_preprocessing", fake_preprocessing)
    return captured


def test_happy_path_writes_parquet_and_sidecar(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    service = MagicMock(spec=TAPService)
    out = mod.ingest_stream1(tmp_path, service=service, download_progress=False)

    assert out == tmp_path / "interim" / "stream1_apogee_gaia.parquet"
    assert out.is_file()
    sidecar = tmp_path / "interim" / "stream1_apogee_gaia.provenance.json"
    assert sidecar.is_file()


def test_merge_is_inner(tmp_path: Path, patched_pipeline: dict[str, object]) -> None:
    """APOGEE has 5 stars, Gaia returns 4 → output must have 4 rows."""
    service = MagicMock(spec=TAPService)
    out = mod.ingest_stream1(tmp_path, service=service, download_progress=False)
    df = pd.read_parquet(out)
    assert len(df) == 4
    assert set(df["source_id"]) == {100, 200, 300, 400}


def test_stage_outputs_threaded_through(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    """Each stage sees the previous stage's output (confirms call ordering)."""
    service = MagicMock(spec=TAPService)
    out = mod.ingest_stream1(tmp_path, service=service, download_progress=False)
    df = pd.read_parquet(out)

    # Mészáros marker survives the merge and the two corrections.
    assert df["meszaros_applied"].all()
    # Lindegren zpt ran on the merged frame (4 rows, not the pre-merge 5).
    assert patched_pipeline["zpt_input_len"] == 4
    # Riello+2021 G-mag correction ran after Lindegren (post-zpt column must be present).
    assert "parallax_corr" in df.columns
    assert "phot_g_mean_mag_corr" in df.columns
    # Derived [C/N] happened before Mészáros (which saw 5 post-cut rows).
    assert patched_pipeline["meszaros_input_len"] == 5
    assert "c_n" in df.columns


def test_enrich_receives_post_cut_source_ids(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    service = MagicMock(spec=TAPService)
    mod.ingest_stream1(tmp_path, service=service, download_progress=False)
    assert patched_pipeline["enrich_service"] is service
    assert patched_pipeline["enrich_source_ids"] == [100, 200, 300, 400, 500]
    assert patched_pipeline["enrich_kwargs"]["batch_size"] == 10_000


def test_custom_batch_size_propagates(tmp_path: Path, patched_pipeline: dict[str, object]) -> None:
    service = MagicMock(spec=TAPService)
    mod.ingest_stream1(tmp_path, service=service, enrich_batch_size=2500, download_progress=False)
    assert patched_pipeline["enrich_kwargs"]["batch_size"] == 2500


def test_custom_quality_cuts_propagated_to_provenance(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    service = MagicMock(spec=TAPService)
    cuts = QualityCuts(min_snr=90.0, teff_min=4200.0)
    mod.ingest_stream1(tmp_path, service=service, cuts=cuts, download_progress=False)

    sidecar = tmp_path / "interim" / "stream1_apogee_gaia.provenance.json"
    meta = json.loads(sidecar.read_text())
    assert "snr > 90.0" in meta["cuts_applied"]
    assert "teff in [4200.0, 5500.0]" in meta["cuts_applied"]


def test_provenance_lists_all_three_corrections(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    service = MagicMock(spec=TAPService)
    mod.ingest_stream1(tmp_path, service=service, download_progress=False)
    sidecar = tmp_path / "interim" / "stream1_apogee_gaia.provenance.json"
    meta = json.loads(sidecar.read_text())

    joined = " | ".join(meta["corrections"]).lower()
    assert "mészáros" in joined or "meszaros" in joined
    assert "lindegren" in joined
    assert "riello" in joined


def test_provenance_records_http_and_tap_sources(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    service = MagicMock(spec=TAPService)
    mod.ingest_stream1(tmp_path, service=service, download_progress=False)
    meta = json.loads((tmp_path / "interim" / "stream1_apogee_gaia.provenance.json").read_text())
    kinds = {s["kind"] for s in meta["sources"]}
    assert {"http", "tap", "local"} <= kinds
    http = next(s for s in meta["sources"] if s["kind"] == "http")
    assert http["sha256"] == "deadbeef"
    tap = next(s for s in meta["sources"] if s["kind"] == "tap")
    assert "IN (__batch__)" in tap["query"]
    assert tap["endpoint"].endswith("/tap")


def test_provenance_row_counts(tmp_path: Path, patched_pipeline: dict[str, object]) -> None:
    service = MagicMock(spec=TAPService)
    mod.ingest_stream1(tmp_path, service=service, download_progress=False)
    meta = json.loads((tmp_path / "interim" / "stream1_apogee_gaia.provenance.json").read_text())
    assert meta["row_count_before"] == 5  # fake_load_dr19 synthesizes 5 rows
    assert meta["row_count_after"] == 4  # post-merge
    assert meta["extra"]["apogee_post_cut"] == 5
    assert meta["extra"]["gaia_enriched"] == 4
    assert meta["extra"]["merged"] == 4


def test_meszaros_runs_on_real_implementation(tmp_path: Path, monkeypatch) -> None:
    """With only the Mészáros patch lifted, the real correction must run and pass.

    Guards against re-introduction of the stub and catches regressions that
    would make the real correction raise on typical DR19 inputs.
    """

    def fake_download(url, dest, **_kw):  # noqa: ANN001
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"x")
        return DownloadInfo(url=url, dest=Path(dest), size_bytes=1, sha256="x")

    def fake_load_dr19(path, **_kw):  # noqa: ANN001
        # Add one real abundance column so the correction actually does work.
        df = _apogee_df([1, 2, 3])
        df["alpha_m_atm"] = np.full(len(df), 0.2, dtype=np.float32)
        return df

    def fake_enrich(service, source_ids, **kwargs):  # noqa: ANN001
        g = _gaia_df(list(source_ids))
        g["nu_eff_used_in_astrometry"] = 1.5
        g["pseudocolour"] = 1.5
        g["ecl_lat"] = 30.0
        g["bp_rp"] = 1.0
        g["phot_g_mean_flux"] = 100.0
        return g

    def fake_zpt(df, **_kw):  # noqa: ANN001
        out = df.copy()
        out["parallax_zpt"] = 0.0
        out["parallax_corr"] = out["parallax"]
        return out

    monkeypatch.setattr(mod, "download", fake_download)
    monkeypatch.setattr(mod, "load_dr19", fake_load_dr19)
    monkeypatch.setattr(mod, "enrich_source_ids", fake_enrich)

    # Post-2026-04-29 unified-preprocessing migration: patch at the new home.
    from arqueogal.data import preprocessing as preproc_mod

    monkeypatch.setattr(preproc_mod, "apply_parallax_zpt", fake_zpt)

    def fake_preprocessing(df, **kwargs):
        return fake_zpt(df)

    monkeypatch.setattr(preproc_mod, "apply_pipeline1_preprocessing", fake_preprocessing)
    monkeypatch.setattr(mod, "apply_pipeline1_preprocessing", fake_preprocessing)

    service = MagicMock(spec=TAPService)
    mod.ingest_stream1(tmp_path, service=service, download_progress=False)
    out = pd.read_parquet(tmp_path / "interim" / "stream1_apogee_gaia.parquet")
    # Mészáros Table 3 shifts alpha_m at Teff=4800 by (a·Teff+b) ≈ -0.02388,
    # so corrected [α/M] ≈ 0.22388 — not 0.2 anymore.
    assert (out["alpha_m_atm"] > 0.2).all(), "Mészáros correction did not run"


def test_output_path_is_atomic_no_part_file_left(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    service = MagicMock(spec=TAPService)
    mod.ingest_stream1(tmp_path, service=service, download_progress=False)
    leftover = list((tmp_path / "interim").glob("*.part"))
    assert not leftover, f"temp files should have been renamed: {leftover}"


def test_checkpoint_dir_recorded_in_provenance(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    service = MagicMock(spec=TAPService)
    mod.ingest_stream1(tmp_path, service=service, download_progress=False)
    meta = json.loads((tmp_path / "interim" / "stream1_apogee_gaia.provenance.json").read_text())
    expected = tmp_path / "interim" / "enrich_batches" / "stream1"
    assert meta["extra"]["enrich_checkpoint_dir"] == str(expected)
    # enrich was called with this as its checkpoint_dir
    assert patched_pipeline["enrich_kwargs"]["checkpoint_dir"] == expected
