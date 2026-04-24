"""Offline tests for arqueogal.data.ingest_stream2.

All network-touching and unimplemented stages are monkeypatched so the
orchestration plumbing is exercised end-to-end without real TAP I/O.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pyvo.dal.tap import TAPService

from arqueogal.data import ingest_stream2 as mod


def _hon_df(tics: list[int]) -> pd.DataFrame:
    n = len(tics)
    return pd.DataFrame(
        {
            "TIC": np.array(tics, dtype=np.int64),
            "RAJ2000": np.linspace(0.0, 10.0, n),
            "DEJ2000": np.linspace(-5.0, 5.0, n),
            "numax": np.full(n, 100.0),
            "e_numax": np.full(n, 1.5),
            "Teff": np.full(n, 4800.0),
            "R": np.full(n, 10.0),
            "Prob": np.full(n, 0.98),
        }
    )


def _tic_df(tics: list[int], *, dr2_for_each: list[int | None] | None = None) -> pd.DataFrame:
    """TIC v8.2 rows; rows whose ``dr2_for_each`` entry is ``None`` get NaN GAIA."""
    if dr2_for_each is None:
        dr2_for_each = [t * 10 for t in tics]
    n = len(tics)
    return pd.DataFrame(
        {
            "TIC": np.array(tics, dtype=np.int64),
            "GAIA": np.array(
                [np.nan if v is None else v for v in dr2_for_each],
                dtype=np.float64,
            ),
            "RAJ2000": np.linspace(0.0, 10.0, n),
            "DEJ2000": np.linspace(-5.0, 5.0, n),
            "Tmag": np.full(n, 12.0),
            "plx": np.full(n, 2.0),
        }
    )


def _xmatch_df(dr2_ids: list[int]) -> pd.DataFrame:
    n = len(dr2_ids)
    return pd.DataFrame(
        {
            "dr2_source_id": np.array(dr2_ids, dtype=np.int64),
            "dr3_source_id": np.array([d + 1 for d in dr2_ids], dtype=np.int64),
            "angular_distance": np.full(n, 50.0),
            "magnitude_difference": np.full(n, 0.01),
            "n_candidates": np.ones(n, dtype=np.int64),
        }
    )


def _gaia_df(source_ids: list[int]) -> pd.DataFrame:
    n = len(source_ids)
    return pd.DataFrame(
        {
            "source_id": np.array(source_ids, dtype=np.int64),
            "ra": np.linspace(0.0, 10.0, n),
            "dec": np.linspace(-5.0, 5.0, n),
            "parallax": np.full(n, 1.0),
            "phot_g_mean_mag": np.full(n, 13.0),
        }
    )


@pytest.fixture
def patched_pipeline(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_fetch_hon2021(service, *, prob_threshold=0.95, **_kw):  # noqa: ANN001
        captured["hon_service"] = service
        captured["hon_prob"] = prob_threshold
        # 4 TICs; TIC=400 will NOT resolve to DR2 (missing GAIA).
        return _hon_df([100, 200, 300, 400])

    def fake_fetch_tic(service, tic_ids, **kwargs):  # noqa: ANN001
        captured["tic_service"] = service
        captured["tic_ids"] = list(tic_ids)
        captured["tic_kwargs"] = kwargs
        return _tic_df(
            list(tic_ids),
            dr2_for_each=[1000, 2000, 3000, None],  # TIC 400 has no DR2
        )

    def fake_xmatch(service, dr2_ids, **kwargs):  # noqa: ANN001
        captured["xmatch_service"] = service
        captured["xmatch_dr2_ids"] = list(dr2_ids)
        captured["xmatch_kwargs"] = kwargs
        # Drop one: DR2=3000 has no DR3 counterpart.
        keep = [d for d in dr2_ids if d in (1000, 2000)]
        return _xmatch_df(keep)

    def fake_enrich(service, source_ids, **kwargs):  # noqa: ANN001
        captured["enrich_service"] = service
        captured["enrich_source_ids"] = list(source_ids)
        captured["enrich_kwargs"] = kwargs
        return _gaia_df(list(source_ids))

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

    monkeypatch.setattr(mod, "fetch_hon2021", fake_fetch_hon2021)
    monkeypatch.setattr(mod, "fetch_tic_v82", fake_fetch_tic)
    monkeypatch.setattr(mod, "crossmatch_dr2_to_dr3", fake_xmatch)
    monkeypatch.setattr(mod, "enrich_source_ids", fake_enrich)
    monkeypatch.setattr(mod, "apply_parallax_zpt", fake_zpt)
    monkeypatch.setattr(mod, "apply_g_mag_correction", fake_g_mag)
    return captured


def test_happy_path_writes_parquet_and_sidecar(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    vz = MagicMock(spec=TAPService)
    ap = MagicMock(spec=TAPService)
    out = mod.ingest_stream2(tmp_path, vizier=vz, aip=ap)

    assert out == tmp_path / "interim" / "stream2_tess_gaia.parquet"
    assert out.is_file()
    sidecar = tmp_path / "interim" / "stream2_tess_gaia.provenance.json"
    assert sidecar.is_file()


def test_tic_without_dr2_is_dropped_before_xmatch(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    """TIC 400 has GAIA=NaN; its DR2 id must never reach the xmatch TAP call."""
    vz = MagicMock(spec=TAPService)
    ap = MagicMock(spec=TAPService)
    mod.ingest_stream2(tmp_path, vizier=vz, aip=ap)
    assert patched_pipeline["xmatch_dr2_ids"] == [1000, 2000, 3000]


def test_inner_joins_drop_unresolved_rows(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    """DR2 3000 fails DR3 resolution → only 2 rows survive to Parquet."""
    vz = MagicMock(spec=TAPService)
    ap = MagicMock(spec=TAPService)
    out = mod.ingest_stream2(tmp_path, vizier=vz, aip=ap)
    df = pd.read_parquet(out)
    assert len(df) == 2
    # DR3 source ids are DR2 + 1 per _xmatch_df
    assert set(df["dr3_source_id"]) == {1001, 2001}


def test_corrections_ran_after_join(tmp_path: Path, patched_pipeline: dict[str, object]) -> None:
    vz = MagicMock(spec=TAPService)
    ap = MagicMock(spec=TAPService)
    out = mod.ingest_stream2(tmp_path, vizier=vz, aip=ap)
    df = pd.read_parquet(out)
    assert "parallax_corr" in df.columns
    assert "phot_g_mean_mag_corr" in df.columns
    # Both correction stages saw the post-join row count.
    assert patched_pipeline["zpt_input_len"] == 2
    assert patched_pipeline["g_mag_input_len"] == 2


def test_custom_prob_threshold_propagates(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    vz = MagicMock(spec=TAPService)
    ap = MagicMock(spec=TAPService)
    mod.ingest_stream2(tmp_path, vizier=vz, aip=ap, prob_threshold=0.99)
    assert patched_pipeline["hon_prob"] == 0.99

    meta = json.loads((tmp_path / "interim" / "stream2_tess_gaia.provenance.json").read_text())
    assert "Hon+2021 Prob > 0.99" in meta["cuts_applied"]


def test_xmatch_cuts_recorded_in_provenance(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    vz = MagicMock(spec=TAPService)
    ap = MagicMock(spec=TAPService)
    mod.ingest_stream2(
        tmp_path,
        vizier=vz,
        aip=ap,
        max_angular_distance_mas=200.0,
        max_mag_diff=0.05,
    )
    meta = json.loads((tmp_path / "interim" / "stream2_tess_gaia.provenance.json").read_text())
    assert "angular_distance < 200.0 mas" in meta["cuts_applied"]
    assert "|magnitude_difference| < 0.05" in meta["cuts_applied"]


def test_provenance_has_all_four_tap_sources(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    vz = MagicMock(spec=TAPService)
    ap = MagicMock(spec=TAPService)
    mod.ingest_stream2(tmp_path, vizier=vz, aip=ap)
    meta = json.loads((tmp_path / "interim" / "stream2_tess_gaia.provenance.json").read_text())
    tap_names = [s["name"] for s in meta["sources"] if s["kind"] == "tap"]
    assert len(tap_names) == 4
    joined = " | ".join(tap_names).lower()
    for kw in ("hon+2021", "tic v8.2", "dr2_neighbourhood", "enrichment"):
        assert kw in joined


def test_provenance_row_counts(tmp_path: Path, patched_pipeline: dict[str, object]) -> None:
    vz = MagicMock(spec=TAPService)
    ap = MagicMock(spec=TAPService)
    mod.ingest_stream2(tmp_path, vizier=vz, aip=ap)
    meta = json.loads((tmp_path / "interim" / "stream2_tess_gaia.provenance.json").read_text())
    extra = meta["extra"]
    assert meta["row_count_before"] == 4
    assert meta["row_count_after"] == 2
    assert extra["hon2021_rows"] == 4
    assert extra["tic_rows_with_dr2"] == 3
    assert extra["dr3_resolved"] == 2
    assert extra["joined"] == 2


def test_enrich_receives_resolved_dr3_ids(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    vz = MagicMock(spec=TAPService)
    ap = MagicMock(spec=TAPService)
    mod.ingest_stream2(tmp_path, vizier=vz, aip=ap)
    assert patched_pipeline["enrich_service"] is ap
    assert patched_pipeline["enrich_source_ids"] == [1001, 2001]


def test_atomic_write_no_part_file_left(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    vz = MagicMock(spec=TAPService)
    ap = MagicMock(spec=TAPService)
    mod.ingest_stream2(tmp_path, vizier=vz, aip=ap)
    leftover = list((tmp_path / "interim").glob("*.part"))
    assert not leftover


def test_checkpoint_dirs_recorded(tmp_path: Path, patched_pipeline: dict[str, object]) -> None:
    vz = MagicMock(spec=TAPService)
    ap = MagicMock(spec=TAPService)
    mod.ingest_stream2(tmp_path, vizier=vz, aip=ap)
    meta = json.loads((tmp_path / "interim" / "stream2_tess_gaia.provenance.json").read_text())
    extra = meta["extra"]
    assert extra["tic_checkpoint_dir"].endswith("stream2_tic")
    assert extra["dr2_xmatch_checkpoint_dir"].endswith("stream2_dr2_nbh")
    assert extra["enrich_checkpoint_dir"].endswith("stream2")
