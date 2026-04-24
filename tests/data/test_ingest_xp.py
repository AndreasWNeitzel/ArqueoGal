"""Offline tests for arqueogal.data.ingest_xp — §11 Level 3.

TAP + Ye+2024 are monkeypatched on the orchestrator module namespace so the
glue is exercised without real network I/O and without loading the vendored
Ye+2024 NN weights. The fake Ye returns the new sampled-flux schema —
``(source_id, corrected_flux, a_v_sfd, ye2024_flag)`` — matching the
contract of :func:`arqueogal.data.gaia_xp.apply_ye2024_correction`.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pyvo.dal.tap import TAPService

from arqueogal.data import ingest_xp as mod
from arqueogal.data.gaia_xp import (
    XP_COEFF_LEN,
    YE2024_FLAG_NO_SYNTH_PHOT,
    YE2024_FLAG_OK,
    YE2024_N_OUTPUT,
    YE2024_SAMPLING_NM,
)


def _raw_xp_df(source_ids: list[int]) -> pd.DataFrame:
    n = len(source_ids)
    rng = np.random.default_rng(0)
    bp_c = [rng.uniform(0.5, 1.5, XP_COEFF_LEN).tolist() for _ in range(n)]
    bp_e = [rng.uniform(0.01, 0.05, XP_COEFF_LEN).tolist() for _ in range(n)]
    rp_c = [rng.uniform(0.5, 1.5, XP_COEFF_LEN).tolist() for _ in range(n)]
    rp_e = [rng.uniform(0.01, 0.05, XP_COEFF_LEN).tolist() for _ in range(n)]
    return pd.DataFrame(
        {
            "source_id": np.asarray(source_ids, dtype=np.int64),
            "bp_coefficients": bp_c,
            "bp_coefficient_errors": bp_e,
            "rp_coefficients": rp_c,
            "rp_coefficient_errors": rp_e,
            "bp_standard_deviation": np.full(n, 0.02),
            "rp_standard_deviation": np.full(n, 0.02),
            "bp_n_measurements": np.full(n, 100),
            "rp_n_measurements": np.full(n, 100),
            "bp_n_relevant_bases": np.full(n, 45),
            "rp_n_relevant_bases": np.full(n, 30),
        }
    )


def _coords_df(source_ids: list[int]) -> pd.DataFrame:
    n = len(source_ids)
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "source_id": np.asarray(source_ids, dtype=np.int64),
            "ra": rng.uniform(0.0, 360.0, n),
            "dec": rng.uniform(-20.0, 20.0, n),
        }
    )


def _fake_ye_output(xp_df: pd.DataFrame) -> pd.DataFrame:
    """Build a plausible Ye+2024 output for ``xp_df`` — sampled-flux schema."""
    n = len(xp_df)
    rng = np.random.default_rng(2)
    flux = [rng.uniform(1.0, 2.0, YE2024_N_OUTPUT).astype(np.float32) for _ in range(n)]
    return pd.DataFrame(
        {
            "source_id": xp_df["source_id"].to_numpy().astype(np.int64),
            "corrected_flux": flux,
            "a_v_sfd": rng.uniform(0.0, 1.0, n).astype(np.float32),
            "ye2024_flag": np.full(n, YE2024_FLAG_OK, dtype=np.int8),
        }
    )


@pytest.fixture
def patched_pipeline(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_fetch(service, source_ids, **kwargs):  # noqa: ANN001
        captured["fetch_service"] = service
        ids = list(source_ids)
        captured["fetch_ids"] = list(np.asarray(ids, dtype=np.int64))
        captured["fetch_kwargs"] = kwargs
        return _raw_xp_df(captured["fetch_ids"])

    def fake_sanity(df):
        captured["sanity_len"] = len(df)
        return {"bp_coefficients_nan_rows": 0, "rp_coefficients_nan_rows": 0}

    def fake_ye(xp_df, coords_df, **kwargs):  # noqa: ANN001
        captured["ye_xp_len"] = len(xp_df)
        captured["ye_coords_len"] = len(coords_df)
        captured["ye_kwargs"] = kwargs
        return _fake_ye_output(xp_df)

    monkeypatch.setattr(mod, "fetch_xp_coefficients", fake_fetch)
    monkeypatch.setattr(mod, "xp_sanity_check", fake_sanity)
    monkeypatch.setattr(mod, "apply_ye2024_correction", fake_ye)
    return captured


def test_happy_path_writes_parquet_and_sidecar(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    svc = MagicMock(spec=TAPService)
    ids = [1, 2, 3]
    out_path, flags = mod.ingest_xp(tmp_path, ids, _coords_df(ids), service=svc)
    assert out_path == tmp_path / "interim" / "xp_sampled_corrected.parquet"
    assert out_path.is_file()
    assert (tmp_path / "interim" / "xp_sampled_corrected.provenance.json").is_file()
    assert flags == {"n_ok": 3, "n_no_synth_phot": 0, "n_calibrate_fail": 0}


def test_output_schema_is_sampled_flux(tmp_path: Path, patched_pipeline: dict[str, object]) -> None:
    svc = MagicMock(spec=TAPService)
    ids = [1, 2, 3]
    out_path, _ = mod.ingest_xp(tmp_path, ids, _coords_df(ids), service=svc)
    df = pd.read_parquet(out_path)
    assert list(df.columns) == ["source_id", "corrected_flux", "a_v_sfd", "ye2024_flag"]
    assert df["source_id"].dtype == np.int64
    # Raw Hermite columns must NOT leak through — Ye emits sampled flux only.
    for col in ("bp_coefficients", "rp_coefficients", "bp_coeffs_norm", "bp_c0_z"):
        assert col not in df.columns
    # corrected_flux rows are length-330 arrays.
    first = np.asarray(df["corrected_flux"].iloc[0])
    assert first.shape == (YE2024_N_OUTPUT,)


def test_coords_df_passed_through_to_ye(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    svc = MagicMock(spec=TAPService)
    ids = [10, 11, 12, 13]
    mod.ingest_xp(tmp_path, ids, _coords_df(ids), service=svc)
    assert patched_pipeline["ye_xp_len"] == len(ids)
    assert patched_pipeline["ye_coords_len"] == len(ids)


def test_missing_coords_column_raises(tmp_path: Path) -> None:
    svc = MagicMock(spec=TAPService)
    bad = pd.DataFrame({"source_id": [1], "ra": [0.0]})  # no dec
    with pytest.raises(KeyError, match="dec"):
        mod.ingest_xp(tmp_path, [1], bad, service=svc)


def test_provenance_has_xp_tap_source(tmp_path: Path, patched_pipeline: dict[str, object]) -> None:
    svc = MagicMock(spec=TAPService)
    ids = [1, 2, 3, 4, 5]
    mod.ingest_xp(tmp_path, ids, _coords_df(ids), service=svc)
    meta = json.loads((tmp_path / "interim" / "xp_sampled_corrected.provenance.json").read_text())
    tap = [s for s in meta["sources"] if s["kind"] == "tap"]
    assert len(tap) == 1
    assert "xp_continuous_mean_spectrum" in tap[0]["name"]
    assert tap[0]["batch_size"] >= 1


def test_provenance_records_ye2024_correction(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    svc = MagicMock(spec=TAPService)
    ids = [1, 2, 3]
    mod.ingest_xp(tmp_path, ids, _coords_df(ids), service=svc)
    meta = json.loads((tmp_path / "interim" / "xp_sampled_corrected.provenance.json").read_text())
    corrections = " | ".join(meta["corrections"]).lower()
    assert "ye+2024" in corrections
    assert "ccm89" in corrections
    assert "sfd" in corrections
    # §6.4 steps 2–5 are intentionally NOT chained here — document via extras.
    assert meta["extra"]["ye2024_sampling_n"] == YE2024_N_OUTPUT


def test_provenance_records_flag_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Flag counts (OK / no-synth-phot / calibrate-fail) land in provenance extras."""
    svc = MagicMock(spec=TAPService)

    def fake_fetch(service, source_ids, **kwargs):  # noqa: ANN001
        return _raw_xp_df(list(np.asarray(list(source_ids), dtype=np.int64)))

    def fake_ye_mixed(xp_df, coords_df, **kwargs):  # noqa: ANN001
        out = _fake_ye_output(xp_df)
        # Mark the second row as no-synth-phot.
        flags = out["ye2024_flag"].to_numpy().copy()
        flags[1] = YE2024_FLAG_NO_SYNTH_PHOT
        out["ye2024_flag"] = flags
        return out

    monkeypatch.setattr(mod, "fetch_xp_coefficients", fake_fetch)
    monkeypatch.setattr(mod, "xp_sanity_check", lambda df: {})
    monkeypatch.setattr(mod, "apply_ye2024_correction", fake_ye_mixed)

    ids = [1, 2, 3]
    _, flags = mod.ingest_xp(tmp_path, ids, _coords_df(ids), service=svc)
    assert flags == {"n_ok": 2, "n_no_synth_phot": 1, "n_calibrate_fail": 0}
    meta = json.loads((tmp_path / "interim" / "xp_sampled_corrected.provenance.json").read_text())
    assert meta["extra"]["ye2024_flag_counts"] == flags


def test_custom_batch_size_propagates(tmp_path: Path, patched_pipeline: dict[str, object]) -> None:
    svc = MagicMock(spec=TAPService)
    ids = [1, 2, 3]
    mod.ingest_xp(tmp_path, ids, _coords_df(ids), service=svc, batch_size=2500)
    assert patched_pipeline["fetch_kwargs"]["batch_size"] == 2500
    assert patched_pipeline["ye_kwargs"]["batch_size"] == 2500


def test_checkpoint_dir_recorded(tmp_path: Path, patched_pipeline: dict[str, object]) -> None:
    svc = MagicMock(spec=TAPService)
    ids = [1, 2, 3]
    mod.ingest_xp(tmp_path, ids, _coords_df(ids), service=svc)
    meta = json.loads((tmp_path / "interim" / "xp_sampled_corrected.provenance.json").read_text())
    expected = tmp_path / "interim" / "enrich_batches" / "xp"
    assert meta["extra"]["xp_checkpoint_dir"] == str(expected)
    assert patched_pipeline["fetch_kwargs"]["checkpoint_dir"] == expected


def test_row_counts_in_provenance(tmp_path: Path, patched_pipeline: dict[str, object]) -> None:
    svc = MagicMock(spec=TAPService)
    ids = list(range(1, 11))
    mod.ingest_xp(tmp_path, ids, _coords_df(ids), service=svc)
    meta = json.loads((tmp_path / "interim" / "xp_sampled_corrected.provenance.json").read_text())
    assert meta["row_count_before"] == 10
    assert meta["extra"]["source_ids_requested"] == 10
    assert meta["extra"]["xp_rows_returned"] == 10
    assert meta["row_count_after"] == 10


def test_atomic_write_no_part_file_left(
    tmp_path: Path, patched_pipeline: dict[str, object]
) -> None:
    svc = MagicMock(spec=TAPService)
    ids = [1, 2, 3]
    mod.ingest_xp(tmp_path, ids, _coords_df(ids), service=svc)
    leftover = list((tmp_path / "interim").glob("*.part"))
    assert not leftover
