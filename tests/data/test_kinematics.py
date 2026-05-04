"""Tests for arqueogal.data.kinematics — §9 orbital parameters.

Integration tests invoke real galpy on very small star samples (N ≤ 3)
to keep runtime bounded — each galpy call is ~1 s on 1–3 stars.

No network I/O. Entirely offline.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from arqueogal.data.kinematics import (
    OUTPUT_COLS,
    R_0_KPC,
    REQUIRED_INPUT_COLS,
    SOLAR_MOTION_KMS,
    STAECKEL_DELTA,
    V_0_KMS,
    Z_0_PC,
    KinematicsConfig,
    compute_actions,
    compute_actions_mc,
)

# ---- helpers -----------------------------------------------------------------


def _sunlike_row(  # noqa: PLR0913 — test helper with keyword-only tuning knobs
    source_id: int = 1,
    *,
    ra: float = 0.0,
    dec: float = 0.0,
    d_pc: float = 100.0,
    pmra: float = 5.0,
    pmdec: float = 2.0,
    rv: float = 10.0,
) -> dict:
    return {
        "source_id": source_id,
        "ra": ra,
        "dec": dec,
        "r_med_photogeo": d_pc,
        "pmra": pmra,
        "pmdec": pmdec,
        "radial_velocity": rv,
    }


def _mc_row(source_id: int = 1) -> dict:
    base = _sunlike_row(source_id)
    base.update(
        {
            "parallax": 1000.0 / base["r_med_photogeo"],
            "parallax_error": 0.05,
            "pmra_error": 0.05,
            "pmdec_error": 0.05,
            "radial_velocity_error": 1.0,
            "ra_dec_corr": 0.0,
            "ra_parallax_corr": 0.0,
            "ra_pmra_corr": 0.0,
            "ra_pmdec_corr": 0.0,
            "dec_parallax_corr": 0.0,
            "dec_pmra_corr": 0.0,
            "dec_pmdec_corr": 0.0,
            "parallax_pmra_corr": 0.0,
            "parallax_pmdec_corr": 0.0,
            "pmra_pmdec_corr": 0.0,
        }
    )
    return base


# ---- constants ---------------------------------------------------------------


def test_default_constants_match_section_9_2() -> None:
    """§9.2: the fitted McMillan+2017 ro/vo and Bennett&Bovy z_0."""
    assert R_0_KPC == 8.21
    assert V_0_KMS == 233.1
    assert Z_0_PC == 20.8
    assert STAECKEL_DELTA == 0.45
    # Schönrich+2010 solar peculiar motion: U_sun sign flipped for galpy.
    assert SOLAR_MOTION_KMS == (-11.1, 12.24, 7.25)


def test_kinematics_config_defaults() -> None:
    cfg = KinematicsConfig()
    assert cfg.ro_kpc == R_0_KPC
    assert cfg.vo_kms == V_0_KMS
    assert cfg.zo_pc == Z_0_PC
    assert cfg.solarmotion_kms == SOLAR_MOTION_KMS
    assert cfg.staeckel_delta == STAECKEL_DELTA
    assert cfg.potential == "mcmillan17"


def test_kinematics_config_is_frozen() -> None:
    cfg = KinematicsConfig()
    with pytest.raises((AttributeError, Exception)):
        cfg.ro_kpc = 9.0  # type: ignore[misc]


def test_required_and_output_cols_shapes() -> None:
    assert REQUIRED_INPUT_COLS[0] == "source_id"
    assert "r_med_photogeo" in REQUIRED_INPUT_COLS
    assert "J_R_kpc_kms" in OUTPUT_COLS
    assert "L_z_kpc_kms" in OUTPUT_COLS
    assert "J_z_kpc_kms" in OUTPUT_COLS
    assert "ecc" in OUTPUT_COLS
    assert OUTPUT_COLS[0] == "source_id"


# ---- validation & NaN filtering ---------------------------------------------


def test_compute_actions_missing_column_raises() -> None:
    df = pd.DataFrame({"source_id": [1], "ra": [0.0]})  # everything else missing
    with pytest.raises(KeyError, match="compute_actions requires columns"):
        compute_actions(df)


def test_compute_actions_all_nan_returns_empty() -> None:
    rows = [_sunlike_row(1), _sunlike_row(2)]
    df = pd.DataFrame(rows)
    df.loc[:, "pmra"] = np.nan
    out = compute_actions(df)
    assert out.empty
    assert list(out.columns) == list(OUTPUT_COLS)


def test_compute_actions_drops_nan_rows(caplog: pytest.LogCaptureFixture) -> None:
    pytest.importorskip("galpy")  # compute_actions imports galpy lazily
    rows = [_sunlike_row(1), _sunlike_row(2, pmra=np.nan), _sunlike_row(3)]
    df = pd.DataFrame(rows)
    with caplog.at_level(logging.INFO, logger="arqueogal.data.kinematics"):
        out = compute_actions(df)
    assert len(out) == 2
    assert list(out["source_id"]) == [1, 3]
    assert any("dropped 1" in rec.message for rec in caplog.records)


def test_compute_actions_mc_missing_corr_raises() -> None:
    df = pd.DataFrame([_sunlike_row(1)])  # missing MC-specific columns
    with pytest.raises(KeyError, match="compute_actions_mc requires columns"):
        compute_actions_mc(df, n_samples=3)


def test_compute_actions_mc_rejects_nonpositive_samples() -> None:
    df = pd.DataFrame([_mc_row(1)])
    with pytest.raises(ValueError, match="n_samples"):
        compute_actions_mc(df, n_samples=0)


# ---- potential dispatch ------------------------------------------------------


def test_unknown_potential_raises() -> None:
    pytest.importorskip("galpy")  # compute_actions imports galpy lazily
    df = pd.DataFrame([_sunlike_row(1)])
    with pytest.raises(ValueError, match="unknown potential"):
        compute_actions(df, config=KinematicsConfig(potential="not-a-pot"))  # type: ignore[arg-type]


# ---- integration: real galpy on a Sun-like star ------------------------------


@pytest.mark.slow
def test_compute_actions_sunlike_orbit_sensible() -> None:
    """A nearby, low-velocity star should come out with low J_R, low J_z,
    L_z near the solar circular value, and small ecc.

    Sanity bounds (generous — galpy conventions/signs can flip):
    - |L_z| ≳ 1000 kpc·km/s  (solar value ≈ R_0 × V_0 ≈ 1913)
    - J_R < 200 kpc·km/s
    - J_z < 100 kpc·km/s
    - 0 ≤ ecc < 0.3
    - r_peri, r_apo within ~3 kpc of R_0
    """
    df = pd.DataFrame([_sunlike_row(1)])
    out = compute_actions(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert list(out.columns) == list(OUTPUT_COLS)

    # R near solar
    assert row["R_galcen_kpc"] == pytest.approx(R_0_KPC, abs=0.2)
    # |L_z| near the circular value
    assert abs(row["L_z_kpc_kms"]) > 1000.0
    assert abs(row["L_z_kpc_kms"]) < 3000.0
    # Radial / vertical actions small for a disk-like orbit
    assert 0.0 <= row["J_R_kpc_kms"] < 500.0
    assert 0.0 <= row["J_z_kpc_kms"] < 200.0
    # Low eccentricity
    assert 0.0 <= row["ecc"] < 0.3
    # Orbit bracket around R_0
    assert row["r_peri_kpc"] < R_0_KPC + 1.0
    assert row["r_apo_kpc"] > R_0_KPC - 1.0
    # Finite values across the board
    for col in OUTPUT_COLS[1:]:
        assert np.isfinite(row[col]), f"{col} is not finite"


@pytest.mark.slow
def test_compute_actions_multiple_stars() -> None:
    rows = [
        _sunlike_row(1, d_pc=50.0),
        _sunlike_row(2, d_pc=100.0, pmra=3.0, pmdec=-1.0),
        _sunlike_row(3, d_pc=200.0, rv=-20.0),
    ]
    df = pd.DataFrame(rows)
    out = compute_actions(df)
    assert len(out) == 3
    assert list(out["source_id"]) == [1, 2, 3]
    # All finite
    for col in OUTPUT_COLS[1:]:
        assert np.all(np.isfinite(out[col])), f"{col} not all finite"


@pytest.mark.slow
def test_compute_actions_mwpotential2014_differs_from_mcmillan17() -> None:
    """Actions depend on the potential — swapping gives different numbers."""
    df = pd.DataFrame([_sunlike_row(1)])
    out_mcmillan = compute_actions(df, config=KinematicsConfig(potential="mcmillan17"))
    out_mw = compute_actions(df, config=KinematicsConfig(potential="mwpotential2014"))
    # At least one action component should differ by > 1% — the two potentials
    # are independently fitted and give meaningfully different answers.
    diff_jr = abs(out_mcmillan["J_R_kpc_kms"].iloc[0] - out_mw["J_R_kpc_kms"].iloc[0])
    diff_jz = abs(out_mcmillan["J_z_kpc_kms"].iloc[0] - out_mw["J_z_kpc_kms"].iloc[0])
    diff_lz = abs(out_mcmillan["L_z_kpc_kms"].iloc[0] - out_mw["L_z_kpc_kms"].iloc[0])
    assert (diff_jr + diff_jz + diff_lz) > 1.0


# ---- integration: MC ---------------------------------------------------------


@pytest.mark.slow
def test_compute_actions_mc_returns_long_format() -> None:
    df = pd.DataFrame([_mc_row(1), _mc_row(2)])
    out = compute_actions_mc(df, n_samples=3, rng_seed=42)
    # 2 stars × 3 draws = 6 rows, long format
    assert len(out) == 6
    assert "draw" in out.columns
    assert sorted(out["draw"].unique().tolist()) == [0, 1, 2]
    # Each star appears exactly n_samples times
    counts = out.groupby("source_id").size()
    assert (counts == 3).all()


@pytest.mark.slow
def test_compute_actions_mc_is_deterministic_with_seed() -> None:
    df = pd.DataFrame([_mc_row(1)])
    df["parallax_error"] = 0.1  # introduce some dispersion
    out1 = compute_actions_mc(df, n_samples=4, rng_seed=123)
    out2 = compute_actions_mc(df, n_samples=4, rng_seed=123)
    pd.testing.assert_frame_equal(out1, out2)


@pytest.mark.slow
def test_compute_actions_mc_produces_dispersion() -> None:
    """With non-zero errors, repeated draws should scatter around the central value."""
    df = pd.DataFrame([_mc_row(1)])
    df["parallax_error"] = 0.2
    df["pmra_error"] = 0.2
    df["pmdec_error"] = 0.2
    out = compute_actions_mc(df, n_samples=10, rng_seed=7)
    assert out["L_z_kpc_kms"].std() > 0.0
    assert out["J_R_kpc_kms"].std() >= 0.0


def test_compute_actions_mc_empty_after_nan_filter() -> None:
    row = _mc_row(1)
    row["parallax"] = np.nan
    df = pd.DataFrame([row])
    out = compute_actions_mc(df, n_samples=2)
    assert out.empty
    assert "draw" in out.columns


# ---- empty input -------------------------------------------------------------


def test_compute_actions_empty_input() -> None:
    df = pd.DataFrame(columns=list(REQUIRED_INPUT_COLS))
    out = compute_actions(df)
    assert out.empty
    assert list(out.columns) == list(OUTPUT_COLS)
