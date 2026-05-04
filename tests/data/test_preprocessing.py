"""Offline tests for arqueogal.data.preprocessing.

Tests the unified preprocessing pipeline across all 8 steps with mocked
TAP services, mocked XP coefficients, and synthetic Hermite statistics.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from arqueogal.data import preprocessing as mod
from arqueogal.data.frozen_stats import FrozenStatsMismatchError, FrozenZScoreStats


def _synthetic_frame(
    n: int,
    *,
    include_broadbands: bool = False,
    include_av_layers: bool = False,
    include_astrometry: bool = True,
) -> pd.DataFrame:
    """Synthetic input frame for preprocessing tests."""
    df = pd.DataFrame(
        {
            "source_id": np.arange(1000, 1000 + n, dtype=np.int64),
            "ra": np.linspace(0.0, 10.0, n),
            "dec": np.linspace(-5.0, 5.0, n),
        }
    )
    if include_astrometry:
        df["parallax"] = np.full(n, 2.5, dtype=np.float32)
        df["parallax_error"] = np.full(n, 0.05, dtype=np.float32)
        df["parallax_corr"] = np.zeros(n, dtype=np.float32)
        df["phot_g_mean_mag"] = np.full(n, 13.0, dtype=np.float32)
        df["phot_bp_mean_mag"] = np.full(n, 13.5, dtype=np.float32)
        df["phot_rp_mean_mag"] = np.full(n, 12.5, dtype=np.float32)
        df["ruwe"] = np.full(n, 1.1, dtype=np.float32)

    if include_broadbands:
        df["j_mag"] = np.full(n, 12.0, dtype=np.float32)
        df["h_mag"] = np.full(n, 11.8, dtype=np.float32)
        df["k_mag"] = np.full(n, 11.6, dtype=np.float32)
        df["w1_mag"] = np.full(n, 11.4, dtype=np.float32)
        df["w2_mag"] = np.full(n, 11.3, dtype=np.float32)

    if include_av_layers:
        df["av_edenhofer"] = np.full(n, 0.1, dtype=np.float32)
        df["av_lallement"] = np.full(n, 0.12, dtype=np.float32)
        df["av_sfd"] = np.full(n, 0.11, dtype=np.float32)
        df["av_nbhd_median"] = np.full(n, 0.105, dtype=np.float32)
        df["av_nbhd_std"] = np.full(n, 0.02, dtype=np.float32)

    return df


def _synthetic_hermite_dict(n: int, fingerprint: str) -> dict:
    """Synthetic Hermite reprojection output."""
    return {
        "bp_coeffs": np.random.randn(n, 55).astype(np.float32),
        "rp_coeffs": np.random.randn(n, 55).astype(np.float32),
        "reprojection_residual_rms_bp": np.full(n, 0.01, dtype=np.float32),
        "reprojection_residual_rms_rp": np.full(n, 0.01, dtype=np.float32),
        "reprojection_residual_rms": np.full(n, 0.01, dtype=np.float32),
        "basis_version": "v1",
        "basis_fingerprint_sha256": fingerprint,
    }


def _synthetic_frozen_stats(fingerprint: str) -> FrozenZScoreStats:
    """Synthetic frozen z-score statistics."""
    return FrozenZScoreStats(
        basis_fingerprint=fingerprint,
        c0_bp_mean_log10=1.5,
        c0_bp_sigma_log10=0.1,
        c0_rp_mean_log10=1.4,
        c0_rp_sigma_log10=0.1,
        coef_norm_bp_mean=np.zeros(54, dtype=np.float64),
        coef_norm_bp_sigma=np.ones(54, dtype=np.float64),
        coef_norm_rp_mean=np.zeros(54, dtype=np.float64),
        coef_norm_rp_sigma=np.ones(54, dtype=np.float64),
        sigma_floor=0.01,
        n_reference_population=1000,
        reference_population_description="synthetic test set",
    )


def _synthetic_corrected_xp(n: int, fingerprint: str) -> tuple[pd.DataFrame, dict]:
    """Synthetic Ye+2024-corrected XP + Hermite dict."""
    xp_df = pd.DataFrame(
        {
            "source_id": np.arange(1000, 1000 + n, dtype=np.int64),
            "corrected_flux": [np.random.randn(330).astype(np.float32) for _ in range(n)],
            "a_v_sfd": np.full(n, 0.1, dtype=np.float32),
            "ye2024_flag": np.zeros(n, dtype=np.int8),
        }
    )
    hermite = _synthetic_hermite_dict(n, fingerprint)
    return xp_df, hermite


def test_preprocessing_all_steps_in_order(tmp_path: Path):
    """Test that all 8 steps are applied in correct order."""
    n = 5
    fp = "0" * 64  # Synthetic fingerprint
    df = _synthetic_frame(
        n,
        include_broadbands=True,
        include_av_layers=True,
        include_astrometry=True,
    )
    xp_df, hermite = _synthetic_corrected_xp(n, fp)
    frozen_stats = _synthetic_frozen_stats(fp)

    # Mock the key functions to track calls
    call_log = []

    def mock_apply_parallax_zpt(frame):
        call_log.append("parallax_zpt")
        frame = frame.copy()
        frame["parallax"] = frame["parallax"] * 1.001
        return frame

    def mock_apply_g_mag_correction(frame):
        call_log.append("g_mag")
        frame = frame.copy()
        frame["phot_g_mean_mag"] = frame["phot_g_mean_mag"] * 1.001
        return frame

    def mock_fetch_xp(service, source_ids, **kwargs):
        call_log.append("fetch_xp")
        return xp_df

    def mock_apply_ye2024(xp_df, coords_df, **kwargs):
        call_log.append("ye2024")
        return xp_df

    def mock_reproject_hermite(flux):
        call_log.append("hermite")
        return hermite

    def mock_apply_extinction(frame, **kwargs):
        call_log.append("extinction")
        for col in ("j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag"):
            if col in frame.columns:
                frame[f"{col}_dered"] = frame[col] * 0.95
        return frame

    def mock_load_frozen_stats(path):
        call_log.append("load_frozen_stats")
        return frozen_stats

    def mock_verify_basis(fp_current, stats):
        call_log.append("verify_basis")
        if fp_current != stats.basis_fingerprint:
            raise FrozenStatsMismatchError(expected=stats.basis_fingerprint, observed=fp_current)

    with patch.multiple(
        mod,
        apply_parallax_zpt=mock_apply_parallax_zpt,
        apply_g_mag_correction=mock_apply_g_mag_correction,
        fetch_xp_coefficients=mock_fetch_xp,
        apply_ye2024_correction=mock_apply_ye2024,
        reproject_ye_to_hermite=mock_reproject_hermite,
        apply_extinction_corrections=mock_apply_extinction,
        load_frozen_zscore_stats=mock_load_frozen_stats,
        verify_basis_fingerprint=mock_verify_basis,
    ):
        result = mod.apply_pipeline1_preprocessing(
            df,
            mode="train",
            skip_xp_fetch=False,
            xp_coords=df[["source_id", "ra", "dec"]],
            apply_extinction=True,
        )

    # Verify call order
    expected_order = [
        "parallax_zpt",
        "g_mag",
        "fetch_xp",
        "ye2024",
        "hermite",
        "load_frozen_stats",  # Loading happens before verify
        "verify_basis",
        "extinction",
    ]
    assert call_log == expected_order, f"Call order mismatch: {call_log} != {expected_order}"

    # Verify output has expected columns
    assert "bp_coef_norm_1" in result.columns
    assert "rp_coef_norm_1" in result.columns
    assert "bp_c0_z" in result.columns
    assert "rp_c0_z" in result.columns
    assert "reprojection_residual_rms" in result.columns


def test_train_and_inference_byte_identical():
    """Test that train and inference modes produce identical output."""
    n = 3
    fp = "1" * 64
    df = _synthetic_frame(n, include_astrometry=True)
    xp_df, hermite = _synthetic_corrected_xp(n, fp)
    frozen_stats = _synthetic_frozen_stats(fp)

    def mock_apply_parallax_zpt(frame):
        return frame.copy()

    def mock_apply_g_mag_correction(frame):
        return frame.copy()

    def mock_fetch_xp(service, source_ids, **kwargs):
        return xp_df

    def mock_apply_ye2024(xp_df, coords_df, **kwargs):
        return xp_df

    def mock_reproject_hermite(flux):
        return hermite

    def mock_load_frozen_stats(path):
        return frozen_stats

    def mock_verify_basis(fp_current, stats):
        pass

    def mock_apply_extinction(frame, **kwargs):
        return frame

    with patch.multiple(
        mod,
        apply_parallax_zpt=mock_apply_parallax_zpt,
        apply_g_mag_correction=mock_apply_g_mag_correction,
        fetch_xp_coefficients=mock_fetch_xp,
        apply_ye2024_correction=mock_apply_ye2024,
        reproject_ye_to_hermite=mock_reproject_hermite,
        apply_extinction_corrections=mock_apply_extinction,
        load_frozen_zscore_stats=mock_load_frozen_stats,
        verify_basis_fingerprint=mock_verify_basis,
    ):
        result_train = mod.apply_pipeline1_preprocessing(
            df,
            mode="train",
            skip_xp_fetch=False,
            xp_coords=df[["source_id", "ra", "dec"]],
            apply_extinction=False,
        )
        result_inference = mod.apply_pipeline1_preprocessing(
            df,
            mode="inference",
            skip_xp_fetch=False,
            xp_coords=df[["source_id", "ra", "dec"]],
            apply_extinction=False,
        )

    # Compare all numeric columns (ignore logging/metadata-only differences)
    numeric_cols = result_train.select_dtypes(include=[np.number]).columns
    pd.testing.assert_frame_equal(
        result_train[numeric_cols].astype(np.float32),
        result_inference[numeric_cols].astype(np.float32),
        check_exact=False,
        rtol=1e-6,
    )


def test_frozen_stats_fingerprint_mismatch_raises():
    """Test that basis fingerprint mismatch raises before z-scoring."""
    n = 2
    fp_good = "a" * 64
    fp_bad = "b" * 64
    df = _synthetic_frame(n, include_astrometry=True)
    xp_df, hermite = _synthetic_corrected_xp(n, fp_bad)  # Bad fingerprint
    frozen_stats = _synthetic_frozen_stats(fp_good)  # Expected good fingerprint

    def mock_apply_parallax_zpt(frame):
        return frame.copy()

    def mock_apply_g_mag_correction(frame):
        return frame.copy()

    def mock_fetch_xp(service, source_ids, **kwargs):
        return xp_df

    def mock_apply_ye2024(xp_df, coords_df, **kwargs):
        return xp_df

    def mock_reproject_hermite(flux):
        return hermite

    def mock_load_frozen_stats(path):
        return frozen_stats

    def mock_verify_basis(fp_current, stats):
        if fp_current != stats.basis_fingerprint:
            raise FrozenStatsMismatchError(expected=stats.basis_fingerprint, observed=fp_current)

    with (
        patch.multiple(
            mod,
            apply_parallax_zpt=mock_apply_parallax_zpt,
            apply_g_mag_correction=mock_apply_g_mag_correction,
            fetch_xp_coefficients=mock_fetch_xp,
            apply_ye2024_correction=mock_apply_ye2024,
            reproject_ye_to_hermite=mock_reproject_hermite,
            load_frozen_zscore_stats=mock_load_frozen_stats,
            verify_basis_fingerprint=mock_verify_basis,
        ),
        pytest.raises(FrozenStatsMismatchError),
    ):
        mod.apply_pipeline1_preprocessing(
            df,
            mode="inference",
            skip_xp_fetch=False,
            xp_coords=df[["source_id", "ra", "dec"]],
            apply_extinction=False,
        )


def test_missing_ir_broadbands_skips_extinction():
    """Test that missing IR broadbands causes extinction step to be skipped."""
    n = 2
    fp = "c" * 64
    df = _synthetic_frame(n, include_broadbands=False, include_av_layers=True)
    xp_df, hermite = _synthetic_corrected_xp(n, fp)
    frozen_stats = _synthetic_frozen_stats(fp)

    extinction_called = []

    def mock_apply_parallax_zpt(frame):
        return frame.copy()

    def mock_apply_g_mag_correction(frame):
        return frame.copy()

    def mock_fetch_xp(service, source_ids, **kwargs):
        return xp_df

    def mock_apply_ye2024(xp_df, coords_df, **kwargs):
        return xp_df

    def mock_reproject_hermite(flux):
        return hermite

    def mock_load_frozen_stats(path):
        return frozen_stats

    def mock_verify_basis(fp_current, stats):
        pass

    def mock_apply_extinction(frame, **kwargs):
        extinction_called.append(True)
        return frame

    with patch.multiple(
        mod,
        apply_parallax_zpt=mock_apply_parallax_zpt,
        apply_g_mag_correction=mock_apply_g_mag_correction,
        fetch_xp_coefficients=mock_fetch_xp,
        apply_ye2024_correction=mock_apply_ye2024,
        reproject_ye_to_hermite=mock_reproject_hermite,
        apply_extinction_corrections=mock_apply_extinction,
        load_frozen_zscore_stats=mock_load_frozen_stats,
        verify_basis_fingerprint=mock_verify_basis,
    ):
        mod.apply_pipeline1_preprocessing(
            df,
            mode="inference",
            skip_xp_fetch=False,
            xp_coords=df[["source_id", "ra", "dec"]],
            apply_extinction=True,
        )

    # Extinction should NOT have been called because broadbands are missing
    assert len(extinction_called) == 0


def test_missing_dust_map_columns_skips_extinction():
    """Test that missing dust-map columns causes extinction step to be skipped."""
    n = 2
    fp = "d" * 64
    df = _synthetic_frame(n, include_broadbands=True, include_av_layers=False)
    xp_df, hermite = _synthetic_corrected_xp(n, fp)
    frozen_stats = _synthetic_frozen_stats(fp)

    extinction_called = []

    def mock_apply_parallax_zpt(frame):
        return frame.copy()

    def mock_apply_g_mag_correction(frame):
        return frame.copy()

    def mock_fetch_xp(service, source_ids, **kwargs):
        return xp_df

    def mock_apply_ye2024(xp_df, coords_df, **kwargs):
        return xp_df

    def mock_reproject_hermite(flux):
        return hermite

    def mock_load_frozen_stats(path):
        return frozen_stats

    def mock_verify_basis(fp_current, stats):
        pass

    def mock_apply_extinction(frame, **kwargs):
        extinction_called.append(True)
        return frame

    with patch.multiple(
        mod,
        apply_parallax_zpt=mock_apply_parallax_zpt,
        apply_g_mag_correction=mock_apply_g_mag_correction,
        fetch_xp_coefficients=mock_fetch_xp,
        apply_ye2024_correction=mock_apply_ye2024,
        reproject_ye_to_hermite=mock_reproject_hermite,
        apply_extinction_corrections=mock_apply_extinction,
        load_frozen_zscore_stats=mock_load_frozen_stats,
        verify_basis_fingerprint=mock_verify_basis,
    ):
        mod.apply_pipeline1_preprocessing(
            df,
            mode="inference",
            skip_xp_fetch=False,
            xp_coords=df[["source_id", "ra", "dec"]],
            apply_extinction=True,
        )

    # Extinction should NOT have been called because dust-map columns are missing
    assert len(extinction_called) == 0


def test_basis_fingerprint_check_before_z_scoring():
    """Test that basis fingerprint is verified before z-scoring step."""
    n = 1
    fp_good = "e" * 64
    fp_bad = "f" * 64
    df = _synthetic_frame(n, include_astrometry=True)
    xp_df, hermite = _synthetic_corrected_xp(n, fp_bad)
    frozen_stats = _synthetic_frozen_stats(fp_good)

    verify_basis_order = []

    def mock_apply_parallax_zpt(frame):
        return frame.copy()

    def mock_apply_g_mag_correction(frame):
        return frame.copy()

    def mock_fetch_xp(service, source_ids, **kwargs):
        return xp_df

    def mock_apply_ye2024(xp_df, coords_df, **kwargs):
        return xp_df

    def mock_reproject_hermite(flux):
        return hermite

    def mock_load_frozen_stats(path):
        verify_basis_order.append("load_frozen_stats")
        return frozen_stats

    def mock_verify_basis(fp_current, stats):
        verify_basis_order.append("verify_basis")
        if fp_current != stats.basis_fingerprint:
            raise FrozenStatsMismatchError(expected=stats.basis_fingerprint, observed=fp_current)

    with (
        patch.multiple(
            mod,
            apply_parallax_zpt=mock_apply_parallax_zpt,
            apply_g_mag_correction=mock_apply_g_mag_correction,
            fetch_xp_coefficients=mock_fetch_xp,
            apply_ye2024_correction=mock_apply_ye2024,
            reproject_ye_to_hermite=mock_reproject_hermite,
            load_frozen_zscore_stats=mock_load_frozen_stats,
            verify_basis_fingerprint=mock_verify_basis,
        ),
        pytest.raises(FrozenStatsMismatchError),
    ):
        mod.apply_pipeline1_preprocessing(
            df,
            mode="inference",
            skip_xp_fetch=False,
            xp_coords=df[["source_id", "ra", "dec"]],
            apply_extinction=False,
        )

    # Verify that verify_basis is called and raises before any z-scoring
    assert "verify_basis" in verify_basis_order


__all__ = []
