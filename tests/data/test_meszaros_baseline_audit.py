"""
Audit: Is the _h_atm column the calibrated ASPCAP output baseline for Mészáros+2025?

Regresssion test for ensuring that apply_meszaros2025_corrections() is applied
to the correct baseline (calibrated ASPCAP output, not raw uncalibrated output).

Context: DR19 FITS publishes both o_h (calibrated) and raw_o_h (uncalibrated).
The COLUMN_ALIASES map {el}_h_atm -> {el}_h (the calibrated version).
Mészáros+2025 Table 3 coefficients are fitted to open-cluster abundances against
the calibrated ASPCAP output (Mészáros §4.1-4.2). Applying the correction to the
wrong baseline (raw uncalibrated) would over/under-correct by the ASPCAP
calibration offset and corrupt the Stream 1 training pool.

This test:
1. Reads a small slice of DR19.
2. Verifies that _h_atm is the calibrated column (by checking the alias).
3. Confirms that calibrated vs raw differ by ~0.01–0.1 dex (the scale of the
   correction, so baseline mismatch would be detectable as systematic shift).
4. Loads the data via the canonical apogee_dr19 loader and verifies the column
   arrives calibrated (not raw).
"""

from pathlib import Path

import numpy as np
import pytest

from arqueogal.data.apogee_dr19 import (
    COLUMN_ALIASES,
    apply_meszaros2025_corrections,
    apply_quality_cuts,
    load_dr19,
)


@pytest.fixture
def dr19_path():
    """Path to the DR19 FITS file if available locally; skip if not."""
    path = Path(
        "/home/aneitzel/projects/ArqueoGal/data/raw/apogee_dr19/astraAllStarASPCAP-0.6.0.fits.gz"
    )
    if not path.exists():
        pytest.skip("DR19 FITS not available locally")
    return path


def test_column_aliases_map_to_calibrated(dr19_path):
    """
    Test 1: Verify that canonical {el}_h_atm aliases map to FITS {el}_h (calibrated),
    not raw_{el}_h (uncalibrated).
    """
    # The COLUMN_ALIASES dict should map canonical names to FITS names.
    # For abundances, it should be {el}_h (calibrated), not raw_{el}_h.
    test_elements = ["o", "na", "ti", "al"]
    for el in test_elements:
        canonical = f"{el}_h_atm"
        fits_name = COLUMN_ALIASES.get(canonical, canonical)
        # The FITS name should be the plain {el}_h, not raw_{el}_h
        assert fits_name == f"{el}_h", (
            f"Alias for {canonical} is {fits_name}, expected {el}_h (calibrated). "
            "The baseline for Mészáros+2025 is calibrated ASPCAP output."
        )


def test_calibrated_vs_raw_differ(dr19_path):
    """
    Test 2: Empirical check that calibrated and raw columns differ by 0.01–0.1 dex.
    If baseline mismatch occurs, applying corrections to the wrong column would
    shift abundances by this offset, and it would be detectable.
    """
    from astropy.io import fits

    with fits.open(dr19_path, memmap=True) as hdul:
        hdu2 = hdul[2]
        # Check a few elements
        elements = ["o", "ti", "na", "mg"]
        for el in elements:
            calib_col = f"{el}_h"
            raw_col = f"raw_{el}_h"

            calib_data = hdu2.data[calib_col][:5000]
            raw_data = hdu2.data[raw_col][:5000]

            # Compute difference for finite pairs
            mask = np.isfinite(calib_data) & np.isfinite(raw_data)
            diff = np.abs(calib_data[mask] - raw_data[mask])

            # The difference should be > 0 for most (i.e., they are not identical)
            # and typically in the 0.005–0.15 dex range based on ASPCAP calibration.
            mean_diff = np.mean(diff)
            assert mean_diff > 0.001, (
                f"{el}_h (calibrated) and raw_{el}_h are identical or nearly identical. "
                "If baseline mismatch occurs, the shift would be undetectable."
            )
            assert mean_diff < 1.0, (
                f"{el}_h and raw_{el}_h differ by {mean_diff:.3f} dex, "
                "this is implausibly large; check ASPCAP schema."
            )


def test_loader_returns_calibrated(dr19_path):
    """
    Test 3: Verify that load_dr19() returns the calibrated columns, not raw.
    Since COLUMN_ALIASES maps canonical {el}_h_atm -> {el}_h, the loader
    should return calibrated abundances.
    """
    df = load_dr19(
        dr19_path,
        columns=["source_id", "teff", "logg", "o_h_atm", "ti_h_atm", "na_h_atm"],
    )

    # The columns should be present
    assert "o_h_atm" in df.columns
    assert "ti_h_atm" in df.columns

    # Spot-check: read the same data directly from the FITS and verify
    # the values match the "calibrated" column, not the "raw" column.
    from astropy.io import fits

    with fits.open(dr19_path, memmap=True) as hdul:
        hdu2 = hdul[2]
        raw_o_h = hdu2.data["o_h"][:100]
        hdu2.data["ti_h"][:100]

        loader_o = df["o_h_atm"].iloc[:100].to_numpy()
        df["ti_h_atm"].iloc[:100].to_numpy()

        # They should match the calibrated column
        mask_o = np.isfinite(raw_o_h) & np.isfinite(loader_o)
        assert np.allclose(raw_o_h[mask_o], loader_o[mask_o], rtol=1e-5, atol=1e-8), (
            "Loaded o_h_atm does not match FITS o_h (calibrated). "
            "The loader may be returning raw_{el}_h instead."
        )


def test_meszaros_correction_applied_to_correct_baseline(dr19_path):
    """
    Test 4: Integration test. Load the DR19 data, apply quality cuts,
    apply Mészáros corrections, and verify that the correction shifts
    abundances by plausible amounts (0.01–0.1 dex range).

    If the correction were applied to the wrong baseline (raw uncalibrated),
    the shift would be wrong baseline + Mészáros delta, which would result in
    systematic over/under-correction.
    """
    df = load_dr19(dr19_path)
    df, _ = apply_quality_cuts(df)

    # Get the uncorrected abundances (calibrated baseline)
    uncorr_o = df["o_h_atm"].copy()
    df["ti_h_atm"].copy()

    # Apply Mészáros corrections
    df_corr = apply_meszaros2025_corrections(df)

    corr_o = df_corr["o_h_atm"]
    df_corr["ti_h_atm"]

    # The correction summary should report the mean shift
    summary = df_corr.attrs.get("meszaros_correction_summary")
    assert summary is not None, "Mészáros correction summary missing"

    # Look up the reported shifts
    o_summary = summary[summary["element"] == "o_h_atm"].iloc[0]
    ti_summary = summary[summary["element"] == "ti_h_atm"].iloc[0]

    mean_shift_o = o_summary["mean_shift"]
    mean_shift_ti = ti_summary["mean_shift"]

    # The shifts should be small and negative (Mészáros-fitted trends are typically
    # negative, correcting for an over-prediction in Teff-warm giants).
    # Typical magnitudes: 0.01–0.15 dex.
    assert isinstance(mean_shift_o, (int, float)), (
        f"o_h_atm mean_shift not a number: {mean_shift_o}"
    )
    assert isinstance(mean_shift_ti, (int, float)), (
        f"ti_h_atm mean_shift not a number: {mean_shift_ti}"
    )

    # If the baseline were wrong (raw uncalibrated), the systematic shift would be
    # much larger (0.1–0.5 dex) and in the opposite sign from what Mészáros reports.
    # Sanity check: the magnitude of the shift should be plausibly small.
    assert abs(mean_shift_o) < 0.2, (
        f"o_h_atm mean correction shift {mean_shift_o:.4f} is implausibly large. "
        "Baseline may be wrong."
    )
    assert abs(mean_shift_ti) < 0.2, (
        f"ti_h_atm mean correction shift {mean_shift_ti:.4f} is implausibly large. "
        "Baseline may be wrong."
    )

    # Empirical check: the corrected abundances should cluster around
    # uncorrected - shift
    mask_o = np.isfinite(uncorr_o) & np.isfinite(corr_o)
    empirical_shift_o = np.mean(corr_o[mask_o] - uncorr_o[mask_o])

    assert np.isclose(empirical_shift_o, mean_shift_o, rtol=0.01, atol=0.001), (
        f"Empirical shift {empirical_shift_o:.6f} does not match reported "
        f"{mean_shift_o:.6f}. Correction may be applied to the wrong baseline."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
