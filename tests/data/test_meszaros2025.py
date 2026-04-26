"""Comprehensive tests for Mészáros+2025 [X/M] Teff-trend corrections.

Validates that the polynomial corrections from Mészáros+2025 (arXiv:2506.07845,
AJ in press) are correctly applied to DR19 [X/M] abundances. Covers linear regime,
boundary offsets, dwarf exclusion, and per-element coefficient structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arqueogal.data.apogee_dr19 import (
    MESZAROS2025_COEFFS,
    MESZAROS2025_LOGG_MAX,
    MESZAROS2025_TEFF_MAX,
    MESZAROS2025_TEFF_MIN,
    apply_meszaros2025_corrections,
)


class TestMeszaros2025CoefficientsStructure:
    """Validate the coefficient dictionary structure and contents."""

    def test_coefficients_dict_not_empty(self) -> None:
        """Coefficients dict is populated."""
        assert len(MESZAROS2025_COEFFS) > 0
        assert len(MESZAROS2025_COEFFS) >= 13

    def test_each_element_has_four_coefficients(self) -> None:
        """Every element entry is a 4-tuple (a, b, hot, cold)."""
        for el, coeff in MESZAROS2025_COEFFS.items():
            assert isinstance(coeff, tuple), f"{el} coefficient is not tuple"
            assert len(coeff) == 4, f"{el} coefficient is not 4-tuple"
            a, b, hot, cold = coeff
            assert isinstance(a, float), f"{el}.a is not float"
            assert isinstance(b, float), f"{el}.b is not float"
            assert isinstance(hot, float), f"{el}.hot is not float"
            assert isinstance(cold, float), f"{el}.cold is not float"

    def test_expected_elements_present(self) -> None:
        """All published elements from Mészáros+2025 Table 3 are included."""
        # Elements published in the paper: alpha, O, Na, Mg, Al, Si, S, K, Ca,
        # Ti, Cr, Mn, Ni, Ce. C, N, Fe, V, Cu deliberately omitted (documented).
        expected = {
            "alpha_m_atm",
            "o_h_atm",
            "na_h_atm",
            "mg_h_atm",
            "al_h_atm",
            "si_h_atm",
            "s_h_atm",
            "k_h_atm",
            "ca_h_atm",
            "ti_h_atm",
            "cr_h_atm",
            "mn_h_atm",
            "ni_h_atm",
            "ce_h_atm",
        }
        actual = set(MESZAROS2025_COEFFS.keys())
        assert expected == actual, f"Mismatch: expected {expected}, got {actual}"

    def test_coefficients_are_plausible_magnitudes(self) -> None:
        """Linear coefficients (a, b) are physically plausible for abundance trends."""
        for el, (a, b, hot, cold) in MESZAROS2025_COEFFS.items():
            # Slopes should be small (dex per K), typically <1e-4
            assert abs(a) < 1e-3, f"{el}.a = {a} is implausibly large"
            # Intercepts are dex units, typically within [-1, 1]
            assert abs(b) < 2.0, f"{el}.b = {b} is implausibly large"
            # Offsets are dex units, typically within [-0.5, 0.5]
            assert abs(hot) < 1.0, f"{el}.hot = {hot} is implausibly large"
            assert abs(cold) < 1.0, f"{el}.cold = {cold} is implausibly large"


class TestMeszaros2025LinearRegime:
    """Test the core linear correction (a·Teff + b) inside [3500, 6000] K."""

    def test_linear_correction_mg_midpoint(self) -> None:
        """Magnesium correction at Teff = 4750 K (center of calibration) is linear."""
        a, b, _, _ = MESZAROS2025_COEFFS["mg_h_atm"]
        teff = np.array([4750.0])
        df = pd.DataFrame({"teff": teff, "logg": [2.0], "mg_h_atm": [0.2]})
        out = apply_meszaros2025_corrections(df, elements=("mg_h_atm",))
        expected_delta = a * 4750.0 + b
        expected_value = 0.2 - expected_delta
        np.testing.assert_allclose(out["mg_h_atm"].to_numpy(), expected_value, rtol=1e-10)

    def test_linear_correction_vectorized_teff(self) -> None:
        """Multiple Teff values in linear regime are corrected independently."""
        a, b, _, _ = MESZAROS2025_COEFFS["alpha_m_atm"]
        teffs = np.array([3500.0, 4000.0, 4500.0, 5000.0, 5500.0, 6000.0])
        df = pd.DataFrame(
            {"teff": teffs, "logg": np.full(len(teffs), 2.0), "alpha_m_atm": np.full(len(teffs), 0.3)}
        )
        out = apply_meszaros2025_corrections(df, elements=("alpha_m_atm",))
        expected = 0.3 - (a * teffs + b)
        np.testing.assert_allclose(out["alpha_m_atm"].to_numpy(), expected, rtol=1e-10)

    def test_linear_correction_multiple_elements(self) -> None:
        """Multiple elements are corrected independently within linear regime."""
        df = pd.DataFrame(
            {
                "teff": [5000.0, 5000.0, 5000.0],
                "logg": [2.0, 2.0, 2.0],
                "mg_h_atm": [0.2, 0.2, 0.2],
                "si_h_atm": [0.1, 0.1, 0.1],
                "ca_h_atm": [0.15, 0.15, 0.15],
            }
        )
        out = apply_meszaros2025_corrections(df)

        # Verify each element's correction
        for el in ["mg_h_atm", "si_h_atm", "ca_h_atm"]:
            a, b, _, _ = MESZAROS2025_COEFFS[el]
            expected = 0.2 if el == "mg_h_atm" else (0.1 if el == "si_h_atm" else 0.15)
            expected = expected - (a * 5000.0 + b)
            np.testing.assert_allclose(
                out[el].to_numpy(), [expected, expected, expected], rtol=1e-10, atol=1e-15
            )


class TestMeszaros2025BoundaryOffsets:
    """Test behavior outside calibrated Teff range [3500, 6000] K."""

    def test_cold_boundary_offset_below_3500k(self) -> None:
        """Teff < 3500 K uses the cold offset from Table 3."""
        _, _, _, cold = MESZAROS2025_COEFFS["mg_h_atm"]
        df = pd.DataFrame({"teff": [3400.0, 3200.0, 3000.0], "logg": [2.0, 2.0, 2.0], "mg_h_atm": [0.5, 0.5, 0.5]})
        out = apply_meszaros2025_corrections(df, elements=("mg_h_atm",))
        expected = 0.5 - cold
        np.testing.assert_allclose(out["mg_h_atm"].to_numpy(), [expected, expected, expected], rtol=1e-10)

    def test_hot_boundary_offset_above_6000k(self) -> None:
        """Teff > 6000 K uses the hot offset from Table 3."""
        _, _, hot, _ = MESZAROS2025_COEFFS["mg_h_atm"]
        df = pd.DataFrame({"teff": [6100.0, 6500.0, 7000.0], "logg": [2.0, 2.0, 2.0], "mg_h_atm": [0.5, 0.5, 0.5]})
        out = apply_meszaros2025_corrections(df, elements=("mg_h_atm",))
        expected = 0.5 - hot
        np.testing.assert_allclose(out["mg_h_atm"].to_numpy(), [expected, expected, expected], rtol=1e-10)

    def test_exact_boundaries_are_linear(self) -> None:
        """At exactly Teff = 3500 and 6000 K the linear formula is used, not offset."""
        a, b, _, _ = MESZAROS2025_COEFFS["mg_h_atm"]
        df = pd.DataFrame({"teff": [3500.0, 6000.0], "logg": [2.0, 2.0], "mg_h_atm": [0.3, 0.3]})
        out = apply_meszaros2025_corrections(df, elements=("mg_h_atm",))
        expected = 0.3 - (a * np.array([3500.0, 6000.0]) + b)
        np.testing.assert_allclose(out["mg_h_atm"].to_numpy(), expected, rtol=1e-10)


class TestMeszaros2025DwarfExclusion:
    """Test that log g >= 3.8 (dwarfs) are left uncorrected."""

    def test_dwarf_logg_threshold_not_corrected(self) -> None:
        """log g >= 3.8 → correction delta is NaN → value unchanged."""
        df = pd.DataFrame(
            {
                "teff": [5000.0, 5000.0, 5000.0],
                "logg": [2.0, 3.8, 4.0],
                "mg_h_atm": [0.3, 0.3, 0.3],
            }
        )
        out = apply_meszaros2025_corrections(df, elements=("mg_h_atm",))

        # First row (logg=2.0) should be corrected
        assert out["mg_h_atm"].iloc[0] != 0.3

        # Second and third rows (logg >= 3.8) should be unchanged
        assert out["mg_h_atm"].iloc[1] == 0.3
        assert out["mg_h_atm"].iloc[2] == 0.3

    def test_dwarf_band_across_teff(self) -> None:
        """Dwarfs remain uncorrected across all Teff values."""
        teffs = np.array([3500.0, 4500.0, 6000.0])
        loggs = np.full(3, 4.0)  # All dwarfs
        df = pd.DataFrame(
            {
                "teff": teffs,
                "logg": loggs,
                "mg_h_atm": np.full(3, 0.2),
            }
        )
        out = apply_meszaros2025_corrections(df, elements=("mg_h_atm",))
        # All should remain 0.2 (uncorrected)
        np.testing.assert_allclose(out["mg_h_atm"].to_numpy(), [0.2, 0.2, 0.2], rtol=1e-10)


class TestMeszaros2025NonFiniteHandling:
    """Test handling of NaN and Inf in Teff/logg."""

    def test_nan_teff_leaves_value_unchanged(self) -> None:
        """NaN in Teff produces NaN delta → value unchanged."""
        df = pd.DataFrame(
            {
                "teff": [5000.0, np.nan, 5000.0],
                "logg": [2.0, 2.0, 2.0],
                "mg_h_atm": [0.2, 0.2, 0.2],
            }
        )
        out = apply_meszaros2025_corrections(df, elements=("mg_h_atm",))
        assert out["mg_h_atm"].iloc[0] != 0.2  # Corrected
        assert out["mg_h_atm"].iloc[1] == 0.2  # Uncorrected (NaN input)
        assert out["mg_h_atm"].iloc[2] != 0.2  # Corrected

    def test_nan_logg_leaves_value_unchanged(self) -> None:
        """NaN in logg produces NaN delta → value unchanged."""
        df = pd.DataFrame(
            {
                "teff": [5000.0, 5000.0],
                "logg": [2.0, np.nan],
                "mg_h_atm": [0.2, 0.2],
            }
        )
        out = apply_meszaros2025_corrections(df, elements=("mg_h_atm",))
        assert out["mg_h_atm"].iloc[0] != 0.2  # Corrected
        assert out["mg_h_atm"].iloc[1] == 0.2  # Uncorrected (NaN logg)

    def test_inf_teff_treated_as_boundary(self) -> None:
        """inf Teff produces NaN delta → value unchanged (inf is non-finite)."""
        df = pd.DataFrame(
            {
                "teff": [np.inf, -np.inf],
                "logg": [2.0, 2.0],
                "mg_h_atm": [0.2, 0.2],
            }
        )
        out = apply_meszaros2025_corrections(df, elements=("mg_h_atm",))
        # Both should be unchanged (inf is non-finite)
        np.testing.assert_allclose(out["mg_h_atm"].to_numpy(), [0.2, 0.2], rtol=1e-10)


class TestMeszaros2025ElementExclusions:
    """Test that C, N, Fe, V, Cu are correctly handled per docstring."""

    def test_fe_not_in_coefficients(self) -> None:
        """Fe deliberately omitted (reference element, Δ[Fe/M] ≡ 0)."""
        assert "fe_h_atm" not in MESZAROS2025_COEFFS
        assert "fe_m_atm" not in MESZAROS2025_COEFFS

    def test_c_not_in_coefficients(self) -> None:
        """C omitted due to first-dredge-up and thermohaline mixing in giants."""
        assert "c_h_atm" not in MESZAROS2025_COEFFS

    def test_n_not_in_coefficients(self) -> None:
        """N omitted due to first-dredge-up and thermohaline mixing in giants."""
        assert "n_h_atm" not in MESZAROS2025_COEFFS

    def test_v_not_in_coefficients(self) -> None:
        """V not published in Mészáros+2025 Table 3."""
        assert "v_h_atm" not in MESZAROS2025_COEFFS

    def test_cu_not_in_coefficients(self) -> None:
        """Cu not published in Mészáros+2025 Table 3."""
        assert "cu_h_atm" not in MESZAROS2025_COEFFS

    def test_requesting_omitted_element_raises(self) -> None:
        """Requesting C, N, Fe, V, or Cu raises KeyError."""
        df = pd.DataFrame({"teff": [5000.0], "logg": [2.0]})
        for omitted in ["c_h_atm", "n_h_atm", "fe_h_atm", "v_h_atm", "cu_h_atm"]:
            with pytest.raises(KeyError, match="no Mészáros"):
                apply_meszaros2025_corrections(df, elements=(omitted,))


class TestMeszaros2025SummaryMetadata:
    """Test that correction summaries are computed and attached correctly."""

    def test_summary_dataframe_attached(self) -> None:
        """Corrected DataFrame has meszaros_correction_summary in attrs."""
        df = pd.DataFrame(
            {
                "teff": [5000.0, 5000.0],
                "logg": [2.0, 2.0],
                "mg_h_atm": [0.2, 0.3],
            }
        )
        out = apply_meszaros2025_corrections(df, elements=("mg_h_atm",))
        assert "meszaros_correction_summary" in out.attrs
        summary = out.attrs["meszaros_correction_summary"]
        assert isinstance(summary, pd.DataFrame)

    def test_summary_contains_required_columns(self) -> None:
        """Summary has element, n_applied, mean_shift, rms_shift columns."""
        df = pd.DataFrame(
            {
                "teff": np.array([4000.0, 5000.0, 6000.0]),
                "logg": [2.0, 2.0, 2.0],
                "mg_h_atm": [0.1, 0.2, 0.3],
            }
        )
        out = apply_meszaros2025_corrections(df, elements=("mg_h_atm",))
        summary = out.attrs["meszaros_correction_summary"]
        required = {"element", "n_applied", "mean_shift", "rms_shift"}
        assert required <= set(summary.columns)

    def test_summary_n_applied_matches_valid_rows(self) -> None:
        """n_applied in summary matches count of corrected rows."""
        df = pd.DataFrame(
            {
                "teff": [5000.0, 5000.0, 5000.0],
                "logg": [2.0, 4.0, 2.0],  # Middle row is dwarf (uncorrected)
                "mg_h_atm": [0.2, 0.2, 0.2],
            }
        )
        out = apply_meszaros2025_corrections(df, elements=("mg_h_atm",))
        summary = out.attrs["meszaros_correction_summary"]
        row = summary[summary["element"] == "mg_h_atm"].iloc[0]
        # Only 2 giants should be corrected
        assert row["n_applied"] == 2


class TestMeszaros2025Integration:
    """Integration tests with full APOGEE-like DataFrames."""

    def test_mixed_regimes_single_frame(self) -> None:
        """Single frame with dwarfs, giants, cold, hot, and NaN."""
        df = pd.DataFrame(
            {
                "teff": [3000.0, 3500.0, 5000.0, 6000.0, 7000.0, np.nan],
                "logg": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                "mg_h_atm": np.full(6, 0.2),
                "si_h_atm": np.full(6, 0.1),
            }
        )
        out = apply_meszaros2025_corrections(df)

        # Row 0 (cold): uses cold offset
        # Row 1 (linear boundary): uses linear formula
        # Row 2 (linear center): uses linear formula
        # Row 3 (hot boundary): uses linear formula
        # Row 4 (hot): uses hot offset
        # Row 5 (NaN): unchanged

        assert out["mg_h_atm"].iloc[0] != 0.2  # Corrected (cold)
        assert out["mg_h_atm"].iloc[4] != 0.2  # Corrected (hot)
        assert out["mg_h_atm"].iloc[5] == 0.2  # Unchanged (NaN)

    def test_no_correction_without_target_columns(self) -> None:
        """If no target columns exist in frame, returns unchanged frame."""
        df = pd.DataFrame(
            {
                "teff": [5000.0],
                "logg": [2.0],
                "teff_err": [50.0],  # Not a correction target
            }
        )
        out = apply_meszaros2025_corrections(df)
        # Should complete without error and return same frame
        assert len(out) == 1
        assert "teff" in out.columns

    def test_correction_is_nondestructive(self) -> None:
        """Original frame is not modified; correction returns a copy."""
        df = pd.DataFrame(
            {
                "teff": [5000.0],
                "logg": [2.0],
                "mg_h_atm": [0.2],
            }
        )
        original_value = df["mg_h_atm"].iloc[0]
        out = apply_meszaros2025_corrections(df)
        # Original should be unchanged
        assert df["mg_h_atm"].iloc[0] == original_value
        # Output may be changed
        assert out is not df


class TestMeszaros2025Constants:
    """Test module-level constants match documented values."""

    def test_teff_min_constant(self) -> None:
        """MESZAROS2025_TEFF_MIN = 3500.0 K."""
        assert MESZAROS2025_TEFF_MIN == 3500.0

    def test_teff_max_constant(self) -> None:
        """MESZAROS2025_TEFF_MAX = 6000.0 K."""
        assert MESZAROS2025_TEFF_MAX == 6000.0

    def test_logg_max_constant(self) -> None:
        """MESZAROS2025_LOGG_MAX = 3.8 (RGB/giant boundary)."""
        assert MESZAROS2025_LOGG_MAX == 3.8
