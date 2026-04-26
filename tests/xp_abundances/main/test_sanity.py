"""Tests for xp_abundances.main.sanity: pre-training gate battery.

Hard-fail checks (1-3) halt training; soft-fail checks (5-7) log and continue.
The harness here is small but each test exercises a documented contract from
the module docstring so future drift surfaces as a clear failure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arqueogal.xp_abundances.main.sanity import (
    PARAMETER_BOUNDS,
    TIER1_ATMOSPHERIC,
    CheckResult,
    check_parameter_bounds,
    check_per_element_nan_rates,
    check_tier1_label_completeness,
    check_xp_feature_nan_invariant,
)


def _xp_norm_columns() -> list[str]:
    return (
        [f"bp_coef_norm_{i}" for i in range(1, 55)]
        + [f"rp_coef_norm_{i}" for i in range(1, 55)]
        + ["bp_c0_z", "rp_c0_z"]
    )


def _make_clean_df(n: int = 8) -> pd.DataFrame:
    """Minimal clean-population fixture: all flags zero, finite values."""
    rng = np.random.default_rng(0)
    base = {
        "ye2024_flag": np.zeros(n, dtype=np.int8),
        "xp_fit_flag_residual_high": pd.array(np.zeros(n, dtype=np.int8), dtype="Int8"),
        "bp_coef_0": np.full(n, 1e-14),
        "rp_coef_0": np.full(n, 1e-14),
        "flag_bad": np.zeros(n, dtype=bool),
        "teff_apogee": rng.uniform(4000, 5500, n),
        "logg_apogee": rng.uniform(1.0, 3.5, n),
        "mh_apogee": rng.uniform(-1.5, 0.5, n),
        "fe_h_apogee": rng.uniform(-1.5, 0.5, n),
        "alpha_m_apogee": rng.uniform(-0.1, 0.4, n),
    }
    for col in _xp_norm_columns():
        base[col] = rng.standard_normal(n).astype(np.float32)
    return pd.DataFrame(base)


# --- check_xp_feature_nan_invariant ------------------------------------------


def test_xp_feature_nan_invariant_passes_on_clean_population() -> None:
    res = check_xp_feature_nan_invariant(_make_clean_df())
    assert isinstance(res, CheckResult)
    assert res.passed
    assert res.level == "HARD"
    assert res.details["n_flagged_out_rows"] == 0


def test_xp_feature_nan_invariant_passes_when_nan_rows_are_flagged() -> None:
    """NaN is allowed when ye2024_flag != 0; that's the documented invariant."""
    df = _make_clean_df()
    df.loc[0, "ye2024_flag"] = 1
    df.loc[0, "bp_coef_norm_1"] = np.nan
    df.loc[0, "rp_c0_z"] = np.nan
    res = check_xp_feature_nan_invariant(df)
    assert res.passed
    assert res.details["n_flagged_out_rows"] == 1


def test_xp_feature_nan_invariant_fails_on_unflagged_nan() -> None:
    """NaN on a flag-clean row is a hard-fail: silent emit-side row loss."""
    df = _make_clean_df()
    df.loc[0, "bp_coef_norm_5"] = np.nan
    res = check_xp_feature_nan_invariant(df)
    assert not res.passed
    assert "bp_coef_norm_5" in res.details["surprise_counts_per_column"]


def test_xp_feature_nan_invariant_treats_nonpositive_c0_as_flagged() -> None:
    """bp_coef_0 <= 0 is a recognised flagged condition per the docstring."""
    df = _make_clean_df()
    df.loc[1, "bp_coef_0"] = 0.0
    df.loc[1, "bp_c0_z"] = np.nan  # NaN is then expected
    res = check_xp_feature_nan_invariant(df)
    assert res.passed


# --- check_tier1_label_completeness ------------------------------------------


def test_tier1_completeness_passes_on_clean_population() -> None:
    res = check_tier1_label_completeness(_make_clean_df())
    assert res.passed
    assert res.level == "HARD"


def test_tier1_completeness_skips_flag_bad_rows() -> None:
    """flag_bad==1 rows are excluded from the gate."""
    df = _make_clean_df()
    df.loc[0, "flag_bad"] = True
    df.loc[0, "teff_apogee"] = np.nan
    res = check_tier1_label_completeness(df)
    assert res.passed


def test_tier1_completeness_fails_when_atmospheric_label_missing_on_clean_row() -> None:
    df = _make_clean_df()
    df.loc[0, "logg_apogee"] = np.nan  # flag_bad=0 row -> contract violation
    res = check_tier1_label_completeness(df)
    assert not res.passed


def test_tier1_atmospheric_set_excludes_fe_h() -> None:
    """fe_h_apogee is intentionally NOT in the Tier-1 gate (DR19 realism)."""
    assert TIER1_ATMOSPHERIC == ("teff_apogee", "logg_apogee", "mh_apogee")
    assert "fe_h_apogee" not in TIER1_ATMOSPHERIC


# --- check_parameter_bounds --------------------------------------------------


def test_parameter_bounds_passes_on_clean_population() -> None:
    res = check_parameter_bounds(_make_clean_df())
    assert res.passed
    assert res.level == "HARD"


def test_parameter_bounds_fails_on_out_of_range_value() -> None:
    df = _make_clean_df()
    df.loc[0, "teff_apogee"] = 9000.0  # > 8000 K upper bound
    res = check_parameter_bounds(df)
    assert not res.passed
    assert "teff_apogee" in res.details["violations"]
    v = res.details["violations"]["teff_apogee"]
    assert v["above_max"] == 1
    assert v["below_min"] == 0


def test_parameter_bounds_dr19_calibrated_envelopes() -> None:
    """The bounds are documented as DR19-calibrated, not textbook-generic."""
    assert PARAMETER_BOUNDS["alpha_m_apogee"] == (-0.8, 0.8)
    assert PARAMETER_BOUNDS["fe_h_apogee"] == (-4.0, 1.1)
    assert PARAMETER_BOUNDS["mh_apogee"] == (-4.0, 1.0)


def test_parameter_bounds_skips_nan_values() -> None:
    """NaN cannot violate a bound. Coverage gate is owned by the completeness check."""
    df = _make_clean_df()
    df.loc[0, "alpha_m_apogee"] = np.nan
    res = check_parameter_bounds(df)
    assert res.passed


# --- check_per_element_nan_rates --------------------------------------------


def test_per_element_nan_rates_is_soft_diagnostic_always_passing() -> None:
    """Documented as report-only: passed=True regardless of NaN rates."""
    df = _make_clean_df()
    res = check_per_element_nan_rates(df)
    assert res.level == "SOFT"
    assert res.passed
    assert "fe_h_apogee" in res.details["rates"]


def test_per_element_nan_rates_records_baseline() -> None:
    """The point of this check is to freeze the baseline; values must round-trip."""
    df = _make_clean_df(n=10)
    df.loc[0:1, "fe_h_apogee"] = np.nan  # 2/10 NaN
    res = check_per_element_nan_rates(df)
    rate = res.details["rates"]["fe_h_apogee"]
    assert rate["n_nan"] == 2
    assert rate["rate"] == pytest.approx(0.2)


# --- CheckResult contract ----------------------------------------------------


def test_check_result_is_frozen_dataclass() -> None:
    res = CheckResult(name="x", level="HARD", passed=True, summary="ok")
    with pytest.raises(Exception):
        res.passed = False  # type: ignore[misc]
    assert res.details == {}
