"""Offline tests for arqueogal.data.master_schema — §10 contract checks.

``PIPELINE2_FEATURES_SCHEMA`` is a legacy schema retained in this repo for
historical compatibility; the active chrono-chemo-kinematic feature
contract now lives in Starfold (separate repo). The tests below continue
to exercise the in-repo schema to prevent accidental breakage of the
historical definition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arqueogal.data.master_schema import (
    APOGEE_ELEMENT_LABELS,
    GAIA_ASTROMETRY_COV_COLS,
    PIPELINE1_INFERENCE_SCHEMA,
    PIPELINE1_TRAINING_SCHEMA,
    PIPELINE2_FEATURES_SCHEMA,
    SCHEMAS,
    XP_ARRAY_COLS,
    XP_N_COEFFS,
    XP_SCALAR_COLS,
    MasterSchema,
    SchemaError,
)

# ---- module-level constants --------------------------------------------------


def test_ten_gaia_astrometry_cov_columns() -> None:
    """§3.6 / gaia_enrich.py ships the 10 upper-triangular correlations."""
    assert len(GAIA_ASTROMETRY_COV_COLS) == 10
    # Spot-check one from each row of the 5x5 upper triangle.
    for col in ("ra_dec_corr", "parallax_pmra_corr", "pmra_pmdec_corr"):
        assert col in GAIA_ASTROMETRY_COV_COLS


def test_xp_array_and_scalar_cols_are_named() -> None:
    assert "bp_coeffs_norm" in XP_ARRAY_COLS
    assert "rp_coeffs_norm" in XP_ARRAY_COLS
    assert "bp_c0_z" in XP_SCALAR_COLS
    assert "rp_c0_z" in XP_SCALAR_COLS
    assert XP_N_COEFFS == 55


def test_apogee_labels_cover_all_elements() -> None:
    """Every element in apogee_dr19.ABUNDANCE_ELEMENTS has an _h_apogee +
    error pair."""
    elements = (
        "c",
        "n",
        "o",
        "na",
        "mg",
        "al",
        "si",
        "s",
        "k",
        "ca",
        "ti",
        "v",
        "cr",
        "mn",
        "fe",
        "ni",
        "ce",
    )
    for el in elements:
        assert f"{el}_h_apogee" in APOGEE_ELEMENT_LABELS
        assert f"e_{el}_h_apogee" in APOGEE_ELEMENT_LABELS
    assert len(APOGEE_ELEMENT_LABELS) == 2 * len(elements)


# ---- schema registry --------------------------------------------------------


def test_schema_registry_has_three_entries() -> None:
    assert set(SCHEMAS) == {
        "pipeline1_training",
        "pipeline1_inference",
        "pipeline2_features",
    }


def test_training_requires_source_id_and_apogee_labels() -> None:
    req = set(PIPELINE1_TRAINING_SCHEMA.required)
    assert {"source_id", "apogee_id", "sdss_id"} <= req
    assert {"teff_apogee", "logg_apogee", "mh_apogee", "alpha_m_apogee"} <= req
    for label in APOGEE_ELEMENT_LABELS:
        assert label in req


def test_inference_excludes_apogee_labels() -> None:
    req = set(PIPELINE1_INFERENCE_SCHEMA.required)
    assert "source_id" in req
    # APOGEE labels must NOT be required on the inference side.
    for label in ("teff_apogee", "logg_apogee", "mh_apogee"):
        assert label not in req
    for label in APOGEE_ELEMENT_LABELS:
        assert label not in req
    # Andrae+2023 diagnostics are required on the inference side.
    for col in ("teff_xgboost", "logg_xgboost", "mh_xgboost"):
        assert col in req


def test_pipeline2_requires_kinematics_and_chemistry() -> None:
    req = set(PIPELINE2_FEATURES_SCHEMA.required)
    for col in ("J_R", "J_z", "L_z", "ecc", "E"):
        assert col in req
    for col in ("fe_h", "mg_fe", "al_fe", "c_n"):
        assert col in req
    # Neitzel+2025 backwards-compat columns still required.
    assert "v_phi" in req
    assert "sqrt_u2_plus_w2" in req


def test_pipeline2_age_is_optional() -> None:
    """Task 4 (asteroseismic ages) has not landed — schema must not block."""
    opt = set(PIPELINE2_FEATURES_SCHEMA.optional)
    assert "age" in opt
    assert "age_err" in opt
    assert "age" not in set(PIPELINE2_FEATURES_SCHEMA.required)


def test_training_and_inference_share_xp_schema() -> None:
    for schema in (PIPELINE1_TRAINING_SCHEMA, PIPELINE1_INFERENCE_SCHEMA):
        assert schema.array_cols == XP_ARRAY_COLS
        assert schema.array_length == XP_N_COEFFS


# ---- validate() --------------------------------------------------------------


def _training_frame(n: int = 3) -> pd.DataFrame:
    row = dict.fromkeys(PIPELINE1_TRAINING_SCHEMA.required, 0.0)
    row["source_id"] = 1
    row["apogee_id"] = "2M000"
    row["sdss_id"] = 42
    df = pd.DataFrame([row] * n)
    for col in XP_ARRAY_COLS:
        df[col] = [np.zeros(XP_N_COEFFS, dtype=np.float32) for _ in range(n)]
    return df


def test_validate_passes_on_complete_training_frame() -> None:
    df = _training_frame()
    PIPELINE1_TRAINING_SCHEMA.validate(df)  # no raise
    PIPELINE1_TRAINING_SCHEMA.validate(df, check_array_lengths=True)


def test_validate_reports_missing_required_columns() -> None:
    df = _training_frame().drop(columns=["teff_apogee"])
    with pytest.raises(SchemaError, match="teff_apogee"):
        PIPELINE1_TRAINING_SCHEMA.validate(df)


def test_validate_reports_xp_array_length_mismatch() -> None:
    df = _training_frame()
    df.at[0, "bp_coeffs_norm"] = np.zeros(10, dtype=np.float32)  # wrong length
    with pytest.raises(SchemaError, match=r"bp_coeffs_norm.*length"):
        PIPELINE1_TRAINING_SCHEMA.validate(df, check_array_lengths=True)


def test_validate_skips_array_length_check_by_default() -> None:
    """Without the opt-in, a malformed array cell does NOT raise — it's too
    expensive to pay for every downstream validate()."""
    df = _training_frame()
    df.at[0, "bp_coeffs_norm"] = np.zeros(10, dtype=np.float32)
    PIPELINE1_TRAINING_SCHEMA.validate(df)  # no raise


def test_validate_allows_extra_columns() -> None:
    df = _training_frame()
    df["some_derived_feature"] = 3.14
    PIPELINE1_TRAINING_SCHEMA.validate(df)  # extras are fine


def test_validate_pipeline2_frame() -> None:
    req = PIPELINE2_FEATURES_SCHEMA.required
    df = pd.DataFrame({col: [0.0, 0.0] for col in req})
    PIPELINE2_FEATURES_SCHEMA.validate(df)
    # Missing a required kinematic column → error.
    with pytest.raises(SchemaError, match="J_R"):
        PIPELINE2_FEATURES_SCHEMA.validate(df.drop(columns=["J_R"]))


def test_validate_pipeline2_age_may_be_absent() -> None:
    req = PIPELINE2_FEATURES_SCHEMA.required
    df = pd.DataFrame({col: [0.0] for col in req})
    # Age columns not present — optional, still valid.
    assert "age" not in df.columns
    PIPELINE2_FEATURES_SCHEMA.validate(df)


# ---- MasterSchema basics -----------------------------------------------------


def test_all_columns_is_required_plus_optional() -> None:
    s = MasterSchema(
        name="t",
        required=("a", "b"),
        optional=("c",),
    )
    assert s.all_columns == ("a", "b", "c")


def test_schema_error_inherits_value_error() -> None:
    assert issubclass(SchemaError, ValueError)
