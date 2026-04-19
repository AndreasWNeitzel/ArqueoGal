"""Tests for population_classifier.main.features — §10.2/§10.3 vectorisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arqueogal.population_classifier.main.features import (
    BASELINE_NEITZEL2025_COLUMNS,
    MAIN_FEATURE_COLUMNS,
    FeatureMatrix,
    FeatureSpec,
    apply_c_n_gate,
    build_feature_matrix,
    standardize,
)


def _synth_frame(n: int = 100, *, with_c_n: bool = True, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {
        "source_id": np.arange(n, dtype=np.int64),
        "age": rng.uniform(1, 12, n),
        "fe_h": rng.normal(-0.2, 0.3, n),
        "mg_fe": rng.normal(0.05, 0.1, n),
        "al_fe": rng.normal(0.0, 0.1, n),
        "J_R": rng.uniform(0, 50, n),
        "J_z": rng.uniform(0, 50, n),
        "L_z": rng.normal(1500, 300, n),
        "ecc": rng.uniform(0, 1, n),
        "E": rng.normal(-1.5e5, 5e4, n),
        "alpha_m": rng.normal(0.05, 0.1, n),
        "v_phi": rng.normal(200, 50, n),
        "sqrt_u2_plus_w2": rng.uniform(0, 100, n),
    }
    if with_c_n:
        cols["c_n"] = rng.normal(0.0, 0.3, n)
    return pd.DataFrame(cols)


# --- presets ---------------------------------------------------------------

def test_feature_spec_main_has_ten_columns() -> None:
    assert len(MAIN_FEATURE_COLUMNS) == 10
    assert "c_n" in MAIN_FEATURE_COLUMNS
    spec = FeatureSpec.main()
    assert "c_n" in spec.gated_columns
    assert spec.name == "main"


def test_feature_spec_baseline_has_five_columns() -> None:
    assert len(BASELINE_NEITZEL2025_COLUMNS) == 5
    spec = FeatureSpec.baseline_neitzel2025()
    assert spec.gated_columns == ()


# --- evolutionary-stage gating ---------------------------------------------

def test_apply_c_n_gate_masks_non_rgb_stars() -> None:
    df = _synth_frame(20)
    probs = np.array([0.1] * 10 + [0.9] * 10)
    gated = apply_c_n_gate(df, evol_stage_probs=probs, rgb_prob_threshold=0.5)
    assert np.all(np.isnan(gated["c_n"].iloc[:10]))
    assert np.all(np.isfinite(gated["c_n"].iloc[10:]))


def test_apply_c_n_gate_noop_when_probs_none() -> None:
    df = _synth_frame(5)
    assert apply_c_n_gate(df, evol_stage_probs=None).equals(df)


def test_apply_c_n_gate_length_mismatch_raises() -> None:
    df = _synth_frame(5)
    with pytest.raises(ValueError, match="evol_stage_probs length"):
        apply_c_n_gate(df, evol_stage_probs=np.zeros(10))


def test_apply_c_n_gate_skips_when_column_absent() -> None:
    df = _synth_frame(5, with_c_n=False)
    out = apply_c_n_gate(df, evol_stage_probs=np.zeros(5))
    assert out.equals(df)


# --- standardize -----------------------------------------------------------

def test_standardize_fits_and_applies() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(5.0, 3.0, size=(200, 4)).astype(np.float32)
    X_s, mu, sd = standardize(X)
    assert X_s.shape == X.shape
    assert np.allclose(X_s.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(X_s.std(axis=0), 1.0, atol=1e-5)
    # Re-apply with stored params — exact same numeric result.
    X_s2, _, _ = standardize(X, mean=mu, std=sd)
    assert np.allclose(X_s, X_s2)


def test_standardize_handles_zero_variance_column() -> None:
    X = np.ones((50, 3), dtype=np.float32)
    X[:, 1] = np.arange(50)
    X_s, mu, sd = standardize(X)
    assert np.isfinite(X_s).all()
    # Constant column should pass through shifted to zero, not NaN.
    assert np.allclose(X_s[:, 0], 0.0)


# --- build_feature_matrix --------------------------------------------------

def test_build_feature_matrix_main_shape() -> None:
    df = _synth_frame(100)
    fm = build_feature_matrix(df)
    assert isinstance(fm, FeatureMatrix)
    assert fm.X.shape == (100, 10)
    assert fm.columns == MAIN_FEATURE_COLUMNS
    assert fm.include_mask.sum() == 100
    assert fm.n_features == 10


def test_build_feature_matrix_baseline_shape() -> None:
    df = _synth_frame(50)
    fm = build_feature_matrix(df, spec=FeatureSpec.baseline_neitzel2025())
    assert fm.X.shape == (50, 5)
    assert fm.spec_name == "baseline_neitzel2025"


def test_build_feature_matrix_excludes_nan_rows() -> None:
    df = _synth_frame(30)
    df.loc[0:4, "age"] = np.nan  # 5 rows with NaN age
    fm = build_feature_matrix(df)
    assert fm.include_mask.sum() == 25
    assert fm.X.shape == (25, 10)


def test_build_feature_matrix_gates_c_n_via_evol_stage() -> None:
    df = _synth_frame(20)
    probs = np.array([0.1] * 10 + [0.9] * 10)
    fm = build_feature_matrix(df, evol_stage_probs=probs)
    # First 10 stars got c_n masked → dropped from included sample.
    assert fm.include_mask[:10].sum() == 0
    assert fm.include_mask[10:].sum() == 10


def test_build_feature_matrix_missing_column_raises() -> None:
    df = _synth_frame(10).drop(columns=["mg_fe"])
    with pytest.raises(ValueError, match="missing required columns"):
        build_feature_matrix(df)


def test_build_feature_matrix_inference_path_reuses_scaler() -> None:
    df_train = _synth_frame(50, seed=0)
    df_infer = _synth_frame(50, seed=1)
    fm_train = build_feature_matrix(df_train)
    fm_infer = build_feature_matrix(
        df_infer, fit_scaler=False, mean=fm_train.mean, std=fm_train.std,
    )
    # Inference scaler matches training's.
    assert np.allclose(fm_infer.mean, fm_train.mean)
    assert np.allclose(fm_infer.std, fm_train.std)


def test_build_feature_matrix_inference_without_params_raises() -> None:
    df = _synth_frame(10)
    with pytest.raises(ValueError, match="fit_scaler=False"):
        build_feature_matrix(df, fit_scaler=False)


def test_feature_matrix_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="X must be 2-D"):
        FeatureMatrix(
            X=np.zeros(10, dtype=np.float32), columns=("a", "b"),
            mean=np.zeros(2), std=np.ones(2), include_mask=np.ones(10, dtype=bool),
        )
    with pytest.raises(ValueError, match="cols"):
        FeatureMatrix(
            X=np.zeros((5, 3), dtype=np.float32), columns=("a", "b"),
            mean=np.zeros(2), std=np.ones(2), include_mask=np.ones(5, dtype=bool),
        )
