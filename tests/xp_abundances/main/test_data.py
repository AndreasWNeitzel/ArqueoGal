"""Tests for xp_abundances.main.data — layout, tiers, stratified split, loader.

Covers the 2026-04-18 frozen feature contract: flat scalar XP columns
(``bp_coef_norm_1..54`` / ``rp_coef_norm_1..54`` + ``bp_c0_z`` / ``rp_c0_z``),
reprojection-residual features, multi-column extinction auxiliaries, and
the ``*_apogee`` label suffix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arqueogal.xp_abundances.main.data import (
    DEFAULT_AUX_COLS,
    DEFAULT_RESIDUAL_COLS,
    DEFAULT_XP_COEF_INDICES,
    DEFAULT_XP_SCALAR_COLS,
    FeatureLayout,
    LabelScaler,
    LabelTiers,
    XpAbundanceDataset,
    load_arrays,
    stratified_split_ids,
)


def test_feature_layout_input_dim_matches_components() -> None:
    layout = FeatureLayout()
    expected = (
        len(layout.xp_bp_indices)
        + len(layout.xp_rp_indices)
        + len(layout.xp_scalar_cols)
        + len(layout.residual_cols)
        + len(layout.aux_cols)
    )
    assert layout.input_dim == expected


def test_feature_layout_required_column_order() -> None:
    layout = FeatureLayout()
    cols = layout.all_required_columns
    n_bp = len(layout.xp_bp_indices)
    n_rp = len(layout.xp_rp_indices)
    n_scalar = len(layout.xp_scalar_cols)
    n_resid = len(layout.residual_cols)
    assert cols[:n_bp] == layout.bp_coef_cols
    assert cols[n_bp : n_bp + n_rp] == layout.rp_coef_cols
    assert cols[n_bp + n_rp : n_bp + n_rp + n_scalar] == layout.xp_scalar_cols
    assert cols[n_bp + n_rp + n_scalar : n_bp + n_rp + n_scalar + n_resid] == layout.residual_cols
    assert cols[-len(layout.aux_cols) :] == layout.aux_cols


def test_feature_layout_defaults_cover_1_to_54() -> None:
    layout = FeatureLayout()
    assert layout.xp_bp_indices == DEFAULT_XP_COEF_INDICES
    assert layout.xp_rp_indices == DEFAULT_XP_COEF_INDICES
    assert layout.xp_bp_indices[0] == 1
    assert layout.xp_bp_indices[-1] == 54
    assert len(layout.xp_bp_indices) == 54


def test_feature_layout_bp_rp_coef_col_names() -> None:
    layout = FeatureLayout(xp_bp_indices=(1, 2, 3), xp_rp_indices=(1, 4))
    assert layout.bp_coef_cols == ("bp_coef_norm_1", "bp_coef_norm_2", "bp_coef_norm_3")
    assert layout.rp_coef_cols == ("rp_coef_norm_1", "rp_coef_norm_4")


def test_feature_layout_truncated_43d() -> None:
    layout = FeatureLayout.truncated_43d()
    assert layout.xp_bp_indices == tuple(range(1, 20))
    assert layout.xp_rp_indices == tuple(range(1, 23))
    assert len(layout.bp_coef_cols) + len(layout.rp_coef_cols) == 19 + 22 == 41
    # Defaults preserved for the non-XP blocks
    assert layout.xp_scalar_cols == DEFAULT_XP_SCALAR_COLS
    assert layout.residual_cols == DEFAULT_RESIDUAL_COLS
    assert layout.aux_cols == DEFAULT_AUX_COLS


def test_feature_layout_truncated_43d_accepts_overrides() -> None:
    layout = FeatureLayout.truncated_43d(aux_cols=("ruwe",))
    assert layout.xp_bp_indices == tuple(range(1, 20))
    assert layout.aux_cols == ("ruwe",)


def test_label_tiers_shape() -> None:
    tiers = LabelTiers()
    assert tiers.n_labels == sum(tiers.tier_sizes)
    assert tiers.tier_sizes == (
        len(tiers.tier1),
        len(tiers.tier2),
        len(tiers.tier3),
    )
    assert len(tiers.all_labels) == tiers.n_labels


def test_label_tiers_all_labels_are_apogee_suffixed() -> None:
    tiers = LabelTiers()
    for name in tiers.all_labels:
        assert name.endswith("_apogee"), f"{name} missing *_apogee suffix"


def test_label_tiers_error_columns_match_labels() -> None:
    tiers = LabelTiers()
    err_cols = tiers.label_error_columns()
    assert len(err_cols) == tiers.n_labels
    for name, e_name in zip(tiers.all_labels, err_cols, strict=True):
        assert e_name == f"e_{name}"


def _synth_frame(n: int, layout: FeatureLayout, tiers: LabelTiers) -> pd.DataFrame:
    """Build a synthetic feature frame matching the flat-scalar contract."""
    rng = np.random.default_rng(0)
    cols: dict[str, np.ndarray] = {
        "source_id": np.arange(1, n + 1, dtype=np.int64),
    }
    for name in layout.bp_coef_cols:
        cols[name] = rng.uniform(-1.0, 1.0, n).astype(np.float32)
    for name in layout.rp_coef_cols:
        cols[name] = rng.uniform(-1.0, 1.0, n).astype(np.float32)
    for name in layout.xp_scalar_cols:
        cols[name] = rng.uniform(-2.0, 2.0, n).astype(np.float32)
    for name in layout.residual_cols:
        cols[name] = rng.uniform(0.0, 1e-14, n).astype(np.float32)
    for name in layout.aux_cols:
        cols[name] = rng.uniform(0.0, 1.0, n).astype(np.float32)
    for label in tiers.all_labels:
        cols[label] = rng.uniform(-1.0, 1.0, n).astype(np.float32)
    for e_col in tiers.label_error_columns():
        cols[e_col] = rng.uniform(0.01, 0.1, n).astype(np.float32)
    cols["b_deg"] = rng.uniform(-60.0, 60.0, n).astype(np.float32)
    return pd.DataFrame(cols)


def test_load_arrays_roundtrip(tmp_path: Path) -> None:
    layout = FeatureLayout()
    tiers = LabelTiers()
    df = _synth_frame(20, layout, tiers)
    path = tmp_path / "t.parquet"
    df.to_parquet(path, index=False)

    arrs = load_arrays(path, layout, tiers)
    assert arrs["X"].shape == (20, layout.input_dim)
    assert arrs["Y"].shape == (20, tiers.n_labels)
    assert arrs["sigma_Y"].shape == (20, tiers.n_labels)
    assert arrs["source_id"].shape == (20,)
    assert arrs["X"].dtype == np.float32
    assert arrs["source_id"].dtype == np.int64


def test_load_arrays_flat_column_order(tmp_path: Path) -> None:
    """Encoder-input order: BP coef → RP coef → scalars → residuals → aux."""
    layout = FeatureLayout(
        xp_bp_indices=(1, 2),
        xp_rp_indices=(1,),
        xp_scalar_cols=("bp_c0_z",),
        residual_cols=("reprojection_residual_rms",),
        aux_cols=("ruwe",),
    )
    tiers = LabelTiers(tier1=("teff_apogee",), tier2=(), tier3=())
    df = _synth_frame(5, layout, tiers)
    path = tmp_path / "t.parquet"
    df.to_parquet(path, index=False)

    arrs = load_arrays(path, layout, tiers)
    X = arrs["X"]
    # 2 (bp) + 1 (rp) + 1 (scalar) + 1 (residual) + 1 (aux) = 6
    assert X.shape == (5, 6)
    np.testing.assert_allclose(X[:, 0], df["bp_coef_norm_1"].to_numpy(np.float32))
    np.testing.assert_allclose(X[:, 1], df["bp_coef_norm_2"].to_numpy(np.float32))
    np.testing.assert_allclose(X[:, 2], df["rp_coef_norm_1"].to_numpy(np.float32))
    np.testing.assert_allclose(X[:, 3], df["bp_c0_z"].to_numpy(np.float32))
    np.testing.assert_allclose(X[:, 4], df["reprojection_residual_rms"].to_numpy(np.float32))
    np.testing.assert_allclose(X[:, 5], df["ruwe"].to_numpy(np.float32))


def test_load_arrays_missing_column_raises(tmp_path: Path) -> None:
    layout = FeatureLayout(
        xp_bp_indices=(1,),
        xp_rp_indices=(),
        xp_scalar_cols=(),
        residual_cols=(),
        aux_cols=(),
    )
    tiers = LabelTiers(tier1=("teff_apogee",), tier2=(), tier3=())
    df = pd.DataFrame(
        {
            "source_id": [1, 2, 3],
            # missing bp_coef_norm_1
            "teff_apogee": [1.0, 2.0, 3.0],
            "e_teff_apogee": [0.1, 0.2, 0.3],
        }
    )
    path = tmp_path / "t.parquet"
    df.to_parquet(path, index=False)
    with pytest.raises((KeyError, ValueError)):
        load_arrays(path, layout, tiers)


def test_stratified_split_partitions_all_source_ids() -> None:
    rng = np.random.default_rng(42)
    n = 400
    df = pd.DataFrame(
        {
            "source_id": np.arange(1, n + 1, dtype=np.int64),
            "fe_h_apogee": rng.normal(-0.2, 0.3, n),
            "teff_apogee": rng.uniform(4000, 5500, n),
            "b_deg": rng.uniform(-60, 60, n),
        }
    )
    splits = stratified_split_ids(df, seed=0)
    total = sum(len(v) for v in splits.values())
    assert total == n
    all_ids = np.concatenate(list(splits.values()))
    assert len(np.unique(all_ids)) == n
    assert set(all_ids) == set(df["source_id"])


def test_stratified_split_reproducible() -> None:
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "source_id": np.arange(1, n + 1, dtype=np.int64),
            "fe_h_apogee": rng.normal(-0.2, 0.3, n),
            "teff_apogee": rng.uniform(4000, 5500, n),
            "b_deg": rng.uniform(-60, 60, n),
        }
    )
    a = stratified_split_ids(df, seed=123)
    b = stratified_split_ids(df, seed=123)
    for k in ("train", "val", "test"):
        np.testing.assert_array_equal(a[k], b[k])


def test_stratified_split_fracs_approximate() -> None:
    rng = np.random.default_rng(42)
    n = 2000
    df = pd.DataFrame(
        {
            "source_id": np.arange(1, n + 1, dtype=np.int64),
            "fe_h_apogee": rng.normal(-0.2, 0.3, n),
            "teff_apogee": rng.uniform(4000, 5500, n),
            "b_deg": rng.uniform(-60, 60, n),
        }
    )
    splits = stratified_split_ids(df, fracs=(0.70, 0.15, 0.15), seed=0)
    assert abs(len(splits["train"]) / n - 0.70) < 0.03
    assert abs(len(splits["val"]) / n - 0.15) < 0.03
    assert abs(len(splits["test"]) / n - 0.15) < 0.03


def test_stratified_split_rejects_bad_fracs() -> None:
    df = pd.DataFrame(
        {
            "source_id": [1, 2, 3],
            "fe_h_apogee": [0.0, 0.1, 0.2],
            "teff_apogee": [4800, 4900, 5000],
            "b_deg": [10, 20, 30],
        }
    )
    with pytest.raises(ValueError, match="fracs must sum"):
        stratified_split_ids(df, fracs=(0.6, 0.2, 0.1))


def test_stratified_split_handles_dec_deg_fallback_for_latitude() -> None:
    rng = np.random.default_rng(42)
    n = 80
    df = pd.DataFrame(
        {
            "source_id": np.arange(1, n + 1, dtype=np.int64),
            "fe_h_apogee": rng.normal(-0.2, 0.3, n),
            "teff_apogee": rng.uniform(4000, 5500, n),
            "dec_deg": rng.uniform(-60, 60, n),  # b_deg missing → falls back to dec_deg
        }
    )
    splits = stratified_split_ids(df, seed=0)
    assert sum(len(v) for v in splits.values()) == n


def test_dataset_returns_tensor_tuples() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 8)).astype(np.float32)
    Y = rng.normal(size=(10, 3)).astype(np.float32)
    sigma = rng.uniform(0.01, 0.1, size=(10, 3)).astype(np.float32)
    ds = XpAbundanceDataset(X=X, Y=Y, sigma_Y=sigma)
    assert len(ds) == 10
    x, y, s = ds[3]
    assert x.shape == (8,)
    assert y.shape == (3,)
    assert s.shape == (3,)


def test_dataset_no_sigma_returns_pair() -> None:
    X = np.zeros((4, 5), dtype=np.float32)
    Y = np.zeros((4, 2), dtype=np.float32)
    ds = XpAbundanceDataset(X=X, Y=Y)
    item = ds[0]
    assert len(item) == 2


def test_dataset_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        XpAbundanceDataset(
            X=np.zeros((4, 5), dtype=np.float32),
            Y=np.zeros((5, 2), dtype=np.float32),
        )


def test_default_aux_cols_exposed() -> None:
    layout = FeatureLayout()
    assert layout.aux_cols == DEFAULT_AUX_COLS


def test_default_residual_cols_exposed() -> None:
    layout = FeatureLayout()
    assert layout.residual_cols == DEFAULT_RESIDUAL_COLS


# --- LabelScaler ------------------------------------------------------------


def _synthetic_labels(seed: int = 0) -> tuple[np.ndarray, tuple[str, ...]]:
    """Three-label matrix with heterogeneous scales and some NaN."""
    rng = np.random.default_rng(seed)
    n = 200
    teff = rng.normal(4600.0, 280.0, n).astype(np.float32)
    logg = rng.normal(2.4, 0.5, n).astype(np.float32)
    mh = rng.normal(-0.3, 0.35, n).astype(np.float32)
    mh[rng.random(n) < 0.1] = np.nan  # sprinkle missing labels
    y = np.column_stack([teff, logg, mh])
    return y, ("teff_apogee", "logg_apogee", "mh_apogee")


def test_label_scaler_fit_matches_nanaware_mean_std() -> None:
    y, names = _synthetic_labels()
    scaler = LabelScaler.fit(y, names)

    for j in range(3):
        col = y[:, j]
        col = col[np.isfinite(col)]
        assert np.isclose(scaler.mean[j], col.mean(), atol=1e-5)
        assert np.isclose(scaler.scale[j], col.std(ddof=0), atol=1e-5)


def test_label_scaler_transform_preserves_nan() -> None:
    y, names = _synthetic_labels()
    scaler = LabelScaler.fit(y, names)
    z = scaler.transform(y)
    nan_mask = ~np.isfinite(y)
    assert np.all(nan_mask == ~np.isfinite(z))


def test_label_scaler_inverse_mean_roundtrips() -> None:
    y, names = _synthetic_labels()
    scaler = LabelScaler.fit(y, names)
    z = scaler.transform(y)
    back = scaler.inverse_mean(z)
    finite = np.isfinite(y)
    np.testing.assert_allclose(back[finite], y[finite], rtol=1e-5, atol=1e-4)


def test_label_scaler_inverse_L_covariance_is_analytic() -> None:
    """Verify Σ_raw[i,j] = scale[i] * scale[j] * Σ_scaled[i,j]."""
    rng = np.random.default_rng(1)
    n = 3
    # Random lower-triangular Cholesky factor in scaled space.
    a = rng.normal(size=(n, n)).astype(np.float32)
    L_scaled = np.tril(a)
    L_scaled[np.arange(n), np.arange(n)] = np.abs(L_scaled[np.arange(n), np.arange(n)]) + 0.1
    sigma_scaled = L_scaled @ L_scaled.T

    scaler = LabelScaler(
        mean=np.array([4600.0, 2.4, -0.3], dtype=np.float32),
        scale=np.array([280.0, 0.5, 0.35], dtype=np.float32),
        label_names=("teff_apogee", "logg_apogee", "mh_apogee"),
    )
    L_raw = scaler.inverse_L(L_scaled)
    sigma_raw = L_raw @ L_raw.T

    s = scaler.scale
    expected = sigma_scaled * s[:, None] * s[None, :]
    np.testing.assert_allclose(sigma_raw, expected, rtol=1e-5, atol=1e-5)


def test_label_scaler_inverse_L_handles_batch() -> None:
    rng = np.random.default_rng(2)
    B, n = 4, 3
    a = rng.normal(size=(B, n, n)).astype(np.float32)
    L_scaled = np.tril(a)
    diag_idx = np.arange(n)
    L_scaled[:, diag_idx, diag_idx] = np.abs(L_scaled[:, diag_idx, diag_idx]) + 0.1

    scaler = LabelScaler(
        mean=np.zeros(n, dtype=np.float32),
        scale=np.array([280.0, 0.5, 0.35], dtype=np.float32),
        label_names=("teff_apogee", "logg_apogee", "mh_apogee"),
    )
    L_raw = scaler.inverse_L(L_scaled)
    for b in range(B):
        single = scaler.inverse_L(L_scaled[b])
        np.testing.assert_allclose(L_raw[b], single, rtol=0, atol=0)


def test_label_scaler_reorder_to_permutes_mean_and_scale() -> None:
    scaler = LabelScaler(
        mean=np.array([4600.0, 2.4, -0.3], dtype=np.float32),
        scale=np.array([280.0, 0.5, 0.35], dtype=np.float32),
        label_names=("teff_apogee", "logg_apogee", "mh_apogee"),
    )
    reordered = scaler.reorder_to(("mh_apogee", "teff_apogee", "logg_apogee"))
    assert reordered.label_names == ("mh_apogee", "teff_apogee", "logg_apogee")
    np.testing.assert_allclose(reordered.mean, [-0.3, 4600.0, 2.4])
    np.testing.assert_allclose(reordered.scale, [0.35, 280.0, 0.5])


def test_label_scaler_is_default_detects_placeholder() -> None:
    default = LabelScaler(
        mean=np.zeros(3, dtype=np.float32),
        scale=np.ones(3, dtype=np.float32),
        label_names=("a", "b", "c"),
    )
    assert default.is_default()
    fit_scaler = LabelScaler(
        mean=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        scale=np.array([0.35, 1.0, 1.0], dtype=np.float32),
        label_names=("a", "b", "c"),
    )
    assert not fit_scaler.is_default()


def test_label_scaler_fit_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="2-D label matrix"):
        LabelScaler.fit(np.zeros(5, dtype=np.float32), ("a",))
    with pytest.raises(ValueError, match="columns"):
        LabelScaler.fit(np.zeros((5, 2), dtype=np.float32), ("a", "b", "c"))


def test_label_scaler_all_nan_column_gets_default_scale() -> None:
    y = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]], dtype=np.float32)
    scaler = LabelScaler.fit(y, ("a", "b"))
    # All-NaN column gets mean=0, scale=1 (the placeholder — safe no-op).
    assert scaler.mean[1] == 0.0
    assert scaler.scale[1] == 1.0
