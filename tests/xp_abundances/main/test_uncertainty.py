"""Tests for xp_abundances.main.uncertainty — calibration + coverage + conformal."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from arqueogal.xp_abundances.main.data import XpAbundanceDataset
from arqueogal.xp_abundances.main.model import (
    CovarianceBlockLayout,
    ModelConfig,
    XpAbundanceModel,
)
from arqueogal.xp_abundances.main.uncertainty import (
    CalibrationArtifacts,
    GpAlphaBundle,
    RegimeBEnvelope,
    apply_calibration,
    apply_gp_alpha,
    bin_by_cells,
    collect_predictions,
    conformal_nonconformity_scores,
    conformal_radius_at_level,
    coverage_at_levels,
    fit_calibration,
    gp_smoothed_per_cell_per_label_scale,
    isotonic_per_label,
    shrunken_per_cell_per_label_scale,
    temperature_scaling_per_cell,
)


def _sample_from_mvn(mu: np.ndarray, L: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one truth sample per star from N(μ_b, L_b L_bᵀ)."""
    B, n = mu.shape
    z = rng.standard_normal(size=(B, n)).astype(np.float32)
    return mu + np.einsum("bij,bj->bi", L, z)


def _batched_cholesky(
    B: int,
    n: int,
    scale: float = 0.3,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((B, n, n), dtype=np.float32)
    for b in range(B):
        a = np.tril(rng.normal(scale=scale, size=(n, n)).astype(np.float32))
        d = np.abs(np.diag(a)) + 0.1
        np.fill_diagonal(a, d)
        out[b] = a
    return out


# --- bin_by_cells ---


def test_bin_by_cells_produces_expected_n_cells() -> None:
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((200, 2)).astype(np.float32)
    codes, defn = bin_by_cells(feats, n_bins=(3, 4))
    assert codes.shape == (200,)
    assert codes.min() >= 0
    assert codes.max() < 3 * 4
    assert defn["n_bins"] == [3, 4]
    assert len(defn["edges_per_col"]) == 2


def test_bin_by_cells_rejects_wrong_ndim() -> None:
    with pytest.raises(ValueError, match="features must be 2D"):
        bin_by_cells(np.zeros(4), n_bins=(2,))
    with pytest.raises(ValueError, match="n_bins length"):
        bin_by_cells(np.zeros((4, 2)), n_bins=(3, 3, 3))


# --- temperature scaling ---


def test_temperature_scaling_recovers_inflation() -> None:
    """If truth is drawn from Σ_true and model predicts 4·Σ_true, recovered s ≈ 0.5."""
    rng = np.random.default_rng(42)
    B, n = 2000, 4
    mu = np.zeros((B, n), dtype=np.float32)
    L_true = _batched_cholesky(B, n, scale=0.5, seed=1)
    y = _sample_from_mvn(mu, L_true, rng)

    L_pred = L_true * 2.0  # predicted σ is 2× truth → s should be ≈ 0.5
    cells = np.zeros(B, dtype=np.int64)
    s_map = temperature_scaling_per_cell(mu, L_pred, y, cells)
    assert abs(s_map[0] - 0.5) < 0.1


def test_temperature_scaling_small_cell_returns_one() -> None:
    """Cells with < n+2 stars get s=1.0 (under-determined)."""
    mu = np.zeros((3, 4), dtype=np.float32)
    L = _batched_cholesky(3, 4, seed=0)
    y = mu.copy()
    s_map = temperature_scaling_per_cell(mu, L, y, np.zeros(3, dtype=np.int64))
    assert s_map[0] == 1.0


# --- shrunken per-cell-per-label scaling ---


def test_shrunken_scale_recovers_heteroscedastic_inflation() -> None:
    """Two cells with different per-label miscalibration → α ≈ truth per cell/label."""
    rng = np.random.default_rng(7)
    n, B_per_cell = 3, 1500
    # Truth is drawn from N(0, Σ_true) with Σ_true = I (for simplicity).
    mu = np.zeros((2 * B_per_cell, n), dtype=np.float32)
    L_true = np.tile(np.eye(n, dtype=np.float32), (2 * B_per_cell, 1, 1))
    y = _sample_from_mvn(mu, L_true, rng)

    # Model predicts an L that is too-wide by a different factor per (cell, label).
    # cell 0: per-label predicted σ = [1.0, 0.5, 2.0] (labels 1&2 miscalibrated)
    # cell 1: per-label predicted σ = [2.0, 1.0, 0.5]
    L_pred = L_true.copy()
    scales_cell0 = np.array([1.0, 0.5, 2.0])
    scales_cell1 = np.array([2.0, 1.0, 0.5])
    for i in range(3):
        L_pred[:B_per_cell, i, i] = scales_cell0[i]
        L_pred[B_per_cell:, i, i] = scales_cell1[i]

    cell_ids = np.concatenate(
        [np.zeros(B_per_cell, dtype=np.int64), np.ones(B_per_cell, dtype=np.int64)]
    )

    out = shrunken_per_cell_per_label_scale(
        mu,
        L_pred,
        y,
        cell_ids,
        tau=1.0,
        min_cell_stars=8,
    )
    # Expected α: predicted σ is `s`, truth σ is `1.0`, so z = y/s has
    # Var(z) = 1/s². Recovered α = √Var(z) = 1/s. So α_0 ≈ 1/scales_cell0.
    alpha_0 = np.array([out["scales"][(0, j)] for j in range(3)])
    alpha_1 = np.array([out["scales"][(1, j)] for j in range(3)])
    np.testing.assert_allclose(alpha_0, 1.0 / scales_cell0, atol=0.08)
    np.testing.assert_allclose(alpha_1, 1.0 / scales_cell1, atol=0.08)


def test_shrunken_scale_sparse_cell_shrinks_to_global() -> None:
    """Sparse cell (n < min_cell_stars) falls back to global α."""
    rng = np.random.default_rng(0)
    n, B_big, B_small = 3, 500, 5
    mu = np.zeros((B_big + B_small, n), dtype=np.float32)
    L = np.tile(np.eye(n, dtype=np.float32), (B_big + B_small, 1, 1))
    y = _sample_from_mvn(mu, L, rng)
    cell_ids = np.concatenate([np.zeros(B_big, dtype=np.int64), np.ones(B_small, dtype=np.int64)])
    out = shrunken_per_cell_per_label_scale(
        mu,
        L,
        y,
        cell_ids,
        tau=50.0,
        min_cell_stars=8,
    )
    # Sparse cell 1 should have α equal to global α (fallback path).
    for j in range(n):
        assert out["scales"][(1, j)] == pytest.approx(out["global_alpha"][j], abs=1e-6)


def test_shrunken_scale_preserves_pd_via_diag_alpha_L() -> None:
    """L' = diag(α) L is lower-triangular with positive diagonal (⇒ Σ' PD)."""
    rng = np.random.default_rng(1)
    B, n = 200, 4
    mu = np.zeros((B, n), dtype=np.float32)
    L = _batched_cholesky(B, n, scale=0.4, seed=3)
    y = _sample_from_mvn(mu, L, rng)
    out = shrunken_per_cell_per_label_scale(
        mu,
        L,
        y,
        np.zeros(B, dtype=np.int64),
        tau=10.0,
    )
    alpha = out["per_star_alpha"]  # (B, n)
    L_prime = alpha[:, :, None] * L  # diag(α) L
    # lower-triangular preserved
    upper = np.triu(L_prime, k=1)
    assert np.allclose(upper, 0.0, atol=1e-6)
    # diagonal positive
    diag = np.einsum("bii->bi", L_prime)
    assert (diag > 0).all()
    # Σ' = L' L'ᵀ → cholesky must succeed (PD verification).
    Sigma_prime = np.einsum("bij,bkj->bik", L_prime, L_prime)
    for b in range(B):
        np.linalg.cholesky(Sigma_prime[b])  # raises if not PD


def test_shrunken_scale_alpha_one_when_perfectly_calibrated() -> None:
    """When predicted σ matches truth exactly, α_{c,j} ≈ 1 everywhere."""
    rng = np.random.default_rng(9)
    B, n = 1500, 3
    mu = np.zeros((B, n), dtype=np.float32)
    L = _batched_cholesky(B, n, scale=0.3, seed=0)
    y = _sample_from_mvn(mu, L, rng)
    cells = (np.arange(B) % 4).astype(np.int64)  # 4 cells, ~375 stars each
    out = shrunken_per_cell_per_label_scale(mu, L, y, cells, tau=50.0)
    per_star_alpha = out["per_star_alpha"]
    # Within 5% of 1.0 is fine at N=375 per cell.
    assert per_star_alpha.max() < 1.10
    assert per_star_alpha.min() > 0.90


# --- GP-smoothed per-cell-per-label scaling ---


def _make_gp_fixture(
    n_cells_per_axis: int = 3,
    stars_per_cell: int = 80,
    seed: int = 11,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """3-D grid of cells with a smooth per-axis α field in label 0.

    Label 0's true miscalibration factor increases smoothly with feature 0.
    Labels 1-2 are perfectly calibrated. Returns (mu, L, y, features, cell_ids).
    """
    rng = np.random.default_rng(seed)
    n = 3
    axes = np.linspace(-1.0, 1.0, n_cells_per_axis)
    X_list: list[np.ndarray] = []
    cell_list: list[int] = []
    alpha_truth_list: list[float] = []
    for i, a in enumerate(axes):
        for j, b in enumerate(axes):
            for k, c in enumerate(axes):
                cid = (i * n_cells_per_axis + j) * n_cells_per_axis + k
                feat = rng.normal(loc=[a, b, c], scale=0.15, size=(stars_per_cell, 3)).astype(
                    np.float32
                )
                X_list.append(feat)
                cell_list.extend([cid] * stars_per_cell)
                alpha_truth_list.append(1.0 + 0.8 * a)  # smooth with feat 0
    features = np.concatenate(X_list, axis=0)
    cell_ids = np.array(cell_list, dtype=np.int64)

    B = features.shape[0]
    mu = np.zeros((B, n), dtype=np.float32)
    L_true = np.tile(np.eye(n, dtype=np.float32), (B, 1, 1))
    # L_true is identity per star, so y = mu + standard normal.
    y = mu + rng.standard_normal(size=(B, n)).astype(np.float32)
    # Model predicts σ_pred = α_truth * σ_true on label 0 only. Thus
    # z = y/σ_pred has Var(z) = 1/α_truth² → recovered α ≈ 1/α_truth.
    alpha_truth_per_star = np.array(
        [alpha_truth_list[c] for c in cell_ids],
        dtype=np.float32,
    )
    L_pred = L_true.copy()
    L_pred[:, 0, 0] = alpha_truth_per_star
    return mu, L_pred, y, features, cell_ids


def test_gp_smoothed_alpha_shape_and_positivity() -> None:
    mu, L, y, feats, cells = _make_gp_fixture()
    out = gp_smoothed_per_cell_per_label_scale(
        mu,
        L,
        y,
        feats,
        cells,
        min_cell_stars_for_training=32,
        min_cell_stars=8,
    )
    assert out["per_star_alpha"].shape == mu.shape
    assert (out["per_star_alpha"] > 0).all()
    assert np.isfinite(out["per_star_alpha"]).all()
    assert isinstance(out["gp_bundle"], GpAlphaBundle)
    assert out["gp_bundle"].cell_centers.shape[1] == feats.shape[1]


def test_gp_smoothed_alpha_matches_smooth_field() -> None:
    """GP should recover smooth 1/α_truth across the feat-0 axis.

    Cell-center broadcast means per-star α is constant within a cell, so
    the correlation check lives on the per-cell scales table: α at cells
    with feat0 axis = -1 (α_truth=0.2) should be much higher than at cells
    with feat0 axis = +1 (α_truth=1.8).
    """
    mu, L, y, feats, cells = _make_gp_fixture(n_cells_per_axis=3, stars_per_cell=120)
    out = gp_smoothed_per_cell_per_label_scale(
        mu,
        L,
        y,
        feats,
        cells,
        min_cell_stars_for_training=32,
        min_cell_stars=8,
    )
    scales = out["scales"]
    # cid = (i*3 + j)*3 + k. i == feat0 axis bin: 0 → -1, 2 → +1.
    low_i_alphas = [scales[(cid, 0)] for cid in range(27) if (cid // 9) == 0]
    high_i_alphas = [scales[(cid, 0)] for cid in range(27) if (cid // 9) == 2]
    assert np.mean(low_i_alphas) > np.mean(high_i_alphas) * 1.5, (
        f"low-feat0 α mean={np.mean(low_i_alphas):.3f} should exceed "
        f"high-feat0 α mean={np.mean(high_i_alphas):.3f} by >1.5×"
    )


def test_gp_smoothed_alpha_labels_with_no_miscalibration_return_near_one() -> None:
    mu, L, y, feats, cells = _make_gp_fixture(n_cells_per_axis=3, stars_per_cell=150)
    out = gp_smoothed_per_cell_per_label_scale(
        mu,
        L,
        y,
        feats,
        cells,
        min_cell_stars_for_training=32,
        min_cell_stars=8,
    )
    # Labels 1 and 2 were perfectly calibrated — α should be near 1 everywhere.
    for j in (1, 2):
        alpha_j = out["per_star_alpha"][:, j]
        assert alpha_j.mean() == pytest.approx(1.0, abs=0.10), (
            f"label {j}: mean α = {alpha_j.mean():.3f}"
        )
        # And the spread should be small relative to the smooth-label case.
        assert alpha_j.std() < 0.15


def test_gp_smoothed_alpha_preserves_pd_via_diag_alpha_L() -> None:
    mu, L, y, feats, cells = _make_gp_fixture()
    out = gp_smoothed_per_cell_per_label_scale(
        mu,
        L,
        y,
        feats,
        cells,
        min_cell_stars_for_training=32,
        min_cell_stars=8,
    )
    alpha = out["per_star_alpha"]
    L_prime = alpha[:, :, None] * L
    upper = np.triu(L_prime, k=1)
    assert np.allclose(upper, 0.0, atol=1e-6)
    Sigma_prime = np.einsum("bij,bkj->bik", L_prime, L_prime)
    # Spot check 30 stars for PD via cholesky.
    for b in np.linspace(0, L.shape[0] - 1, 30).astype(int):
        np.linalg.cholesky(Sigma_prime[b])


def test_gp_smoothed_alpha_sparse_cell_borrows_from_neighbors() -> None:
    """A cell below min_cell_stars_for_training gets α from GP interpolation."""
    mu, L, y, feats, cells = _make_gp_fixture(n_cells_per_axis=3, stars_per_cell=120)
    # Down-sample one cell (id=13, interior) to 10 stars only.
    target_cell = 13
    keep_mask = np.ones(cells.size, dtype=bool)
    target_idx = np.where(cells == target_cell)[0]
    np.random.default_rng(0).shuffle(target_idx)
    drop_idx = target_idx[10:]
    keep_mask[drop_idx] = False
    mu_s, L_s, y_s = mu[keep_mask], L[keep_mask], y[keep_mask]
    feats_s, cells_s = feats[keep_mask], cells[keep_mask]

    out = gp_smoothed_per_cell_per_label_scale(
        mu_s,
        L_s,
        y_s,
        feats_s,
        cells_s,
        min_cell_stars_for_training=32,
        min_cell_stars=8,
    )
    # Cell 13 should NOT be in train_cell_ids (10 < 32).
    assert target_cell not in set(out["train_cell_ids"])
    # But α_GP at its stars should still be finite and near its neighbors.
    target_stars_mask = cells_s == target_cell
    alpha_target = out["per_star_alpha"][target_stars_mask, 0]
    assert np.isfinite(alpha_target).all()
    assert (alpha_target > 0).all()


def test_gp_bundle_roundtrip_and_apply_consistency() -> None:
    """Bundle round-trip: apply_gp_alpha on the dict-restored bundle matches origin.

    Note: fit-time ``per_star_alpha`` is the cell-center GP value broadcast by
    cell_id (per-cell constant), whereas ``apply_gp_alpha`` evaluates the GP
    at per-star features (smooth field) — these are *different* by design.
    This test verifies only that bundle serialisation survives a round-trip
    in ``apply_gp_alpha``'s per-star semantics.
    """
    mu, L, y, feats, cells = _make_gp_fixture()
    out = gp_smoothed_per_cell_per_label_scale(
        mu,
        L,
        y,
        feats,
        cells,
        min_cell_stars_for_training=32,
        min_cell_stars=8,
    )
    bundle = out["gp_bundle"]
    blob = bundle.to_dict()
    bundle2 = GpAlphaBundle.from_dict(blob)

    alpha_orig = apply_gp_alpha(feats, bundle)
    alpha_reapplied = apply_gp_alpha(feats, bundle2)
    np.testing.assert_allclose(alpha_reapplied, alpha_orig, atol=1e-6)


def test_gp_smoothed_alpha_handles_nonfinite_features() -> None:
    """NaN feature rows must not crash; cell-center broadcast still assigns α."""
    mu, L, y, feats, cells = _make_gp_fixture(n_cells_per_axis=3, stars_per_cell=80)
    feats_bad = feats.copy()
    feats_bad[::50, 0] = np.nan
    out = gp_smoothed_per_cell_per_label_scale(
        mu,
        L,
        y,
        feats_bad,
        cells,
        min_cell_stars_for_training=32,
        min_cell_stars=8,
    )
    nan_rows = np.where(~np.isfinite(feats_bad).all(axis=1))[0]
    # Cell-center eval derives each cell's centre from the *finite* rows in
    # that cell, so NaN-feature rows still inherit the cell's α.
    for j in range(mu.shape[1]):
        assert np.isfinite(out["per_star_alpha"][nan_rows, j]).all()
        assert (out["per_star_alpha"][nan_rows, j] > 0).all()


# --- regime B exclusion envelope ---


def test_regime_b_envelope_captures_plane_warm_rgb() -> None:
    env = RegimeBEnvelope()
    # Stars inside envelope: |b|<5, Teff>4750, logg<2.1.
    teff = np.array([5000.0, 4800.0, 4900.0, 4600.0])
    logg = np.array([2.0, 1.8, 2.0, 1.9])
    b = np.array([2.0, -4.0, 0.5, 10.0])
    inside = env.mask(teff, logg, b)
    assert inside.tolist() == [True, True, True, False]
    rel = env.tier1_release_flag(teff, logg, b)
    assert rel.tolist() == [False, False, False, True]


def test_regime_b_envelope_respects_all_three_thresholds() -> None:
    env = RegimeBEnvelope()
    # Must violate *all three* conditions to be inside the envelope.
    cases = [
        (4700.0, 2.0, 2.0),  # Teff too cool
        (5000.0, 2.3, 2.0),  # logg too high
        (5000.0, 2.0, 6.0),  # |b| too large
    ]
    for teff, logg, b in cases:
        assert not env.mask(
            np.array([teff]),
            np.array([logg]),
            np.array([b]),
        )[0]


def test_regime_b_envelope_roundtrip() -> None:
    env = RegimeBEnvelope(b_deg_max=3.0, teff_k_min=4820.0, logg_dex_max=2.05)
    blob = env.to_dict()
    env2 = RegimeBEnvelope.from_dict(blob)
    assert env2.b_deg_max == 3.0
    assert env2.teff_k_min == 4820.0
    assert env2.logg_dex_max == 2.05


def test_regime_b_envelope_handles_absolute_b() -> None:
    env = RegimeBEnvelope()
    # Both +3 and -3 are inside.
    teff = np.array([5000.0, 5000.0])
    logg = np.array([2.0, 2.0])
    b = np.array([3.0, -3.0])
    inside = env.mask(teff, logg, b)
    assert inside.tolist() == [True, True]


# --- isotonic ---


def test_isotonic_per_label_monotone_non_decreasing() -> None:
    rng = np.random.default_rng(0)
    B, n = 500, 3
    mu = np.zeros((B, n), dtype=np.float32)
    L = _batched_cholesky(B, n, seed=0)
    y = _sample_from_mvn(mu, L, rng)
    iso = isotonic_per_label(mu, L, y)
    assert len(iso) == n
    for j in range(n):
        y_thresh = iso[j]["y"]
        diffs = np.diff(y_thresh)
        assert (diffs >= -1e-6).all(), f"label {j} isotonic not monotone: min diff {diffs.min()}"


# --- coverage ---


def test_coverage_at_levels_nominal_on_calibrated_data() -> None:
    """When truth is sampled from predicted Σ, coverage should match nominal."""
    rng = np.random.default_rng(0)
    B, n = 4000, 3
    mu = rng.standard_normal((B, n)).astype(np.float32)
    L = _batched_cholesky(B, n, seed=1)
    y = _sample_from_mvn(mu, L, rng)
    cov = coverage_at_levels(mu, L, y, levels=(0.68, 0.95))
    for lvl in (0.68, 0.95):
        assert abs(cov["joint"][lvl] - lvl) < 0.03, (
            f"joint coverage at {lvl}: got {cov['joint'][lvl]}"
        )
        assert np.all(np.abs(cov["per_label"][lvl] - lvl) < 0.04)


def test_coverage_miscalibrated_data_deviates() -> None:
    rng = np.random.default_rng(0)
    B, n = 2000, 3
    mu = np.zeros((B, n), dtype=np.float32)
    L_true = _batched_cholesky(B, n, seed=0)
    y = _sample_from_mvn(mu, L_true, rng)
    L_pred = L_true * 0.3  # under-estimate σ → coverage should plummet
    cov = coverage_at_levels(mu, L_pred, y, levels=(0.95,))
    assert cov["joint"][0.95] < 0.5


# --- conformal ---


def test_conformal_scores_shape_and_nonnegative() -> None:
    rng = np.random.default_rng(0)
    B, n = 200, 4
    mu = rng.standard_normal((B, n)).astype(np.float32)
    L = _batched_cholesky(B, n, seed=0)
    y = _sample_from_mvn(mu, L, rng)
    s = conformal_nonconformity_scores(mu, L, y)
    assert s.shape == (B,)
    assert (s >= 0).all()
    assert np.isfinite(s).all()


def test_conformal_radius_increases_with_level() -> None:
    rng = np.random.default_rng(0)
    B, n = 200, 3
    mu = np.zeros((B, n), dtype=np.float32)
    L = _batched_cholesky(B, n, seed=1)
    y = _sample_from_mvn(mu, L, rng)
    art = fit_calibration(mu, L, y, fit_isotonic=False, fit_conformal=True)
    r68 = conformal_radius_at_level(art, 0.68)
    r95 = conformal_radius_at_level(art, 0.95)
    assert r95 > r68


def test_conformal_radius_empty_scores_raises() -> None:
    art = CalibrationArtifacts()
    with pytest.raises(ValueError, match="conformal_scores empty"):
        conformal_radius_at_level(art, 0.95)


# --- apply_calibration ---


def test_apply_calibration_scales_cholesky() -> None:
    B, n = 50, 3
    mu = np.zeros((B, n), dtype=np.float32)
    L = _batched_cholesky(B, n, seed=0)
    art = CalibrationArtifacts(temperature_per_cell={0: 2.0, 1: 0.5})
    cell_ids = np.array([0] * 25 + [1] * 25, dtype=np.int64)
    mu_cal, L_cal = apply_calibration(mu, L, art, cell_ids=cell_ids)
    assert np.allclose(mu_cal, mu)
    assert np.allclose(L_cal[:25], L[:25] * 2.0, atol=1e-5)
    assert np.allclose(L_cal[25:], L[25:] * 0.5, atol=1e-5)


def test_apply_calibration_unknown_cell_falls_back_to_one() -> None:
    B, n = 4, 2
    mu = np.zeros((B, n), dtype=np.float32)
    L = _batched_cholesky(B, n, seed=0)
    art = CalibrationArtifacts(temperature_per_cell={})  # no entries
    _, L_cal = apply_calibration(mu, L, art, cell_ids=np.zeros(B, dtype=np.int64))
    assert np.allclose(L_cal, L)


# --- end-to-end fit_calibration ---


def test_fit_calibration_roundtrip_checkpoint_schema() -> None:
    rng = np.random.default_rng(0)
    B, n = 300, 3
    mu = rng.standard_normal((B, n)).astype(np.float32)
    L = _batched_cholesky(B, n, seed=0)
    y = _sample_from_mvn(mu, L, rng)
    cell_features = rng.standard_normal((B, 2)).astype(np.float32)

    art = fit_calibration(mu, L, y, cell_features=cell_features, cell_n_bins=(2, 2))
    blob = art.as_checkpoint_dict()
    assert set(blob) == {
        "temperature_per_cell",
        "isotonic_per_label",
        "conformal_scores",
        "cell_definition",
    }
    assert len(art.temperature_per_cell) >= 1
    assert len(art.isotonic_per_label) == n
    assert art.conformal_scores.shape == (B,)


def test_fit_calibration_temperature_recovery_endtoend() -> None:
    """Full fit: truth drawn from Σ_true; predictions use 2·Σ_true → s ≈ 0.5."""
    rng = np.random.default_rng(0)
    B, n = 2000, 3
    mu = np.zeros((B, n), dtype=np.float32)
    L_true = _batched_cholesky(B, n, seed=0)
    y = _sample_from_mvn(mu, L_true, rng)
    L_pred = L_true * 2.0

    art = fit_calibration(mu, L_pred, y, fit_isotonic=False, fit_conformal=False)
    assert abs(art.temperature_per_cell[0] - 0.5) < 0.1


# --- collect_predictions ---


def test_collect_predictions_round_trip() -> None:
    cfg = ModelConfig(
        input_dim=16,
        block_layout=CovarianceBlockLayout.anonymous(block_sizes=(3, 3), n_diagonal_only=4),
    )
    model = XpAbundanceModel(cfg)
    model.eval()

    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 16)).astype(np.float32)
    Y = rng.standard_normal((20, 10)).astype(np.float32)
    sig = rng.uniform(0.01, 0.1, (20, 10)).astype(np.float32)
    ds = XpAbundanceDataset(X=X, Y=Y, sigma_Y=sig)
    loader = DataLoader(ds, batch_size=5)
    preds = collect_predictions(model, loader, device=torch.device("cpu"))
    assert preds["mu"].shape == (20, 10)
    assert preds["L"].shape == (20, 10, 10)
    assert preds["y"].shape == (20, 10)
    assert preds["sigma_Y"].shape == (20, 10)


def test_collect_predictions_without_sigma() -> None:
    cfg = ModelConfig(
        input_dim=8,
        block_layout=CovarianceBlockLayout.anonymous(block_sizes=(2, 2), n_diagonal_only=2),
    )
    model = XpAbundanceModel(cfg)
    model.eval()
    X = torch.randn(10, 8)
    Y = torch.randn(10, 6)
    ds = TensorDataset(X, Y)
    loader = DataLoader(ds, batch_size=5)
    preds = collect_predictions(model, loader, device=torch.device("cpu"))
    assert "sigma_Y" not in preds
