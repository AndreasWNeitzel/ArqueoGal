"""Tests for xp_abundances.main.ood — Mahalanobis + ensemble OOD flags."""

from __future__ import annotations

import numpy as np
import pytest

from arqueogal.xp_abundances.main.ood import (
    MahalanobisOODBundle,
    combined_ood_status,
    ensemble_disagreement_ratio,
    fit_mahalanobis_ood,
    flag_ensemble_ood,
    flag_mahalanobis_ood,
    score_mahalanobis_ood,
)


# --- Mahalanobis OOD: fit ---------------------------------------------------

def test_fit_mahalanobis_recovers_mean_and_precision() -> None:
    """On isotropic Gaussian training data, μ ≈ 0, Σ⁻¹ ≈ I."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5000, 10)).astype(np.float32)
    bundle = fit_mahalanobis_ood(X, p_threshold=0.99, regularization=1e-6)
    np.testing.assert_allclose(bundle.feature_mean, 0.0, atol=0.1)
    # precision should be close to identity (since Σ ≈ I).
    np.testing.assert_allclose(
        bundle.feature_precision, np.eye(10), atol=0.1,
    )


def test_fit_mahalanobis_threshold_is_p_quantile() -> None:
    """The fitted threshold matches the empirical p_threshold quantile of distances."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((10_000, 8)).astype(np.float32)
    bundle = fit_mahalanobis_ood(X, p_threshold=0.99)
    # Training-set distances used for the quantile — re-score and compare.
    dists = score_mahalanobis_ood(X, bundle)
    frac_below = float(np.mean(dists <= bundle.threshold))
    assert abs(frac_below - 0.99) < 0.005


def test_fit_mahalanobis_drops_non_finite_rows() -> None:
    rng = np.random.default_rng(2)
    X = rng.standard_normal((200, 5)).astype(np.float32)
    X[0, 0] = np.nan
    X[10, 2] = np.inf
    bundle = fit_mahalanobis_ood(X, p_threshold=0.95)
    assert bundle.n_training == 198


def test_fit_mahalanobis_rejects_underdetermined_input() -> None:
    X = np.zeros((3, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="need at least"):
        fit_mahalanobis_ood(X)


def test_fit_mahalanobis_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="must be 2D"):
        fit_mahalanobis_ood(np.zeros(5))


# --- Mahalanobis OOD: score & flag ------------------------------------------

def test_score_mahalanobis_shape_and_finite() -> None:
    rng = np.random.default_rng(3)
    X = rng.standard_normal((1000, 6)).astype(np.float32)
    bundle = fit_mahalanobis_ood(X)
    scores = score_mahalanobis_ood(X, bundle)
    assert scores.shape == (1000,)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()


def test_score_mahalanobis_out_of_dist_exceeds_threshold() -> None:
    """A point at 10·(mean + unit) in input space blows past the p99 threshold."""
    rng = np.random.default_rng(4)
    X = rng.standard_normal((5000, 8)).astype(np.float32)
    bundle = fit_mahalanobis_ood(X, p_threshold=0.99)
    ood_point = np.full((1, 8), 10.0, dtype=np.float32)
    score_ood = score_mahalanobis_ood(ood_point, bundle)[0]
    assert score_ood > bundle.threshold * 5


def test_flag_mahalanobis_reports_ood() -> None:
    rng = np.random.default_rng(5)
    X_train = rng.standard_normal((3000, 5)).astype(np.float32)
    bundle = fit_mahalanobis_ood(X_train, p_threshold=0.99)
    # In-distribution test: ~1% flagged (matches p99 quantile).
    X_test_in = rng.standard_normal((2000, 5)).astype(np.float32)
    flags_in = flag_mahalanobis_ood(X_test_in, bundle)
    assert 0.003 < flags_in.mean() < 0.035
    # OOD test: all flagged.
    X_test_ood = np.full((500, 5), 8.0, dtype=np.float32)
    flags_ood = flag_mahalanobis_ood(X_test_ood, bundle)
    assert flags_ood.all()


def test_flag_mahalanobis_flags_non_finite_rows() -> None:
    rng = np.random.default_rng(6)
    X = rng.standard_normal((500, 4)).astype(np.float32)
    bundle = fit_mahalanobis_ood(X)
    X_test = rng.standard_normal((10, 4)).astype(np.float32)
    X_test[0, 0] = np.nan
    X_test[5, 2] = np.inf
    flags = flag_mahalanobis_ood(X_test, bundle)
    assert flags[0]  # nan → flagged
    assert flags[5]  # inf → flagged


def test_score_mahalanobis_rejects_wrong_dim() -> None:
    rng = np.random.default_rng(7)
    X = rng.standard_normal((200, 5)).astype(np.float32)
    bundle = fit_mahalanobis_ood(X)
    X_wrong = rng.standard_normal((10, 4)).astype(np.float32)
    with pytest.raises(ValueError, match="feature dim"):
        score_mahalanobis_ood(X_wrong, bundle)


def test_mahalanobis_bundle_roundtrip() -> None:
    rng = np.random.default_rng(8)
    X = rng.standard_normal((1000, 6)).astype(np.float32)
    bundle = fit_mahalanobis_ood(X, p_threshold=0.95, regularization=1e-5)
    blob = bundle.to_dict()
    bundle2 = MahalanobisOODBundle.from_dict(blob)
    # Scores should match bit-exactly via float64 promotion.
    X_test = rng.standard_normal((100, 6)).astype(np.float32)
    s1 = score_mahalanobis_ood(X_test, bundle)
    s2 = score_mahalanobis_ood(X_test, bundle2)
    np.testing.assert_allclose(s1, s2, atol=1e-5)
    assert bundle2.p_threshold == 0.95
    assert bundle2.regularization == 1e-5


# --- Ensemble disagreement --------------------------------------------------

def test_ensemble_disagreement_ratio_calibrated_ensemble() -> None:
    """Members agreeing on μ (low epistemic) → low ratio."""
    rng = np.random.default_rng(10)
    M, B, n = 5, 200, 3
    mu = np.tile(rng.standard_normal((B, n)), (M, 1, 1)).astype(np.float32)
    # Small epistemic perturbation — 0.01 across members.
    mu += 0.01 * rng.standard_normal((M, B, n))
    sigma = np.full((M, B, n), 0.5, dtype=np.float32)
    ratio = ensemble_disagreement_ratio(mu, sigma)
    assert ratio.shape == (B,)
    assert ratio.max() < 0.1  # epistemic 0.01 vs aleatoric 0.5 → tiny


def test_ensemble_disagreement_ratio_disagreeing_ensemble() -> None:
    """Members disagreeing heavily → high ratio."""
    rng = np.random.default_rng(11)
    M, B, n = 5, 100, 2
    mu = rng.standard_normal((M, B, n)).astype(np.float32) * 10.0  # huge spread
    sigma = np.full((M, B, n), 0.1, dtype=np.float32)  # tiny aleatoric
    ratio = ensemble_disagreement_ratio(mu, sigma)
    assert ratio.min() > 0.9


def test_ensemble_disagreement_requires_multiple_members() -> None:
    mu = np.zeros((1, 10, 3), dtype=np.float32)
    sigma = np.ones((1, 10, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="≥2"):
        ensemble_disagreement_ratio(mu, sigma)


def test_ensemble_disagreement_rejects_shape_mismatch() -> None:
    mu = np.zeros((3, 10, 2), dtype=np.float32)
    sigma = np.zeros((3, 10, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        ensemble_disagreement_ratio(mu, sigma)


def test_flag_ensemble_ood_threshold_semantics() -> None:
    rng = np.random.default_rng(12)
    M, B, n = 5, 50, 2
    # Half the stars get large μ spread, half get tiny.
    mu = np.zeros((M, B, n), dtype=np.float32)
    mu[:, :25, :] = rng.standard_normal((M, 25, n)) * 5.0  # high epistemic
    mu[:, 25:, :] = rng.standard_normal((M, 25, n)) * 0.01  # low epistemic
    sigma = np.full((M, B, n), 0.5, dtype=np.float32)
    flags = flag_ensemble_ood(mu, sigma, threshold=0.5)
    assert flags[:25].all()
    assert not flags[25:].any()


# --- Combined status --------------------------------------------------------

def test_combined_ood_status_level_counts() -> None:
    mahal = np.array([False, True, False, True])
    ensemble = np.array([False, False, True, True])
    status = combined_ood_status(mahal, ensemble)
    assert status.tolist() == [0, 1, 1, 2]


def test_combined_ood_status_dtype() -> None:
    mahal = np.zeros(5, dtype=bool)
    ens = np.zeros(5, dtype=bool)
    status = combined_ood_status(mahal, ens)
    assert status.dtype == np.int8
