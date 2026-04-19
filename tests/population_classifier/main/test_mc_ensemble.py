"""Tests for population_classifier.main.mc_ensemble — MC uncertainty propagation."""

from __future__ import annotations

import numpy as np
import pytest

from arqueogal.population_classifier.main.mc_ensemble import (
    MCEnsembleConfig,
    MCEnsembleResult,
    run_mc_ensemble,
    sample_feature_posteriors,
)

# --- posterior sampling --------------------------------------------------


def test_sample_posteriors_shape_with_diagonal_sigma() -> None:
    X = np.zeros((10, 3), dtype=np.float64)
    sig = np.ones_like(X) * 0.1
    samples = sample_feature_posteriors(X, sig, n_mc=7, seed=0)
    assert samples.shape == (7, 10, 3)


def test_sample_posteriors_shape_with_full_covariance() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 4))
    # Construct per-star covariance as LᵀL for random L so Cholesky works.
    A = rng.standard_normal((8, 4, 4))
    cov = np.einsum("nij,nkj->nik", A, A) + np.eye(4) * 0.1
    samples = sample_feature_posteriors(X, cov, n_mc=5, seed=0)
    assert samples.shape == (5, 8, 4)


def test_sample_posteriors_respects_mean_and_scale() -> None:
    """Draw many samples, check empirical mean + std recover inputs."""
    rng = np.random.default_rng(0)
    N, D = 50, 3
    X = rng.standard_normal((N, D))
    # Bounded sigma so the empirical-mean standard error stays small
    # enough that the max over N×D entries is tight.
    sig = np.abs(rng.standard_normal((N, D))) * 0.1 + 0.1  # σ ∈ [0.1, 0.5]
    n_mc = 4000
    samples = sample_feature_posteriors(X, sig, n_mc=n_mc, seed=0)
    assert samples.shape == (n_mc, N, D)
    # Per-star empirical mean should converge to X. SE ≈ σ/√n_mc ≲ 0.008.
    emp_mean = samples.mean(axis=0)
    assert np.abs(emp_mean - X).max() < 0.05
    # Per-star empirical std should converge to sig.
    emp_std = samples.std(axis=0)
    assert np.abs(emp_std - sig).max() < 0.05


def test_sample_posteriors_rejects_shape_mismatch() -> None:
    X = np.zeros((10, 3))
    bad = np.ones((10, 4))
    with pytest.raises(ValueError, match="sigma shape"):
        sample_feature_posteriors(X, bad, n_mc=2)


def test_sample_posteriors_rejects_1d_input() -> None:
    with pytest.raises(ValueError, match="X_mean must be 2-D"):
        sample_feature_posteriors(np.zeros(10), np.zeros(10), n_mc=2)


# --- run_mc_ensemble -----------------------------------------------------


def _fixed_predict_soft(soft: np.ndarray):
    """Return a ``predict_soft_fn`` that ignores its input and always
    returns the same soft-membership matrix — useful for testing that
    the aggregator behaves correctly when inputs vary but outputs don't."""

    def inner(_X: np.ndarray) -> np.ndarray:
        return soft

    return inner


def test_run_mc_ensemble_zero_std_for_deterministic_predictor() -> None:
    rng = np.random.default_rng(0)
    N, K = 20, 3
    soft_ref = rng.uniform(0, 1, (N, K)).astype(np.float32)
    soft_ref /= soft_ref.sum(axis=1, keepdims=True)

    X_mean = rng.standard_normal((N, 4)).astype(np.float64)
    sig = np.ones_like(X_mean) * 0.05

    res = run_mc_ensemble(
        X_mean, sig, _fixed_predict_soft(soft_ref),
        cluster_ids=(10, 11, 12),
        config=MCEnsembleConfig(n_mc=6, seed=0),
    )
    assert isinstance(res, MCEnsembleResult)
    assert res.mean_soft.shape == (N, K)
    assert np.allclose(res.mean_soft, soft_ref, atol=1e-5)
    assert np.allclose(res.std_soft, 0.0, atol=1e-5)
    assert res.n_mc == 6
    assert res.cluster_ids == (10, 11, 12)


def test_run_mc_ensemble_std_grows_with_predictor_noise() -> None:
    N = 15
    X_mean = np.zeros((N, 3), dtype=np.float64)
    sig = np.ones_like(X_mean) * 0.1

    def noisy_predictor(X: np.ndarray) -> np.ndarray:
        # Soft memberships depend on a projection of X — so MC noise in X
        # propagates directly to non-trivial std in the soft memberships.
        p = np.clip(X[:, 0], -3, 3) / 6 + 0.5
        return np.column_stack([p, 1 - p]).astype(np.float32)

    res = run_mc_ensemble(
        X_mean, sig, noisy_predictor, cluster_ids=(0, 1),
        config=MCEnsembleConfig(n_mc=50, seed=0),
    )
    # Every star should have non-zero MC spread given σ = 0.1 on axis 0.
    assert (res.std_soft > 0).all()


def test_run_mc_ensemble_consensus_uses_argmax_with_threshold() -> None:
    N = 4
    # Star 0: clear cluster 0 (p=0.9). Star 1: ambiguous (all ≈ 0.33). Star 2
    # peak-but-below-threshold. Star 3: clear cluster 2.
    soft = np.array([
        [0.9, 0.05, 0.05],
        [0.34, 0.33, 0.33],
        [0.45, 0.30, 0.25],
        [0.05, 0.05, 0.90],
    ], dtype=np.float32)
    X_mean = np.zeros((N, 2), dtype=np.float64)
    sig = np.zeros_like(X_mean)

    res = run_mc_ensemble(
        X_mean, sig, _fixed_predict_soft(soft),
        cluster_ids=(7, 8, 9),
        config=MCEnsembleConfig(n_mc=3, seed=0),
        assign_threshold=0.5,
    )
    # Star 0 → id 7, star 1 → -1 (no peak ≥ 0.5), star 2 → -1 (0.45 < 0.5),
    # star 3 → id 9.
    assert res.consensus_labels.tolist() == [7, -1, -1, 9]


def test_run_mc_ensemble_boundary_flag_by_mc_std_threshold() -> None:
    N = 5
    X_mean = np.zeros((N, 2), dtype=np.float64)
    sig = np.ones_like(X_mean) * 2.0  # large feature σ → large MC spread

    def proj_predictor(X: np.ndarray) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-X[:, 0]))  # sigmoid
        return np.column_stack([p, 1 - p]).astype(np.float32)

    res_loose = run_mc_ensemble(
        X_mean, sig, proj_predictor, cluster_ids=(0, 1),
        config=MCEnsembleConfig(n_mc=50, seed=0,
                                mc_boundary_threshold=0.5),
    )
    res_tight = run_mc_ensemble(
        X_mean, sig, proj_predictor, cluster_ids=(0, 1),
        config=MCEnsembleConfig(n_mc=50, seed=0,
                                mc_boundary_threshold=0.05),
    )
    # Tight threshold → more boundary flags.
    assert res_tight.mc_boundary_flag.sum() >= res_loose.mc_boundary_flag.sum()


def test_run_mc_ensemble_rejects_nan_input() -> None:
    X = np.full((5, 3), np.nan)
    sig = np.ones_like(X)
    with pytest.raises(ValueError, match="NaN"):
        run_mc_ensemble(
            X, sig, _fixed_predict_soft(np.zeros((5, 1), dtype=np.float32)),
            cluster_ids=(0,),
            config=MCEnsembleConfig(n_mc=2),
        )


def test_run_mc_ensemble_rejects_bad_predict_shape() -> None:
    X = np.zeros((5, 3))
    sig = np.ones_like(X)
    with pytest.raises(ValueError, match="predict_soft_fn returned"):
        run_mc_ensemble(
            X, sig, _fixed_predict_soft(np.zeros((4, 2), dtype=np.float32)),
            cluster_ids=(0, 1),
            config=MCEnsembleConfig(n_mc=2),
        )


def test_run_mc_ensemble_full_covariance_path() -> None:
    rng = np.random.default_rng(0)
    N, D = 6, 2
    X = rng.standard_normal((N, D))
    A = rng.standard_normal((N, D, D))
    cov = np.einsum("nij,nkj->nik", A, A) + np.eye(D) * 0.1

    def trivial_soft(_X: np.ndarray) -> np.ndarray:
        return np.ones((N, 1), dtype=np.float32)

    res = run_mc_ensemble(
        X, cov, trivial_soft, cluster_ids=(0,),
        config=MCEnsembleConfig(n_mc=4, seed=0),
    )
    assert res.mean_soft.shape == (N, 1)
    assert np.allclose(res.mean_soft, 1.0)
