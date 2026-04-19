"""Tests for population_classifier.main.clustering — HDBSCAN + soft + GLOSH."""

from __future__ import annotations

import numpy as np
import pytest

from arqueogal.population_classifier.main.clustering import (
    ClusteringResult,
    HDBSCANConfig,
    cluster_hdbscan,
)


def _two_clusters(n_per: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(-5.0, 0.3, (n_per, 2))
    b = rng.normal(+5.0, 0.3, (n_per, 2))
    return np.vstack([a, b]).astype(np.float32)


def _three_clusters(n_per: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.array([[-6, -6], [6, -6], [0, 6]], dtype=np.float32)
    pts = np.vstack([
        rng.normal(loc=c, scale=0.3, size=(n_per, 2)) for c in centers
    ])
    return pts.astype(np.float32)


# --- basic shape -----------------------------------------------------------

def test_cluster_hdbscan_runs_on_two_blobs() -> None:
    Z = _two_clusters(40)
    res = cluster_hdbscan(Z, HDBSCANConfig(min_cluster_size=10))
    assert isinstance(res, ClusteringResult)
    assert res.n_clusters == 2
    assert len(res.cluster_ids) == 2
    assert res.labels.shape == (80,)
    assert res.glosh.shape == (80,)
    assert res.soft_memberships.shape == (80, 2)
    assert res.probabilities.shape == (80,)
    assert res.boundary_flag.shape == (80,)


def test_cluster_hdbscan_detects_three_populations() -> None:
    Z = _three_clusters(40)
    res = cluster_hdbscan(Z, HDBSCANConfig(min_cluster_size=15))
    assert res.n_clusters == 3
    assert res.soft_memberships.shape == (120, 3)


# --- soft membership semantics --------------------------------------------

def test_soft_memberships_row_sum_bounded_by_one() -> None:
    """Per hdbscan: rows sum to ≤ 1, with deficit = implicit noise membership."""
    Z = _two_clusters(40)
    res = cluster_hdbscan(Z, HDBSCANConfig(min_cluster_size=10))
    sums = res.soft_memberships.sum(axis=1)
    assert (sums <= 1.0 + 1e-5).all()
    assert (sums >= 0.0).all()


def test_soft_memberships_bounded_zero_one() -> None:
    Z = _two_clusters(40)
    res = cluster_hdbscan(Z, HDBSCANConfig(min_cluster_size=10))
    assert (res.soft_memberships >= 0.0).all()
    assert (res.soft_memberships <= 1.0 + 1e-6).all()


# --- boundary flag --------------------------------------------------------

def test_boundary_flag_default_threshold() -> None:
    Z = _two_clusters(40)
    res = cluster_hdbscan(Z, HDBSCANConfig(min_cluster_size=10))
    expected = res.soft_memberships.max(axis=1) < 0.7
    assert np.array_equal(res.boundary_flag, expected)


def test_boundary_flag_custom_threshold_stricter_triggers_more() -> None:
    Z = _two_clusters(40)
    lax = cluster_hdbscan(Z, HDBSCANConfig(min_cluster_size=10),
                          boundary_threshold=0.5)
    strict = cluster_hdbscan(Z, HDBSCANConfig(min_cluster_size=10),
                             boundary_threshold=0.95)
    assert strict.boundary_flag.sum() >= lax.boundary_flag.sum()


# --- noise handling -------------------------------------------------------

def test_noise_points_have_label_minus_one() -> None:
    rng = np.random.default_rng(0)
    Z = np.concatenate([
        _two_clusters(40),
        rng.uniform(-15, 15, size=(20, 2)).astype(np.float32),
    ])
    res = cluster_hdbscan(Z, HDBSCANConfig(min_cluster_size=15))
    assert (res.labels == -1).any()
    assert res.noise_fraction > 0.0


def test_no_clusters_found_returns_empty_soft() -> None:
    rng = np.random.default_rng(0)
    # Uniform noise, min_cluster_size larger than plausibly findable cluster.
    Z = rng.uniform(-20, 20, size=(40, 2)).astype(np.float32)
    res = cluster_hdbscan(
        Z, HDBSCANConfig(min_cluster_size=35, min_samples=5),
    )
    if res.n_clusters == 0:
        assert res.soft_memberships.shape == (40, 0)
        assert res.boundary_flag.all()
    else:
        pytest.skip("hdbscan found a cluster in noise on this RNG seed")


# --- GLOSH ---------------------------------------------------------------

def test_glosh_scores_are_finite_and_bounded() -> None:
    Z = _two_clusters(40)
    res = cluster_hdbscan(Z, HDBSCANConfig(min_cluster_size=10))
    assert np.isfinite(res.glosh).all()
    assert (res.glosh >= 0).all()
    assert (res.glosh <= 1 + 1e-6).all()


# --- validation -----------------------------------------------------------

def test_cluster_hdbscan_rejects_1d_input() -> None:
    with pytest.raises(ValueError, match="Z must be 2-D"):
        cluster_hdbscan(np.zeros(10, dtype=np.float32))


def test_clustering_result_shape_validation() -> None:
    with pytest.raises(ValueError, match="not aligned"):
        ClusteringResult(
            labels=np.zeros(10, dtype=np.int64),
            probabilities=np.zeros(10, dtype=np.float32),
            soft_memberships=np.zeros((7, 2), dtype=np.float32),
            glosh=np.zeros(10, dtype=np.float32),
            boundary_flag=np.zeros(10, dtype=bool),
            n_clusters=2,
            cluster_ids=(0, 1),
        )


def test_clustering_result_noise_fraction_computed_correctly() -> None:
    res = ClusteringResult(
        labels=np.array([-1, 0, 0, -1, 1], dtype=np.int64),
        probabilities=np.ones(5, dtype=np.float32),
        soft_memberships=np.zeros((5, 2), dtype=np.float32),
        glosh=np.zeros(5, dtype=np.float32),
        boundary_flag=np.zeros(5, dtype=bool),
        n_clusters=2,
        cluster_ids=(0, 1),
    )
    assert res.noise_fraction == pytest.approx(0.4)
