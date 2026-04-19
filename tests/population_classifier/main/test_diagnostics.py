"""Tests for population_classifier.main.diagnostics — six-tool stack."""

from __future__ import annotations

import numpy as np
import pytest

from arqueogal.population_classifier.main.diagnostics import (
    BootstrapStabilityReport,
    DiagnosticStackReport,
    bootstrap_cluster_stability,
    held_out_feature_consistency,
    literature_cross_reference,
    null_model_comparison,
    permutation_feature_causal,
)

# --- helpers -------------------------------------------------------------

def _two_well_separated_blobs(
    n_per: int = 40, d: int = 3, seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(-5, 0.3, (n_per, d))
    b = rng.normal(+5, 0.3, (n_per, d))
    return np.vstack([a, b]).astype(np.float32)


def _axis0_kmeans2_labels(X: np.ndarray) -> np.ndarray:
    """Cheap deterministic 2-cluster "clusterer": splits on axis-0 sign."""
    return (X[:, 0] > X[:, 0].mean()).astype(np.int64)


def _uniform_noise(n: int = 80, d: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=(n, d)).astype(np.float32)


# --- 1. bootstrap stability ----------------------------------------------


def test_bootstrap_stability_high_for_trivially_stable_clusters() -> None:
    X = _two_well_separated_blobs(40)
    report = bootstrap_cluster_stability(
        X, _axis0_kmeans2_labels, n_bootstrap=10, seed=0,
    )
    assert isinstance(report, BootstrapStabilityReport)
    assert report.n_bootstrap == 10
    assert report.pairwise_ari.size == 10 * 9 // 2
    assert report.median_ari > 0.75
    assert report.stable
    assert not report.artefact


def test_bootstrap_stability_low_for_unstable_clusterer() -> None:
    """Random projection on uniform noise → every refit yields a different
    partition, so pairwise ARI should be near zero."""
    X = _uniform_noise(80, d=3, seed=1)

    def unstable_projection(X_in: np.ndarray) -> np.ndarray:
        # Entropy-seeded so each call produces an independent partition.
        local = np.random.default_rng()
        direction = local.standard_normal(X_in.shape[1])
        proj = X_in @ direction
        return (proj > np.median(proj)).astype(np.int64)

    report = bootstrap_cluster_stability(
        X, unstable_projection, n_bootstrap=10, seed=0,
    )
    assert report.median_ari < 0.5
    assert report.artefact
    assert not report.stable


def test_bootstrap_stability_rejects_1d_input() -> None:
    with pytest.raises(ValueError, match="X must be 2-D"):
        bootstrap_cluster_stability(
            np.zeros(10), _axis0_kmeans2_labels, n_bootstrap=3,
        )


def test_bootstrap_stability_quantile_ordering() -> None:
    X = _two_well_separated_blobs(40)
    report = bootstrap_cluster_stability(
        X, _axis0_kmeans2_labels, n_bootstrap=6, seed=0,
    )
    assert report.q05_ari <= report.median_ari <= report.q95_ari


# --- 3. permutation-feature causal ---------------------------------------


def test_permutation_feature_causal_identifies_informative_axis() -> None:
    """Axis 0 is the only axis carrying cluster structure → ARI drops
    dramatically under its shuffle, stays high under other shuffles."""
    X = _two_well_separated_blobs(40, d=3)
    baseline = _axis0_kmeans2_labels(X)
    report = permutation_feature_causal(
        X, baseline, _axis0_kmeans2_labels, seed=0,
    )
    assert report.ari_by_feature[0] < 0.5
    assert report.ari_by_feature[1] > 0.9
    assert report.ari_by_feature[2] > 0.9
    # ranked most-causal first.
    assert report.ranked[0][0] == 0


def test_permutation_feature_causal_rejects_length_mismatch() -> None:
    X = _two_well_separated_blobs(40)
    with pytest.raises(ValueError, match="baseline_labels length"):
        permutation_feature_causal(
            X, np.zeros(7, dtype=np.int64), _axis0_kmeans2_labels,
        )


def test_permutation_feature_causal_ari_drop_complement_of_ari() -> None:
    X = _two_well_separated_blobs(40)
    baseline = _axis0_kmeans2_labels(X)
    report = permutation_feature_causal(
        X, baseline, _axis0_kmeans2_labels, seed=0,
    )
    for k, v in report.ari_by_feature.items():
        assert report.ari_drop_by_feature[k] == pytest.approx(1.0 - v)


def test_permutation_feature_causal_respects_feature_indices_arg() -> None:
    X = _two_well_separated_blobs(40)
    baseline = _axis0_kmeans2_labels(X)
    report = permutation_feature_causal(
        X, baseline, _axis0_kmeans2_labels,
        feature_indices=[0, 2], seed=0,
    )
    assert set(report.ari_by_feature.keys()) == {0, 2}


# --- 4. null-model comparison --------------------------------------------


def test_null_model_copula_real_exceeds_null_for_filament() -> None:
    """Correlation-dependent cluster: a tight 3-D filament. Copula-shuffling
    destroys the cross-axis correlations so the line collapses to a diffuse
    Gaussian cloud with no density-based clusters — classic §10.5 test-4 case.
    """
    from arqueogal.population_classifier.main.clustering import (
        HDBSCANConfig,
        cluster_hdbscan,
    )
    rng = np.random.default_rng(0)
    # Compact 3-D filament along the diagonal: uniformly dense along t,
    # tightly pinched in the perpendicular directions. Cross-axis
    # correlation is what ties the density into a detectable cluster.
    t = rng.uniform(-1.0, 1.0, 120)
    perp = rng.normal(0.0, 0.03, (120, 3))
    X = (np.column_stack([t, t, t]) + perp).astype(np.float32)

    def hdbscan_clusterer(X_in: np.ndarray) -> np.ndarray:
        return cluster_hdbscan(
            X_in.astype(np.float32),
            HDBSCANConfig(min_cluster_size=20, min_samples=10),
        ).labels

    report = null_model_comparison(
        X, hdbscan_clusterer, method="copula", n_null=8, seed=0,
    )
    assert report.real_n_clusters >= 1
    assert report.null_median <= report.real_n_clusters
    assert report.method == "copula"
    assert report.null_n_clusters.shape == (8,)


def test_null_model_mvn_method_runs() -> None:
    X = _two_well_separated_blobs(40)
    report = null_model_comparison(
        X, _axis0_kmeans2_labels, method="mvn", n_null=4, seed=0,
    )
    assert report.method == "mvn"
    assert report.null_n_clusters.shape == (4,)


def test_null_model_rejects_1d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        null_model_comparison(
            np.zeros(10), _axis0_kmeans2_labels, n_null=2,
        )


# --- 5. held-out feature consistency -------------------------------------


def test_held_out_feature_consistency_reports_separation() -> None:
    """Build data where feature 2 is identical to feature 0 (so the
    clustering learnt from {1,2} still separates on feature 0)."""
    rng = np.random.default_rng(0)
    a = rng.normal(-5, 0.3, (40, 2))
    b = rng.normal(+5, 0.3, (40, 2))
    core = np.vstack([a, b])  # (80, 2)
    X = np.hstack([core, core[:, [0]]])  # feature-2 mirrors feature-0

    report = held_out_feature_consistency(
        X, _axis0_kmeans2_labels, held_out_index=2,
    )
    assert report.held_out_index == 2
    assert report.labels.shape == (80,)
    assert np.isfinite(report.kw_h)
    assert report.kw_pvalue < 0.05
    assert report.separation_ratio > 1.0
    assert len(report.cluster_ids) == 2


def test_held_out_feature_consistency_rejects_bad_index() -> None:
    X = _two_well_separated_blobs(40)
    with pytest.raises(ValueError, match="out of range"):
        held_out_feature_consistency(X, _axis0_kmeans2_labels, held_out_index=99)


def test_held_out_feature_returns_nan_when_single_cluster() -> None:
    X = _uniform_noise(40)

    def one_cluster(X_in: np.ndarray) -> np.ndarray:
        return np.zeros(X_in.shape[0], dtype=np.int64)

    report = held_out_feature_consistency(X, one_cluster, held_out_index=0)
    assert np.isnan(report.kw_h)
    assert np.isnan(report.kw_pvalue)


# --- 6. literature cross-reference ---------------------------------------


def test_literature_cross_reference_perfect_match_gives_ari_one() -> None:
    pred = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    lit = np.array(["gse", "gse", "sequoia", "sequoia", "thamnos", "thamnos"])
    report = literature_cross_reference(pred, lit)
    assert report.ari == pytest.approx(1.0)
    assert report.contingency.shape == (3, 3)
    assert report.n_matched == 6
    assert set(report.predicted_ids) == {0, 1, 2}


def test_literature_cross_reference_excludes_noise_by_default() -> None:
    pred = np.array([-1, 0, 0, 1, 1], dtype=np.int64)
    lit = np.array(["noise", "gse", "gse", "seq", "seq"])
    report = literature_cross_reference(pred, lit)
    assert report.n_matched == 4
    assert -1 not in report.predicted_ids


def test_literature_cross_reference_precision_recall_semantics() -> None:
    pred = np.array([0, 0, 0, 1], dtype=np.int64)
    lit = np.array(["gse", "gse", "seq", "seq"])
    report = literature_cross_reference(pred, lit)
    # cluster 0 is 2 gse + 1 seq → precision 2/3.
    # cluster 1 is 1 seq → precision 1.
    pred_idx = report.predicted_ids.index(0)
    assert report.precision[pred_idx] == pytest.approx(2 / 3)
    # gse is fully in cluster 0 → recall 2/2. seq is 1 in 0, 1 in 1 → recall 1/2.
    lit_idx_gse = report.literature_ids.index("gse")
    assert report.recall[lit_idx_gse] == pytest.approx(1.0)
    lit_idx_seq = report.literature_ids.index("seq")
    assert report.recall[lit_idx_seq] == pytest.approx(0.5)


def test_literature_cross_reference_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        literature_cross_reference(
            np.zeros(5, dtype=np.int64), np.zeros(6, dtype=np.int64),
        )


def test_literature_cross_reference_handles_missing_values() -> None:
    pred = np.array([0, 0, 1, 1], dtype=np.int64)
    lit = np.array(["gse", None, "seq", "seq"], dtype=object)
    report = literature_cross_reference(pred, lit)
    assert report.n_matched == 3


# --- aggregated report container -----------------------------------------


def test_diagnostic_stack_report_defaults() -> None:
    r = DiagnosticStackReport()
    assert r.bootstrap is None
    assert r.dbcv is None
    assert r.feature_causal is None
    assert r.null_model is None
    assert r.held_out == {}
    assert r.literature is None
