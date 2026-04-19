"""Tests for xp_abundances.main.tier_promotion — §3.3 six-test protocol."""

from __future__ import annotations

import json

import numpy as np
import pytest

from arqueogal.xp_abundances.main.tier_promotion import (
    TestResult,
    TierPromotionReport,
    audit_gate,
    cluster_precision,
    conditional_mi_bootstrap,
    cross_catalogue_consistency,
    holdout_rmse,
    physical_gate,
    tier_promotion_report,
)

# --- Test 1: physical gate --------------------------------------------------

def test_physical_gate_passes_when_absorption_present() -> None:
    r = physical_gate("Mg", has_absorption=True)
    assert r.passed
    assert r.detail["element"] == "Mg"


def test_physical_gate_fails_when_no_absorption() -> None:
    r = physical_gate("Xe", has_absorption=False)
    assert not r.passed


# --- Test 2: hold-out RMSE stratified ---------------------------------------

def test_holdout_rmse_passes_on_uniform_prediction_quality() -> None:
    rng = np.random.default_rng(0)
    N = 1000
    y = rng.standard_normal(N)
    mu = y + 0.02 * rng.standard_normal(N)  # small, unbiased residuals
    cells = rng.integers(0, 5, size=N)
    r = holdout_rmse(mu, y, cells)
    assert r.passed
    assert r.statistic < 2.0


def test_holdout_rmse_fails_on_heterogeneous_cells() -> None:
    """One cell has 10× the noise of others → worst/median ratio busts 2.0×."""
    rng = np.random.default_rng(0)
    N = 1000
    cells = rng.integers(0, 5, size=N)
    y = rng.standard_normal(N)
    mu = y.copy()
    for c in range(5):
        noise = 0.02 if c != 0 else 0.5
        mu[cells == c] = y[cells == c] + noise * rng.standard_normal(int((cells == c).sum()))
    r = holdout_rmse(mu, y, cells)
    assert not r.passed


def test_holdout_rmse_fails_on_systematic_bias() -> None:
    rng = np.random.default_rng(0)
    N = 500
    cells = rng.integers(0, 3, size=N)
    y = rng.standard_normal(N)
    mu = y + 0.1  # 0.1 dex bias — above the 0.05 dex limit
    r = holdout_rmse(mu, y, cells)
    assert not r.passed
    assert r.detail["worst_bias"] >= 0.05


def test_holdout_rmse_shape_validation() -> None:
    with pytest.raises(ValueError, match="must be 1-D"):
        holdout_rmse(np.zeros((4, 2)), np.zeros((4, 2)), np.zeros(4, dtype=np.int64))


# --- Test 3: cluster precision floor ----------------------------------------

def test_cluster_precision_passes_with_tight_intra_cluster_scatter() -> None:
    rng = np.random.default_rng(0)
    # 5 clusters, 10 members each, tight scatter (σ = 0.02).
    cluster_ids = np.repeat(np.arange(5), 10)
    y_pred = 0.02 * rng.standard_normal(50) + np.repeat(rng.standard_normal(5), 10)
    r = cluster_precision(y_pred, cluster_ids, apogee_sigma=0.03)
    assert r.passed
    assert r.statistic < r.threshold


def test_cluster_precision_fails_with_loose_scatter() -> None:
    rng = np.random.default_rng(0)
    cluster_ids = np.repeat(np.arange(5), 10)
    y_pred = rng.standard_normal(50)  # σ ~ 1.0 intra-cluster
    r = cluster_precision(y_pred, cluster_ids, apogee_sigma=0.03)
    assert not r.passed


def test_cluster_precision_empty_returns_fail() -> None:
    r = cluster_precision(
        np.zeros(5), np.arange(5), apogee_sigma=0.03, min_members=10,
    )
    assert not r.passed
    assert "no clusters" in r.detail.get("reason", "")


# --- Test 4: audit gate ----------------------------------------------------

def test_audit_gate_passes_with_strong_signal() -> None:
    perm = np.array([0.05, 0.03, 0.04, 0.001, 0.001])  # 3 coefs > 0.02
    looco = np.array([0.1, 0.05, 0.02, 0.0, 0.0])
    r = audit_gate(
        permutation_importance=perm,
        looco_delta_rmse=looco,
        null_skill_ratio=0.1,
        decorrelated_r2_ratio=0.7,
    )
    assert r.passed


def test_audit_gate_fails_with_too_few_informative_coefs() -> None:
    perm = np.array([0.05, 0.03, 0.001, 0.001])  # only 2 > 0.02
    r = audit_gate(
        permutation_importance=perm,
        looco_delta_rmse=np.array([0.1, 0.05, 0.01, 0.0]),
        null_skill_ratio=0.1,
        decorrelated_r2_ratio=0.7,
    )
    assert not r.passed


def test_audit_gate_fails_on_prior_driven_label() -> None:
    """null_skill_ratio > 0.2 → label is prior-driven → fail."""
    perm = np.array([0.05, 0.05, 0.05])
    r = audit_gate(
        permutation_importance=perm,
        looco_delta_rmse=np.array([0.1, 0.1, 0.1]),
        null_skill_ratio=0.8,  # model does almost as well on shuffled data
        decorrelated_r2_ratio=0.7,
    )
    assert not r.passed


def test_audit_gate_fails_when_decorrelated_skill_drops() -> None:
    perm = np.array([0.05, 0.05, 0.05])
    r = audit_gate(
        permutation_importance=perm,
        looco_delta_rmse=np.array([0.1, 0.1, 0.1]),
        null_skill_ratio=0.1,
        decorrelated_r2_ratio=0.2,  # far below 0.5
    )
    assert not r.passed


# --- Test 5: cross-catalogue consistency ------------------------------------

def test_cross_catalogue_consistency_passes_on_aligned_catalogues() -> None:
    rng = np.random.default_rng(0)
    N = 500
    pred = rng.standard_normal(N)
    external = {
        "aspgap": pred + 0.01 * rng.standard_normal(N),
        "galah": pred + 0.01 * rng.standard_normal(N),
    }
    r = cross_catalogue_consistency(pred, external, apogee_sigma=0.03)
    assert r.passed


def test_cross_catalogue_consistency_fails_on_biased_external() -> None:
    rng = np.random.default_rng(0)
    N = 500
    pred = rng.standard_normal(N)
    external = {"aspgap": pred + 0.2}  # 0.2 dex systematic bias
    r = cross_catalogue_consistency(pred, external, apogee_sigma=0.03)
    assert not r.passed


def test_cross_catalogue_consistency_empty_fails() -> None:
    r = cross_catalogue_consistency(np.zeros(5), {}, apogee_sigma=0.03)
    assert not r.passed


def test_cross_catalogue_consistency_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        cross_catalogue_consistency(
            np.zeros(5), {"bad": np.zeros(7)}, apogee_sigma=0.03,
        )


# --- Test 6: conditional MI bootstrap ---------------------------------------

def test_conditional_mi_bootstrap_passes_when_xp_carries_residual_signal() -> None:
    """Construct X = Z + signal, Y = signal + eps → I(X;Y|Z) > 0."""
    rng = np.random.default_rng(0)
    N = 400
    z = rng.standard_normal((N, 2))
    signal = rng.standard_normal(N)
    xp = np.stack([z[:, 0] + 0.5 * signal, z[:, 1] + 0.2 * signal], axis=1)
    element = signal + 0.3 * rng.standard_normal(N)
    r = conditional_mi_bootstrap(xp, element, z, n_boot=50, seed=0)
    assert r.passed
    assert r.detail["bootstrap_ci_lower"] > 0.0


def test_conditional_mi_bootstrap_fails_when_element_independent_given_z() -> None:
    rng = np.random.default_rng(0)
    N = 400
    z = rng.standard_normal((N, 2))
    xp = z + 0.3 * rng.standard_normal((N, 2))
    element = z[:, 0] + 0.3 * rng.standard_normal(N)  # only depends on z
    r = conditional_mi_bootstrap(xp, element, z, n_boot=50, seed=0)
    # CI lower should drop to / near zero → fail.
    assert r.detail["bootstrap_ci_lower"] < 0.1


def test_conditional_mi_bootstrap_shape_validation() -> None:
    with pytest.raises(ValueError, match="xp must be 2-D"):
        conditional_mi_bootstrap(np.zeros(10), np.zeros(10), np.zeros((10, 1)))
    with pytest.raises(ValueError, match="element must be 1-D"):
        conditional_mi_bootstrap(np.zeros((10, 1)), np.zeros((10, 1)), np.zeros((10, 1)))
    with pytest.raises(ValueError, match="stellar_params must be 2-D"):
        conditional_mi_bootstrap(np.zeros((10, 1)), np.zeros(10), np.zeros(10))


# --- decision tree --------------------------------------------------------

def _pass(passed: bool = True) -> TestResult:
    return TestResult(passed=passed, statistic=1.0, threshold=0.0)


def test_decision_tree_all_pass_plus_calibration_gives_tier_1() -> None:
    report = tier_promotion_report(
        "Mg",
        test1=_pass(), test2=_pass(), test3=_pass(),
        test4=_pass(), test5=_pass(), test6=_pass(),
        calibration_ok=True,
    )
    assert report.tier == "tier_1"


def test_decision_tree_all_pass_without_calibration_gives_tier_2() -> None:
    report = tier_promotion_report(
        "Mg",
        test1=_pass(), test2=_pass(), test3=_pass(),
        test4=_pass(), test5=_pass(), test6=_pass(),
    )
    assert report.tier == "tier_2"


def test_decision_tree_fails_5_or_6_gives_tier_3_internal() -> None:
    report = tier_promotion_report(
        "Al",
        test1=_pass(), test2=_pass(), test3=_pass(),
        test4=_pass(), test5=_pass(False), test6=_pass(),
    )
    assert report.tier == "tier_3_internal"
    report2 = tier_promotion_report(
        "Al",
        test1=_pass(), test2=_pass(), test3=_pass(),
        test4=_pass(), test5=_pass(), test6=_pass(False),
    )
    assert report2.tier == "tier_3_internal"


def test_decision_tree_fails_physical_gate_rejects() -> None:
    report = tier_promotion_report(
        "Xe",
        test1=_pass(False), test2=_pass(), test3=_pass(),
        test4=_pass(), test5=_pass(), test6=_pass(),
    )
    assert report.tier == "tier_3_rejected"


def test_decision_tree_fails_cluster_or_holdout_rejects() -> None:
    report = tier_promotion_report(
        "Na",
        test1=_pass(), test2=_pass(False), test3=_pass(),
        test4=_pass(), test5=_pass(), test6=_pass(),
    )
    assert report.tier == "tier_3_rejected"


def test_decision_tree_fails_audit_gate_rejects() -> None:
    report = tier_promotion_report(
        "Na",
        test1=_pass(), test2=_pass(), test3=_pass(),
        test4=_pass(False), test5=_pass(), test6=_pass(),
    )
    assert report.tier == "tier_3_rejected"


def test_tier_promotion_report_json_roundtrip() -> None:
    report = tier_promotion_report(
        "Mg",
        test1=_pass(), test2=_pass(), test3=_pass(),
        test4=_pass(), test5=_pass(), test6=_pass(),
        calibration_ok=True,
    )
    assert isinstance(report, TierPromotionReport)
    blob = report.as_dict()
    roundtrip = json.loads(json.dumps(blob))
    assert roundtrip["tier"] == "tier_1"
    assert roundtrip["element"] == "Mg"
    assert set(roundtrip["test1_physical"]) == {"passed", "statistic", "threshold", "detail"}
