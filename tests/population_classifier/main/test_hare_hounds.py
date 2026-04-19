"""Tests for population_classifier.main.hare_hounds — FIRE-2 validation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from arqueogal.population_classifier.main.hare_hounds import (
    HareHoundsReport,
    compute_hare_hounds_metrics,
)

# --- perfect / mismatch sanity ------------------------------------------


def test_perfect_match_gives_unit_metrics() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    y_pred = np.array([5, 5, 6, 6, 7, 7], dtype=np.int64)  # relabeled perfect
    report = compute_hare_hounds_metrics(y_pred, y_true)
    assert isinstance(report, HareHoundsReport)
    assert report.ari == pytest.approx(1.0)
    assert report.ami == pytest.approx(1.0)
    assert report.mcc == pytest.approx(1.0)
    assert report.youden_j == pytest.approx(1.0)
    assert report.n_stars_compared == 6
    # Hungarian best-match should pair each predicted id with its true id.
    assert report.match == {5: 0, 6: 1, 7: 2}


def test_complete_disagreement_gives_low_ari() -> None:
    # Predicted label unrelated to truth: ARI should be ≈ 0.
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    report = compute_hare_hounds_metrics(y_pred, y_true)
    assert report.ari < 0.1
    assert abs(report.mcc) < 0.1


# --- ignore rules --------------------------------------------------------


def test_ignore_predicted_noise_drops_minus_one() -> None:
    y_true = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    y_pred = np.array([-1, 0, 1, 1, 1], dtype=np.int64)
    report = compute_hare_hounds_metrics(y_pred, y_true)
    assert report.n_stars_compared == 4
    assert -1 not in report.predicted_ids


def test_ignore_true_drops_specified_labels() -> None:
    y_true = np.array([9, 0, 0, 1, 1], dtype=np.int64)  # 9 is "merger" — drop.
    y_pred = np.array([3, 3, 3, 4, 4], dtype=np.int64)
    report = compute_hare_hounds_metrics(y_pred, y_true, ignore_true=(9,))
    assert report.n_stars_compared == 4
    assert 9 not in report.true_ids


# --- contingency & Hungarian match --------------------------------------


def test_contingency_shape_and_counts() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    y_pred = np.array([5, 5, 6, 7, 7, 7], dtype=np.int64)
    report = compute_hare_hounds_metrics(y_pred, y_true)
    assert report.contingency.shape == (3, 3)
    assert report.contingency.sum() == 6
    # Predicted 5 → true 0 (best 2/2)
    i5 = report.predicted_ids.index(5)
    j0 = report.true_ids.index(0)
    assert report.contingency[i5, j0] == 2


def test_hungarian_match_returns_dict_keyed_by_predicted_id() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_pred = np.array([9, 9, 8, 8], dtype=np.int64)
    report = compute_hare_hounds_metrics(y_pred, y_true)
    # With 2×2 perfect overlap, each predicted id maps to exactly one true id.
    assert set(report.match.keys()) == {9, 8}
    assert set(report.match.values()) == {0, 1}


# --- per-cluster precision / recall -------------------------------------


def test_per_cluster_precision_and_recall_semantics() -> None:
    y_true = np.array([0, 0, 0, 1], dtype=np.int64)
    y_pred = np.array([5, 5, 5, 6], dtype=np.int64)
    report = compute_hare_hounds_metrics(y_pred, y_true)
    # predicted cluster 5 (size 3, 3 of true 0) → precision = 3/3 = 1.0
    i5 = report.predicted_ids.index(5)
    assert report.per_cluster_precision[i5] == pytest.approx(1.0)
    # true label 0 (size 3, 3 in predicted 5) → recall = 3/3 = 1.0
    j0 = report.true_ids.index(0)
    assert report.per_cluster_recall[j0] == pytest.approx(1.0)


# --- guard rails ---------------------------------------------------------


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_hare_hounds_metrics(
            np.zeros(5, dtype=np.int64), np.zeros(6, dtype=np.int64),
        )


def test_nan_output_when_fewer_than_two_stars() -> None:
    # After dropping noise, only one star survives — metrics must be NaN.
    y_pred = np.array([-1, -1, -1, 0], dtype=np.int64)
    y_true = np.array([0, 0, 0, 0], dtype=np.int64)
    report = compute_hare_hounds_metrics(y_pred, y_true)
    assert np.isnan(report.ari)
    assert np.isnan(report.ami)
    assert np.isnan(report.mcc)
    assert np.isnan(report.youden_j)
    assert report.n_stars_compared == 1
    assert report.contingency.shape == (0, 0)


def test_multiclass_three_populations_match_identity() -> None:
    rng = np.random.default_rng(0)
    # Three populations, 40 stars each, predictions mostly agree (90%).
    y_true_list: list[int] = []
    y_pred_list: list[int] = []
    for c in range(3):
        for _ in range(40):
            y_true_list.append(c)
            y_pred_list.append(c if rng.uniform() < 0.9 else (c + 1) % 3)
    y_true = np.asarray(y_true_list, dtype=np.int64)
    y_pred = np.asarray(y_pred_list, dtype=np.int64)
    report = compute_hare_hounds_metrics(y_pred, y_true)
    assert report.ari > 0.5
    assert report.ami > 0.5
    assert report.mcc > 0.5
    # Identity mapping is the correct Hungarian assignment.
    assert report.match == {0: 0, 1: 1, 2: 2}


def test_predicted_and_true_id_tuples_are_sorted_unique() -> None:
    y_true = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    y_pred = np.array([7, 7, 8, 8, 9], dtype=np.int64)
    report = compute_hare_hounds_metrics(y_pred, y_true)
    assert list(report.predicted_ids) == sorted(set(report.predicted_ids))
    assert list(report.true_ids) == sorted(set(report.true_ids))
