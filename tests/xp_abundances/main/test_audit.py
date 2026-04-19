"""Tests for xp_abundances.main.audit — §9.2 information-content report card."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from arqueogal.xp_abundances.main.audit import (
    AuditReport,
    audit_report,
    conditional_mi_ksg,
    decorrelated_subsample,
    leave_one_coeff_out,
    mutual_information_ksg,
    permutation_feature_importance,
    shuffled_spectrum_null,
)

# --- tiny deterministic model ----------------------------------------------

class _LinearMockModel(nn.Module):
    """Deterministic ``mu = x @ W + b`` model returning ``(mu, L, h, z)``.

    Lets us pin which features are relevant by constructing ``W``. The ``L``
    return is an identity Cholesky (audit ignores it); ``h`` and ``z`` echo
    the input so ``(mu, L, h, z)`` matches the XpAbundanceModel signature.
    """

    def __init__(self, W: np.ndarray, b: np.ndarray | None = None) -> None:
        super().__init__()
        W_t = torch.as_tensor(W, dtype=torch.float32)
        b_t = torch.as_tensor(
            b if b is not None else np.zeros(W.shape[1]), dtype=torch.float32,
        )
        self.register_buffer("W", W_t)
        self.register_buffer("b", b_t)
        self.n_labels = W.shape[1]

    def forward(self, x: torch.Tensor):  # noqa: ANN001 — torch module contract
        mu = x @ self.W + self.b
        B = x.shape[0]
        L = torch.eye(self.n_labels).expand(B, -1, -1)
        return mu, L, x, x


def _loader_from_arrays(X: np.ndarray, Y: np.ndarray, batch_size: int = 32) -> DataLoader:
    ds = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(Y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size)


# --- §9.2 Test 1: LOOCO -----------------------------------------------------

def test_leave_one_coeff_out_shapes_and_nonzero_for_relevant_coeff() -> None:
    rng = np.random.default_rng(0)
    N, D, n_labels = 100, 5, 2
    W = np.zeros((D, n_labels), dtype=np.float32)
    W[0, 0] = 1.0  # label 0 ONLY depends on coefficient 0
    W[1, 1] = 1.0  # label 1 ONLY depends on coefficient 1
    model = _LinearMockModel(W)
    X = rng.standard_normal((N, D)).astype(np.float32)
    Y = X @ W
    loader = _loader_from_arrays(X, Y)

    out = leave_one_coeff_out(model, loader, coeff_indices=[0, 1, 2])
    assert out["baseline_mu"].shape == (N, n_labels)
    assert out["per_coeff_delta_rmse"].shape == (3, n_labels)
    # Zeroing coefficient 0 must move label-0 prediction; label-1 unchanged.
    assert out["per_coeff_delta_rmse"][0, 0] > 1e-3
    assert out["per_coeff_delta_rmse"][0, 1] < 1e-6
    # Coefficient 2 is irrelevant to both labels.
    assert np.all(out["per_coeff_delta_rmse"][2] < 1e-6)


# --- §9.2 Test 2: Permutation feature importance ----------------------------

def test_permutation_importance_positive_for_relevant_feature() -> None:
    rng = np.random.default_rng(0)
    N, D = 200, 4
    W = np.zeros((D, 1), dtype=np.float32)
    W[0, 0] = 1.0  # only feature 0 matters
    model = _LinearMockModel(W)
    X = rng.standard_normal((N, D)).astype(np.float32)
    Y = X @ W
    loader = _loader_from_arrays(X, Y)

    out = permutation_feature_importance(model, loader, feature_indices=[0, 2])
    assert out["baseline_rmse"].shape == (1,)
    assert out["permuted_rmse"].shape == (2, 1)
    assert out["importance"].shape == (2, 1)
    # Permuting feature 0 (relevant) must raise RMSE materially.
    assert out["importance"][0, 0] > 0.1
    # Permuting feature 2 (irrelevant) leaves RMSE unchanged.
    assert abs(out["importance"][1, 0]) < 1e-6


# --- §9.2 Test 4: Shuffled-spectrum null ------------------------------------

def test_shuffled_spectrum_null_inflates_rmse_for_spectrum_driven_label() -> None:
    rng = np.random.default_rng(0)
    N, D = 300, 6
    W = np.zeros((D, 1), dtype=np.float32)
    # Label is entirely driven by "spectrum" columns 0 and 1.
    W[0, 0] = 1.0
    W[1, 0] = -0.5
    model = _LinearMockModel(W)
    X = rng.standard_normal((N, D)).astype(np.float32)
    Y = X @ W
    loader = _loader_from_arrays(X, Y)

    out = shuffled_spectrum_null(model, loader, spectrum_indices=[0, 1])
    # Within-batch shuffle destroys the signal → RMSE >> baseline (≈0).
    assert out["null_rmse"].shape == (1,)
    assert out["null_rmse"][0] > 0.5


def test_shuffled_spectrum_null_preserves_label_if_only_nonspectrum_drives_it() -> None:
    rng = np.random.default_rng(0)
    N, D = 200, 5
    W = np.zeros((D, 1), dtype=np.float32)
    W[4, 0] = 1.0  # only the last (non-spectrum) feature matters
    model = _LinearMockModel(W)
    X = rng.standard_normal((N, D)).astype(np.float32)
    Y = X @ W
    loader = _loader_from_arrays(X, Y)
    out = shuffled_spectrum_null(model, loader, spectrum_indices=[0, 1, 2, 3])
    # Shuffling spectrum cols leaves column 4 untouched → RMSE stays ≈0.
    assert out["null_rmse"][0] < 1e-5


def test_shuffled_spectrum_null_cell_length_validation() -> None:
    rng = np.random.default_rng(0)
    N, D = 20, 4
    model = _LinearMockModel(np.zeros((D, 1), dtype=np.float32))
    loader = _loader_from_arrays(
        rng.standard_normal((N, D)).astype(np.float32),
        np.zeros((N, 1), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="cell_ids length"):
        shuffled_spectrum_null(
            model, loader, spectrum_indices=[0],
            cell_ids=np.zeros(999, dtype=np.int64),
        )


# --- §9.2 Test 5: KSG mutual information ------------------------------------

def test_ksg_mi_linear_gaussian_matches_analytic() -> None:
    """For ρ-correlated bivariate Gaussian, true MI = -0.5 ln(1-ρ²)."""
    rng = np.random.default_rng(0)
    rho = 0.8
    cov = np.array([[1.0, rho], [rho, 1.0]])
    samples = rng.multivariate_normal([0, 0], cov, size=3000)
    mi_est = mutual_information_ksg(samples[:, 0], samples[:, 1], k=5)
    mi_true = -0.5 * np.log(1.0 - rho**2)
    assert abs(mi_est - mi_true) < 0.1, f"KSG MI {mi_est} vs analytic {mi_true}"


def test_ksg_mi_independent_variables_near_zero() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    y = rng.standard_normal(2000)
    mi_est = mutual_information_ksg(x, y, k=5)
    assert mi_est < 0.05


def test_ksg_mi_rejects_undersized_sample() -> None:
    x = np.zeros(3)
    y = np.zeros(3)
    with pytest.raises(ValueError, match="at least k"):
        mutual_information_ksg(x, y, k=5)


def test_conditional_mi_zero_when_xy_independent_given_z() -> None:
    """X = Z + εx, Y = Z + εy with εx ⟂ εy → I(X;Y|Z) ≈ 0."""
    rng = np.random.default_rng(0)
    N = 2000
    z = rng.standard_normal(N)
    x = z + 0.3 * rng.standard_normal(N)
    y = z + 0.3 * rng.standard_normal(N)
    cmi = conditional_mi_ksg(x, y, z, k=5)
    # Unconditional MI should be clearly positive (shared driver z).
    mi_unc = mutual_information_ksg(x, y, k=5)
    assert mi_unc > 0.3
    # Conditioning on z should collapse it near zero.
    assert cmi < 0.1, f"CMI {cmi} should be near zero"


def test_conditional_mi_empty_z_equals_unconditional_mi() -> None:
    rng = np.random.default_rng(0)
    N = 1000
    x = rng.standard_normal(N)
    y = 0.5 * x + rng.standard_normal(N)
    z_empty = np.empty((N, 0))
    cmi = conditional_mi_ksg(x, y, z_empty, k=5)
    mi = mutual_information_ksg(x, y, k=5)
    assert abs(cmi - mi) < 1e-9  # empty-z path delegates to MI


# --- §9.2 Test 6: Decorrelated sub-sample -----------------------------------

def test_decorrelated_subsample_smaller_and_reduces_correlation() -> None:
    rng = np.random.default_rng(0)
    N = 2000
    prior = rng.standard_normal((N, 1))
    # Moderate correlation: each prior-bin still contains multiple label bins,
    # so the function can reweight. Near-perfect coupling is legitimately
    # undecorrelatable.
    label = 0.5 * prior[:, 0] + rng.standard_normal(N)

    idx = decorrelated_subsample(label, prior, n_bins=4, seed=0)
    assert idx.size > 0
    assert idx.size < N
    rho_full = np.corrcoef(label, prior[:, 0])[0, 1]
    rho_sub = np.corrcoef(label[idx], prior[idx, 0])[0, 1]
    assert abs(rho_sub) < abs(rho_full)


def test_decorrelated_subsample_validates_shapes() -> None:
    with pytest.raises(ValueError, match="labels must be 1D"):
        decorrelated_subsample(np.zeros((10, 2)), np.zeros((10, 1)))
    with pytest.raises(ValueError, match="priors must be 2D"):
        decorrelated_subsample(np.zeros(10), np.zeros(10))


# --- orchestrator -----------------------------------------------------------

def test_audit_report_schema_and_json_serialisable() -> None:
    rng = np.random.default_rng(0)
    N, D, n_labels = 150, 6, 2
    W = np.zeros((D, n_labels), dtype=np.float32)
    W[0, 0] = 1.0
    W[3, 1] = 1.0
    model = _LinearMockModel(W)
    X = rng.standard_normal((N, D)).astype(np.float32)
    Y = X @ W
    loader = _loader_from_arrays(X, Y)

    report = audit_report(
        model, loader,
        label_names=("label_a", "label_b"),
        feature_names=tuple(f"f{i}" for i in range(D)),
        coefficient_indices=[0, 1],
        spectrum_indices=[0, 1, 2],
        permutation_feature_indices=[0, 3],
        cell_ids=np.zeros(N, dtype=np.int64),
    )
    assert isinstance(report, AuditReport)
    assert report.baseline_rmse.shape == (n_labels,)
    assert report.shuffled_null_rmse.shape == (n_labels,)
    assert report.permutation_importance.shape == (2, n_labels)
    assert report.looco_delta_rmse.shape == (2, n_labels)

    # JSON roundtrip check — this is the release-artefact contract.
    blob = report.as_dict()
    round_trip = json.loads(json.dumps(blob))
    assert set(round_trip) == {
        "label_names", "feature_names", "coefficient_indices",
        "baseline_rmse", "shuffled_null_rmse", "permutation_importance",
        "looco_delta_rmse", "mi_conditional",
    }


def test_audit_report_without_optional_inputs() -> None:
    """No coefficient_indices, no spectrum_indices → looco/null fields empty/NaN."""
    rng = np.random.default_rng(0)
    N, D = 60, 3
    W = np.zeros((D, 1), dtype=np.float32)
    W[0, 0] = 1.0
    model = _LinearMockModel(W)
    X = rng.standard_normal((N, D)).astype(np.float32)
    Y = X @ W
    loader = _loader_from_arrays(X, Y)

    report = audit_report(
        model, loader,
        label_names=("only_label",),
        feature_names=("f0", "f1", "f2"),
    )
    assert report.looco_delta_rmse.shape == (0, 1)
    assert np.isnan(report.shuffled_null_rmse).all()
    assert report.permutation_importance.shape == (D, 1)
