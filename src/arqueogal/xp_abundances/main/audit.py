"""Information-content audit — research_brief §9.2 per-label report card.

Six tests (research_brief §9.2):

1. **LOOCO** — Leave-one-coefficient-out at inference. :func:`leave_one_coeff_out`.
2. **Permutation feature importance** — :func:`permutation_feature_importance`.
3. **SHAP values** — *deferred* (requires the ``shap`` library which is not
   in the pinned env; keep a stub placeholder in :func:`audit_report`).
4. **Shuffled-spectrum null** — within-cell permutation of spectrum columns.
   :func:`shuffled_spectrum_null`.
5. **Mutual information** with conditional form. :func:`mutual_information_ksg`
   and :func:`conditional_mi_ksg`.
6. **Decorrelated sub-sample test** — :func:`decorrelated_subsample`.

:func:`audit_report` aggregates tests 1, 2, 4, 5, 6 into one JSON-serialisable
dict — the report card DESIGN §Release gates requires for every released label.

Style notes
-----------
Everything is deliberately numpy/torch-level (no sklearn coupling beyond
k-NN in the KSG estimator). Tests aim to be fast — per-call cost < 1 s on a
10 k-star audit slice — so each metric can be bootstrapped without blowing
the audit budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.special import digamma
from torch import nn
from torch.utils.data import DataLoader

_EPS: float = 1e-12


# --- helpers ----------------------------------------------------------------

def _collect_mu_y(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper: stack ``(mu, y, x)`` across a loader. Model must return ``(mu, L, h, z)``."""
    model.eval().to(device)
    mus, ys, xs = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1]
            mu, _L, _h, _z = model(x)
            mus.append(mu.cpu().numpy())
            ys.append(y.numpy())
            xs.append(x.cpu().numpy())
    return (np.concatenate(mus, 0), np.concatenate(ys, 0), np.concatenate(xs, 0))


def _rmse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-label RMSE between two (N, n) arrays."""
    return np.sqrt(np.maximum(((a - b) ** 2).mean(axis=0), 0.0))


# --- §9.2 Test 1: LOOCO ------------------------------------------------------

def leave_one_coeff_out(
    model: nn.Module,
    loader: DataLoader,
    coeff_indices: list[int] | np.ndarray,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Per-coefficient shift in predicted labels when coefficient is zeroed.

    Parameters
    ----------
    coeff_indices
        Column indices of the flattened input vector that correspond to XP
        coefficients. Every other feature is left intact.

    Returns
    -------
    dict with keys:
        ``"baseline_mu"`` : ``(N, n_labels)`` — predictions with all coeffs.
        ``"per_coeff_delta_rmse"`` : ``(len(coeff_indices), n_labels)`` —
            per-label RMSE of the shift induced by zeroing each coefficient.
    """
    device = device or torch.device("cpu")
    coeff_indices = np.asarray(coeff_indices, dtype=np.int64)
    _, _, x_all = _collect_mu_y(model, loader, device)
    x_t = torch.as_tensor(x_all, device=device)

    with torch.no_grad():
        baseline, _L, _h, _z = model(x_t)
    baseline_np = baseline.cpu().numpy()

    deltas = np.empty((len(coeff_indices), baseline_np.shape[1]), dtype=np.float64)
    for i, c in enumerate(coeff_indices):
        x_masked = x_t.clone()
        x_masked[:, int(c)] = 0.0
        with torch.no_grad():
            mu_masked, _L, _h, _z = model(x_masked)
        deltas[i] = _rmse(mu_masked.cpu().numpy(), baseline_np)
    return {"baseline_mu": baseline_np, "per_coeff_delta_rmse": deltas}


# --- §9.2 Test 2: Permutation feature importance ----------------------------

def permutation_feature_importance(
    model: nn.Module,
    loader: DataLoader,
    feature_indices: list[int] | np.ndarray,
    *,
    device: torch.device | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Per-feature RMSE increase when that feature is randomly permuted.

    Returns
    -------
    dict with:
        ``"baseline_rmse"`` — ``(n_labels,)`` baseline per-label RMSE.
        ``"permuted_rmse"`` — ``(len(feature_indices), n_labels)``.
        ``"importance"`` — ``permuted_rmse - baseline_rmse``.
    """
    device = device or torch.device("cpu")
    rng = np.random.default_rng(seed)
    feature_indices = np.asarray(feature_indices, dtype=np.int64)
    mu0, y, x_all = _collect_mu_y(model, loader, device)
    baseline = _rmse(mu0, y)
    x_t = torch.as_tensor(x_all, device=device)

    out = np.empty((len(feature_indices), baseline.shape[0]), dtype=np.float64)
    for i, f in enumerate(feature_indices):
        perm = rng.permutation(x_all.shape[0])
        x_perm = x_t.clone()
        x_perm[:, int(f)] = x_t[perm, int(f)]
        with torch.no_grad():
            mu_perm, _L, _h, _z = model(x_perm)
        out[i] = _rmse(mu_perm.cpu().numpy(), y)

    return {
        "baseline_rmse": baseline,
        "permuted_rmse": out,
        "importance": out - baseline,
    }


# --- §9.2 Test 4: Shuffled-spectrum null -------------------------------------

def shuffled_spectrum_null(  # noqa: PLR0913 — per-test knobs are explicit by design
    model: nn.Module,
    loader: DataLoader,
    spectrum_indices: list[int] | np.ndarray,
    *,
    cell_ids: np.ndarray | None = None,
    device: torch.device | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Permute every spectrum column jointly within each cell; measure RMSE.

    Labels whose RMSE stays low under this null are prior-driven — the model
    recovered them from auxiliaries alone. Cells with < 4 stars are skipped
    (permutation impossible).
    """
    device = device or torch.device("cpu")
    rng = np.random.default_rng(seed)
    spectrum_indices = np.asarray(spectrum_indices, dtype=np.int64)
    _mu0, y, x_all = _collect_mu_y(model, loader, device)
    if cell_ids is None:
        cell_ids = np.zeros(x_all.shape[0], dtype=np.int64)
    if cell_ids.shape[0] != x_all.shape[0]:
        raise ValueError(
            f"cell_ids length {cell_ids.shape[0]} != N={x_all.shape[0]}"
        )

    x_shuf = x_all.copy()
    for c in np.unique(cell_ids):
        mask = np.flatnonzero(cell_ids == c)
        if mask.size < 4:
            continue
        perm = rng.permutation(mask)
        x_shuf[np.ix_(mask, spectrum_indices)] = x_all[np.ix_(perm, spectrum_indices)]

    with torch.no_grad():
        mu_null, _L, _h, _z = model(torch.as_tensor(x_shuf, device=device))
    return {"null_rmse": _rmse(mu_null.cpu().numpy(), y)}


# --- §9.2 Test 5: Mutual information (KSG) ----------------------------------

def mutual_information_ksg(
    x: np.ndarray, y: np.ndarray, *, k: int = 5,
) -> float:
    """KSG Algorithm 1 mutual information estimator (Kraskov+2004).

    ``x`` and ``y`` may be 1-D or 2-D (``(N,)`` or ``(N, d)``). Returns a
    non-negative scalar nat-valued MI estimate. For small samples or near-zero
    MI the estimator can produce small negative numbers; we clamp to 0.
    """
    x = np.atleast_2d(x).T if x.ndim == 1 else x
    y = np.atleast_2d(y).T if y.ndim == 1 else y
    n = x.shape[0]
    if n < k + 2:
        raise ValueError(f"need at least k+2={k + 2} samples, got {n}")

    xy = np.concatenate([x, y], axis=1)
    # Max-norm distance to the k-th neighbour in joint space.
    tree_xy = cKDTree(xy)
    eps = tree_xy.query(xy, k=k + 1, p=np.inf)[0][:, -1]
    # Strictly less than eps per KSG Algorithm 1.
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)
    nx = np.array([
        len(tree_x.query_ball_point(x[i], eps[i] - _EPS, p=np.inf))
        for i in range(n)
    ], dtype=np.float64)
    ny = np.array([
        len(tree_y.query_ball_point(y[i], eps[i] - _EPS, p=np.inf))
        for i in range(n)
    ], dtype=np.float64)
    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
    return float(max(mi, 0.0))


def conditional_mi_ksg(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, *, k: int = 5,
) -> float:
    """Conditional MI ``I(X;Y | Z)`` via KSG (Frenzel & Pompe 2007).

    Same domain conventions as :func:`mutual_information_ksg`. ``z`` carries
    the conditioning variables; pass an empty ``(N, 0)`` array to recover
    unconditional MI.
    """
    x = np.atleast_2d(x).T if x.ndim == 1 else x
    y = np.atleast_2d(y).T if y.ndim == 1 else y
    z = np.atleast_2d(z).T if z.ndim == 1 else z
    if z.shape[1] == 0:
        return mutual_information_ksg(x, y, k=k)

    n = x.shape[0]
    xyz = np.concatenate([x, y, z], axis=1)
    xz = np.concatenate([x, z], axis=1)
    yz = np.concatenate([y, z], axis=1)

    tree_xyz = cKDTree(xyz)
    eps = tree_xyz.query(xyz, k=k + 1, p=np.inf)[0][:, -1]
    tree_xz, tree_yz, tree_z = cKDTree(xz), cKDTree(yz), cKDTree(z)
    nz = np.array([
        len(tree_z.query_ball_point(z[i], eps[i] - _EPS, p=np.inf))
        for i in range(n)
    ], dtype=np.float64)
    nxz = np.array([
        len(tree_xz.query_ball_point(xz[i], eps[i] - _EPS, p=np.inf))
        for i in range(n)
    ], dtype=np.float64)
    nyz = np.array([
        len(tree_yz.query_ball_point(yz[i], eps[i] - _EPS, p=np.inf))
        for i in range(n)
    ], dtype=np.float64)
    cmi = (
        digamma(k)
        + np.mean(digamma(nz + 1) - digamma(nxz + 1) - digamma(nyz + 1))
    )
    return float(max(cmi, 0.0))


# --- §9.2 Test 6: Decorrelated sub-sample -----------------------------------

def decorrelated_subsample(
    labels: np.ndarray,
    priors: np.ndarray,
    *,
    n_bins: int = 4,
    seed: int = 0,
) -> np.ndarray:
    """Return indices of a sub-sample where ``labels`` is decorrelated from ``priors``.

    Joint-bins ``priors``, then within each prior-cell samples uniformly over
    the ``labels`` distribution (reweighting per cell to approximate a
    matched-propensity sample). The intent: retain a subsample where the
    prior's explanatory power for the label has been neutralised; the model's
    residual skill on this subsample is spectrum-driven.
    """
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got {labels.shape}")
    if priors.ndim != 2:
        raise ValueError(f"priors must be 2D, got {priors.shape}")

    rng = np.random.default_rng(seed)
    n = labels.shape[0]
    # Bin each prior column independently, then joint-encode.
    codes = np.zeros(n, dtype=np.int64)
    for j in range(priors.shape[1]):
        col = priors[:, j]
        edges = np.quantile(col, np.linspace(0, 1, n_bins + 1)[1:-1])
        idx = np.clip(np.digitize(col, edges), 0, n_bins - 1)
        codes = codes * n_bins + idx

    label_bins = np.clip(
        np.digitize(labels, np.quantile(labels, np.linspace(0, 1, n_bins + 1)[1:-1])),
        0, n_bins - 1,
    )

    # Within each (prior-cell, label-bin) keep up to min-cell-size samples to
    # break the label↔prior correlation while preserving each joint-bin mass.
    keep: list[int] = []
    for jb in np.unique(codes):
        per_label_counts = [
            int((label_bins[codes == jb] == lb).sum()) for lb in range(n_bins)
        ]
        if not per_label_counts or min(per_label_counts) == 0:
            continue
        target = min(per_label_counts)
        for lb in range(n_bins):
            mask = np.flatnonzero((codes == jb) & (label_bins == lb))
            keep.extend(rng.choice(mask, size=target, replace=False))
    return np.asarray(sorted(keep), dtype=np.int64)


# --- §9.2 Orchestrator -------------------------------------------------------

@dataclass
class AuditReport:
    """Container for the per-label §9.2 report card.

    Stored as plain dicts/arrays so :meth:`as_dict` can be JSON-dumped for
    release documentation.
    """

    label_names: tuple[str, ...]
    baseline_rmse: np.ndarray
    shuffled_null_rmse: np.ndarray
    permutation_importance: np.ndarray
    feature_names: tuple[str, ...]
    looco_delta_rmse: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    coefficient_indices: tuple[int, ...] = ()
    mi_conditional: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label_names": list(self.label_names),
            "feature_names": list(self.feature_names),
            "coefficient_indices": list(self.coefficient_indices),
            "baseline_rmse": self.baseline_rmse.tolist(),
            "shuffled_null_rmse": self.shuffled_null_rmse.tolist(),
            "permutation_importance": self.permutation_importance.tolist(),
            "looco_delta_rmse": self.looco_delta_rmse.tolist(),
            "mi_conditional": dict(self.mi_conditional),
        }


def audit_report(  # noqa: PLR0913 — orchestrator with per-test knobs
    model: nn.Module,
    loader: DataLoader,
    *,
    label_names: tuple[str, ...],
    feature_names: tuple[str, ...],
    coefficient_indices: list[int] | np.ndarray | None = None,
    spectrum_indices: list[int] | np.ndarray | None = None,
    permutation_feature_indices: list[int] | np.ndarray | None = None,
    cell_ids: np.ndarray | None = None,
    device: torch.device | None = None,
    seed: int = 0,
) -> AuditReport:
    """Run §9.2 tests 1, 2, 4 and return a structured :class:`AuditReport`.

    MI tests 5 and 6 aren't run inside this coordinator — they need
    off-model data (labels, priors, sub-sample), and users wire them up
    in release scripts with :func:`conditional_mi_ksg` and
    :func:`decorrelated_subsample` directly.
    """
    device = device or torch.device("cpu")
    mu0, y, _x = _collect_mu_y(model, loader, device)
    baseline = _rmse(mu0, y)

    spectrum_indices = (
        np.asarray(spectrum_indices, dtype=np.int64)
        if spectrum_indices is not None else np.asarray([], dtype=np.int64)
    )
    permutation_feature_indices = (
        np.asarray(permutation_feature_indices, dtype=np.int64)
        if permutation_feature_indices is not None
        else np.arange(len(feature_names), dtype=np.int64)
    )

    null_rmse = (
        shuffled_spectrum_null(
            model, loader, spectrum_indices, cell_ids=cell_ids,
            device=device, seed=seed,
        )["null_rmse"] if spectrum_indices.size > 0
        else np.full(len(label_names), np.nan, dtype=np.float64)
    )

    perm = permutation_feature_importance(
        model, loader, permutation_feature_indices, device=device, seed=seed,
    )

    looco = (
        leave_one_coeff_out(
            model, loader, coefficient_indices, device=device,
        )["per_coeff_delta_rmse"] if coefficient_indices is not None
        else np.zeros((0, len(label_names)))
    )

    return AuditReport(
        label_names=label_names,
        feature_names=feature_names,
        coefficient_indices=(tuple(coefficient_indices)
                             if coefficient_indices is not None else ()),
        baseline_rmse=baseline,
        shuffled_null_rmse=null_rmse,
        permutation_importance=perm["importance"],
        looco_delta_rmse=looco,
    )


__all__ = [
    "AuditReport",
    "audit_report",
    "conditional_mi_ksg",
    "decorrelated_subsample",
    "leave_one_coeff_out",
    "mutual_information_ksg",
    "permutation_feature_importance",
    "shuffled_spectrum_null",
]
