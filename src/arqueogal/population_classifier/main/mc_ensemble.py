"""MC ensemble over Pipeline-1 feature uncertainties — research_brief §10.3.

D-Cat-d requires per-star population-membership probabilities that
*include* the propagated uncertainty of the Pipeline-1 chemical
abundances, ages, and kinematics. This module orchestrates the
standard §10.3 / DESIGN §10.3 procedure:

1. Fit Parametric UMAP + HDBSCAN **once** on the mean feature matrix
   ``X_mean`` — that fixes the *reference* cluster structure.
2. Draw ``N_MC = 50`` posterior realisations of the feature matrix
   ``X_k ~ N(mu_i, Σ_i)`` per star using the Pipeline-1 calibrated
   covariance.
3. For each ``X_k`` run a user-supplied ``predict_soft_fn`` — usually
   ``ParametricUMAP.transform`` followed by
   ``hdbscan.approximate_predict`` + ``prediction_data`` — that returns
   a ``(N, K)`` soft-membership matrix *aligned* to the reference
   cluster ids.
4. Aggregate: per-star mean and std of soft memberships across the
   ``N_MC`` realisations, plus the DESIGN §10.3 boundary flag
   ``std_over_mc > 0.15``.

**Why the reference is fixed.** Re-fitting UMAP + HDBSCAN on every MC
sample fragments the cluster structure differently in each realisation
— there's no canonical matching across MC runs. Fixing the reference
makes the MC a pure *uncertainty-propagation* computation: the
quantity of interest is the per-star soft-membership distribution
against the already-validated §10.5 diagnostic stack, not the
hyperparameter sensitivity (that lives in :mod:`.hyperparameter`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PredictSoftFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class MCEnsembleConfig:
    """§10.3 knobs.

    ``n_mc``: research_brief §10.3 defaults to 50.
    ``seed``: reproducibility.
    ``mc_boundary_threshold``: DESIGN §10.3 — per-star boundary flag
    triggers when any cluster's std-across-MC exceeds this value
    (default 0.15).
    """

    n_mc: int = 50
    seed: int = 0
    mc_boundary_threshold: float = 0.15


@dataclass
class MCEnsembleResult:
    """Output of :func:`run_mc_ensemble`.

    ``mean_soft`` / ``std_soft``: ``(N, K)`` per-star soft-membership
    mean and std across the ``N_MC`` realisations. ``K`` is set by the
    reference clusterer.
    ``consensus_labels``: ``argmax(mean_soft)`` with ``-1`` where the
    max falls below ``assign_threshold``.
    ``mc_boundary_flag``: ``(N,)`` True where ``max_k std_soft[:, k] >
    mc_boundary_threshold``. Combines additively with the §10.3
    max-soft-membership threshold (that flag lives in
    :mod:`.clustering`).
    ``cluster_ids``: ordered ids corresponding to the columns of
    ``mean_soft`` / ``std_soft``.
    ``n_clusters``: ``K``.
    ``n_mc``: number of MC realisations run.
    ``extras``: caller-populated passthrough (per-run timings, etc.).
    """

    mean_soft: np.ndarray
    std_soft: np.ndarray
    consensus_labels: np.ndarray
    mc_boundary_flag: np.ndarray
    cluster_ids: tuple[int, ...]
    n_clusters: int
    n_mc: int
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mean_soft.shape != self.std_soft.shape:
            raise ValueError(
                f"mean_soft {self.mean_soft.shape} / std_soft "
                f"{self.std_soft.shape} shape mismatch",
            )
        n = self.mean_soft.shape[0]
        if self.consensus_labels.shape != (n,):
            raise ValueError(
                f"consensus_labels shape {self.consensus_labels.shape} != ({n},)",
            )
        if self.mc_boundary_flag.shape != (n,):
            raise ValueError(
                f"mc_boundary_flag shape {self.mc_boundary_flag.shape} != ({n},)",
            )


def _sample_one_realisation(
    X_mean: np.ndarray,
    cov_or_sigma: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one ``X_k`` realisation ``(N, D)``.

    ``cov_or_sigma`` shape dispatch:
    - ``(N, D)``: per-star diagonal σ (1σ).
    - ``(N, D, D)``: full per-star covariance.
    """
    N, D = X_mean.shape
    z = rng.standard_normal(size=(N, D))
    if cov_or_sigma.ndim == 2:
        if cov_or_sigma.shape != (N, D):
            raise ValueError(
                f"sigma shape {cov_or_sigma.shape} != ({N}, {D})",
            )
        return X_mean + cov_or_sigma * z
    if cov_or_sigma.ndim == 3:
        if cov_or_sigma.shape != (N, D, D):
            raise ValueError(
                f"cov shape {cov_or_sigma.shape} != ({N}, {D}, {D})",
            )
        # Per-star Cholesky → X_k = mu + L @ z.
        L = np.linalg.cholesky(cov_or_sigma)
        return X_mean + np.einsum("nij,nj->ni", L, z)
    raise ValueError(
        f"cov_or_sigma must be 2-D or 3-D, got {cov_or_sigma.ndim}-D",
    )


def sample_feature_posteriors(
    X_mean: np.ndarray,
    cov_or_sigma: np.ndarray,
    *,
    n_mc: int = 50,
    seed: int = 0,
) -> np.ndarray:
    """Draw ``n_mc`` posterior realisations — shape ``(n_mc, N, D)``.

    Useful standalone when a caller wants to inspect the posterior
    samples directly; :func:`run_mc_ensemble` calls the single-sample
    helper itself to avoid holding the entire ``(n_mc, N, D)`` array in
    memory for large N.
    """
    if X_mean.ndim != 2:
        raise ValueError(f"X_mean must be 2-D, got shape {X_mean.shape}")
    rng = np.random.default_rng(seed)
    N, D = X_mean.shape
    out = np.empty((n_mc, N, D), dtype=np.float64)
    for k in range(n_mc):
        out[k] = _sample_one_realisation(X_mean, cov_or_sigma, rng=rng)
    return out


def run_mc_ensemble(  # noqa: PLR0913 — algorithm-driven signature
    X_mean: np.ndarray,
    cov_or_sigma: np.ndarray,
    predict_soft_fn: PredictSoftFn,
    *,
    cluster_ids: tuple[int, ...],
    config: MCEnsembleConfig | None = None,
    assign_threshold: float = 0.5,
) -> MCEnsembleResult:
    """Full MC orchestration per research_brief §10.3.

    ``predict_soft_fn(X_k) -> (N, K)`` is a pre-fitted callable returning
    soft memberships over the fixed reference cluster ids. The caller
    is responsible for alignment — typically by wrapping a pre-fitted
    :class:`.ParametricUMAP` + ``hdbscan`` pair.
    """
    if X_mean.ndim != 2:
        raise ValueError(f"X_mean must be 2-D, got shape {X_mean.shape}")
    if not np.isfinite(X_mean).all():
        raise ValueError("X_mean contains NaN/Inf; impute before MC")
    if not np.isfinite(cov_or_sigma).all():
        raise ValueError("cov_or_sigma contains NaN/Inf")

    cfg = config or MCEnsembleConfig()
    rng = np.random.default_rng(cfg.seed)
    N = X_mean.shape[0]
    K = len(cluster_ids)

    running_sum = np.zeros((N, K), dtype=np.float64)
    running_sqsum = np.zeros((N, K), dtype=np.float64)

    for _ in range(cfg.n_mc):
        X_k = _sample_one_realisation(X_mean, cov_or_sigma, rng=rng)
        soft_k_raw = np.asarray(predict_soft_fn(X_k))
        if soft_k_raw.shape != (N, K):
            raise ValueError(
                f"predict_soft_fn returned shape {soft_k_raw.shape}, "
                f"expected ({N}, {K})",
            )
        soft_k = soft_k_raw.astype(np.float64, copy=False)
        running_sum += soft_k
        running_sqsum += soft_k * soft_k

    mean_soft = (running_sum / cfg.n_mc).astype(np.float32)
    # Population variance → std (we want the spread across realisations).
    var = running_sqsum / cfg.n_mc - (running_sum / cfg.n_mc) ** 2
    std_soft = np.sqrt(np.clip(var, 0.0, None)).astype(np.float32)

    if K > 0:
        max_prob = mean_soft.max(axis=1)
        argmax = mean_soft.argmax(axis=1)
        consensus = np.where(
            max_prob >= assign_threshold,
            np.asarray(cluster_ids, dtype=np.int64)[argmax],
            -1,
        ).astype(np.int64)
        mc_boundary = (std_soft.max(axis=1) > cfg.mc_boundary_threshold)
    else:
        consensus = np.full(N, -1, dtype=np.int64)
        mc_boundary = np.ones(N, dtype=bool)

    return MCEnsembleResult(
        mean_soft=mean_soft,
        std_soft=std_soft,
        consensus_labels=consensus,
        mc_boundary_flag=mc_boundary.astype(bool),
        cluster_ids=cluster_ids,
        n_clusters=K,
        n_mc=cfg.n_mc,
    )


__all__ = [
    "MCEnsembleConfig",
    "MCEnsembleResult",
    "run_mc_ensemble",
    "sample_feature_posteriors",
]
