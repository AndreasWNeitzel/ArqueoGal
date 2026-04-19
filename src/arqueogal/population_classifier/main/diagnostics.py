"""Pipeline-2 unsupervised diagnostic stack — research_brief §10.5 / §11.

Six tools that answer the user's interpretation pain-point — *"we lack an
objective, unsupervised way to interpret our results"* — without visual
inspection:

1. :func:`bootstrap_cluster_stability` — Hennig 2007. Resample with
   replacement ``N=500`` times, refit, compute pairwise ARI on the
   overlap of sample indices. Median ARI > 0.75 stable; < 0.5 = artefact.
2. ``dbcv_score`` — already lives in :mod:`.hyperparameter` and is reused
   here as a first-class diagnostic.
3. :func:`permutation_feature_causal` — shuffle one feature across the
   catalogue, refit, ARI to original. Chemistry-driven clusters vanish
   under ``[Fe/H]`` shuffle; clusters robust to every shuffle are
   density artefacts.
4. :func:`null_model_comparison` — fit multivariate Gaussian and/or
   Gaussian-copula null (marginals preserved, correlations destroyed),
   draw samples, apply pipeline. Real clusters appear far above null.
5. :func:`held_out_feature_consistency` — unsupervised cross-validation
   analogue. Cluster on ``D−1`` features, check the held-out feature
   separates cleanly per cluster via effect-size metrics.
6. :func:`literature_cross_reference` — confusion / precision-recall
   against named-population labels from Dodd+2023, Myeong+2019,
   Horta+2021, Ceccarelli+2024.

All functions take a callable ``cluster_fn: X -> labels`` so the pipeline
is pluggable: Parametric UMAP + HDBSCAN for production, a cheap toy
clusterer for tests. DBCV scoring already lives in
:mod:`.hyperparameter` and is intentionally *not* duplicated here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy import stats
from sklearn.metrics import adjusted_rand_score

ClusterFn = Callable[[np.ndarray], np.ndarray]
NullMethod = Literal["mvn", "copula"]


# --------------------------------------------------------------------------
# 1. Bootstrap cluster stability — Hennig 2007
# --------------------------------------------------------------------------


@dataclass
class BootstrapStabilityReport:
    """Output of :func:`bootstrap_cluster_stability`.

    ``pairwise_ari``: length ``n_bootstrap*(n_bootstrap-1)/2`` array of
    pairwise ARIs on the overlap of indices.
    ``median_ari`` / ``mean_ari`` / ``q05_ari`` / ``q95_ari``: summary stats.
    ``n_bootstrap``: how many bootstraps were drawn.
    ``stable``: median_ari > 0.75.
    ``artefact``: median_ari < 0.5.
    """

    pairwise_ari: np.ndarray
    median_ari: float
    mean_ari: float
    q05_ari: float
    q95_ari: float
    n_bootstrap: int
    stable: bool
    artefact: bool


def _bootstrap_label_map(
    X: np.ndarray,
    cluster_fn: ClusterFn,
    idx: np.ndarray,
) -> dict[int, int]:
    """Fit ``cluster_fn`` on ``X[idx]`` and return ``{original_row: label}``.

    Duplicates (same original row sampled multiple times) keep the first
    label — they are identical features so any deterministic clusterer
    gives identical labels anyway.
    """
    labels = np.asarray(cluster_fn(X[idx]))
    if labels.shape != (idx.shape[0],):
        raise ValueError(
            f"cluster_fn returned shape {labels.shape}, expected {idx.shape}",
        )
    mapping: dict[int, int] = {}
    for pos, original_row in enumerate(idx):
        if original_row not in mapping:
            mapping[int(original_row)] = int(labels[pos])
    return mapping


def bootstrap_cluster_stability(
    X: np.ndarray,
    cluster_fn: ClusterFn,
    *,
    n_bootstrap: int = 500,
    seed: int = 0,
) -> BootstrapStabilityReport:
    """Pairwise-ARI bootstrap stability per Hennig 2007 / research_brief §10.5.

    Each bootstrap resamples ``N`` rows *with replacement* from ``X`` and
    refits ``cluster_fn``. Pairwise ARI is computed on the *intersection*
    of the unique indices that appear in both bootstraps — i.e. on stars
    that survived into both resamples. This is the standard fpc /
    clusterboot approach.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    n = X.shape[0]
    rng = np.random.default_rng(seed)

    runs: list[dict[int, int]] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        runs.append(_bootstrap_label_map(X, cluster_fn, idx))

    aris: list[float] = []
    for i in range(n_bootstrap):
        for j in range(i + 1, n_bootstrap):
            shared = runs[i].keys() & runs[j].keys()
            if len(shared) < 2:
                continue
            shared_list = sorted(shared)
            a = np.fromiter((runs[i][s] for s in shared_list), dtype=np.int64)
            b = np.fromiter((runs[j][s] for s in shared_list), dtype=np.int64)
            aris.append(float(adjusted_rand_score(a, b)))

    arr = np.asarray(aris, dtype=np.float64)
    median = float(np.median(arr)) if arr.size else float("nan")
    mean = float(np.mean(arr)) if arr.size else float("nan")
    q05 = float(np.quantile(arr, 0.05)) if arr.size else float("nan")
    q95 = float(np.quantile(arr, 0.95)) if arr.size else float("nan")
    return BootstrapStabilityReport(
        pairwise_ari=arr,
        median_ari=median,
        mean_ari=mean,
        q05_ari=q05,
        q95_ari=q95,
        n_bootstrap=n_bootstrap,
        stable=median > 0.75 if not np.isnan(median) else False,
        artefact=median < 0.5 if not np.isnan(median) else False,
    )


# --------------------------------------------------------------------------
# 3. Permutation-feature causal attribution
# --------------------------------------------------------------------------


@dataclass
class FeatureCausalReport:
    """Output of :func:`permutation_feature_causal`.

    ``ari_by_feature``: ``{feature_index: ARI}`` after shuffling that
    feature. Low ARI = clusters depend on the feature.
    ``ari_drop_by_feature``: ``1 - ARI`` for convenience.
    ``ranked``: list of ``(feature_index, ari)`` sorted ascending by ARI
    (most causal first).
    """

    ari_by_feature: dict[int, float]
    ari_drop_by_feature: dict[int, float]
    ranked: list[tuple[int, float]]


def permutation_feature_causal(
    X: np.ndarray,
    baseline_labels: np.ndarray,
    cluster_fn: ClusterFn,
    *,
    feature_indices: Sequence[int] | None = None,
    seed: int = 0,
) -> FeatureCausalReport:
    """Shuffle one feature at a time; ARI to ``baseline_labels``.

    Interpretation per §10.5: chemistry-driven clusters vanish under
    ``[Fe/H]`` shuffle (ARI ≈ 0); clusters robust to every shuffle are
    density artefacts. This is causal decomposition that visual
    inspection cannot provide.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if baseline_labels.shape[0] != X.shape[0]:
        raise ValueError(
            f"baseline_labels length {baseline_labels.shape[0]} "
            f"!= X N={X.shape[0]}",
        )
    rng = np.random.default_rng(seed)
    indices = (
        list(range(X.shape[1]))
        if feature_indices is None
        else list(feature_indices)
    )
    ari_by: dict[int, float] = {}
    for fi in indices:
        X_shuf = X.copy()
        X_shuf[:, fi] = rng.permutation(X_shuf[:, fi])
        labels = np.asarray(cluster_fn(X_shuf))
        ari_by[fi] = float(adjusted_rand_score(baseline_labels, labels))

    drop = {k: 1.0 - v for k, v in ari_by.items()}
    ranked = sorted(ari_by.items(), key=lambda kv: kv[1])
    return FeatureCausalReport(
        ari_by_feature=ari_by,
        ari_drop_by_feature=drop,
        ranked=ranked,
    )


# --------------------------------------------------------------------------
# 4. Null-model comparison
# --------------------------------------------------------------------------


@dataclass
class NullModelReport:
    """Output of :func:`null_model_comparison`.

    ``null_n_clusters``: per-null count of non-noise clusters.
    ``null_median`` / ``null_q95``: summary stats.
    ``real_n_clusters``: the count observed on the real data.
    ``passes``: True iff ``real_n_clusters > null_q95``.
    ``method``: which null was used.
    """

    null_n_clusters: np.ndarray
    null_median: float
    null_q95: float
    real_n_clusters: int
    passes: bool
    method: NullMethod


def _sample_mvn_null(
    X: np.ndarray, n_samples: int, *, rng: np.random.Generator,
) -> np.ndarray:
    mean = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    return rng.multivariate_normal(mean, cov, size=n_samples)


def _sample_copula_null(
    X: np.ndarray, n_samples: int, *, rng: np.random.Generator,
) -> np.ndarray:
    """Independent-column bootstrap — destroys cross-feature correlations
    while preserving marginals exactly. Equivalent to a fully-independent
    Gaussian copula null sampled via the empirical marginal CDFs.
    """
    out = np.empty((n_samples, X.shape[1]), dtype=X.dtype)
    for j in range(X.shape[1]):
        out[:, j] = rng.choice(X[:, j], size=n_samples, replace=True)
    return out


def null_model_comparison(
    X: np.ndarray,
    cluster_fn: ClusterFn,
    *,
    method: NullMethod = "copula",
    n_null: int = 100,
    seed: int = 0,
) -> NullModelReport:
    """Compare real-data cluster count to null-model cluster count."""
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")

    real_labels = np.asarray(cluster_fn(X))
    real_n = int(np.unique(real_labels[real_labels >= 0]).size)

    rng = np.random.default_rng(seed)
    counts = np.empty(n_null, dtype=np.int64)
    for i in range(n_null):
        sampler = _sample_mvn_null if method == "mvn" else _sample_copula_null
        X_null = sampler(X, X.shape[0], rng=rng).astype(X.dtype, copy=False)
        labels = np.asarray(cluster_fn(X_null))
        counts[i] = int(np.unique(labels[labels >= 0]).size)

    q95 = float(np.quantile(counts, 0.95))
    return NullModelReport(
        null_n_clusters=counts,
        null_median=float(np.median(counts)),
        null_q95=q95,
        real_n_clusters=real_n,
        passes=real_n > q95,
        method=method,
    )


# --------------------------------------------------------------------------
# 5. Held-out feature consistency
# --------------------------------------------------------------------------


@dataclass
class HeldOutFeatureReport:
    """Output of :func:`held_out_feature_consistency`.

    ``held_out_index``: which feature was withheld.
    ``labels``: cluster labels from the reduced fit (length ``N``).
    ``kw_h``: Kruskal-Wallis H across clusters on the held-out feature.
    ``kw_pvalue``: Kruskal-Wallis p-value.
    ``separation_ratio``: between-cluster-spread / within-cluster-spread
    of held-out feature. High = cleanly separated.
    ``per_cluster_mean`` / ``per_cluster_std``: shape ``(K,)`` per-cluster
    mean/std of the held-out feature.
    ``cluster_ids``: non-noise cluster ids.
    """

    held_out_index: int
    labels: np.ndarray
    kw_h: float
    kw_pvalue: float
    separation_ratio: float
    per_cluster_mean: np.ndarray
    per_cluster_std: np.ndarray
    cluster_ids: tuple[int, ...]


def held_out_feature_consistency(
    X: np.ndarray,
    cluster_fn: ClusterFn,
    *,
    held_out_index: int,
) -> HeldOutFeatureReport:
    """Cluster on all-but-one features; check the held-out feature separates.

    Unsupervised cross-validation analogue (§10.5 test 5). A clustering
    that fell out *without* information from feature ``j`` should still
    predict different distributions of feature ``j`` across clusters;
    otherwise those clusters were over-fit to the remaining features.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if not (0 <= held_out_index < X.shape[1]):
        raise ValueError(
            f"held_out_index {held_out_index} out of range for D={X.shape[1]}",
        )
    mask = np.ones(X.shape[1], dtype=bool)
    mask[held_out_index] = False
    X_reduced = X[:, mask]
    labels = np.asarray(cluster_fn(X_reduced))
    held = X[:, held_out_index]

    cluster_ids_arr = np.unique(labels[labels >= 0])
    n_clusters = int(cluster_ids_arr.size)
    if n_clusters < 2:
        return HeldOutFeatureReport(
            held_out_index=held_out_index,
            labels=labels,
            kw_h=float("nan"),
            kw_pvalue=float("nan"),
            separation_ratio=float("nan"),
            per_cluster_mean=np.zeros(n_clusters, dtype=np.float64),
            per_cluster_std=np.zeros(n_clusters, dtype=np.float64),
            cluster_ids=tuple(int(c) for c in cluster_ids_arr.tolist()),
        )
    groups = [held[labels == c] for c in cluster_ids_arr]
    kw = stats.kruskal(*groups)
    means = np.array([g.mean() for g in groups], dtype=np.float64)
    stds = np.array([g.std(ddof=1) if g.size > 1 else 0.0 for g in groups],
                    dtype=np.float64)
    between = float(np.std(means, ddof=0))
    within = float(np.mean(stds)) if stds.size else 0.0
    sep = between / within if within > 0 else float("inf")
    return HeldOutFeatureReport(
        held_out_index=held_out_index,
        labels=labels,
        kw_h=float(kw.statistic),
        kw_pvalue=float(kw.pvalue),
        separation_ratio=sep,
        per_cluster_mean=means,
        per_cluster_std=stds,
        cluster_ids=tuple(int(c) for c in cluster_ids_arr.tolist()),
    )


# --------------------------------------------------------------------------
# 6. Literature cross-reference
# --------------------------------------------------------------------------


@dataclass
class LiteratureCrossReferenceReport:
    """Output of :func:`literature_cross_reference`.

    ``contingency``: ``(K_pred, K_lit)`` count matrix, aligned with
    ``predicted_ids`` / ``literature_ids`` in row / column order.
    ``predicted_ids`` / ``literature_ids``: ordered ids; literature
    ids correspond to ``literature_labels`` unique values.
    ``precision``: ``(K_pred,)`` best-match precision per predicted
    cluster — fraction of the cluster that belongs to its majority
    literature population.
    ``recall``: ``(K_lit,)`` best-match recall per literature population.
    ``best_lit_for_pred``: ``{pred_id: literature_id}`` best match.
    ``ari``: overall adjusted-Rand index between the two labelings on the
    subset where literature labels are available.
    """

    contingency: np.ndarray
    predicted_ids: tuple[int, ...]
    literature_ids: tuple[Any, ...]
    precision: np.ndarray
    recall: np.ndarray
    best_lit_for_pred: dict[int, Any]
    ari: float
    n_matched: int


def literature_cross_reference(
    predicted_labels: np.ndarray,
    literature_labels: np.ndarray,
    *,
    ignore_predicted: Sequence[int] = (-1,),
    ignore_literature: Sequence[Any] = (),
    literature_missing_value: Any = None,
) -> LiteratureCrossReferenceReport:
    """Confusion-matrix / ARI cross-reference against literature labels.

    Stars without a literature label (``literature_labels == missing``)
    are excluded from the match but not from the overall catalog.
    """
    if predicted_labels.shape != literature_labels.shape:
        raise ValueError(
            f"shape mismatch: predicted {predicted_labels.shape} vs "
            f"literature {literature_labels.shape}",
        )
    if literature_missing_value is None:
        have_lit = np.array(
            [lbl is not None for lbl in literature_labels], dtype=bool,
        )
    else:
        have_lit = literature_labels != literature_missing_value
    in_pred = ~np.isin(predicted_labels, list(ignore_predicted))
    in_lit = ~np.isin(literature_labels, list(ignore_literature))
    mask = have_lit & in_pred & in_lit

    pred = predicted_labels[mask]
    lit = literature_labels[mask]

    pred_ids = np.unique(pred)
    lit_ids = np.unique(lit)

    K, L = pred_ids.size, lit_ids.size
    cont = np.zeros((K, L), dtype=np.int64)
    for ki, pid in enumerate(pred_ids):
        sel_p = pred == pid
        for li, lid in enumerate(lit_ids):
            cont[ki, li] = int((sel_p & (lit == lid)).sum())

    row_sums = cont.sum(axis=1).clip(min=1)
    col_sums = cont.sum(axis=0).clip(min=1)
    best_per_row = cont.max(axis=1) if K else np.zeros(0, dtype=np.int64)
    best_per_col = cont.max(axis=0) if L else np.zeros(0, dtype=np.int64)
    precision = best_per_row / row_sums
    recall = best_per_col / col_sums
    best_match = (
        {int(pred_ids[ki]): lit_ids[int(cont[ki].argmax())] for ki in range(K)}
        if K and L
        else {}
    )

    if mask.any():
        pred_int = pred.astype(np.int64, copy=False)
        lit_key = {v: i for i, v in enumerate(lit_ids)}
        lit_int = np.fromiter((lit_key[v] for v in lit), dtype=np.int64)
        ari = float(adjusted_rand_score(pred_int, lit_int))
    else:
        ari = float("nan")

    return LiteratureCrossReferenceReport(
        contingency=cont,
        predicted_ids=tuple(int(p) for p in pred_ids.tolist()),
        literature_ids=tuple(lit_ids.tolist()),
        precision=precision.astype(np.float64),
        recall=recall.astype(np.float64),
        best_lit_for_pred=best_match,
        ari=ari,
        n_matched=int(mask.sum()),
    )


# --------------------------------------------------------------------------
# Aggregated six-tool report
# --------------------------------------------------------------------------


@dataclass
class DiagnosticStackReport:
    """Aggregated output of the research_brief §10.5 six-tool stack.

    Individual entries default to ``None`` when the release script opts
    to skip one (e.g. literature cross-reference when no literature
    labels are available yet).
    """

    bootstrap: BootstrapStabilityReport | None = None
    dbcv: float | None = None
    feature_causal: FeatureCausalReport | None = None
    null_model: NullModelReport | None = None
    held_out: dict[int, HeldOutFeatureReport] = field(default_factory=dict)
    literature: LiteratureCrossReferenceReport | None = None


__all__ = [
    "BootstrapStabilityReport",
    "DiagnosticStackReport",
    "FeatureCausalReport",
    "HeldOutFeatureReport",
    "LiteratureCrossReferenceReport",
    "NullModelReport",
    "bootstrap_cluster_stability",
    "held_out_feature_consistency",
    "literature_cross_reference",
    "null_model_comparison",
    "permutation_feature_causal",
]
