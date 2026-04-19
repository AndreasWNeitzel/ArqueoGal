"""DBCV-maximising hyperparameter grid search — research_brief §10.4.

The rule DESIGN §10.4 fixes: hyperparameters for UMAP + HDBSCAN are selected
by **Density-Based Clustering Validation** (DBCV, Moulavi+2014). Persistence
score alone is rejected — it favours trivial dense clusters (§10.7).
Visual selection is rejected — not reproducible, not releasable.

Grid from research_brief §10.4 (full):

- ``n_neighbors ∈ {15, 30, 50, 100, 200}``
- ``min_dist ∈ {0.0, 0.05, 0.1}``
- UMAP output dim ``∈ {2, 3, 5}``
- ``min_cluster_size ∈ {50, 100, 200, 500}``
- ``min_samples ∈ {5, 10, 20}``
- ``cluster_selection_epsilon ∈ {0.0, 0.1, 0.2}``
- ``cluster_selection_method ∈ {eom, leaf}``

3240 combinations; full sweep on 10⁵ stars takes days on RTX 3060.
This module provides the *machinery*; compute
budget management (coarse → fine, cuML HDBSCAN, parallel jobs) is
release-script concern and sits above this module.

The ``embed_fn`` callable lets callers inject anything that produces an
embedding from raw features — Parametric UMAP for production, a cheap
identity-projection for tests. Keeping the grid search decoupled from the
embedder means DBCV optimisation isn't locked to one specific UMAP impl.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from hdbscan.validity import validity_index

from arqueogal.population_classifier.main.clustering import (
    ClusteringResult,
    HDBSCANConfig,
    cluster_hdbscan,
)

SelectionMethod = Literal["eom", "leaf"]


@dataclass(frozen=True, slots=True)
class UMAPHyperparams:
    """Subset of parametric-UMAP hyperparameters the DBCV grid explores."""

    n_neighbors: int = 15
    min_dist: float = 0.1
    n_components: int = 2


@dataclass(frozen=True, slots=True)
class HyperparameterGrid:
    """Full §10.4 grid; defaults match the research_brief exactly.

    Override any axis to tighten the sweep (``grid = HyperparameterGrid(
    n_neighbors=(15, 30), ...)``). The product across every axis is the
    exhaustive enumeration.
    """

    n_neighbors: tuple[int, ...] = (15, 30, 50, 100, 200)
    min_dist: tuple[float, ...] = (0.0, 0.05, 0.10)
    n_components: tuple[int, ...] = (2, 3, 5)
    min_cluster_size: tuple[int, ...] = (50, 100, 200, 500)
    min_samples: tuple[int, ...] = (5, 10, 20)
    cluster_selection_epsilon: tuple[float, ...] = (0.0, 0.1, 0.2)
    cluster_selection_method: tuple[SelectionMethod, ...] = ("eom", "leaf")

    def n_combinations(self) -> int:
        return (
            len(self.n_neighbors) * len(self.min_dist) * len(self.n_components)
            * len(self.min_cluster_size) * len(self.min_samples)
            * len(self.cluster_selection_epsilon)
            * len(self.cluster_selection_method)
        )

    def iter_combinations(self) -> Iterable[tuple[UMAPHyperparams, HDBSCANConfig]]:
        """Yield one ``(UMAPHyperparams, HDBSCANConfig)`` per grid cell."""
        for nn in self.n_neighbors:
            for md in self.min_dist:
                for nc in self.n_components:
                    umap_hp = UMAPHyperparams(
                        n_neighbors=nn, min_dist=md, n_components=nc,
                    )
                    for mcs in self.min_cluster_size:
                        for ms in self.min_samples:
                            for eps in self.cluster_selection_epsilon:
                                for method in self.cluster_selection_method:
                                    yield umap_hp, HDBSCANConfig(
                                        min_cluster_size=mcs,
                                        min_samples=ms,
                                        cluster_selection_epsilon=eps,
                                        cluster_selection_method=method,
                                    )


@dataclass
class GridCell:
    """One grid-search outcome: config + embedding + clustering + DBCV."""

    umap: UMAPHyperparams
    hdbscan: HDBSCANConfig
    dbcv: float
    n_clusters: int
    noise_fraction: float
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "umap": {
                "n_neighbors": self.umap.n_neighbors,
                "min_dist": self.umap.min_dist,
                "n_components": self.umap.n_components,
            },
            "hdbscan": {
                "min_cluster_size": self.hdbscan.min_cluster_size,
                "min_samples": self.hdbscan.min_samples,
                "cluster_selection_epsilon": self.hdbscan.cluster_selection_epsilon,
                "cluster_selection_method": self.hdbscan.cluster_selection_method,
            },
            "dbcv": float(self.dbcv),
            "n_clusters": int(self.n_clusters),
            "noise_fraction": float(self.noise_fraction),
        }


def dbcv_score(Z: np.ndarray, labels: np.ndarray) -> float:
    """DBCV (Moulavi+2014) for embedding ``Z`` with hard labels ``labels``.

    Returns ``-1.0`` if DBCV is undefined (all-noise or 1-cluster cases);
    the grid search treats these as worst-case.
    """
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2-D, got shape {Z.shape}")
    if labels.shape[0] != Z.shape[0]:
        raise ValueError(
            f"labels length {labels.shape[0]} != Z N={Z.shape[0]}",
        )
    n_clusters = int(np.unique(labels[labels >= 0]).size)
    if n_clusters < 2:
        return -1.0
    # validity_index requires double-precision and no NaN/Inf.
    Z_d = np.asarray(Z, dtype=np.float64)
    try:
        score = float(validity_index(Z_d, labels.astype(np.int64)))
    except (ValueError, FloatingPointError):
        return -1.0
    if not np.isfinite(score):
        return -1.0
    return score


def grid_search(
    X: np.ndarray,
    grid: HyperparameterGrid,
    embed_fn: Callable[[np.ndarray, UMAPHyperparams], np.ndarray],
    *,
    top_k: int | None = None,
    progress_callback: Callable[[int, int, GridCell], None] | None = None,
) -> list[GridCell]:
    """Full grid sweep; return cells sorted by descending DBCV.

    ``embed_fn`` is invoked once per *unique* ``UMAPHyperparams`` cell and
    cached — important because the UMAP fit is the expensive bit, while
    HDBSCAN re-fits over the same embedding are comparatively cheap.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")

    embed_cache: dict[UMAPHyperparams, np.ndarray] = {}
    results: list[GridCell] = []
    total = grid.n_combinations()

    for idx, (umap_hp, hdb_cfg) in enumerate(grid.iter_combinations()):
        if umap_hp not in embed_cache:
            embed_cache[umap_hp] = np.asarray(embed_fn(X, umap_hp))
        Z = embed_cache[umap_hp]
        cluster: ClusteringResult = cluster_hdbscan(Z, hdb_cfg)
        score = dbcv_score(Z, cluster.labels)
        cell = GridCell(
            umap=umap_hp, hdbscan=hdb_cfg, dbcv=score,
            n_clusters=cluster.n_clusters,
            noise_fraction=cluster.noise_fraction,
        )
        results.append(cell)
        if progress_callback is not None:
            progress_callback(idx, total, cell)

    results.sort(key=lambda c: c.dbcv, reverse=True)
    return results[:top_k] if top_k is not None else results


__all__ = [
    "GridCell",
    "HyperparameterGrid",
    "UMAPHyperparams",
    "dbcv_score",
    "grid_search",
]
