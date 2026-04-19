"""HDBSCAN clustering with soft memberships + GLOSH — research_brief §10.1/§10.3.

D-Cat-d requires per-star *population membership probabilities*, not hard
cluster labels. This module wraps ``hdbscan.HDBSCAN`` so the three artefacts
DESIGN §10.3 calls for come out together:

1. **Hard labels** (``labels``) — for plotting and cluster-level diagnostics.
2. **Soft memberships** (``soft_memberships``) via
   ``hdbscan.all_points_membership_vectors`` — ``(N, K)`` probability
   matrix needed for boundary stars.
3. **GLOSH outlier scores** (``glosh``) — ``(N,)`` per-star outlier
   probability per Campello+2015.

Boundary-star flag (DESIGN §10.3): ``max(soft_memberships, axis=1) < 0.7``
OR ``std_over_mc > 0.15`` (std comes from the MC ensemble, not this module;
we supply the first half of the flag here).

cuML path: research_brief §10.1 wants cuML HDBSCAN on CUDA when available.
We keep the CPU path as the default here — GPU is orthogonal and can be
added as a drop-in ``engine="cuml"`` later without touching tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import hdbscan
import numpy as np

SelectionMethod = Literal["eom", "leaf"]


@dataclass(frozen=True, slots=True)
class HDBSCANConfig:
    """Knob set from research_brief §10.4."""

    min_cluster_size: int = 50
    min_samples: int | None = None  # None → defaults to min_cluster_size in hdbscan
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: SelectionMethod = "eom"
    metric: str = "euclidean"
    allow_single_cluster: bool = False


@dataclass
class ClusteringResult:
    """Outputs of one HDBSCAN fit.

    ``labels``: ``(N,)`` int array, ``-1`` = noise per the hdbscan convention.
    ``probabilities``: ``(N,)`` hdbscan "membership strength" (diagonal of
    the soft membership matrix ∼ how strongly this star sits inside *its*
    cluster, not a cross-cluster distribution).
    ``soft_memberships``: ``(N, K)`` float matrix where ``K`` is the number
    of non-noise clusters. Rows sum to 1 (hdbscan's convention).
    ``glosh``: ``(N,)`` GLOSH outlier scores.
    ``boundary_flag``: ``(N,)`` bool array, True if ``max(soft) < 0.7``.
    ``n_clusters``: count of non-noise clusters.
    """

    labels: np.ndarray
    probabilities: np.ndarray
    soft_memberships: np.ndarray
    glosh: np.ndarray
    boundary_flag: np.ndarray
    n_clusters: int
    cluster_ids: tuple[int, ...]
    config: HDBSCANConfig = field(default_factory=HDBSCANConfig)

    def __post_init__(self) -> None:
        n = self.labels.shape[0]
        for name, arr in [
            ("probabilities", self.probabilities),
            ("glosh", self.glosh),
            ("boundary_flag", self.boundary_flag),
        ]:
            if arr.shape != (n,):
                raise ValueError(f"{name} shape {arr.shape} != ({n},)")
        if self.soft_memberships.ndim != 2 or self.soft_memberships.shape[0] != n:
            raise ValueError(
                f"soft_memberships shape {self.soft_memberships.shape} "
                f"not aligned with labels ({n},)",
            )

    @property
    def noise_fraction(self) -> float:
        return float((self.labels == -1).mean())


def cluster_hdbscan(
    Z: np.ndarray,
    config: HDBSCANConfig | None = None,
    *,
    boundary_threshold: float = 0.7,
) -> ClusteringResult:
    """Run HDBSCAN on ``Z`` (typically a UMAP embedding) and return the full triple.

    ``boundary_threshold`` sets the DESIGN §10.3 max-soft-membership cut-off
    above which a star is *not* flagged as a boundary star.
    """
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2-D, got shape {Z.shape}")
    config = config or HDBSCANConfig()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        cluster_selection_epsilon=config.cluster_selection_epsilon,
        cluster_selection_method=config.cluster_selection_method,
        metric=config.metric,
        allow_single_cluster=config.allow_single_cluster,
        prediction_data=True,  # required for all_points_membership_vectors
    )
    labels = clusterer.fit_predict(Z).astype(np.int64)
    probabilities = np.asarray(clusterer.probabilities_, dtype=np.float32)
    glosh = np.asarray(clusterer.outlier_scores_, dtype=np.float32)

    cluster_ids_arr = np.unique(labels[labels >= 0])
    n_clusters = int(cluster_ids_arr.size)
    if n_clusters > 0:
        soft = np.asarray(
            hdbscan.all_points_membership_vectors(clusterer), dtype=np.float32,
        )
        # Edge case: when K=1 hdbscan returns 1-D; normalise to 2-D.
        if soft.ndim == 1:
            soft = soft[:, None]
        boundary_flag = (soft.max(axis=1) < boundary_threshold)
    else:
        soft = np.zeros((Z.shape[0], 0), dtype=np.float32)
        boundary_flag = np.ones(Z.shape[0], dtype=bool)

    return ClusteringResult(
        labels=labels,
        probabilities=probabilities,
        soft_memberships=soft,
        glosh=glosh,
        boundary_flag=boundary_flag.astype(bool),
        n_clusters=n_clusters,
        cluster_ids=tuple(int(c) for c in cluster_ids_arr.tolist()),
        config=config,
    )


__all__ = [
    "ClusteringResult",
    "HDBSCANConfig",
    "cluster_hdbscan",
]
