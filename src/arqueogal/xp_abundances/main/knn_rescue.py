"""Latent-kNN rescue for high-σ Pipeline-1 predictions.

For each Stream-3 star, find its ``K`` nearest neighbours in the encoder's
projection space (z, L2-normalised) within the Stream-1 training set, then
report the actual APOGEE label distribution of those neighbours. This bypasses
the regression head entirely, which is the right move for stars whose
regression-head σ has inflated above the prior-collapse threshold (release.py
``_PER_ELEMENT_SIGMA_INFLATED_THRESHOLD``).

The kNN is **principled**:

- The encoder is trained with a SupCon loss term, so stars nearby in z-space
  are physically similar per the encoder's learned similarity metric (which
  is anchored to APOGEE labels via SupCon).
- The kNN does NOT extrapolate. Predictions are bounded by the training-set
  support; stars whose nearest neighbours span a wide label range get a wide
  rescue prediction (large IQR), which is the correct behaviour for genuinely
  out-of-support stars.

Strategy:

1. Compute z (projection vector, L2-normalised) for the training set.
2. Compute z for the inference set (Stream 3) using the same encoder.
3. GPU brute-force kNN via ``torch.topk`` on cosine similarity. For ~614 k
   queries against ~290 k references with K=50 the cost is ~20 s on an
   RTX 3060 (vs ~110 min on CPU brute-force). **Cosine, not Euclidean**: the
   SupCon training objective normalises projection vectors and aligns
   *directions* in z-space, not magnitudes; cosine is the metric the encoder
   itself was trained against, so the kNN inherits the encoder's learned
   notion of similarity rather than a Euclidean prior the encoder was never
   constrained to. Implementation is dot-product on L2-normalised z (so
   ``cos(a, b) = a @ b`` and ``cos_distance = 1 − a @ b``); see
   :func:`gpu_knn_search` for the explicit normalisation step.
4. Per-element neighbour-label statistics (median, p25, p75, IQR, std).

This module is **not** a hybrid composer. The composer (which decides when to
trust the regression head vs the kNN) lives in
``arqueogal.data.release_pipeline.run_hybrid_release_pipeline`` so the
contract here stays narrow: produce a kNN-rescue artefact, no regression-head
substitution.

References
----------
- HIGH_SIGMA_RESCUE_REPORT.md (2026-04-25): empirical justification for the
  K=50 cosine-kNN choice and the 85 % structure-recovery on the high-σ subset.
- DESIGN.md §latent_knn_rescue: contract.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import torch

__all__ = [
    "KnnRescueArtifact",
    "LABEL_NAMES",
    "compute_latents",
    "gpu_knn_search",
    "summarize_neighbors",
    "write_artifact",
]


LABEL_NAMES: Final[tuple[str, ...]] = ("teff", "logg", "mh", "alpha_m", "mg_h")
"""Label order assumed by ``summarize_neighbors``. Must match the encoder's
training label-tier order (``LabelTiers.five_label()``)."""

_DEFAULT_K: Final[int] = 50
"""Default number of neighbours to retrieve. K=50 was chosen empirically (see
HIGH_SIGMA_RESCUE_REPORT) as the smallest K at which the median-of-neighbours
recovers the disc bimodality on the high-σ subset; K∈[30, 100] is stable."""

_DEFAULT_BATCH: Final[int] = 2048
"""Query-batch size for GPU kNN. 2048 × 290 k × float32 ≈ 2.4 GB sim matrix,
which fits in 6 GB VRAM (RTX 3060) alongside the reference and query tensors."""


@dataclass(frozen=True)
class KnnRescueArtifact:
    """Structured output of a kNN-rescue run.

    Attributes
    ----------
    source_id
        ``int64`` source identifiers for the inference (query) set, length ``N``.
    summaries
        Per-element neighbour-label summaries: maps element → ``(N, 5)`` array
        of columns ``[median, p25, p75, iqr, std]``. Element keys are
        :data:`LABEL_NAMES`.
    top_distance
        ``(N,)`` distance to the closest neighbour (cosine distance,
        ``1 - cos``).
    median_distance
        ``(N,)`` median of the K cosine distances per query.
    k
        Number of neighbours used.
    """

    source_id: np.ndarray
    summaries: dict[str, np.ndarray]
    top_distance: np.ndarray
    median_distance: np.ndarray
    k: int

    def to_dataframe(self) -> pd.DataFrame:
        """Serialise to a DataFrame with the schema-aligned column names."""
        out = pd.DataFrame({"source_id": self.source_id.astype(np.int64)})
        for elem in LABEL_NAMES:
            stats = self.summaries[elem]
            out[f"knn_{elem}_med"] = stats[:, 0]
            out[f"knn_{elem}_p25"] = stats[:, 1]
            out[f"knn_{elem}_p75"] = stats[:, 2]
            out[f"knn_{elem}_iqr"] = stats[:, 3]
            out[f"knn_{elem}_std"] = stats[:, 4]
        out["knn_top_distance"] = self.top_distance
        out["knn_median_distance"] = self.median_distance
        return out


def compute_latents(
    model: torch.nn.Module,
    X: np.ndarray,
    *,
    device: torch.device,
    batch: int = 4096,
) -> np.ndarray:
    """Push ``X`` through the encoder and return the L2-normalised projection ``z``.

    Parameters
    ----------
    model
        Pipeline-1 model. Must expose ``model.encoder(x) -> (h, z)`` where
        ``z`` is the SupCon projection used as the contrastive metric.
    X
        ``(N, F)`` feature matrix in the same column order as the encoder's
        training data. NaN handling is the caller's responsibility (the
        boundary contract is mirror-of-training: ``np.nan_to_num`` upstream).
    device
        Torch device the model has been moved to.
    batch
        Forward-pass batch size. 4096 fits comfortably on 6 GB VRAM.

    Returns
    -------
    np.ndarray
        ``(N, D)`` projection vectors as ``float32``. The encoder normalises
        ``z`` internally; we re-normalise defensively in
        :func:`gpu_knn_search` so cosine-similarity downstream is exact.
    """
    z_out: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i : i + batch]).to(device).float()
            _, z = model.encoder(xb)
            z_out.append(z.cpu().numpy())
    return np.concatenate(z_out, axis=0).astype(np.float32, copy=False)


def gpu_knn_search(
    z_train: np.ndarray,
    z_query: np.ndarray,
    *,
    k: int = _DEFAULT_K,
    device: torch.device | None = None,
    batch: int = _DEFAULT_BATCH,
    progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU brute-force cosine-similarity kNN via ``torch.topk``.

    Both ``z_train`` and ``z_query`` are L2-normalised internally so cosine
    distance equals ``1 - z_query @ z_train.T``. NaN/Inf in latents are
    sanitised to zero (defensive; degenerate inputs would otherwise produce
    ``nan`` similarity).

    Parameters
    ----------
    z_train
        ``(M, D)`` reference latents (training set).
    z_query
        ``(N, D)`` query latents (inference set).
    k
        Number of neighbours per query. Default 50 (see :data:`_DEFAULT_K`).
    device
        Torch device. Defaults to ``cuda`` if available, else ``cpu``. Note
        that CPU brute-force is ~110× slower; this routine assumes GPU.
    batch
        Query-batch size (rows of ``z_query`` per ``topk`` call).
    progress
        If True, prints a one-line status every 30 batches.

    Returns
    -------
    distances : np.ndarray
        ``(N, k)`` cosine distances ``1 - cos`` to the k nearest neighbours,
        sorted ascending (closest first).
    indices : np.ndarray
        ``(N, k)`` integer indices into ``z_train``.
    """
    if z_train.shape[1] != z_query.shape[1]:
        raise ValueError(
            f"latent dim mismatch: z_train.shape[1]={z_train.shape[1]} vs "
            f"z_query.shape[1]={z_query.shape[1]}"
        )
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > len(z_train):
        raise ValueError(f"k={k} exceeds reference-set size {len(z_train)}")

    z_train = z_train.astype(np.float32, copy=True)
    z_query = z_query.astype(np.float32, copy=True)
    np.nan_to_num(z_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(z_query, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    # L2-normalise (defensive even if encoder already normalised z).
    z_train /= np.linalg.norm(z_train, axis=1, keepdims=True).clip(min=1e-12)
    z_query /= np.linalg.norm(z_query, axis=1, keepdims=True).clip(min=1e-12)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z_train_t = torch.from_numpy(z_train).to(device)
    z_query_t = torch.from_numpy(z_query).to(device)
    n_query = len(z_query_t)
    distances = np.empty((n_query, k), dtype=np.float32)
    indices = np.empty((n_query, k), dtype=np.int64)

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n_query, batch):
            end = min(i + batch, n_query)
            sim = z_query_t[i:end] @ z_train_t.T
            top_v, top_i = torch.topk(sim, k, dim=1, largest=True, sorted=True)
            distances[i:end] = (1.0 - top_v.cpu().numpy()).astype(np.float32)
            indices[i:end] = top_i.cpu().numpy().astype(np.int64)
            if progress and i % (batch * 30) == 0:
                rate = (end + 1) / max(time.time() - t0, 1e-3)
                eta = (n_query - end) / max(rate, 1.0)
                print(f"  knn {end}/{n_query} ({rate:.0f} qps, ETA {eta:.0f}s)")

    return distances, indices


def summarize_neighbors(
    Y_train: np.ndarray,
    indices: np.ndarray,
    distances: np.ndarray,
    *,
    source_id: np.ndarray,
    k: int | None = None,
) -> KnnRescueArtifact:
    """Per-element median/IQR/std of training-set labels under the K neighbours.

    Parameters
    ----------
    Y_train
        ``(M, 5)`` training-set labels in :data:`LABEL_NAMES` order. Rows
        with NaN labels should already be dropped upstream (kNN over training
        data with NaN labels would propagate to neighbour statistics).
    indices
        ``(N, k)`` neighbour indices from :func:`gpu_knn_search`.
    distances
        ``(N, k)`` neighbour cosine distances (sorted ascending).
    source_id
        ``(N,)`` query source identifiers, used to label the artefact.
    k
        Override the inferred k (defaults to ``indices.shape[1]``).

    Returns
    -------
    KnnRescueArtifact
        Per-element statistics ready for parquet emission.
    """
    if Y_train.shape[1] != len(LABEL_NAMES):
        raise ValueError(
            f"Y_train must have {len(LABEL_NAMES)} columns (LABEL_NAMES), got {Y_train.shape[1]}"
        )
    if indices.shape != distances.shape:
        raise ValueError(f"indices/distances shape mismatch: {indices.shape} vs {distances.shape}")
    if len(source_id) != len(indices):
        raise ValueError(
            f"source_id length {len(source_id)} does not match indices length {len(indices)}"
        )

    k_eff = k if k is not None else indices.shape[1]
    if k_eff > indices.shape[1]:
        raise ValueError(f"k={k_eff} exceeds neighbour count {indices.shape[1]}")

    neighbors = Y_train[indices[:, :k_eff]]  # (N, k, 5)
    # All-finite fast path: callers (run_knn_rescue.py) drop training rows with
    # any NaN label upstream, so Y_train is guaranteed finite by contract. We
    # sort once per element and reuse the sorted axis for median + p25 + p75,
    # which is ~10× faster than three independent np.nan{median,quantile} calls
    # on a (614k, 50) array. Falls back to the NaN-aware path if the input is
    # not all finite.
    summaries: dict[str, np.ndarray] = {}
    finite = np.isfinite(neighbors).all()
    if finite:
        sorted_neighbors = np.sort(neighbors, axis=1)  # (N, k, 5)
        # Quantile positions: linear interpolation between adjacent sorted ranks
        n_k = sorted_neighbors.shape[1]
        for j, elem in enumerate(LABEL_NAMES):
            col = sorted_neighbors[:, :, j]
            qs = np.quantile(col, [0.25, 0.5, 0.75], axis=1)
            p25, median, p75 = qs[0], qs[1], qs[2]
            iqr = p75 - p25
            std = neighbors[:, :, j].std(axis=1, ddof=0)
            summaries[elem] = np.column_stack([median, p25, p75, iqr, std]).astype(
                np.float32, copy=False
            )
            del col, qs
        del sorted_neighbors
    else:
        for j, elem in enumerate(LABEL_NAMES):
            col = neighbors[:, :, j]
            median = np.nanmedian(col, axis=1)
            p25 = np.nanquantile(col, 0.25, axis=1)
            p75 = np.nanquantile(col, 0.75, axis=1)
            iqr = p75 - p25
            std = np.nanstd(col, axis=1)
            summaries[elem] = np.column_stack([median, p25, p75, iqr, std]).astype(
                np.float32, copy=False
            )

    return KnnRescueArtifact(
        source_id=np.asarray(source_id, dtype=np.int64),
        summaries=summaries,
        top_distance=distances[:, 0].astype(np.float32, copy=False),
        median_distance=np.median(distances[:, :k_eff], axis=1).astype(np.float32, copy=False),
        k=int(k_eff),
    )


def write_artifact(artifact: KnnRescueArtifact, path: Path) -> Path:
    """Write the kNN-rescue artefact to parquet.

    Parameters
    ----------
    artifact
        Output of :func:`summarize_neighbors`.
    path
        Destination parquet path. Parent directory is created if missing.

    Returns
    -------
    pathlib.Path
        The path written, for chaining / logging.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact.to_dataframe().to_parquet(path)
    return path
