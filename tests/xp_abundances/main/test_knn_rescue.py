"""Tests for ``xp_abundances.main.knn_rescue``.

Exercises the four pure functions:

- :func:`compute_latents` (dummy encoder pass-through)
- :func:`gpu_knn_search` (correctness on a small synthetic set)
- :func:`summarize_neighbors` (per-element statistics)
- :func:`write_artifact` (parquet round-trip + schema-aligned columns)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from arqueogal.xp_abundances.main.knn_rescue import (
    LABEL_NAMES,
    KnnRescueArtifact,
    compute_latents,
    gpu_knn_search,
    summarize_neighbors,
    write_artifact,
)


class _DummyEncoderModel(torch.nn.Module):
    """Encoder that returns ``(h, z)`` where ``z`` is a deterministic linear
    projection of the input. Used to make :func:`compute_latents` testable
    without needing a full Pipeline-1 model."""

    def __init__(self, in_dim: int, out_dim: int = 8):
        super().__init__()
        self.encoder_module = torch.nn.Linear(in_dim, out_dim, bias=False)
        torch.nn.init.eye_(self.encoder_module.weight) if in_dim == out_dim else None

    @property
    def encoder(self):
        # Return a callable matching the (h, z) contract.
        def _fn(x):
            z = self.encoder_module(x)
            return z, torch.nn.functional.normalize(z, dim=1)

        return _fn


def test_compute_latents_shape_and_normalisation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(17, 8)).astype(np.float32)
    model = _DummyEncoderModel(in_dim=8, out_dim=8)
    z = compute_latents(model, X, device=torch.device("cpu"), batch=4)
    assert z.shape == (17, 8)
    assert z.dtype == np.float32
    # Encoder L2-normalises z; norms should all be ~1.
    norms = np.linalg.norm(z, axis=1)
    np.testing.assert_allclose(norms, np.ones(17), rtol=1e-5)


def test_gpu_knn_search_basic_correctness():
    """Identity references → first neighbour is the query itself with distance ≈ 0."""
    rng = np.random.default_rng(1)
    z = rng.normal(size=(20, 4)).astype(np.float32)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    distances, indices = gpu_knn_search(z, z, k=3, device=torch.device("cpu"))
    assert distances.shape == (20, 3)
    assert indices.shape == (20, 3)
    # Closest neighbour is itself.
    assert (indices[:, 0] == np.arange(20)).all()
    np.testing.assert_allclose(distances[:, 0], 0.0, atol=1e-5)
    # Distances are non-negative (allow float32 epsilon) and sorted ascending.
    assert (distances >= -1e-5).all()
    assert (np.diff(distances, axis=1) >= -1e-5).all()


def test_gpu_knn_search_finds_known_close_neighbour():
    """A query that is a small perturbation of training point ``i`` retrieves ``i``
    as its top-1 neighbour."""
    rng = np.random.default_rng(2)
    # 50 reference points; queries are perturbations of the first 5.
    z_train = rng.normal(size=(50, 6)).astype(np.float32)
    z_train /= np.linalg.norm(z_train, axis=1, keepdims=True)
    jitter = 0.01 * rng.normal(size=(5, 6)).astype(np.float32)
    z_query = z_train[:5] + jitter
    z_query /= np.linalg.norm(z_query, axis=1, keepdims=True)

    distances, indices = gpu_knn_search(z_train, z_query, k=1, device=torch.device("cpu"))
    assert distances.shape == (5, 1)
    assert indices.shape == (5, 1)
    np.testing.assert_array_equal(indices[:, 0], np.arange(5))


def test_gpu_knn_search_validates_arguments():
    z = np.random.default_rng(3).normal(size=(10, 4)).astype(np.float32)
    with pytest.raises(ValueError, match="latent dim mismatch"):
        gpu_knn_search(z, np.zeros((10, 6), dtype=np.float32), device=torch.device("cpu"))
    with pytest.raises(ValueError, match="k must be positive"):
        gpu_knn_search(z, z, k=0, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="exceeds reference-set size"):
        gpu_knn_search(z, z, k=11, device=torch.device("cpu"))


def test_gpu_knn_search_handles_nan_in_latents():
    """NaN in latents is sanitised to zero rather than propagating."""
    z_train = np.random.default_rng(4).normal(size=(10, 4)).astype(np.float32)
    z_train /= np.linalg.norm(z_train, axis=1, keepdims=True)
    z_query = z_train[:3].copy()
    z_query[0, 0] = np.nan  # one NaN element in one query
    distances, indices = gpu_knn_search(z_train, z_query, k=2, device=torch.device("cpu"))
    assert np.isfinite(distances).all()
    assert np.isfinite(indices.astype(np.float32)).all()


def test_summarize_neighbors_basic():
    """Per-element median / IQR / std of training labels under the K-nn."""
    rng = np.random.default_rng(5)
    n_train, n_query, k = 30, 5, 4
    Y_train = rng.normal(size=(n_train, len(LABEL_NAMES))).astype(np.float32)
    indices = rng.integers(low=0, high=n_train, size=(n_query, k)).astype(np.int64)
    distances = rng.uniform(size=(n_query, k)).astype(np.float32)
    distances.sort(axis=1)
    sid = np.arange(n_query, dtype=np.int64)

    artefact = summarize_neighbors(Y_train, indices, distances, source_id=sid)
    assert isinstance(artefact, KnnRescueArtifact)
    assert artefact.k == k
    assert artefact.source_id.tolist() == list(range(n_query))
    np.testing.assert_array_equal(artefact.top_distance, distances[:, 0])

    # Per-element stats: spot-check that median for the first query and first
    # element matches np.median(Y_train[indices[0]][:, 0]).
    for j, elem in enumerate(LABEL_NAMES):
        stats = artefact.summaries[elem]
        assert stats.shape == (n_query, 5)
        for i in range(n_query):
            neighbours = Y_train[indices[i], j]
            np.testing.assert_allclose(stats[i, 0], np.median(neighbours), rtol=1e-5)
            np.testing.assert_allclose(stats[i, 1], np.quantile(neighbours, 0.25), rtol=1e-5)
            np.testing.assert_allclose(stats[i, 2], np.quantile(neighbours, 0.75), rtol=1e-5)
            np.testing.assert_allclose(stats[i, 4], np.std(neighbours), rtol=1e-5)


def test_summarize_neighbors_k_override():
    """The ``k=`` override must use the first k neighbours, not all of them."""
    rng = np.random.default_rng(6)
    Y_train = rng.normal(size=(20, 5)).astype(np.float32)
    indices = rng.integers(low=0, high=20, size=(3, 8)).astype(np.int64)
    distances = rng.uniform(size=(3, 8)).astype(np.float32)
    distances.sort(axis=1)
    sid = np.arange(3, dtype=np.int64)

    full = summarize_neighbors(Y_train, indices, distances, source_id=sid, k=8)
    truncated = summarize_neighbors(Y_train, indices, distances, source_id=sid, k=4)
    assert full.k == 8
    assert truncated.k == 4
    # Truncated must use only the first 4 neighbours.
    for elem in LABEL_NAMES:
        assert not np.allclose(full.summaries[elem], truncated.summaries[elem])


def test_summarize_neighbors_validates_shapes():
    Y_train = np.zeros((10, 5), dtype=np.float32)
    indices = np.zeros((3, 4), dtype=np.int64)
    distances = np.zeros((3, 4), dtype=np.float32)
    sid = np.arange(3)

    with pytest.raises(ValueError, match="LABEL_NAMES"):
        summarize_neighbors(np.zeros((10, 4)), indices, distances, source_id=sid)
    with pytest.raises(ValueError, match="indices/distances shape mismatch"):
        summarize_neighbors(Y_train, indices, np.zeros((3, 5)), source_id=sid)
    with pytest.raises(ValueError, match="source_id length"):
        summarize_neighbors(Y_train, indices, distances, source_id=np.arange(5))
    with pytest.raises(ValueError, match="exceeds neighbour count"):
        summarize_neighbors(Y_train, indices, distances, source_id=sid, k=10)


def test_write_artifact_round_trip(tmp_path: Path):
    rng = np.random.default_rng(7)
    Y_train = rng.normal(size=(20, len(LABEL_NAMES))).astype(np.float32)
    indices = rng.integers(low=0, high=20, size=(4, 5)).astype(np.int64)
    distances = rng.uniform(size=(4, 5)).astype(np.float32)
    distances.sort(axis=1)
    sid = np.array([100, 200, 300, 400], dtype=np.int64)

    artefact = summarize_neighbors(Y_train, indices, distances, source_id=sid)
    out = write_artifact(artefact, tmp_path / "rescue.parquet")
    assert out.exists()
    df = pd.read_parquet(out)
    assert df["source_id"].tolist() == sid.tolist()

    expected_cols = ["source_id"]
    for elem in LABEL_NAMES:
        for stat in ("med", "p25", "p75", "iqr", "std"):
            expected_cols.append(f"knn_{elem}_{stat}")
    expected_cols.extend(["knn_top_distance", "knn_median_distance"])
    assert list(df.columns) == expected_cols


def test_artifact_columns_align_with_master_schema():
    """The DataFrame's column set must be a subset of the master-schema kNN-rescue
    optional columns; if a new column is added in :mod:`master_schema`, this
    test surfaces the drift before it ships."""
    from arqueogal.data.master_schema import _PIPELINE1_KNN_RESCUE_COLS

    rng = np.random.default_rng(8)
    Y_train = rng.normal(size=(10, len(LABEL_NAMES))).astype(np.float32)
    indices = rng.integers(low=0, high=10, size=(2, 3)).astype(np.int64)
    distances = rng.uniform(size=(2, 3)).astype(np.float32)
    distances.sort(axis=1)
    sid = np.array([1, 2], dtype=np.int64)
    artefact = summarize_neighbors(Y_train, indices, distances, source_id=sid)
    df = artefact.to_dataframe()
    df_cols = set(df.columns) - {"source_id"}
    schema_cols = set(_PIPELINE1_KNN_RESCUE_COLS)
    assert df_cols == schema_cols, (
        f"kNN artefact columns drift from master_schema: "
        f"in df not schema = {df_cols - schema_cols}; "
        f"in schema not df = {schema_cols - df_cols}"
    )
