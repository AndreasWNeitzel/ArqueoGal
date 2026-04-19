"""Tests for population_classifier.main.embedding — Parametric UMAP."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from arqueogal.population_classifier.main.embedding import (
    ParametricUMAP,
    ParametricUMAPConfig,
    ParametricUMAPEncoder,
)


def _two_blob_data(n: int = 60, d: int = 5, seed: int = 0) -> np.ndarray:
    """Two well-separated Gaussian blobs — a trivial structure UMAP must recover."""
    rng = np.random.default_rng(seed)
    a = rng.normal(-3.0, 0.5, (n // 2, d)).astype(np.float32)
    b = rng.normal(+3.0, 0.5, (n // 2, d)).astype(np.float32)
    X = np.concatenate([a, b], axis=0)
    return X


# --- encoder module --------------------------------------------------------

def test_encoder_forward_shape() -> None:
    enc = ParametricUMAPEncoder(input_dim=5, n_components=2, hidden_dims=(16,))
    x = torch.randn(10, 5)
    z = enc(x)
    assert z.shape == (10, 2)


def test_encoder_is_differentiable() -> None:
    enc = ParametricUMAPEncoder(input_dim=4, n_components=2, hidden_dims=(8,))
    x = torch.randn(6, 4)
    z = enc(x)
    loss = z.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in enc.parameters()]
    assert all(g is not None and g.abs().sum() > 0 for g in grads)


# --- fit / transform ------------------------------------------------------

@pytest.fixture
def tiny_config() -> ParametricUMAPConfig:
    return ParametricUMAPConfig(
        n_components=2, n_neighbors=5, min_dist=0.1,
        hidden_dims=(16,), n_epochs=3, batch_size=32,
        learning_rate=1e-2, negative_sample_rate=3, seed=0,
    )


def test_fit_transform_shape(tiny_config: ParametricUMAPConfig) -> None:
    X = _two_blob_data(n=40, d=5)
    pu = ParametricUMAP(tiny_config)
    Z = pu.fit_transform(X, device=torch.device("cpu"))
    assert Z.shape == (40, 2)
    assert np.isfinite(Z).all()
    assert pu.a is not None and pu.b is not None
    assert pu.input_dim == 5


def test_transform_without_fit_raises(tiny_config: ParametricUMAPConfig) -> None:
    pu = ParametricUMAP(tiny_config)
    with pytest.raises(RuntimeError, match="not fitted"):
        pu.transform(np.zeros((5, 4), dtype=np.float32))


def test_transform_input_dim_validation(tiny_config: ParametricUMAPConfig) -> None:
    pu = ParametricUMAP(tiny_config)
    pu.fit(_two_blob_data(n=30, d=5), device=torch.device("cpu"))
    with pytest.raises(ValueError, match="cols"):
        pu.transform(np.zeros((5, 7), dtype=np.float32))


def test_fit_rejects_1d_input(tiny_config: ParametricUMAPConfig) -> None:
    pu = ParametricUMAP(tiny_config)
    with pytest.raises(ValueError, match="X must be 2-D"):
        pu.fit(np.zeros(10, dtype=np.float32))


def test_fit_loss_history_recorded(tiny_config: ParametricUMAPConfig) -> None:
    pu = ParametricUMAP(tiny_config)
    pu.fit(_two_blob_data(n=40, d=4), device=torch.device("cpu"))
    assert len(pu.history) == tiny_config.n_epochs
    assert all(np.isfinite(h) for h in pu.history)


def test_out_of_sample_transform_runs(tiny_config: ParametricUMAPConfig) -> None:
    rng = np.random.default_rng(0)
    X_train = _two_blob_data(n=40, d=5, seed=0)
    X_new = rng.standard_normal((7, 5)).astype(np.float32)
    pu = ParametricUMAP(tiny_config)
    pu.fit(X_train, device=torch.device("cpu"))
    Z_new = pu.transform(X_new)
    assert Z_new.shape == (7, 2)
    assert np.isfinite(Z_new).all()


def test_fit_separates_two_well_separated_blobs() -> None:
    """With stronger training the two-blob structure should survive embedding."""
    cfg = ParametricUMAPConfig(
        n_components=2, n_neighbors=5, min_dist=0.1, hidden_dims=(32,),
        n_epochs=30, batch_size=64, learning_rate=5e-3,
        negative_sample_rate=5, seed=0,
    )
    X = _two_blob_data(n=60, d=5, seed=0)
    pu = ParametricUMAP(cfg)
    Z = pu.fit_transform(X, device=torch.device("cpu"))
    # Compare average pairwise distance within vs between blobs.
    c1 = Z[:30].mean(axis=0)
    c2 = Z[30:].mean(axis=0)
    inter_centroid = np.linalg.norm(c1 - c2)
    intra_1 = np.linalg.norm(Z[:30] - c1, axis=1).mean()
    intra_2 = np.linalg.norm(Z[30:] - c2, axis=1).mean()
    assert inter_centroid > (intra_1 + intra_2)


# --- persistence ----------------------------------------------------------

def test_save_and_load_roundtrip(
    tiny_config: ParametricUMAPConfig, tmp_path: Path,
) -> None:
    X = _two_blob_data(n=40, d=5)
    pu = ParametricUMAP(tiny_config)
    pu.fit(X, device=torch.device("cpu"))
    Z_before = pu.transform(X)
    path = tmp_path / "encoder.pt"
    pu.save(path)
    pu2 = ParametricUMAP.load(path, device=torch.device("cpu"))
    Z_after = pu2.transform(X)
    assert np.allclose(Z_before, Z_after, atol=1e-5)
    assert pu2.config == tiny_config
    assert pu2.input_dim == 5


def test_save_without_fit_raises(
    tiny_config: ParametricUMAPConfig, tmp_path: Path,
) -> None:
    pu = ParametricUMAP(tiny_config)
    with pytest.raises(RuntimeError, match="not fitted"):
        pu.save(tmp_path / "never.pt")
