"""Tests for xp_abundances.main.halfway_umap — Run A's halfway gate (#132).

UMAP is heavy and non-deterministic in edge cases, so :func:`compute_halfway_embedding`
is exercised here with a stub ``umap.UMAP`` that fit-transforms to a deterministic
``(N, 2)`` array. This lets us verify the trunk wiring, finite-label accounting,
and plot filename contract without pulling ``umap-learn`` into the test loop.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch

from arqueogal.xp_abundances.main.adapter import XpFeatureAdapter
from arqueogal.xp_abundances.main.data import FeatureLayout
from arqueogal.xp_abundances.main.halfway_umap import (
    HalfwayEmbedding,
    compute_halfway_embedding,
    save_halfway_plots,
)
from arqueogal.xp_abundances.main.model import ModelConfig, XpAbundanceModel

# --- umap stub ---------------------------------------------------------------


class _StubReducer:
    """Minimal ``umap.UMAP`` stand-in — projects to first two columns of ``H``.

    Returns a deterministic ``(N, 2)`` array so we can assert on it without
    invoking the real UMAP. Matches the call surface used by
    :func:`compute_halfway_embedding`.
    """

    def __init__(self, **_: object) -> None:
        self.kwargs = _

    def fit_transform(self, H: np.ndarray) -> np.ndarray:
        return H[:, :2].astype(np.float32)


def _install_umap_stub(monkeypatch) -> None:
    stub_mod = types.ModuleType("umap")
    stub_mod.UMAP = _StubReducer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "umap", stub_mod)


# --- HalfwayEmbedding --------------------------------------------------------


def test_halfway_embedding_dataclass_construction() -> None:
    emb = np.zeros((5, 2), dtype=np.float32)
    labels = {
        "teff_apogee": np.array([4500.0, 4700.0, np.nan, 4900.0, 5000.0]),
        "mh_apogee": np.array([-0.5, -0.3, -0.1, 0.0, 0.2]),
        "logg_apogee": np.array([2.0, np.nan, np.nan, 2.5, 3.0]),
    }
    he = HalfwayEmbedding(
        embedding=emb,
        labels=labels,
        n_stars=5,
        n_finite={"teff_apogee": 4, "mh_apogee": 5, "logg_apogee": 3},
    )
    assert he.embedding.shape == (5, 2)
    assert set(he.labels) == {"teff_apogee", "mh_apogee", "logg_apogee"}
    assert he.n_stars == 5
    assert he.n_finite["logg_apogee"] == 3


def test_halfway_embedding_default_n_finite_is_empty() -> None:
    """``n_finite`` has a default factory; zero-arg form is legal."""
    he = HalfwayEmbedding(
        embedding=np.zeros((1, 2), dtype=np.float32),
        labels={"teff_apogee": np.array([4500.0])},
        n_stars=1,
    )
    assert he.n_finite == {}


# --- compute_halfway_embedding ----------------------------------------------


def _tiny_model() -> tuple[XpAbundanceModel, XpFeatureAdapter, FeatureLayout]:
    """Build the smallest legal model + adapter — a trunk that runs in microseconds."""
    layout = FeatureLayout()
    cfg = ModelConfig(
        input_dim=layout.input_dim,
        trunk_hidden=(32, 16),
        latent_dim=8,
        head_hidden=16,
    )
    model = XpAbundanceModel(cfg)
    adapter = XpFeatureAdapter(layout, use_c0_scalars=False)
    return model, adapter, layout


def test_compute_halfway_embedding_shape_and_n_finite(monkeypatch) -> None:
    _install_umap_stub(monkeypatch)
    model, adapter, layout = _tiny_model()

    rng = np.random.default_rng(0)
    N = 40
    X = rng.standard_normal((N, layout.input_dim)).astype(np.float32)

    teff = rng.uniform(4000, 5500, size=N)
    mh = rng.uniform(-1.0, 0.3, size=N)
    logg = rng.uniform(1.5, 3.5, size=N)
    # Inject NaNs in known positions so n_finite has something to count.
    teff[:3] = np.nan
    logg[[10, 20, 30, 35]] = np.nan

    labels = {"teff_apogee": teff, "mh_apogee": mh, "logg_apogee": logg}

    he = compute_halfway_embedding(
        model,
        adapter,
        X,
        labels,
        device=torch.device("cpu"),
        n_neighbors=5,
        min_dist=0.1,
        umap_seed=0,
        batch_size=16,
    )

    assert isinstance(he, HalfwayEmbedding)
    assert he.embedding.shape == (N, 2)
    assert he.embedding.dtype == np.float32
    assert he.n_stars == N
    assert he.n_finite == {"teff_apogee": N - 3, "mh_apogee": N, "logg_apogee": N - 4}
    assert set(he.labels) == {"teff_apogee", "mh_apogee", "logg_apogee"}


def test_compute_halfway_embedding_batching_is_invariant(monkeypatch) -> None:
    """Trunk output must not depend on the batch_size knob."""
    _install_umap_stub(monkeypatch)
    model, adapter, layout = _tiny_model()

    rng = np.random.default_rng(1)
    N = 24
    X = rng.standard_normal((N, layout.input_dim)).astype(np.float32)
    labels = {
        "teff_apogee": rng.uniform(4000, 5500, size=N),
        "mh_apogee": rng.uniform(-1.0, 0.3, size=N),
        "logg_apogee": rng.uniform(1.5, 3.5, size=N),
    }

    he_big = compute_halfway_embedding(
        model,
        adapter,
        X,
        labels,
        device=torch.device("cpu"),
        batch_size=N,
        n_neighbors=5,
    )
    he_small = compute_halfway_embedding(
        model,
        adapter,
        X,
        labels,
        device=torch.device("cpu"),
        batch_size=7,
        n_neighbors=5,
    )
    np.testing.assert_allclose(he_big.embedding, he_small.embedding, atol=1e-6)


def test_compute_halfway_embedding_uses_h_not_z(monkeypatch) -> None:
    """Sanity: the stub reducer receives ``h`` (trunk), not ``z`` (L2-normalised).

    Captured via a spy: ``z`` rows are unit-norm so its first two columns would all
    sit inside [-1, 1]; ``h`` is unbounded. We check that the returned embedding's
    range is consistent with ``h``.
    """
    _install_umap_stub(monkeypatch)
    model, adapter, layout = _tiny_model()

    rng = np.random.default_rng(2)
    N = 20
    # Large-amplitude input ⇒ the trunk output drifts outside [-1, 1] reliably.
    X = (rng.standard_normal((N, layout.input_dim)) * 5.0).astype(np.float32)
    labels = {
        "teff_apogee": rng.uniform(4000, 5500, size=N),
        "mh_apogee": rng.uniform(-1.0, 0.3, size=N),
        "logg_apogee": rng.uniform(1.5, 3.5, size=N),
    }

    # Compare to what ``z`` would look like (bounded) to confirm ``h`` was used.
    model.eval()
    with torch.no_grad():
        h, z = model.encoder(adapter(torch.as_tensor(X)))
    assert z.pow(2).sum(dim=-1).sqrt().max().item() <= 1.0 + 1e-5  # unit norm

    he = compute_halfway_embedding(
        model,
        adapter,
        X,
        labels,
        device=torch.device("cpu"),
        batch_size=N,
        n_neighbors=5,
    )
    # Stub projects first two columns of H; equality proves h (not z) was passed.
    np.testing.assert_allclose(he.embedding, h.numpy()[:, :2], atol=1e-5)


def test_compute_halfway_embedding_keeps_labels_verbatim(monkeypatch) -> None:
    """Labels dict is stored unchanged for downstream plotting."""
    _install_umap_stub(monkeypatch)
    model, adapter, layout = _tiny_model()

    rng = np.random.default_rng(3)
    N = 16
    X = rng.standard_normal((N, layout.input_dim)).astype(np.float32)
    labels = {
        "teff_apogee": rng.uniform(4000, 5500, size=N),
        "mh_apogee": rng.uniform(-1.0, 0.3, size=N),
        "logg_apogee": rng.uniform(1.5, 3.5, size=N),
    }

    he = compute_halfway_embedding(
        model,
        adapter,
        X,
        labels,
        device=torch.device("cpu"),
        batch_size=N,
        n_neighbors=5,
    )
    for k, v in labels.items():
        np.testing.assert_array_equal(he.labels[k], v)


# --- save_halfway_plots -------------------------------------------------------


def _make_he(n: int = 12) -> HalfwayEmbedding:
    rng = np.random.default_rng(4)
    emb = rng.standard_normal((n, 2)).astype(np.float32)
    labels = {
        "teff_apogee": rng.uniform(4000, 5500, size=n),
        "mh_apogee": rng.uniform(-1.0, 0.3, size=n),
        "logg_apogee": rng.uniform(1.5, 3.5, size=n),
    }
    # Inject NaN so the "finite subset" plot path is exercised.
    labels["teff_apogee"][0] = np.nan
    return HalfwayEmbedding(
        embedding=emb,
        labels=labels,
        n_stars=n,
        n_finite={k: int(np.isfinite(v).sum()) for k, v in labels.items()},
    )


def test_save_halfway_plots_writes_expected_filenames(tmp_path: Path) -> None:
    he = _make_he()
    paths = save_halfway_plots(he, tmp_path, prefix="halfway")
    names = sorted(p.name for p in paths)
    assert names == [
        "halfway_umap_logg_apogee.png",
        "halfway_umap_mh_apogee.png",
        "halfway_umap_teff_apogee.png",
    ]
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0


def test_save_halfway_plots_custom_prefix(tmp_path: Path) -> None:
    he = _make_he()
    paths = save_halfway_plots(he, tmp_path, prefix="run_a")
    for p in paths:
        assert p.name.startswith("run_a_umap_")
        assert p.suffix == ".png"


def test_save_halfway_plots_creates_missing_dir(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "halfway"
    assert not out.exists()
    he = _make_he()
    paths = save_halfway_plots(he, out)
    assert out.is_dir()
    assert len(paths) == 3


def test_save_halfway_plots_returns_three_tier1_paths(tmp_path: Path) -> None:
    he = _make_he()
    paths = save_halfway_plots(he, tmp_path)
    assert len(paths) == 3
    # Contract: one file per label key in ``he.labels``.
    assert {p.name for p in paths} == {f"halfway_umap_{k}.png" for k in he.labels}
