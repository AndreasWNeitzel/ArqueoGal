"""Tests for population_classifier.main.hyperparameter — DBCV grid search."""

from __future__ import annotations

import numpy as np
import pytest

from arqueogal.population_classifier.main.hyperparameter import (
    GridCell,
    HyperparameterGrid,
    UMAPHyperparams,
    dbcv_score,
    grid_search,
)

# --- grid combinatorics ---------------------------------------------------

def test_default_grid_n_combinations_matches_section_10_4() -> None:
    """5 × 3 × 3 × 4 × 3 × 3 × 2 = 3240 per research_brief §10.4."""
    grid = HyperparameterGrid()
    assert grid.n_combinations() == 5 * 3 * 3 * 4 * 3 * 3 * 2 == 3240


def test_iter_combinations_yields_expected_count() -> None:
    grid = HyperparameterGrid(
        n_neighbors=(15, 30), min_dist=(0.1,), n_components=(2,),
        min_cluster_size=(50,), min_samples=(5,),
        cluster_selection_epsilon=(0.0,), cluster_selection_method=("eom",),
    )
    combos = list(grid.iter_combinations())
    assert len(combos) == 2
    assert grid.n_combinations() == 2


def test_iter_combinations_produces_unique_umap_hdb_pairs() -> None:
    grid = HyperparameterGrid(
        n_neighbors=(15, 30), min_dist=(0.1, 0.2), n_components=(2,),
        min_cluster_size=(50, 100), min_samples=(5,),
        cluster_selection_epsilon=(0.0,), cluster_selection_method=("eom",),
    )
    combos = list(grid.iter_combinations())
    # 2 × 2 × 1 × 2 × 1 × 1 × 1 = 8 unique pairs.
    assert len({(u, h) for u, h in combos}) == 8


def test_iter_combinations_unique_umap_cells_equals_umap_axes_product() -> None:
    grid = HyperparameterGrid(
        n_neighbors=(15, 30, 50), min_dist=(0.0, 0.1), n_components=(2, 3),
        min_cluster_size=(50,), min_samples=(5,),
        cluster_selection_epsilon=(0.0,), cluster_selection_method=("eom",),
    )
    umap_cells = {u for u, _ in grid.iter_combinations()}
    assert len(umap_cells) == 3 * 2 * 2


# --- dbcv_score ----------------------------------------------------------

def test_dbcv_score_undefined_for_all_noise() -> None:
    Z = np.random.default_rng(0).standard_normal((20, 2))
    labels = -np.ones(20, dtype=np.int64)
    assert dbcv_score(Z, labels) == -1.0


def test_dbcv_score_undefined_for_single_cluster() -> None:
    Z = np.random.default_rng(0).standard_normal((20, 2))
    labels = np.zeros(20, dtype=np.int64)
    assert dbcv_score(Z, labels) == -1.0


def test_dbcv_score_positive_for_well_separated_clusters() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(-5, 0.2, (30, 2))
    b = rng.normal(+5, 0.2, (30, 2))
    Z = np.vstack([a, b])
    labels = np.array([0] * 30 + [1] * 30, dtype=np.int64)
    score = dbcv_score(Z, labels)
    assert np.isfinite(score)
    assert score > 0.0


def test_dbcv_score_rejects_1d_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        dbcv_score(np.zeros(10), np.zeros(10, dtype=np.int64))


def test_dbcv_score_rejects_mismatched_labels() -> None:
    with pytest.raises(ValueError, match="labels length"):
        dbcv_score(np.zeros((10, 2)), np.zeros(7, dtype=np.int64))


# --- grid_search end-to-end ----------------------------------------------

def _identity_embed(X: np.ndarray, hp: UMAPHyperparams) -> np.ndarray:
    """Cheap embed_fn for tests: take the first ``n_components`` columns."""
    return X[:, : hp.n_components].astype(np.float32)


def _two_well_separated_clusters(n_per: int = 40, d: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(-5.0, 0.2, (n_per, d))
    b = rng.normal(+5.0, 0.2, (n_per, d))
    return np.vstack([a, b]).astype(np.float32)


def test_grid_search_returns_cells_sorted_by_descending_dbcv() -> None:
    X = _two_well_separated_clusters()
    grid = HyperparameterGrid(
        n_neighbors=(15,), min_dist=(0.1,), n_components=(2, 3),
        min_cluster_size=(10, 20), min_samples=(5,),
        cluster_selection_epsilon=(0.0,), cluster_selection_method=("eom",),
    )
    results = grid_search(X, grid, _identity_embed)
    assert len(results) == grid.n_combinations()
    dbcvs = [c.dbcv for c in results]
    assert dbcvs == sorted(dbcvs, reverse=True)


def test_grid_search_caches_embedding_per_umap_cell() -> None:
    X = _two_well_separated_clusters()
    grid = HyperparameterGrid(
        n_neighbors=(15, 30), min_dist=(0.1,), n_components=(2,),
        min_cluster_size=(10, 20), min_samples=(5,),
        cluster_selection_epsilon=(0.0,), cluster_selection_method=("eom",),
    )
    calls: list[UMAPHyperparams] = []

    def counting_embed(X_in: np.ndarray, hp: UMAPHyperparams) -> np.ndarray:
        calls.append(hp)
        return _identity_embed(X_in, hp)

    grid_search(X, grid, counting_embed)
    # 2 unique UMAP cells, 2 HDBSCAN variants → 4 total combos but 2 embed calls.
    assert len(calls) == 2
    assert len(set(calls)) == 2


def test_grid_search_top_k_truncates() -> None:
    X = _two_well_separated_clusters()
    grid = HyperparameterGrid(
        n_neighbors=(15,), min_dist=(0.1,), n_components=(2, 3),
        min_cluster_size=(10, 20), min_samples=(5,),
        cluster_selection_epsilon=(0.0,), cluster_selection_method=("eom",),
    )
    all_cells = grid_search(X, grid, _identity_embed)
    top2 = grid_search(X, grid, _identity_embed, top_k=2)
    assert len(top2) == 2
    assert [c.dbcv for c in top2] == [c.dbcv for c in all_cells[:2]]


def test_grid_search_progress_callback_invoked_per_cell() -> None:
    X = _two_well_separated_clusters()
    grid = HyperparameterGrid(
        n_neighbors=(15,), min_dist=(0.1,), n_components=(2,),
        min_cluster_size=(10, 20), min_samples=(5,),
        cluster_selection_epsilon=(0.0,), cluster_selection_method=("eom",),
    )
    seen: list[tuple[int, int]] = []

    def cb(idx: int, total: int, _cell: GridCell) -> None:
        seen.append((idx, total))

    grid_search(X, grid, _identity_embed, progress_callback=cb)
    assert len(seen) == grid.n_combinations()
    assert seen[0] == (0, grid.n_combinations())
    assert seen[-1] == (grid.n_combinations() - 1, grid.n_combinations())


def test_grid_search_rejects_1d_input() -> None:
    grid = HyperparameterGrid()
    with pytest.raises(ValueError, match="X must be 2-D"):
        grid_search(np.zeros(10), grid, _identity_embed)


def test_grid_cell_as_dict_json_serialisable() -> None:
    import json

    X = _two_well_separated_clusters()
    grid = HyperparameterGrid(
        n_neighbors=(15,), min_dist=(0.1,), n_components=(2,),
        min_cluster_size=(10,), min_samples=(5,),
        cluster_selection_epsilon=(0.0,), cluster_selection_method=("eom",),
    )
    cells = grid_search(X, grid, _identity_embed)
    payload = cells[0].as_dict()
    s = json.dumps(payload)
    reloaded = json.loads(s)
    assert reloaded["umap"]["n_neighbors"] == 15
    assert reloaded["hdbscan"]["min_cluster_size"] == 10
    assert "dbcv" in reloaded
    assert "n_clusters" in reloaded
    assert "noise_fraction" in reloaded
