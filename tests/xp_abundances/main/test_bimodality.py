"""Tests for ``arqueogal.xp_abundances.main.bimodality``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from arqueogal.xp_abundances.main.bimodality import (
    BimodalityGrid,
    _is_cell_bimodal,
    fit_bimodality_grid,
)


def _make_bimodal_sample(rng: np.random.Generator, n: int) -> np.ndarray:
    """Two-mode sample at +0.05 and +0.25 with equal weights."""
    k = rng.binomial(1, 0.5, size=n)
    mu = np.where(k == 0, 0.05, 0.25)
    return rng.normal(mu, 0.02, size=n)


def _make_unimodal_sample(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.normal(0.10, 0.04, size=n)


def test_is_cell_bimodal_flags_clearly_bimodal_distribution():
    rng = np.random.default_rng(0)
    a = _make_bimodal_sample(rng, 500)
    flag, stats = _is_cell_bimodal(
        a,
        min_minor_weight=0.15,
        min_mean_sep=0.08,
        bic_delta_min=4.0,
    )
    assert flag is True
    assert stats["bic_delta"] > 4.0
    assert stats["mean_sep"] > 0.08


def test_is_cell_bimodal_does_not_flag_unimodal_distribution():
    rng = np.random.default_rng(1)
    a = _make_unimodal_sample(rng, 500)
    flag, _ = _is_cell_bimodal(
        a,
        min_minor_weight=0.15,
        min_mean_sep=0.08,
        bic_delta_min=4.0,
    )
    assert flag is False


def test_is_cell_bimodal_rejects_too_small_minor_mode():
    """A 97:3 mixture is dominated by one mode — should NOT be flagged."""
    rng = np.random.default_rng(2)
    n = 500
    k = rng.binomial(1, 0.03, size=n)  # 3% minority
    mu = np.where(k == 0, 0.05, 0.30)
    a = rng.normal(mu, 0.02, size=n)
    flag, _ = _is_cell_bimodal(
        a,
        min_minor_weight=0.15,
        min_mean_sep=0.08,
        bic_delta_min=4.0,
    )
    assert flag is False


def test_is_cell_bimodal_too_few_points():
    rng = np.random.default_rng(3)
    flag, stats = _is_cell_bimodal(
        rng.normal(0.10, 0.04, size=5),
        min_minor_weight=0.15,
        min_mean_sep=0.08,
        bic_delta_min=4.0,
    )
    assert flag is False
    assert stats["reason"] == "too few points"


def test_fit_bimodality_grid_detects_synthetic_bimodal_cell():
    """Two cells: one bimodal, one unimodal. The fitter flags only the bimodal one."""
    rng = np.random.default_rng(10)
    n = 400

    # Single-cell grid — points all fall in one cell per concatenation
    # Bimodal cell: Teff 4800, logg 2.5, [M/H] -0.4
    t1 = np.full(n, 4800.0)
    l1 = np.full(n, 2.5)
    m1 = np.full(n, -0.4)
    a1 = _make_bimodal_sample(rng, n)

    # Unimodal cell: Teff 4800, logg 2.5, [M/H] -1.8
    t2 = np.full(n, 4800.0)
    l2 = np.full(n, 2.5)
    m2 = np.full(n, -1.8)
    a2 = _make_unimodal_sample(rng, n)

    teff = np.concatenate([t1, t2])
    logg = np.concatenate([l1, l2])
    mh = np.concatenate([m1, m2])
    alpha = np.concatenate([a1, a2])

    grid = fit_bimodality_grid(
        teff,
        logg,
        mh,
        alpha,
        teff_edges=np.array([4500.0, 5100.0]),
        logg_edges=np.array([2.2, 2.8]),
        mh_edges=np.array([-2.0, -1.5, -0.5, 0.0]),
        min_cell_n=50,
    )
    # Exactly one cell (the [M/H]-in-[-0.5, 0) one) should be flagged.
    assert grid.is_bimodal.sum() == 1
    assert grid.is_bimodal[0, 0, 2]
    assert not grid.is_bimodal[0, 0, 0]
    flag_bi, in_bi = grid.query(
        np.array([4800.0]),
        np.array([2.5]),
        np.array([-0.4]),
    )
    assert flag_bi[0] and in_bi[0]
    flag_uni, in_uni = grid.query(
        np.array([4800.0]),
        np.array([2.5]),
        np.array([-1.8]),
    )
    assert in_uni[0] and not flag_uni[0]


def test_query_out_of_grid_returns_false_and_not_in_grid():
    teff_edges = np.array([4500.0, 4700.0, 4900.0])
    logg_edges = np.array([2.0, 3.0])
    mh_edges = np.array([-1.0, 0.0])
    grid = BimodalityGrid(
        teff_edges=teff_edges,
        logg_edges=logg_edges,
        mh_edges=mh_edges,
        is_bimodal=np.ones((2, 1, 1), dtype=bool),
        n_per_cell=np.full((2, 1, 1), 100, dtype=np.int32),
        min_cell_n=50,
        min_minor_weight=0.15,
        min_mean_sep=0.08,
        bic_delta_min=4.0,
    )
    # Inside grid — should return True
    flag, in_grid = grid.query(
        np.array([4600.0]),
        np.array([2.5]),
        np.array([-0.5]),
    )
    assert flag[0] and in_grid[0]
    # Outside grid in Teff
    flag, in_grid = grid.query(
        np.array([6000.0]),
        np.array([2.5]),
        np.array([-0.5]),
    )
    assert not flag[0] and not in_grid[0]
    # NaN entries must also be marked out-of-grid (not flag-propagating)
    flag, in_grid = grid.query(
        np.array([np.nan]),
        np.array([2.5]),
        np.array([-0.5]),
    )
    assert not flag[0] and not in_grid[0]


def test_query_mismatched_shapes_raises():
    grid = BimodalityGrid(
        teff_edges=np.array([4000.0, 5000.0]),
        logg_edges=np.array([2.0, 3.0]),
        mh_edges=np.array([-1.0, 0.0]),
        is_bimodal=np.zeros((1, 1, 1), dtype=bool),
        n_per_cell=np.zeros((1, 1, 1), dtype=np.int32),
        min_cell_n=50,
        min_minor_weight=0.15,
        min_mean_sep=0.08,
        bic_delta_min=4.0,
    )
    with pytest.raises(ValueError, match="share shape"):
        grid.query(np.array([4500.0, 4600.0]), np.array([2.5]), np.array([-0.5]))


def test_save_load_round_trip(tmp_path: Path):
    g = BimodalityGrid(
        teff_edges=np.array([4000.0, 4500.0, 5000.0]),
        logg_edges=np.array([1.0, 2.0, 3.0]),
        mh_edges=np.array([-1.0, -0.5, 0.0]),
        is_bimodal=np.array(
            [
                [[True, False], [False, True]],
                [[False, True], [True, False]],
            ]
        ),
        n_per_cell=np.full((2, 2, 2), 100, dtype=np.int32),
        min_cell_n=50,
        min_minor_weight=0.15,
        min_mean_sep=0.08,
        bic_delta_min=4.0,
    )
    path = tmp_path / "grid.npz"
    g.save(path, provenance={"source": "round-trip test"})
    assert path.is_file()
    sidecar = path.with_suffix(path.suffix + ".provenance.json")
    assert sidecar.is_file()
    prov = json.loads(sidecar.read_text())
    assert prov["source"] == {"source": "round-trip test"}
    assert prov["counts"]["cells_bimodal"] == int(g.is_bimodal.sum())

    back = BimodalityGrid.load(path)
    np.testing.assert_array_equal(back.teff_edges, g.teff_edges)
    np.testing.assert_array_equal(back.logg_edges, g.logg_edges)
    np.testing.assert_array_equal(back.mh_edges, g.mh_edges)
    np.testing.assert_array_equal(back.is_bimodal, g.is_bimodal)
    np.testing.assert_array_equal(back.n_per_cell, g.n_per_cell)
    assert back.min_cell_n == g.min_cell_n
    assert back.min_minor_weight == g.min_minor_weight
    assert back.min_mean_sep == g.min_mean_sep
    assert back.bic_delta_min == g.bic_delta_min


def test_fit_bimodality_grid_skips_small_cells():
    """Cells with N < min_cell_n must be left unflagged (not evaluated)."""
    rng = np.random.default_rng(11)
    # Synthesize 30 bimodal points in one cell (< min_cell_n=50)
    teff = rng.uniform(4600, 4800, size=30)
    logg = rng.uniform(2.3, 2.5, size=30)
    mh = rng.uniform(-0.5, -0.3, size=30)
    alpha = _make_bimodal_sample(rng, 30)

    grid = fit_bimodality_grid(
        teff,
        logg,
        mh,
        alpha,
        teff_edges=np.array([4500.0, 4900.0]),
        logg_edges=np.array([2.2, 2.6]),
        mh_edges=np.array([-0.5, -0.3]),
        min_cell_n=50,
    )
    assert not grid.is_bimodal.any()
    assert int(grid.n_per_cell.sum()) == 30


def test_fit_bimodality_grid_drops_nans():
    rng = np.random.default_rng(12)
    n = 200
    teff = rng.uniform(4600, 4800, size=n)
    logg = rng.uniform(2.3, 2.5, size=n)
    mh = rng.uniform(-0.5, -0.3, size=n)
    alpha = _make_bimodal_sample(rng, n)
    # Corrupt half the rows with NaN
    alpha[: n // 2] = np.nan

    grid = fit_bimodality_grid(
        teff,
        logg,
        mh,
        alpha,
        teff_edges=np.array([4500.0, 4900.0]),
        logg_edges=np.array([2.2, 2.6]),
        mh_edges=np.array([-0.5, -0.3]),
        min_cell_n=50,
    )
    assert int(grid.n_per_cell.sum()) == n // 2
