"""Tests for arqueogal.data.stream3_selection — §5.3 stratified sampler.

Fully offline. Uses synthetic star frames so bin population per cell is
controllable and the stratification contract is checked directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arqueogal.data.stream3_selection import (
    DEFAULT_BINS_G,
    DEFAULT_BINS_LOGG,
    DEFAULT_BINS_MH,
    DEFAULT_BINS_TEFF,
    DEFAULT_PER_CELL,
    StratificationResult,
    stratified_subsample,
)

# ---- helpers -----------------------------------------------------------------


def _synthetic(n: int, *, rng_seed: int = 0) -> pd.DataFrame:
    """Uniformly fill the default bin box with ``n`` stars."""
    rng = np.random.default_rng(rng_seed)
    return pd.DataFrame(
        {
            "source_id": np.arange(n, dtype=np.int64),
            "teff_xgboost": rng.uniform(DEFAULT_BINS_TEFF[0], DEFAULT_BINS_TEFF[-1], n),
            "logg_xgboost": rng.uniform(DEFAULT_BINS_LOGG[0], DEFAULT_BINS_LOGG[-1], n),
            "mh_xgboost": rng.uniform(DEFAULT_BINS_MH[0], DEFAULT_BINS_MH[-1], n),
            "phot_g_mean_mag": rng.uniform(DEFAULT_BINS_G[0], DEFAULT_BINS_G[-1], n),
        }
    )


# ---- §5.3 default bin spec ---------------------------------------------------


def test_default_bins_match_section_5_3() -> None:
    """§5.3 literals."""
    np.testing.assert_array_equal(DEFAULT_BINS_TEFF, np.linspace(4000.0, 5500.0, 7))
    np.testing.assert_array_equal(DEFAULT_BINS_LOGG, np.linspace(1.0, 3.5, 6))
    np.testing.assert_array_equal(DEFAULT_BINS_MH, np.linspace(-2.0, 0.5, 6))
    np.testing.assert_array_equal(DEFAULT_BINS_G, np.linspace(7.0, 16.0, 10))


def test_default_per_cell_matches_section_5_3() -> None:
    assert DEFAULT_PER_CELL == 600


def test_default_cells_total_2520() -> None:
    """§5.3: 6 × 5 × 5 × 9 = 1350 cells in the defaults (bins = edges - 1).

    Note the docstring in data_acquisition.md §5.3 rounds to "≈ 2520" but the
    literal linspaces give 6×5×5×9 = 1350 bins. This test pins the actual
    cell count and documents the discrepancy.
    """
    n_cells = (
        (len(DEFAULT_BINS_TEFF) - 1)
        * (len(DEFAULT_BINS_LOGG) - 1)
        * (len(DEFAULT_BINS_MH) - 1)
        * (len(DEFAULT_BINS_G) - 1)
    )
    assert n_cells == 1350


# ---- basic contract ----------------------------------------------------------


def test_returns_stratification_result() -> None:
    df = _synthetic(100)
    result = stratified_subsample(df, per_cell=5, rng_seed=7)
    assert isinstance(result, StratificationResult)
    assert isinstance(result.sample, pd.DataFrame)
    assert isinstance(result.cell_counts, pd.DataFrame)
    assert result.per_cell == 5
    assert result.rng_seed == 7


def test_output_row_count_within_bounds() -> None:
    """With many stars and a cap of k per cell, output ≤ k × n_cells."""
    df = _synthetic(5_000, rng_seed=42)
    result = stratified_subsample(df, per_cell=3, rng_seed=0)
    n_cells = result.cell_counts.shape[0]
    assert len(result.sample) <= 3 * n_cells
    assert len(result.sample) == result.cell_counts["n_selected"].sum()


def test_all_stars_taken_when_per_cell_exceeds_bin_population() -> None:
    """If every cell has < per_cell stars, every valid star is in the output."""
    df = _synthetic(50, rng_seed=1)
    result = stratified_subsample(df, per_cell=10_000, rng_seed=0)
    # Every row is within the default box and finite, so all taken.
    assert len(result.sample) == 50


def test_cell_cap_respected_for_oversampled_bin() -> None:
    """A single-cell DataFrame of 100 stars and per_cell=5 → exactly 5 out."""
    # Build a frame where every star lands in the same (0,0,0,0) cell.
    n = 100
    df = pd.DataFrame(
        {
            "source_id": np.arange(n),
            "teff_xgboost": np.full(n, DEFAULT_BINS_TEFF[0] + 1.0),
            "logg_xgboost": np.full(n, DEFAULT_BINS_LOGG[0] + 0.01),
            "mh_xgboost": np.full(n, DEFAULT_BINS_MH[0] + 0.01),
            "phot_g_mean_mag": np.full(n, DEFAULT_BINS_G[0] + 0.01),
        }
    )
    result = stratified_subsample(df, per_cell=5, rng_seed=0)
    assert len(result.sample) == 5
    assert len(result.cell_counts) == 1
    assert result.cell_counts["n_available"].iloc[0] == 100
    assert result.cell_counts["n_selected"].iloc[0] == 5


# ---- reproducibility ---------------------------------------------------------


def test_subsample_is_deterministic_with_seed() -> None:
    df = _synthetic(2_000, rng_seed=99)
    r1 = stratified_subsample(df, per_cell=3, rng_seed=42)
    r2 = stratified_subsample(df, per_cell=3, rng_seed=42)
    pd.testing.assert_frame_equal(r1.sample, r2.sample)
    pd.testing.assert_frame_equal(
        r1.cell_counts.sort_values(["i_teff", "i_logg", "i_mh", "i_g"]).reset_index(drop=True),
        r2.cell_counts.sort_values(["i_teff", "i_logg", "i_mh", "i_g"]).reset_index(drop=True),
    )


def test_different_seeds_give_different_picks() -> None:
    df = _synthetic(2_000, rng_seed=99)
    r1 = stratified_subsample(df, per_cell=3, rng_seed=1)
    r2 = stratified_subsample(df, per_cell=3, rng_seed=2)
    # At least one star selected in one but not the other.
    s1 = set(r1.sample["source_id"])
    s2 = set(r2.sample["source_id"])
    assert s1 != s2


# ---- exclusion rules ---------------------------------------------------------


def test_out_of_range_stars_are_excluded() -> None:
    """Stars outside the bin box are dropped and don't contribute to counts."""
    df = _synthetic(10, rng_seed=3)
    # Push 5 stars out of range.
    df.loc[:4, "teff_xgboost"] = 10_000.0  # above upper Teff edge
    result = stratified_subsample(df, per_cell=100, rng_seed=0)
    # Only the 5 in-range stars should appear.
    assert len(result.sample) == 5


def test_nan_stars_are_excluded() -> None:
    df = _synthetic(10, rng_seed=4)
    df.loc[:4, "logg_xgboost"] = np.nan
    result = stratified_subsample(df, per_cell=100, rng_seed=0)
    assert len(result.sample) == 5


def test_empty_input_returns_empty_result() -> None:
    df = pd.DataFrame(
        columns=["teff_xgboost", "logg_xgboost", "mh_xgboost", "phot_g_mean_mag"]
    )
    result = stratified_subsample(df)
    assert result.sample.empty
    assert result.cell_counts.empty


def test_all_out_of_range_returns_empty_sample() -> None:
    df = pd.DataFrame(
        {
            "source_id": np.arange(3),
            "teff_xgboost": [10_000.0, 10_000.0, 10_000.0],
            "logg_xgboost": [2.0, 2.0, 2.0],
            "mh_xgboost": [0.0, 0.0, 0.0],
            "phot_g_mean_mag": [12.0, 12.0, 12.0],
        }
    )
    result = stratified_subsample(df)
    assert result.sample.empty


# ---- validation --------------------------------------------------------------


def test_missing_stratification_column_raises() -> None:
    df = pd.DataFrame({"teff_xgboost": [4500.0]})
    with pytest.raises(KeyError, match="stratified_subsample requires columns"):
        stratified_subsample(df)


def test_nonpositive_per_cell_raises() -> None:
    df = _synthetic(5)
    with pytest.raises(ValueError, match="per_cell"):
        stratified_subsample(df, per_cell=0)


def test_custom_column_names_honored() -> None:
    df = _synthetic(50).rename(
        columns={
            "teff_xgboost": "T",
            "logg_xgboost": "g",
            "mh_xgboost": "m",
            "phot_g_mean_mag": "mag",
        }
    )
    result = stratified_subsample(
        df, teff_col="T", logg_col="g", mh_col="m", g_col="mag", per_cell=5,
    )
    assert len(result.sample) > 0
    assert result.columns == ("T", "g", "m", "mag")


# ---- provenance metadata -----------------------------------------------------


def test_to_provenance_is_serialisable_and_complete() -> None:
    df = _synthetic(200, rng_seed=11)
    result = stratified_subsample(df, per_cell=5, rng_seed=123)
    prov = result.to_provenance()
    for key in (
        "method", "bins_teff", "bins_logg", "bins_mh", "bins_g",
        "per_cell", "rng_seed", "n_available", "n_selected",
        "n_nonempty_cells", "stratification_columns",
    ):
        assert key in prov
    assert prov["rng_seed"] == 123
    assert prov["per_cell"] == 5
    assert prov["n_selected"] == len(result.sample)
    # Must be JSON-safe (no ndarrays, no non-primitives).
    import json
    json.dumps(prov)


# ---- stratification effect ---------------------------------------------------


def test_stratification_flattens_distribution() -> None:
    """An input skewed toward one cell should have most of that cell's excess
    capped by per_cell, producing a more uniform cell-count distribution."""
    rng = np.random.default_rng(0)
    # Half the sample is concentrated in a single cell.
    n_tall = 500
    n_spread = 500
    tall = pd.DataFrame(
        {
            "source_id": np.arange(n_tall),
            "teff_xgboost": np.full(n_tall, 4100.0),
            "logg_xgboost": np.full(n_tall, 1.2),
            "mh_xgboost": np.full(n_tall, -1.8),
            "phot_g_mean_mag": np.full(n_tall, 7.5),
        }
    )
    spread = pd.DataFrame(
        {
            "source_id": n_tall + np.arange(n_spread),
            "teff_xgboost": rng.uniform(4000, 5500, n_spread),
            "logg_xgboost": rng.uniform(1.0, 3.5, n_spread),
            "mh_xgboost": rng.uniform(-2.0, 0.5, n_spread),
            "phot_g_mean_mag": rng.uniform(7, 16, n_spread),
        }
    )
    df = pd.concat([tall, spread], ignore_index=True)

    # Without stratification cap (per_cell=∞): sample reflects the skew.
    uncapped = stratified_subsample(df, per_cell=10_000, rng_seed=0)
    # With cap: the tall bin contributes at most per_cell.
    capped = stratified_subsample(df, per_cell=10, rng_seed=0)
    assert capped.cell_counts["n_selected"].max() <= 10
    # The uncapped sample is larger than the capped one (tall cell dominates).
    assert len(uncapped.sample) > len(capped.sample)
