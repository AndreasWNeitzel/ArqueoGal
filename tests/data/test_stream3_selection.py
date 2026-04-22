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
    DEFAULT_DISTANCE_CUT_KPC,
    DEFAULT_PER_CELL,
    StratificationResult,
    VolumeLimitedResult,
    stratified_subsample,
    volume_limited_subsample,
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


# ---- volume-limited sampler --------------------------------------------------


def _volume_synthetic(
    n: int,
    *,
    d_min: float = 0.05,
    d_max: float = 5.0,
    rng_seed: int = 0,
    distance_col: str = "d_photogeo_kpc",
) -> pd.DataFrame:
    """Synthetic catalogue uniformly distributed in distance over [d_min, d_max] kpc."""
    rng = np.random.default_rng(rng_seed)
    return pd.DataFrame(
        {
            "source_id": np.arange(n, dtype=np.int64),
            distance_col: rng.uniform(d_min, d_max, n),
            "teff": rng.uniform(4000.0, 5500.0, n),
        }
    )


def test_volume_limited_default_cut_is_2_5_kpc() -> None:
    assert DEFAULT_DISTANCE_CUT_KPC == 2.5


def test_volume_limited_no_selected_star_above_cut() -> None:
    df = _volume_synthetic(2_000, rng_seed=0)
    result = volume_limited_subsample(
        df, distance_col="d_photogeo_kpc",
        distance_cut_kpc=2.5, n_target=500, seed=7,
    )
    assert (result.sample["d_photogeo_kpc"] < 2.5).all()


def test_volume_limited_deterministic_with_seed() -> None:
    df = _volume_synthetic(3_000, rng_seed=99)
    r1 = volume_limited_subsample(
        df, distance_col="d_photogeo_kpc",
        distance_cut_kpc=2.5, n_target=400, seed=42,
    )
    r2 = volume_limited_subsample(
        df, distance_col="d_photogeo_kpc",
        distance_cut_kpc=2.5, n_target=400, seed=42,
    )
    pd.testing.assert_frame_equal(r1.sample, r2.sample)


def test_volume_limited_different_seeds_give_different_picks() -> None:
    df = _volume_synthetic(5_000, rng_seed=1)
    r1 = volume_limited_subsample(
        df, distance_col="d_photogeo_kpc",
        distance_cut_kpc=2.5, n_target=400, seed=1,
    )
    r2 = volume_limited_subsample(
        df, distance_col="d_photogeo_kpc",
        distance_cut_kpc=2.5, n_target=400, seed=2,
    )
    s1 = set(r1.sample["source_id"])
    s2 = set(r2.sample["source_id"])
    assert s1 != s2


def test_volume_limited_selection_size_matches_min_rule() -> None:
    df = _volume_synthetic(4_000, d_min=0.1, d_max=4.0, rng_seed=3)
    n_below_expected = int((df["d_photogeo_kpc"] < 2.5).sum())
    # Pool > n_target.
    r_cap = volume_limited_subsample(
        df, distance_col="d_photogeo_kpc",
        distance_cut_kpc=2.5, n_target=100, seed=0,
    )
    assert len(r_cap.sample) == 100
    assert r_cap.n_below_cut == n_below_expected
    assert r_cap.n_selected == 100
    # Pool <= n_target.
    r_full = volume_limited_subsample(
        df, distance_col="d_photogeo_kpc",
        distance_cut_kpc=2.5, n_target=10 * n_below_expected, seed=0,
    )
    assert len(r_full.sample) == n_below_expected
    assert r_full.n_selected == n_below_expected


def test_volume_limited_preserves_input_schema() -> None:
    df = _volume_synthetic(500, rng_seed=0)
    df["extra_col"] = "foo"
    result = volume_limited_subsample(
        df, distance_col="d_photogeo_kpc",
        distance_cut_kpc=2.5, n_target=50, seed=0,
    )
    assert list(result.sample.columns) == list(df.columns)
    assert (result.sample["extra_col"] == "foo").all()


def test_volume_limited_returns_full_pool_when_below_n_target() -> None:
    # Only 3 stars below cut, n_target=100 → returns 3.
    df = pd.DataFrame(
        {
            "source_id": np.arange(10),
            "d_photogeo_kpc": np.concatenate(
                [np.array([0.1, 0.5, 1.0]), np.full(7, 5.0)]
            ),
        }
    )
    result = volume_limited_subsample(
        df, distance_col="d_photogeo_kpc",
        distance_cut_kpc=2.5, n_target=100, seed=0,
    )
    assert len(result.sample) == 3
    assert result.n_below_cut == 3


def test_volume_limited_nan_distance_excluded() -> None:
    df = pd.DataFrame(
        {
            "source_id": np.arange(5),
            "d_photogeo_kpc": [0.5, np.nan, 1.0, np.nan, 2.0],
        }
    )
    result = volume_limited_subsample(
        df, distance_col="d_photogeo_kpc",
        distance_cut_kpc=2.5, n_target=100, seed=0,
    )
    # Only the 3 finite, below-cut stars should be kept.
    assert len(result.sample) == 3
    assert result.sample["d_photogeo_kpc"].notna().all()


def test_volume_limited_returns_result_type() -> None:
    df = _volume_synthetic(200, rng_seed=0)
    result = volume_limited_subsample(
        df, distance_col="d_photogeo_kpc",
        distance_cut_kpc=2.5, n_target=50, seed=11,
    )
    assert isinstance(result, VolumeLimitedResult)
    assert result.distance_col == "d_photogeo_kpc"
    assert result.distance_cut_kpc == 2.5
    assert result.n_target == 50
    assert result.rng_seed == 11


def test_volume_limited_to_provenance_is_serialisable() -> None:
    import json
    df = _volume_synthetic(200, rng_seed=0)
    result = volume_limited_subsample(
        df, distance_col="d_photogeo_kpc",
        distance_cut_kpc=2.5, n_target=50, seed=11,
    )
    prov = result.to_provenance()
    for key in (
        "method", "distance_col", "distance_cut_kpc", "n_target",
        "n_input", "n_below_cut", "n_selected", "rng_seed",
    ):
        assert key in prov
    json.dumps(prov)  # must be JSON-safe


def test_volume_limited_missing_column_raises() -> None:
    df = pd.DataFrame({"source_id": [1, 2]})
    with pytest.raises(KeyError, match="volume_limited_subsample requires column"):
        volume_limited_subsample(
            df, distance_col="d_photogeo_kpc",
            distance_cut_kpc=2.5, n_target=10,
        )


def test_volume_limited_nonpositive_cut_raises() -> None:
    df = _volume_synthetic(100, rng_seed=0)
    with pytest.raises(ValueError, match="distance_cut_kpc"):
        volume_limited_subsample(
            df, distance_col="d_photogeo_kpc",
            distance_cut_kpc=0.0, n_target=10,
        )


def test_volume_limited_nonpositive_ntarget_raises() -> None:
    df = _volume_synthetic(100, rng_seed=0)
    with pytest.raises(ValueError, match="n_target"):
        volume_limited_subsample(
            df, distance_col="d_photogeo_kpc",
            distance_cut_kpc=2.5, n_target=0,
        )


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
