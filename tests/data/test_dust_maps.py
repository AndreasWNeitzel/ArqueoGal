"""Offline tests for arqueogal.data.dust_maps — §8.2 composition and §8.3
neighborhood-median A_G.

No external data (no dustmaps maps downloaded, no network I/O). The §8.2
tests pass plain callables as query stand-ins, so the routing logic is
fully exercised without any map fixtures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arqueogal.data.dust_maps import (
    DEFAULT_NEIGHBORHOOD_RADIUS_PC,
    FAR_BOUNDARY_KPC,
    MIN_NEIGHBORS_FOR_MEDIAN,
    NEAR_BOUNDARY_KPC,
    SFD_TO_AV_COEFF,
    ComposedAv,
    NeighborhoodAvFeatures,
    compose_av,
    galactic_to_xyz,
    neighborhood_av_features,
)

# ---- galactic_to_xyz ---------------------------------------------------------


def test_galactic_to_xyz_origin() -> None:
    xyz = galactic_to_xyz(np.array([0.0]), np.array([0.0]), np.array([100.0]))
    np.testing.assert_allclose(xyz, [[100.0, 0.0, 0.0]])


def test_galactic_to_xyz_north_pole() -> None:
    xyz = galactic_to_xyz(np.array([12.3]), np.array([90.0]), np.array([50.0]))
    np.testing.assert_allclose(xyz, [[0.0, 0.0, 50.0]], atol=1e-12)


def test_galactic_to_xyz_shape() -> None:
    n = 7
    xyz = galactic_to_xyz(np.zeros(n), np.linspace(-90, 90, n), np.full(n, 10.0))
    assert xyz.shape == (n, 3)


def test_galactic_to_xyz_distance_scaling() -> None:
    ra = np.array([45.0])
    dec = np.array([30.0])
    xyz1 = galactic_to_xyz(ra, dec, np.array([1.0]))
    xyz2 = galactic_to_xyz(ra, dec, np.array([100.0]))
    np.testing.assert_allclose(xyz2, xyz1 * 100.0)


# ---- neighborhood_av_features, basic behavior --------------------------------


def _make_linear_stars(n: int, spacing_pc: float, av_fn) -> dict[str, np.ndarray]:
    """Stars on a straight line along +x axis at (i*spacing, 0, 0)."""
    distances = np.arange(1, n + 1, dtype=float) * spacing_pc
    ra = np.zeros(n)  # +x axis is (RA, Dec) = (0, 0).
    dec = np.zeros(n)
    av = np.asarray([av_fn(i) for i in range(n)], dtype=float)
    return {"ra": ra, "dec": dec, "distance": distances, "av": av}


def test_neighborhood_constant_field_recovers_the_constant() -> None:
    """All stars have A_G = 0.5 → median should be 0.5, std 0."""
    n = 40
    data = _make_linear_stars(n, spacing_pc=5.0, av_fn=lambda _i: 0.5)

    feats = neighborhood_av_features(
        data["ra"],
        data["dec"],
        data["distance"],
        data["av"],
        radius_pc=30.0,
    )

    # Ends have fewer neighbours; pick a star well inside the line.
    mid = n // 2
    assert feats.n_neighbors[mid] >= MIN_NEIGHBORS_FOR_MEDIAN
    assert feats.av_nbhd_median[mid] == pytest.approx(0.5)
    assert feats.av_nbhd_std[mid] == pytest.approx(0.0, abs=1e-6)


def test_neighborhood_count_includes_only_points_inside_radius() -> None:
    # Spacing 10 pc, radius 25 pc → neighbours are at most ±2 steps, so
    # interior stars see 4 neighbours (excluding self).
    n = 20
    data = _make_linear_stars(n, spacing_pc=10.0, av_fn=lambda i: float(i))

    feats = neighborhood_av_features(
        data["ra"],
        data["dec"],
        data["distance"],
        data["av"],
        radius_pc=25.0,
        min_neighbors=1,
    )
    # Interior star at index 10: neighbours at 8,9,11,12 → count 4.
    assert feats.n_neighbors[10] == 4


def test_neighborhood_excludes_self_by_default() -> None:
    """Self exclusion: median over neighbours only, not incl. the star itself."""
    # 5 stars spaced 10 pc, A_G = [0, 1, 100, 1, 0]. Radius 50 pc → each
    # interior star sees 4 neighbours. The middle star (i=2) has av=100;
    # its neighbours are [0, 1, 1, 0] → median 0.5, std 0.5.
    ra = np.zeros(5)
    dec = np.zeros(5)
    distance = np.arange(1, 6) * 10.0
    av = np.array([0.0, 1.0, 100.0, 1.0, 0.0])

    feats = neighborhood_av_features(
        ra,
        dec,
        distance,
        av,
        radius_pc=50.0,
        min_neighbors=1,
    )
    assert feats.n_neighbors[2] == 4
    assert feats.av_nbhd_median[2] == pytest.approx(0.5)
    assert feats.av_nbhd_std[2] == pytest.approx(0.5)


def test_neighborhood_include_self_changes_result() -> None:
    ra = np.zeros(3)
    dec = np.zeros(3)
    distance = np.array([10.0, 20.0, 30.0])
    av = np.array([0.0, 10.0, 0.0])

    excl = neighborhood_av_features(
        ra,
        dec,
        distance,
        av,
        radius_pc=15.0,
        min_neighbors=1,
        include_self=False,
    )
    incl = neighborhood_av_features(
        ra,
        dec,
        distance,
        av,
        radius_pc=15.0,
        min_neighbors=1,
        include_self=True,
    )
    # Middle star: neighbours excl self = [0, 0] → median 0. Incl self = [0, 10, 0] → median 0.
    # Check count differs: excl=2, incl=3.
    assert excl.n_neighbors[1] == 2
    assert incl.n_neighbors[1] == 3


# ---- NaN / missing handling --------------------------------------------------


def test_neighborhood_below_min_neighbors_gives_nan() -> None:
    """A radius too small → fewer than min_neighbors → NaN median/std."""
    n = 10
    data = _make_linear_stars(n, spacing_pc=50.0, av_fn=lambda i: float(i))

    feats = neighborhood_av_features(
        data["ra"],
        data["dec"],
        data["distance"],
        data["av"],
        radius_pc=10.0,
        min_neighbors=5,  # radius < spacing, no neighbours at all
    )
    assert np.all(np.isnan(feats.av_nbhd_median))
    assert np.all(np.isnan(feats.av_nbhd_std))


def test_neighborhood_nan_distance_excluded_from_tree() -> None:
    """A star with NaN distance gets NaN outputs and is not a neighbour of others."""
    ra = np.array([0.0, 0.0, 0.0])
    dec = np.array([0.0, 0.0, 0.0])
    distance = np.array([100.0, np.nan, 200.0])
    av = np.array([0.5, 0.9, 0.5])

    feats = neighborhood_av_features(
        ra,
        dec,
        distance,
        av,
        radius_pc=150.0,
        min_neighbors=1,
    )
    assert np.isnan(feats.av_nbhd_median[1])
    assert feats.n_neighbors[1] == 0


def test_neighborhood_nan_av_still_gets_neighbourhood_stat() -> None:
    """A star with NaN A_G can still have a valid neighbourhood median."""
    ra = np.zeros(5)
    dec = np.zeros(5)
    distance = np.arange(1, 6) * 10.0
    av = np.array([0.5, 0.5, np.nan, 0.5, 0.5])

    feats = neighborhood_av_features(
        ra,
        dec,
        distance,
        av,
        radius_pc=50.0,
        min_neighbors=1,
    )
    assert feats.av_nbhd_median[2] == pytest.approx(0.5)
    assert feats.n_neighbors[2] == 4  # 4 finite neighbours


def test_neighborhood_nan_av_neighbours_are_dropped_from_median() -> None:
    """NaN A_G in a neighbour is excluded from the median computation."""
    ra = np.zeros(4)
    dec = np.zeros(4)
    distance = np.array([10.0, 20.0, 30.0, 40.0])
    av = np.array([1.0, np.nan, 3.0, 5.0])

    feats = neighborhood_av_features(
        ra,
        dec,
        distance,
        av,
        radius_pc=50.0,
        min_neighbors=1,
        include_self=False,
    )
    # Star at index 0: neighbours [1, 2, 3] → AVs [nan, 3, 5] → dropping nan → median 4.
    assert feats.n_neighbors[0] == 2
    assert feats.av_nbhd_median[0] == pytest.approx(4.0)


# ---- validation --------------------------------------------------------------


def test_neighborhood_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="mismatched lengths"):
        neighborhood_av_features(
            np.zeros(3),
            np.zeros(4),
            np.zeros(3),
            np.zeros(3),
        )


def test_neighborhood_rejects_nonpositive_radius() -> None:
    with pytest.raises(ValueError, match="radius_pc"):
        neighborhood_av_features(
            np.zeros(2),
            np.zeros(2),
            np.ones(2),
            np.zeros(2),
            radius_pc=0.0,
        )


def test_neighborhood_rejects_zero_min_neighbors() -> None:
    with pytest.raises(ValueError, match="min_neighbors"):
        neighborhood_av_features(
            np.zeros(2),
            np.zeros(2),
            np.ones(2),
            np.zeros(2),
            min_neighbors=0,
        )


# ---- dtype and shape contract ------------------------------------------------


def test_neighborhood_returns_float32_and_int32() -> None:
    n = 20
    data = _make_linear_stars(n, spacing_pc=5.0, av_fn=lambda i: 0.1 * i)
    feats = neighborhood_av_features(
        data["ra"],
        data["dec"],
        data["distance"],
        data["av"],
        radius_pc=30.0,
    )
    assert feats.av_nbhd_median.dtype == np.float32
    assert feats.av_nbhd_std.dtype == np.float32
    assert feats.n_neighbors.dtype == np.int32
    assert feats.av_nbhd_median.shape == (n,)


# ---- NeighborhoodAvFeatures.to_frame -----------------------------------------


def test_features_to_frame_without_source_ids() -> None:
    n = 5
    data = _make_linear_stars(n, spacing_pc=5.0, av_fn=lambda _i: 1.0)
    feats = neighborhood_av_features(
        data["ra"],
        data["dec"],
        data["distance"],
        data["av"],
        radius_pc=30.0,
    )
    df = feats.to_frame()
    assert list(df.columns) == ["av_nbhd_median", "av_nbhd_std", "n_neighbors"]
    assert len(df) == n


def test_features_to_frame_with_source_ids() -> None:
    n = 4
    sids = np.array([101, 102, 103, 104])
    feats = NeighborhoodAvFeatures(
        av_nbhd_median=np.zeros(n, dtype=np.float32),
        av_nbhd_std=np.zeros(n, dtype=np.float32),
        n_neighbors=np.zeros(n, dtype=np.int32),
    )
    df = feats.to_frame(source_ids=sids)
    assert list(df.columns[:1]) == ["source_id"]
    assert list(df["source_id"]) == [101, 102, 103, 104]


# ---- defaults ----------------------------------------------------------------


def test_default_radius_is_in_range_8_3() -> None:
    """§8.3 recommends 50–100 pc."""
    assert 50.0 <= DEFAULT_NEIGHBORHOOD_RADIUS_PC <= 100.0


def test_min_neighbors_default_is_sensible() -> None:
    assert MIN_NEIGHBORS_FOR_MEDIAN >= 3  # below 3, median is trivially noisy


# ---- empty input -------------------------------------------------------------


def test_neighborhood_empty_input() -> None:
    feats = neighborhood_av_features(
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
    )
    assert feats.av_nbhd_median.shape == (0,)
    assert feats.av_nbhd_std.shape == (0,)
    assert feats.n_neighbors.shape == (0,)


def test_neighborhood_all_nan_distances() -> None:
    n = 5
    feats = neighborhood_av_features(
        np.zeros(n),
        np.zeros(n),
        np.full(n, np.nan),
        np.zeros(n),
    )
    assert np.all(np.isnan(feats.av_nbhd_median))
    assert np.all(feats.n_neighbors == 0)


# ---- integration-style: realistic cluster of stars ---------------------------


def test_neighborhood_isolated_cluster() -> None:
    """Two clusters 500 pc apart — neighbourhoods should not cross."""
    rng = np.random.default_rng(42)
    # Cluster A centered at (100 pc, 0, 0), 30 stars within 20 pc, av ~ 0.3
    # Cluster B centered at (600 pc, 0, 0), 30 stars within 20 pc, av ~ 1.5
    n_each = 30
    offsets_a = rng.uniform(-20, 20, size=(n_each, 3))
    offsets_b = rng.uniform(-20, 20, size=(n_each, 3))
    xyz_a = offsets_a + np.array([100.0, 0.0, 0.0])
    xyz_b = offsets_b + np.array([600.0, 0.0, 0.0])
    xyz_all = np.vstack([xyz_a, xyz_b])
    av_all = np.concatenate([rng.normal(0.3, 0.01, n_each), rng.normal(1.5, 0.01, n_each)])

    ra = np.rad2deg(np.arctan2(xyz_all[:, 1], xyz_all[:, 0]))
    distance = np.linalg.norm(xyz_all, axis=1)
    dec = np.rad2deg(np.arcsin(xyz_all[:, 2] / distance))

    feats = neighborhood_av_features(
        ra,
        dec,
        distance,
        av_all,
        radius_pc=30.0,
    )
    # Cluster A median should hover near 0.3, cluster B near 1.5.
    a_median = np.nanmedian(feats.av_nbhd_median[:n_each])
    b_median = np.nanmedian(feats.av_nbhd_median[n_each:])
    assert a_median == pytest.approx(0.3, abs=0.05)
    assert b_median == pytest.approx(1.5, abs=0.05)
    # And they should be clearly separated (no cross-contamination).
    assert b_median - a_median > 1.0


# ---- pandas interop ----------------------------------------------------------


def test_neighborhood_accepts_pandas_series() -> None:
    n = 10
    df = pd.DataFrame(
        {
            "ra": np.zeros(n),
            "dec": np.zeros(n),
            "distance": np.arange(1, n + 1, dtype=float) * 10.0,
            "av": np.full(n, 0.5),
        }
    )
    feats = neighborhood_av_features(
        df["ra"],
        df["dec"],
        df["distance"],
        df["av"],
        radius_pc=50.0,
    )
    assert feats.av_nbhd_median.shape == (n,)


# =============================================================================
# §8.2 line-of-sight composition
# =============================================================================


def _constant_query(value: float):
    """Return a fake DustQuery that yields ``value`` for every star."""

    def q(coords) -> np.ndarray:  # noqa: ANN001
        n = len(coords)
        return np.full(n, value, dtype=float)

    return q


def test_compose_av_boundary_constants_match_section_8_2() -> None:
    assert NEAR_BOUNDARY_KPC == 1.25
    assert FAR_BOUNDARY_KPC == 3.0
    assert pytest.approx(2.742, rel=1e-3) == SFD_TO_AV_COEFF


def test_compose_av_routes_by_distance() -> None:
    """Each distance bin goes to the correct backend; source flag reflects it."""
    ra = np.array([10.0, 20.0, 30.0])
    dec = np.array([0.0, 0.0, 0.0])
    distance_pc = np.array([500.0, 2000.0, 5000.0])  # near, mid, far

    out = compose_av(
        ra,
        dec,
        distance_pc,
        near_query=_constant_query(0.3),
        mid_query=_constant_query(0.9),
        far_query=_constant_query(0.1),  # E(B−V)=0.1 → A_V=0.2742
    )
    assert list(out.source) == [0, 1, 2]
    assert out.av[0] == pytest.approx(0.3)
    assert out.av[1] == pytest.approx(0.9)
    assert out.av[2] == pytest.approx(0.1 * SFD_TO_AV_COEFF, rel=1e-5)


def test_compose_av_boundary_inclusivity() -> None:
    """Boundaries: distance == near_boundary → mid; distance == far_boundary → far."""
    ra = np.zeros(2)
    dec = np.zeros(2)
    # Exactly 1250 pc and 3000 pc.
    distance_pc = np.array([1250.0, 3000.0])

    out = compose_av(
        ra,
        dec,
        distance_pc,
        near_query=_constant_query(0.3),
        mid_query=_constant_query(0.9),
        far_query=_constant_query(0.2),
    )
    # 1.25 kpc falls into mid bin (near = d < near_boundary, strict inequality).
    assert out.source[0] == 1
    # 3.0 kpc falls into far bin.
    assert out.source[1] == 2


def test_compose_av_nan_distance_gets_source_minus_one() -> None:
    ra = np.array([0.0, 0.0])
    dec = np.array([0.0, 0.0])
    distance_pc = np.array([500.0, np.nan])
    out = compose_av(
        ra,
        dec,
        distance_pc,
        near_query=_constant_query(0.4),
        mid_query=_constant_query(0.4),
        far_query=_constant_query(0.4),
    )
    assert out.source[0] == 0
    assert out.source[1] == -1
    assert np.isnan(out.av[1])


def test_compose_av_negative_distance_excluded() -> None:
    ra = np.array([0.0])
    dec = np.array([0.0])
    distance_pc = np.array([-100.0])
    out = compose_av(
        ra,
        dec,
        distance_pc,
        near_query=_constant_query(0.4),
        mid_query=_constant_query(0.4),
        far_query=_constant_query(0.4),
    )
    assert out.source[0] == -1
    assert np.isnan(out.av[0])


def test_compose_av_all_far_uses_sfd_conversion() -> None:
    """A pure SFD-regime batch applies the E(B−V) → A_V factor."""
    n = 4
    ra = np.zeros(n)
    dec = np.zeros(n)
    distance_pc = np.full(n, 5000.0)  # all in far bin
    ebv = 0.25
    out = compose_av(
        ra,
        dec,
        distance_pc,
        near_query=_constant_query(99.0),  # should never be called
        mid_query=_constant_query(99.0),
        far_query=_constant_query(ebv),
    )
    assert np.allclose(out.av, ebv * SFD_TO_AV_COEFF)
    assert (out.source == 2).all()


def test_compose_av_query_receives_only_its_bin() -> None:
    """Each backend query must be called with just the stars in its bin."""
    seen_lengths: dict[str, list[int]] = {"near": [], "mid": [], "far": []}

    def track(name: str, value: float):
        def q(coords) -> np.ndarray:  # noqa: ANN001
            seen_lengths[name].append(len(coords))
            return np.full(len(coords), value, dtype=float)

        return q

    ra = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    dec = np.zeros(5)
    distance_pc = np.array([100.0, 500.0, 2000.0, 4000.0, 6000.0])  # 2 near, 1 mid, 2 far

    compose_av(
        ra,
        dec,
        distance_pc,
        near_query=track("near", 0.3),
        mid_query=track("mid", 0.9),
        far_query=track("far", 0.1),
    )
    assert seen_lengths["near"] == [2]
    assert seen_lengths["mid"] == [1]
    assert seen_lengths["far"] == [2]


def test_compose_av_empty_bin_skips_query() -> None:
    """If no stars fall in a bin, that bin's query is never invoked."""
    called: list[str] = []

    def mark(name: str):
        def q(coords) -> np.ndarray:  # noqa: ANN001
            called.append(name)
            return np.full(len(coords), 0.0, dtype=float)

        return q

    ra = np.array([0.0, 1.0])
    dec = np.zeros(2)
    distance_pc = np.array([100.0, 200.0])  # both near

    compose_av(
        ra,
        dec,
        distance_pc,
        near_query=mark("near"),
        mid_query=mark("mid"),
        far_query=mark("far"),
    )
    assert called == ["near"]


def test_compose_av_query_nan_yields_source_minus_one() -> None:
    """A backend that returns NaN for some stars → those stars get source=-1."""

    def partial_nan_query(coords) -> np.ndarray:  # noqa: ANN001
        n = len(coords)
        out = np.full(n, 0.5, dtype=float)
        out[::2] = np.nan
        return out

    ra = np.arange(4, dtype=float)
    dec = np.zeros(4)
    distance_pc = np.full(4, 500.0)  # all near
    out = compose_av(
        ra,
        dec,
        distance_pc,
        near_query=partial_nan_query,
        mid_query=_constant_query(0.9),
        far_query=_constant_query(0.1),
    )
    # Even stars: NaN result → source=-1. Odd stars: finite → source=0.
    assert list(out.source) == [-1, 0, -1, 0]
    assert np.isnan(out.av[0])
    assert out.av[1] == pytest.approx(0.5)


def test_compose_av_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="mismatched lengths"):
        compose_av(
            np.zeros(3),
            np.zeros(4),
            np.zeros(3),
            near_query=_constant_query(0.0),
            mid_query=_constant_query(0.0),
            far_query=_constant_query(0.0),
        )


def test_compose_av_rejects_bad_boundaries() -> None:
    with pytest.raises(ValueError, match="near_boundary"):
        compose_av(
            np.zeros(1),
            np.zeros(1),
            np.array([100.0]),
            near_query=_constant_query(0.0),
            mid_query=_constant_query(0.0),
            far_query=_constant_query(0.0),
            near_boundary_kpc=0.0,
            far_boundary_kpc=3.0,
        )
    with pytest.raises(ValueError, match="near_boundary"):
        compose_av(
            np.zeros(1),
            np.zeros(1),
            np.array([100.0]),
            near_query=_constant_query(0.0),
            mid_query=_constant_query(0.0),
            far_query=_constant_query(0.0),
            near_boundary_kpc=2.0,
            far_boundary_kpc=1.0,
        )


def test_compose_av_empty_input() -> None:
    out = compose_av(
        np.array([]),
        np.array([]),
        np.array([]),
        near_query=_constant_query(0.0),
        mid_query=_constant_query(0.0),
        far_query=_constant_query(0.0),
    )
    assert out.av.shape == (0,)
    assert out.source.shape == (0,)


def test_compose_av_returns_float32_av() -> None:
    out = compose_av(
        np.array([0.0]),
        np.array([0.0]),
        np.array([100.0]),
        near_query=_constant_query(0.5),
        mid_query=_constant_query(0.0),
        far_query=_constant_query(0.0),
    )
    assert out.av.dtype == np.float32
    assert out.source.dtype == np.int8


def test_composed_av_to_frame() -> None:
    feats = ComposedAv(
        av=np.array([0.3, 0.9, 0.2], dtype=np.float32),
        source=np.array([0, 1, 2], dtype=np.int8),
    )
    df = feats.to_frame(source_ids=np.array([111, 222, 333]))
    assert list(df.columns) == ["source_id", "av_los", "av_los_source"]
    assert list(df["source_id"]) == [111, 222, 333]
    assert list(df["av_los_source"]) == [0, 1, 2]


def test_composed_av_to_frame_without_source_ids() -> None:
    feats = ComposedAv(
        av=np.array([0.3], dtype=np.float32),
        source=np.array([0], dtype=np.int8),
    )
    df = feats.to_frame()
    assert list(df.columns) == ["av_los", "av_los_source"]
