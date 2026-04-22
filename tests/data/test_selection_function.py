"""Offline tests for arqueogal.data.selection_function.

Covers:

- v1 (Ye+2024 ``NO_SYNTH_PHOT`` retention): artefact load, shape, range,
  monotonicity, input clamping, sign-symmetry.
- v1.1 (IR-completeness + compound): artefact load, range, monotonicity
  where physically expected (brighter → higher, higher |b| → higher),
  NaN Teff/log g fallback to the |b|×G marginal, compound = product of
  components, and 0/1 gate propagation (parallax / extinction).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arqueogal.data.selection_function import (
    COMPOUND_PROB_FLOOR,
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_IR_ARTIFACT_PATH,
    SELECTION_PROB_CEIL,
    SELECTION_PROB_FLOOR,
    _load_grid,
    score_compound_selection_prob,
    score_ir_completeness,
    score_selection_prob,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_ARTIFACT_PATH.exists(),
    reason=(
        f"v1 artefact not built yet at {DEFAULT_ARTIFACT_PATH}. "
        "Run: PYTHONPATH=src python scripts/build_selection_function_v1.py"
    ),
)


# ---- (a) artefact loads ------------------------------------------------------


def test_artifact_loads_and_has_required_columns() -> None:
    df = pd.read_parquet(DEFAULT_ARTIFACT_PATH)
    assert len(df) > 0
    for col in ("b_lo", "b_hi", "g_lo", "g_hi", "n_total", "n_flagged",
                "flag_rate", "selection_prob"):
        assert col in df.columns, f"missing column: {col}"


def test_load_grid_returns_consistent_shape() -> None:
    b_edges, g_edges, prob = _load_grid(str(DEFAULT_ARTIFACT_PATH))
    assert b_edges[0] == 0.0
    assert b_edges[-1] == 90.0
    assert prob.shape == (len(b_edges) - 1, len(g_edges) - 1)
    assert np.isfinite(prob).all()


# ---- (b) shape -----------------------------------------------------------


def test_score_shape_matches_input_1d() -> None:
    b = np.linspace(-80.0, 80.0, 13)
    g = np.linspace(5.0, 17.0, 13)
    out = score_selection_prob(b, g)
    assert out.shape == b.shape


def test_score_shape_matches_input_scalar_like() -> None:
    out = score_selection_prob(np.array([3.0]), np.array([16.0]))
    assert out.shape == (1,)


def test_score_raises_on_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        score_selection_prob(np.zeros(4), np.zeros(5))


def test_score_accepts_pandas_series() -> None:
    b = pd.Series([3.0, -3.0, 40.0])
    g = pd.Series([16.0, 16.0, 10.0])
    out = score_selection_prob(b, g)
    assert out.shape == (3,)


# ---- (c) range ----------------------------------------------------------


def test_scores_are_in_allowed_range() -> None:
    # Cover the full (b, G) support plus out-of-range inputs to exercise clamping.
    rng = np.random.default_rng(seed=42)
    b = rng.uniform(-95.0, 95.0, size=500)
    g = rng.uniform(1.0, 18.5, size=500)
    out = score_selection_prob(b, g)
    assert np.all(out >= SELECTION_PROB_FLOOR - 1e-12)
    assert np.all(out <= SELECTION_PROB_CEIL + 1e-12)
    assert np.isfinite(out).all()


def test_all_grid_cells_respect_floor() -> None:
    df = pd.read_parquet(DEFAULT_ARTIFACT_PATH)
    assert (df["selection_prob"] >= SELECTION_PROB_FLOOR).all()
    assert (df["selection_prob"] <= SELECTION_PROB_CEIL).all()


# ---- (d) monotonicity smoke check --------------------------------------


def test_plane_faint_scores_lower_than_cap_bright() -> None:
    # Plane + faint corner vs cap + bright corner: must differ in the right
    # direction and by a lot (Thread-1 established ~134x rate ratio globally).
    b_plane = np.array([2.0])   # |b| ~ 2 deg
    g_faint = np.array([17.0])
    b_cap = np.array([60.0])
    g_bright = np.array([10.0])
    p_plane_faint = score_selection_prob(b_plane, g_faint)[0]
    p_cap_bright = score_selection_prob(b_cap, g_bright)[0]
    assert p_cap_bright > p_plane_faint
    assert p_cap_bright - p_plane_faint > 0.1  # generous margin; Thread-1 gave ~0.4


def test_sign_of_b_does_not_matter() -> None:
    b_pos = np.array([3.0, 25.0, -60.0, 60.0])
    b_neg = -b_pos
    g = np.full_like(b_pos, 16.0)
    np.testing.assert_array_equal(
        score_selection_prob(b_pos, g), score_selection_prob(b_neg, g)
    )


def test_out_of_range_inputs_are_clamped() -> None:
    # |b| = 100 deg and G = 20 mag both outside the grid — must still return a
    # valid probability equal to the nearest edge cell's value.
    out_of_range_b = score_selection_prob(np.array([100.0]), np.array([16.0]))
    edge_b = score_selection_prob(np.array([89.999]), np.array([16.0]))
    np.testing.assert_allclose(out_of_range_b, edge_b)

    out_of_range_g = score_selection_prob(np.array([3.0]), np.array([20.0]))
    edge_g = score_selection_prob(np.array([3.0]), np.array([17.64]))
    np.testing.assert_allclose(out_of_range_g, edge_g)


# ---- misc: default path resolves under repo ----------------------------


def test_default_artifact_path_is_repo_relative() -> None:
    assert DEFAULT_ARTIFACT_PATH.is_absolute()
    assert DEFAULT_ARTIFACT_PATH.name == "selection_function_v1.parquet"
    # four levels up from src/arqueogal/data/selection_function.py is the repo.
    repo_root = Path(__file__).resolve().parents[2]
    assert DEFAULT_ARTIFACT_PATH.is_relative_to(repo_root)


# ---------------------------------------------------------------------------
# v1.1 IR-completeness and compound scorers
# ---------------------------------------------------------------------------


ir_available = pytest.mark.skipif(
    not DEFAULT_IR_ARTIFACT_PATH.exists(),
    reason=(
        f"IR-completeness artefact not built yet at {DEFAULT_IR_ARTIFACT_PATH}. "
        "Run: PYTHONPATH=src python scripts/build_selection_function_v11.py"
    ),
)


@ir_available
def test_ir_artifact_has_required_schema() -> None:
    df = pd.read_parquet(DEFAULT_IR_ARTIFACT_PATH)
    assert len(df) > 0
    for col in ("b_lo", "b_hi", "g_lo", "g_hi", "p_ir_complete", "grid"):
        assert col in df.columns, f"missing column: {col}"
    # Both marginal and 4-D views must exist.
    assert (df["grid"] == "bg").any()
    assert (df["grid"] == "4d").any()


@ir_available
def test_ir_scores_are_in_allowed_range() -> None:
    rng = np.random.default_rng(seed=123)
    n = 500
    b = rng.uniform(-95.0, 95.0, size=n)
    g = rng.uniform(1.0, 18.5, size=n)
    t = rng.uniform(3500.0, 6000.0, size=n)
    lg = rng.uniform(0.5, 4.5, size=n)
    out = score_ir_completeness(b, g, t, lg)
    assert out.shape == (n,)
    assert np.all(out >= SELECTION_PROB_FLOOR - 1e-12)
    assert np.all(out <= SELECTION_PROB_CEIL + 1e-12)
    assert np.isfinite(out).all()


@ir_available
def test_ir_monotonicity_bright_beats_faint() -> None:
    # At the same |b|, cool giants, a bright G should score higher than faint.
    b = np.array([30.0, 30.0])
    g = np.array([11.5, 17.0])
    t = np.array([4500.0, 4500.0])
    lg = np.array([2.3, 2.3])
    p_bright, p_faint = score_ir_completeness(b, g, t, lg)
    assert p_bright >= p_faint
    # At the plane-faint vs off-plane-faint, off-plane should score higher.
    b2 = np.array([3.0, 40.0])
    g2 = np.array([16.0, 16.0])
    t2 = np.array([4500.0, 4500.0])
    lg2 = np.array([2.3, 2.3])
    p_plane, p_cap = score_ir_completeness(b2, g2, t2, lg2)
    assert p_cap >= p_plane


@ir_available
def test_ir_nan_teff_logg_falls_back_to_bg_marginal() -> None:
    b = np.array([3.0, 3.0, 3.0])
    g = np.array([16.0, 16.0, 16.0])
    # Finite Teff+logg yields a 4-D or bg-fallback lookup; NaN in either
    # yields the |b|×G marginal. The two NaN cases must produce the same
    # probability (the marginal), and that probability must equal what we
    # read directly from the bg rows for that cell.
    t = np.array([4500.0, np.nan, 4500.0])
    lg = np.array([2.3, 2.3, np.nan])
    p = score_ir_completeness(b, g, t, lg)
    assert p[1] == p[2], "NaN-Teff and NaN-logg paths must agree on the bg marginal"

    df = pd.read_parquet(DEFAULT_IR_ARTIFACT_PATH)
    bg = df[df["grid"] == "bg"]
    cell = bg[(bg["b_lo"] <= 3.0) & (bg["b_hi"] > 3.0)
              & (bg["g_lo"] <= 16.0) & (bg["g_hi"] > 16.0)]
    assert len(cell) == 1
    p_bg = float(cell["p_ir_complete"].iloc[0])
    np.testing.assert_allclose(p[1], np.clip(p_bg, SELECTION_PROB_FLOOR, SELECTION_PROB_CEIL))


@ir_available
def test_ir_full_nan_teff_logg_all_fall_back() -> None:
    n = 20
    rng = np.random.default_rng(seed=7)
    b = rng.uniform(-80.0, 80.0, size=n)
    g = rng.uniform(5.0, 17.0, size=n)
    t_finite = rng.uniform(4000.0, 5500.0, size=n)
    lg_finite = rng.uniform(1.0, 3.5, size=n)
    # All-finite 4-D path vs all-NaN fallback path: both must lie in range
    # and the fallback path equals the |b|×G marginal of the artefact.
    out_finite = score_ir_completeness(b, g, t_finite, lg_finite)
    out_nan = score_ir_completeness(b, g, np.full(n, np.nan), np.full(n, np.nan))
    assert np.all(out_finite >= SELECTION_PROB_FLOOR - 1e-12)
    assert np.all(out_finite <= SELECTION_PROB_CEIL + 1e-12)
    assert np.all(out_nan >= SELECTION_PROB_FLOOR - 1e-12)
    assert np.all(out_nan <= SELECTION_PROB_CEIL + 1e-12)
    # Cross-check: NaN path = direct lookup of the |b|×G marginal cell.
    df = pd.read_parquet(DEFAULT_IR_ARTIFACT_PATH)
    bg = df[df["grid"] == "bg"].reset_index(drop=True)
    abs_b = np.abs(b)
    for i in range(n):
        cell = bg[(bg["b_lo"] <= abs_b[i]) & (bg["b_hi"] > abs_b[i])
                  & (bg["g_lo"] <= g[i]) & (bg["g_hi"] > g[i])]
        if len(cell) != 1:
            # Edge row (rare with random uniforms); ignore.
            continue
        expected = float(np.clip(cell["p_ir_complete"].iloc[0],
                                 SELECTION_PROB_FLOOR, SELECTION_PROB_CEIL))
        np.testing.assert_allclose(out_nan[i], expected)


@ir_available
def test_ir_sign_of_b_does_not_matter() -> None:
    b_pos = np.array([3.0, 25.0, 60.0])
    b_neg = -b_pos
    g = np.full_like(b_pos, 16.0)
    t = np.full_like(b_pos, 4500.0)
    lg = np.full_like(b_pos, 2.3)
    np.testing.assert_array_equal(
        score_ir_completeness(b_pos, g, t, lg),
        score_ir_completeness(b_neg, g, t, lg),
    )


# ---- compound scorer ----------------------------------------------------


@ir_available
def test_compound_product_of_components() -> None:
    b, g, t, lg = 3.0, 16.0, 4500.0, 2.3
    res = score_compound_selection_prob(
        b, g, t, lg, parallax_over_error=12.0, av_missing=False,
    )
    # With parallax available and extinction present, compound = p_ye * p_ir.
    expected = res["p_ye_retained"] * res["p_ir_complete"]
    np.testing.assert_allclose(res["p_compound"], np.clip(expected, COMPOUND_PROB_FLOOR, 1.0))
    # Components dict agrees with the product.
    c = res["components"]
    np.testing.assert_allclose(
        res["p_compound"],
        np.clip(c["ye"] * c["ir"] * c["parallax"] * c["extinction"],
                COMPOUND_PROB_FLOOR, 1.0),
    )


@ir_available
def test_compound_av_missing_zeroes_output() -> None:
    res = score_compound_selection_prob(
        30.0, 11.0, 4800.0, 2.5, parallax_over_error=20.0, av_missing=True,
    )
    assert res["p_compound"] == 0.0
    assert res["components"]["extinction"] == 0.0
    # Other components remain informative.
    assert res["p_ye_retained"] > 0.0
    assert res["p_ir_complete"] > 0.0
    assert res["components"]["parallax"] == 1.0


@ir_available
def test_compound_parallax_missing_zeroes_output() -> None:
    res = score_compound_selection_prob(
        30.0, 11.0, 4800.0, 2.5, parallax_over_error=None, av_missing=False,
    )
    assert res["p_compound"] == 0.0
    assert res["components"]["parallax"] == 0.0
    assert res["components"]["extinction"] == 1.0


@ir_available
def test_compound_bounds_are_respected() -> None:
    rng = np.random.default_rng(seed=99)
    for _ in range(50):
        b = float(rng.uniform(-90.0, 90.0))
        g = float(rng.uniform(2.5, 17.5))
        t = float(rng.uniform(4000.0, 5500.0))
        lg = float(rng.uniform(1.0, 3.5))
        res = score_compound_selection_prob(
            b, g, t, lg, parallax_over_error=True, av_missing=False,
        )
        assert 0.0 <= res["p_compound"] <= 1.0
        # With both gates on, the positive floor applies.
        assert res["p_compound"] >= COMPOUND_PROB_FLOOR - 1e-12


@ir_available
def test_compound_nan_teff_logg_still_bounded() -> None:
    # NaN Teff/log g must not blow up the compound; they just route IR to
    # the |b|×G marginal fallback.
    res = score_compound_selection_prob(
        3.0, 16.0, float("nan"), float("nan"),
        parallax_over_error=10.0, av_missing=False,
    )
    assert 0.0 <= res["p_compound"] <= 1.0
    assert np.isfinite(res["p_ir_complete"])
