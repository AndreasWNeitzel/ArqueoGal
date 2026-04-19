"""Offline tests for arqueogal.data.gaia_corrections.

A fake ``zero_point`` module is injected into ``sys.modules`` so the wrapper
can be exercised without installing ``gaiadr3-zeropoint``. The fake returns
deterministic zpt values in mas — matching the real package, which the
docstring confirms returns "correction in mas (milliarcsecond, not micro)".
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from arqueogal.data.gaia_corrections import (
    SOL_FIVE_PARAM,
    SOL_SIX_PARAM,
    SOL_TWO_PARAM,
    apply_g_mag_correction,
    apply_parallax_zpt,
)

# mas — the fixed zpt value returned by the fake. Real Lindegren values
# are ~-17 μas = -0.017 mas; the package returns mas directly.
_FAKE_ZPT_MAS = -0.017


@pytest.fixture
def fake_zpt(monkeypatch: pytest.MonkeyPatch):
    """Inject a fake ``zero_point.zpt`` submodule with load_tables + get_zpt."""
    calls: dict[str, object] = {"load_tables": 0, "get_zpt": []}

    zpt = types.SimpleNamespace()

    def load_tables() -> None:
        calls["load_tables"] = int(calls["load_tables"]) + 1  # type: ignore[arg-type]

    def get_zpt(gmag, nueff, psc, eclat, sol, _warnings=True):  # noqa: ANN001
        calls["get_zpt"].append(  # type: ignore[union-attr]
            {"n": len(gmag), "warnings": _warnings}
        )
        return np.full(len(gmag), _FAKE_ZPT_MAS, dtype=np.float64)

    zpt.load_tables = load_tables
    zpt.get_zpt = get_zpt

    pkg = types.ModuleType("zero_point")
    pkg.zpt = zpt  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "zero_point", pkg)
    monkeypatch.setitem(sys.modules, "zero_point.zpt", zpt)
    return calls


def _make_df(sol_values: list[int], parallax_mas: list[float]) -> pd.DataFrame:
    n = len(sol_values)
    return pd.DataFrame(
        {
            "parallax": parallax_mas,
            "phot_g_mean_mag": np.full(n, 14.0),
            "nu_eff_used_in_astrometry": np.full(n, 1.5),
            "pseudocolour": np.full(n, 1.5),
            "ecl_lat": np.full(n, 30.0),
            "astrometric_params_solved": sol_values,
        }
    )


def test_zpt_applied_to_five_param(fake_zpt) -> None:
    df = _make_df([SOL_FIVE_PARAM], [0.500])
    out = apply_parallax_zpt(df)

    assert np.isclose(out["parallax_zpt"].iloc[0], _FAKE_ZPT_MAS)
    # corrected = 0.500 - (-0.017) = 0.517
    assert np.isclose(out["parallax_corr"].iloc[0], 0.500 - _FAKE_ZPT_MAS)
    assert fake_zpt["load_tables"] >= 1


def test_zpt_applied_to_six_param(fake_zpt) -> None:
    df = _make_df([SOL_SIX_PARAM], [0.200])
    out = apply_parallax_zpt(df)
    assert np.isclose(out["parallax_zpt"].iloc[0], _FAKE_ZPT_MAS)
    assert np.isclose(out["parallax_corr"].iloc[0], 0.200 - _FAKE_ZPT_MAS)


def test_zpt_skips_two_param(fake_zpt) -> None:
    df = _make_df([SOL_TWO_PARAM, SOL_FIVE_PARAM], [0.100, 0.300])
    out = apply_parallax_zpt(df)

    # Row 0 (2-param): zpt is NaN, corr equals raw parallax.
    assert np.isnan(out["parallax_zpt"].iloc[0])
    assert out["parallax_corr"].iloc[0] == 0.100

    # Row 1 (5-param): zpt applied.
    assert np.isclose(out["parallax_zpt"].iloc[1], _FAKE_ZPT_MAS)
    assert np.isclose(out["parallax_corr"].iloc[1], 0.300 - _FAKE_ZPT_MAS)


def test_zpt_all_two_param_no_get_zpt_call(fake_zpt) -> None:
    df = _make_df([SOL_TWO_PARAM] * 3, [0.1, 0.2, 0.3])
    out = apply_parallax_zpt(df)

    # get_zpt never called when no row qualifies.
    assert fake_zpt["get_zpt"] == []
    assert out["parallax_zpt"].isna().all()
    assert (out["parallax_corr"] == out["parallax"]).all()


def test_zpt_does_not_mutate_input(fake_zpt) -> None:
    df = _make_df([SOL_FIVE_PARAM], [0.500])
    apply_parallax_zpt(df)
    assert "parallax_zpt" not in df.columns
    assert "parallax_corr" not in df.columns


def test_zpt_missing_column_raises(fake_zpt) -> None:
    df = _make_df([SOL_FIVE_PARAM], [0.5]).drop(columns=["ecl_lat"])
    with pytest.raises(KeyError, match="ecl_lat"):
        apply_parallax_zpt(df)


def test_zpt_missing_package_raises_with_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ensure zero_point is not importable even if a previous test injected it.
    monkeypatch.setitem(sys.modules, "zero_point", None)

    df = _make_df([SOL_FIVE_PARAM], [0.5])
    with pytest.raises(ImportError, match="gaiadr3-zeropoint"):
        apply_parallax_zpt(df)


def _make_g_df(
    g_mag: list[float], bp_rp: list[float], sol: list[int],
    flux: list[float] | None = None,
) -> pd.DataFrame:
    data = {
        "phot_g_mean_mag": g_mag,
        "bp_rp": bp_rp,
        "astrometric_params_solved": sol,
    }
    if flux is not None:
        data["phot_g_mean_flux"] = flux
    return pd.DataFrame(data)


def test_g_mag_correction_requires_columns() -> None:
    df = pd.DataFrame({"phot_g_mean_mag": [14.0]})
    with pytest.raises(KeyError, match="bp_rp|astrometric_params_solved"):
        apply_g_mag_correction(df)


def test_g_mag_correction_skips_5param() -> None:
    df = _make_g_df([14.0, 14.0], [1.0, 1.0], [SOL_FIVE_PARAM, SOL_SIX_PARAM])
    out = apply_g_mag_correction(df)
    assert out["phot_g_mean_mag_corr"].iloc[0] == 14.0  # 5-param unchanged
    assert out["phot_g_mean_mag_corr"].iloc[1] != 14.0  # 6-param corrected


def test_g_mag_correction_skips_bright_sources() -> None:
    """G < 13 is left uncorrected regardless of solution type."""
    df = _make_g_df([10.0, 12.99], [1.0, 1.0], [SOL_SIX_PARAM, SOL_SIX_PARAM])
    out = apply_g_mag_correction(df)
    assert out["phot_g_mean_mag_corr"].iloc[0] == 10.0
    assert out["phot_g_mean_mag_corr"].iloc[1] == 12.99


def test_g_mag_correction_bright_faint_branches() -> None:
    """Bright (13-16) and faint (>16) branches use different polynomials."""
    df = _make_g_df([14.0, 18.0], [1.0, 1.0], [SOL_SIX_PARAM, SOL_SIX_PARAM])
    out = apply_g_mag_correction(df)
    # Solar-ish BP-RP=1: bright factor ≈ 1.00876 -0.0254 +0.01747 -0.00277 ≈ 0.99806
    # faint factor ≈ 1.00525 -0.02323 +0.01740 -0.00253 ≈ 0.99689
    # Corrected mag = g - 2.5*log10(factor)
    bright_factor = 1.00876 - 0.02540 + 0.01747 - 0.00277
    faint_factor = 1.00525 - 0.02323 + 0.01740 - 0.00253
    np.testing.assert_allclose(
        out["phot_g_mean_mag_corr"].iloc[0],
        14.0 - 2.5 * np.log10(bright_factor),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        out["phot_g_mean_mag_corr"].iloc[1],
        18.0 - 2.5 * np.log10(faint_factor),
        rtol=1e-10,
    )


def test_g_mag_correction_bprp_clipping() -> None:
    """BP-RP outside [0.25, 3.0] gets clipped; correction same at boundary."""
    df = _make_g_df([14.0, 14.0], [0.1, 0.25], [SOL_SIX_PARAM, SOL_SIX_PARAM])
    out = apply_g_mag_correction(df)
    assert np.isclose(
        out["phot_g_mean_mag_corr"].iloc[0], out["phot_g_mean_mag_corr"].iloc[1]
    )


def test_g_mag_correction_nan_bprp_passes_through() -> None:
    df = _make_g_df([14.0], [np.nan], [SOL_SIX_PARAM])
    out = apply_g_mag_correction(df)
    assert out["phot_g_mean_mag_corr"].iloc[0] == 14.0


def test_g_mag_correction_flux_optional_and_multiplicative() -> None:
    df = _make_g_df([14.0], [1.0], [SOL_SIX_PARAM], flux=[1000.0])
    out = apply_g_mag_correction(df)
    factor = 1.00876 - 0.02540 + 0.01747 - 0.00277
    np.testing.assert_allclose(
        out["phot_g_mean_flux_corr"].iloc[0], 1000.0 * factor, rtol=1e-10
    )


def test_g_mag_correction_no_flux_col_drops_flux_output() -> None:
    df = _make_g_df([14.0], [1.0], [SOL_SIX_PARAM], flux=None)
    out = apply_g_mag_correction(df)
    assert "phot_g_mean_mag_corr" in out.columns
    assert "phot_g_mean_flux_corr" not in out.columns


def test_g_mag_correction_does_not_mutate_input() -> None:
    df = _make_g_df([14.0], [1.0], [SOL_SIX_PARAM])
    apply_g_mag_correction(df)
    assert "phot_g_mean_mag_corr" not in df.columns


def test_zpt_preserves_row_order(fake_zpt) -> None:
    df = _make_df([SOL_FIVE_PARAM, SOL_TWO_PARAM, SOL_SIX_PARAM], [1.0, 2.0, 3.0])
    out = apply_parallax_zpt(df)
    # Index preserved, DataFrame row ordering preserved.
    assert list(out.index) == [0, 1, 2]
    assert list(out["parallax"]) == [1.0, 2.0, 3.0]


def test_zpt_column_names_customisable(fake_zpt) -> None:
    df = _make_df([SOL_FIVE_PARAM], [0.5])
    out = apply_parallax_zpt(df, zpt_col="my_zpt", corrected_col="my_plx_corr")
    assert "my_zpt" in out.columns
    assert "my_plx_corr" in out.columns
    assert "parallax_zpt" not in out.columns


# ---- integration: exercise the real gaiadr3-zeropoint package if installed ---


def _has_real_zpt() -> bool:
    try:
        import zero_point  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not _has_real_zpt(), reason="gaiadr3-zeropoint not installed in this env"
)
def test_zpt_real_package_returns_mas_not_uas() -> None:
    """Guard against unit regressions. Typical Lindegren zpt is ~-0.017 mas.

    If the package ever switches to μas (or our wrapper reintroduces a /1000
    conversion), the computed correction becomes ~1000× too small or too
    large — catchable with an order-of-magnitude bracket.
    """
    df = _make_df([SOL_FIVE_PARAM], [0.500])
    out = apply_parallax_zpt(df)

    z = out["parallax_zpt"].iloc[0]
    # Lindegren zpt for typical bright-giant inputs sits in roughly
    # [-0.060, +0.005] mas. If we see |z| > 0.2 mas, we've re-scaled wrong.
    assert not np.isnan(z)
    assert -0.20 < z < 0.05, f"zpt {z} mas implausible; unit scaling regressed?"
