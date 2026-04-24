"""Offline tests for arqueogal.data.apogee_dr19.

Build a tiny in-memory FITS table that mimics the DR19 ASPCAP HDU 2 layout,
write it to a tmp path, and exercise the loader + cuts + derivations against
it. No network, no astropy TAP.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from arqueogal.data.apogee_dr19 import (
    QualityCuts,
    apply_meszaros2025_corrections,
    apply_quality_cuts,
    derive_c_n,
    kept_columns,
    load_dr19,
)


def _build_dr19_fits(path: Path, n_rows: int = 8) -> None:
    """Minimal DR19-like table covering every column load_dr19 requests."""
    rng = np.random.default_rng(42)
    n = n_rows

    cols: list[fits.Column] = []

    # Identifiers
    cols.append(fits.Column(name="sdss_id", format="K", array=np.arange(n, dtype=np.int64)))
    cols.append(
        fits.Column(
            name="apogee_id",
            format="30A",
            array=np.array([f"2M{i:08d}" for i in range(n)]),
        )
    )
    cols.append(
        fits.Column(
            name="source_id",
            format="K",
            array=np.arange(100_000, 100_000 + n, dtype=np.int64),
        )
    )

    # Position + Gaia DR3 astrometry (DR19 bakes these into HDU 2).
    cols.append(
        fits.Column(name="ra", format="E", array=np.linspace(0.0, 359.0, n, dtype=np.float32))
    )
    cols.append(
        fits.Column(name="dec", format="E", array=np.linspace(-45.0, 45.0, n, dtype=np.float32))
    )
    cols.append(
        fits.Column(name="plx", format="E", array=np.linspace(0.2, 2.0, n, dtype=np.float32))
    )
    cols.append(fits.Column(name="e_plx", format="E", array=np.full(n, 0.02, dtype=np.float32)))
    cols.append(fits.Column(name="pmra", format="E", array=np.full(n, -1.0, dtype=np.float32)))
    cols.append(fits.Column(name="e_pmra", format="E", array=np.full(n, 0.03, dtype=np.float32)))
    cols.append(fits.Column(name="pmde", format="E", array=np.full(n, 2.0, dtype=np.float32)))
    cols.append(fits.Column(name="e_pmde", format="E", array=np.full(n, 0.03, dtype=np.float32)))

    # Photometry: Gaia + 2MASS + WISE (all 1:1 names with DR19).
    for c, val in (
        ("g_mag", 12.0),
        ("bp_mag", 12.5),
        ("rp_mag", 11.5),
        ("j_mag", 10.5),
        ("e_j_mag", 0.02),
        ("h_mag", 10.0),
        ("e_h_mag", 0.02),
        ("k_mag", 9.8),
        ("e_k_mag", 0.02),
        ("w1_mag", 9.7),
        ("e_w1_mag", 0.02),
        ("w2_mag", 9.7),
        ("e_w2_mag", 0.02),
    ):
        cols.append(fits.Column(name=c, format="E", array=np.full(n, val, dtype=np.float32)))

    # Per-star dust (baked into DR19 HDU 2 — 3D + 2D, no external fetch).
    for c, val in (
        ("ebv", 0.12),
        ("e_ebv", 0.02),
        ("ebv_edenhofer_2023", 0.1),
        ("e_ebv_edenhofer_2023", 0.01),
        ("ebv_bayestar_2019", 0.11),
        ("e_ebv_bayestar_2019", 0.015),
        ("ebv_zhang_2023", 0.09),
        ("e_ebv_zhang_2023", 0.02),
        ("ebv_sfd", 0.15),
        ("e_ebv_sfd", 0.02),
    ):
        cols.append(fits.Column(name=c, format="E", array=np.full(n, val, dtype=np.float32)))

    # Bailer-Jones+2021 distances (pre-joined in DR19).
    for c, val in (
        ("r_med_geo", 2000.0),
        ("r_lo_geo", 1900.0),
        ("r_hi_geo", 2100.0),
        ("r_med_photogeo", 2050.0),
        ("r_lo_photogeo", 1950.0),
        ("r_hi_photogeo", 2150.0),
    ):
        cols.append(fits.Column(name=c, format="E", array=np.full(n, val, dtype=np.float32)))

    # Atmos
    teff = np.linspace(3500, 6000, n)  # crosses both Teff bounds
    logg = np.linspace(0.5, 4.0, n)  # crosses both logg bounds
    m_h = np.linspace(-2.5, 1.0, n)  # crosses both [M/H] bounds
    cols.append(fits.Column(name="teff", format="E", array=teff.astype(np.float32)))
    cols.append(fits.Column(name="e_teff", format="E", array=np.full(n, 50.0, dtype=np.float32)))
    cols.append(fits.Column(name="logg", format="E", array=logg.astype(np.float32)))
    cols.append(fits.Column(name="e_logg", format="E", array=np.full(n, 0.05, dtype=np.float32)))
    cols.append(fits.Column(name="m_h_atm", format="E", array=m_h.astype(np.float32)))
    cols.append(fits.Column(name="e_m_h_atm", format="E", array=np.full(n, 0.02, dtype=np.float32)))
    alpha = rng.normal(0.1, 0.1, n).astype(np.float32)
    cols.append(fits.Column(name="alpha_m_atm", format="E", array=alpha))
    cols.append(
        fits.Column(name="e_alpha_m_atm", format="E", array=np.full(n, 0.02, dtype=np.float32))
    )
    for c in ("vsini", "vmicro"):
        cols.append(fits.Column(name=c, format="E", array=np.full(n, 2.0, dtype=np.float32)))

    # Abundances
    elements = (
        "c",
        "n",
        "o",
        "na",
        "mg",
        "al",
        "si",
        "s",
        "k",
        "ca",
        "ti",
        "v",
        "cr",
        "mn",
        "fe",
        "ni",
        "ce",
    )
    for el in elements:
        cols.append(
            fits.Column(
                name=f"{el}_h_atm",
                format="E",
                array=rng.normal(0.0, 0.1, n).astype(np.float32),
            )
        )
        cols.append(
            fits.Column(name=f"e_{el}_h_atm", format="E", array=np.full(n, 0.03, dtype=np.float32))
        )

    # C/N derivation inputs
    c_fe = rng.normal(0.0, 0.1, n).astype(np.float32)
    n_fe = rng.normal(0.2, 0.1, n).astype(np.float32)
    cols.append(fits.Column(name="c_fe", format="E", array=c_fe))
    cols.append(fits.Column(name="e_c_fe", format="E", array=np.full(n, 0.03, dtype=np.float32)))
    cols.append(fits.Column(name="n_fe", format="E", array=n_fe))
    cols.append(fits.Column(name="e_n_fe", format="E", array=np.full(n, 0.04, dtype=np.float32)))

    # Flags — mix passing + failing
    flag_bad = np.zeros(n, dtype=np.int32)
    flag_bad[-1] = 1  # last row fails flag_bad
    cols.append(fits.Column(name="flag_bad", format="J", array=flag_bad))

    snr = np.full(n, 120.0, dtype=np.float32)
    snr[0] = 30.0  # first row fails SNR
    cols.append(fits.Column(name="snr", format="E", array=snr))

    for c in (
        "flag_warn",
        "result_flags",
        "initial_flags",
        "calibrated_flags",
    ):
        cols.append(fits.Column(name=c, format="J", array=np.zeros(n, dtype=np.int32)))
    vhel = np.full(n, -10.0, dtype=np.float32)
    cols.append(fits.Column(name="vhelio_avg", format="E", array=vhel))
    cols.append(fits.Column(name="vhelio_err", format="E", array=np.full(n, 0.5, dtype=np.float32)))

    # Meta
    cols.append(fits.Column(name="v_astra", format="10A", array=np.array(["0.6.0"] * n)))
    cols.append(fits.Column(name="task_id", format="K", array=np.arange(n, dtype=np.int64)))
    cols.append(fits.Column(name="spectrum_pk", format="K", array=np.arange(n, dtype=np.int64)))

    primary = fits.PrimaryHDU()
    # HDU 1 is an empty placeholder; HDU 2 holds the catalogue (matches DR19 layout).
    placeholder = fits.BinTableHDU(data=np.array([(0,)], dtype=[("x", np.int32)]))
    catalogue = fits.BinTableHDU.from_columns(cols)
    fits.HDUList([primary, placeholder, catalogue]).writeto(path, overwrite=True)


@pytest.fixture
def dr19_fits(tmp_path: Path) -> Path:
    path = tmp_path / "astraAllStarASPCAP-test.fits"
    _build_dr19_fits(path, n_rows=8)
    return path


def test_load_dr19_returns_expected_columns(dr19_fits: Path) -> None:
    df = load_dr19(dr19_fits)
    assert len(df) == 8
    for col in kept_columns():
        assert col in df.columns, f"missing {col}"


def test_load_dr19_custom_columns_subset(dr19_fits: Path) -> None:
    df = load_dr19(dr19_fits, columns=["source_id", "teff", "logg"])
    assert list(df.columns) == ["source_id", "teff", "logg"]
    assert df["source_id"].iloc[0] == 100_000


def test_load_dr19_missing_column_raises(dr19_fits: Path) -> None:
    with pytest.raises(KeyError, match="not_a_real_column"):
        load_dr19(dr19_fits, columns=["source_id", "not_a_real_column"])


def test_apply_quality_cuts_counts(dr19_fits: Path) -> None:
    df = load_dr19(dr19_fits)
    out, stats = apply_quality_cuts(df)

    assert stats["before"] == 8
    # Row 0 fails SNR; row 7 fails flag_bad; rows outside Teff/logg/[M/H] also drop.
    # Exact count depends on the linspaces — assert the invariants, not the count.
    assert stats["after"] < stats["before"]
    assert stats["after_flag_bad"] == 7  # one row fails flag_bad
    assert (out["snr"] > 70).all()
    assert out["teff"].between(4000, 5500).all()
    assert out["logg"].between(1.0, 3.5).all()
    assert out["m_h_atm"].between(-2.0, 0.5).all()
    assert (out["flag_bad"] == 0).all()


def test_apply_quality_cuts_custom_bounds(dr19_fits: Path) -> None:
    df = load_dr19(dr19_fits)
    tight = QualityCuts(min_snr=80.0, teff_min=4500, teff_max=5200)
    out, _ = apply_quality_cuts(df, tight)
    assert out["teff"].between(4500, 5200).all()


def test_quality_cuts_predicates_are_strings() -> None:
    preds = QualityCuts().as_predicates()
    assert all(isinstance(p, str) for p in preds)
    assert any("flag_bad" in p for p in preds)


def test_derive_c_n_adds_column_and_propagates_error(dr19_fits: Path) -> None:
    df = load_dr19(dr19_fits)
    out = derive_c_n(df)
    assert "c_n" in out.columns
    assert "e_c_n" in out.columns
    np.testing.assert_allclose(out["c_n"], out["c_fe"] - out["n_fe"], rtol=1e-6)
    np.testing.assert_allclose(
        out["e_c_n"], np.sqrt(out["e_c_fe"] ** 2 + out["e_n_fe"] ** 2), rtol=1e-6
    )


def test_derive_c_n_noops_if_present(dr19_fits: Path) -> None:
    df = load_dr19(dr19_fits)
    df["c_n"] = 99.0
    out = derive_c_n(df)
    assert (out["c_n"] == 99.0).all()


def test_derive_c_n_missing_inputs_raises() -> None:
    import pandas as pd

    with pytest.raises(KeyError, match="c_fe.*n_fe|n_fe.*c_fe|c_fe"):
        derive_c_n(pd.DataFrame({"a": [1.0]}))


def test_meszaros_requires_teff_logg() -> None:
    import pandas as pd

    with pytest.raises(KeyError, match="teff"):
        apply_meszaros2025_corrections(pd.DataFrame({"mg_h_atm": [0.1]}))


def test_meszaros_linear_regime() -> None:
    """Inside [3500, 6000] K with log g < 3.8 the correction is (a·Teff + b)."""
    import pandas as pd

    from arqueogal.data.apogee_dr19 import MESZAROS2025_COEFFS

    a, b, _, _ = MESZAROS2025_COEFFS["mg_h_atm"]
    teff = np.array([3500.0, 4800.0, 6000.0])
    df = pd.DataFrame({"teff": teff, "logg": [2.0, 2.0, 2.0], "mg_h_atm": [0.2, 0.2, 0.2]})
    out = apply_meszaros2025_corrections(df, elements=("mg_h_atm",))
    expected = 0.2 - (a * teff + b)
    np.testing.assert_allclose(out["mg_h_atm"].to_numpy(), expected, rtol=1e-10)


def test_meszaros_out_of_range_uses_boundary_offsets() -> None:
    import pandas as pd

    from arqueogal.data.apogee_dr19 import MESZAROS2025_COEFFS

    _, _, hot, cold = MESZAROS2025_COEFFS["mg_h_atm"]
    df = pd.DataFrame({"teff": [3000.0, 7000.0], "logg": [2.0, 2.0], "mg_h_atm": [0.5, 0.5]})
    out = apply_meszaros2025_corrections(df, elements=("mg_h_atm",))
    np.testing.assert_allclose(out["mg_h_atm"].to_numpy(), [0.5 - cold, 0.5 - hot], rtol=1e-10)


def test_meszaros_dwarfs_left_uncorrected() -> None:
    """log g ≥ 3.8 → Δ is NaN → raw values pass through unchanged."""
    import pandas as pd

    df = pd.DataFrame({"teff": [5000.0, 5000.0], "logg": [2.0, 4.0], "mg_h_atm": [0.3, 0.3]})
    out = apply_meszaros2025_corrections(df, elements=("mg_h_atm",))
    assert out["mg_h_atm"].iloc[0] != 0.3
    assert out["mg_h_atm"].iloc[1] == 0.3


def test_meszaros_unknown_element_raises() -> None:
    import pandas as pd

    df = pd.DataFrame({"teff": [4800.0], "logg": [2.0]})
    with pytest.raises(KeyError, match="no Mészáros"):
        apply_meszaros2025_corrections(df, elements=("fe_h_atm",))


def test_meszaros_attaches_summary(dr19_fits: Path) -> None:
    df = load_dr19(dr19_fits)
    out, _ = apply_quality_cuts(df)
    corrected = apply_meszaros2025_corrections(out)
    summary = corrected.attrs["meszaros_correction_summary"]
    assert {"element", "n_applied", "mean_shift", "rms_shift"} <= set(summary.columns)
    assert (summary["n_applied"] > 0).any()
