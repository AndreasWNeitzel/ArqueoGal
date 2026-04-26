"""Offline tests for arqueogal.data.gaia_xp.

No TAP traffic: ``run_sync`` / ``run_async`` are monkeypatched to return
fake Tables, and the Ye+2024 stub is exercised directly to confirm it halts.
The §6.4 normalisation and error-propagation math is tested against
closed-form values computed by hand.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from astropy.table import Table
from pyvo.dal.tap import TAPService

from arqueogal.data import tap as tap_mod
from arqueogal.data.gaia_xp import (
    HERMITE_BP_RANGE_NM,
    HERMITE_N_BASIS,
    HERMITE_REPROJECTION_VERSION,
    HERMITE_RP_RANGE_NM,
    XP_BATCH_SIZE,
    XP_COEFF_LEN,
    XP_QUERY_ADQL,
    YE2024_N_OUTPUT,
    YE2024_SAMPLING_NM,
    XpC0Stats,
    _build_hermite_basis,  # noqa: PLC2701 — internal, tested
    apply_ye2024_correction,
    fetch_xp_coefficients,
    normalise_xp,
    reproject_ye_to_hermite,
    xp_sanity_check,
    zscore_c0,
)
from arqueogal.data.tap import BATCH_PLACEHOLDER

# ---- helpers -----------------------------------------------------------------


def _fake_xp_row(source_id: int, c0_bp: float = 1e-15, c0_rp: float = 2e-15) -> dict[str, object]:
    """Build one §6.3-shaped row with deterministic coefficients."""
    bp = np.zeros(XP_COEFF_LEN, dtype=np.float64)
    rp = np.zeros(XP_COEFF_LEN, dtype=np.float64)
    bp[0] = c0_bp
    rp[0] = c0_rp
    bp[1:] = np.linspace(1e-17, 5e-17, XP_COEFF_LEN - 1)
    rp[1:] = np.linspace(2e-17, 6e-17, XP_COEFF_LEN - 1)
    bp_err = np.full(XP_COEFF_LEN, 1e-18)
    rp_err = np.full(XP_COEFF_LEN, 1e-18)
    return {
        "source_id": source_id,
        "bp_coefficients": bp,
        "bp_coefficient_errors": bp_err,
        "rp_coefficients": rp,
        "rp_coefficient_errors": rp_err,
        "bp_standard_deviation": 0.1,
        "rp_standard_deviation": 0.1,
        "bp_n_measurements": 30,
        "rp_n_measurements": 30,
        "bp_n_relevant_bases": 40,
        "rp_n_relevant_bases": 35,
    }


def _fake_xp_df(source_ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame([_fake_xp_row(sid) for sid in source_ids])


# ---- §6.3 fetch --------------------------------------------------------------


def test_fetch_rejects_missing_placeholder() -> None:
    service = MagicMock(spec=TAPService)
    with pytest.raises(ValueError, match="placeholder"):
        fetch_xp_coefficients(service, [1, 2], adql="SELECT * FROM xp")


def test_fetch_rejects_zero_batch_size() -> None:
    service = MagicMock(spec=TAPService)
    with pytest.raises(ValueError, match="batch_size"):
        fetch_xp_coefficients(service, [1], batch_size=0)


def test_fetch_empty_input_short_circuits() -> None:
    service = MagicMock(spec=TAPService)

    # Poison the TAP runners so any call blows up — proves we never hit them.
    def boom(*_a, **_kw):
        raise AssertionError("TAP runner should not be called for empty input")

    result = fetch_xp_coefficients(service, [])
    assert result.empty


def test_fetch_batches_correctly_and_concatenates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[int]] = []

    def fake_async(_svc, adql, **_kw):  # noqa: ANN001
        import re

        ids = [int(x) for x in re.search(r"IN \(([^)]+)\)", adql).group(1).split(",")]
        calls.append(ids)
        return Table.from_pandas(_fake_xp_df(ids))

    monkeypatch.setattr(tap_mod, "run_async", fake_async)
    monkeypatch.setattr(tap_mod, "run_sync", lambda *_a, **_kw: pytest.fail("sync should not fire"))

    service = MagicMock(spec=TAPService)
    out = fetch_xp_coefficients(service, list(range(1, 8)), batch_size=3, mode="async")

    assert [c for c in calls] == [[1, 2, 3], [4, 5, 6], [7]]
    assert len(out) == 7
    assert list(out["source_id"]) == [1, 2, 3, 4, 5, 6, 7]


def test_fetch_checkpoint_reuse(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Once a batch_NNNN.parquet exists, the TAP runner must be skipped."""
    ckpt = tmp_path / "xp_batches"
    ckpt.mkdir()
    # Pre-populate batch_0000 checkpoint with a DataFrame the test will read back.
    pre = _fake_xp_df([10, 11])
    pre.to_parquet(ckpt / "xp_batch_0000.parquet", index=False)

    def fail_async(*_a, **_kw):
        raise AssertionError("async should not run when checkpoint exists")

    monkeypatch.setattr(tap_mod, "run_async", fail_async)

    service = MagicMock(spec=TAPService)
    out = fetch_xp_coefficients(service, [10, 11], batch_size=5, checkpoint_dir=ckpt)
    assert list(out["source_id"]) == [10, 11]


def test_fetch_batch_size_default_is_5000() -> None:
    assert XP_BATCH_SIZE == 5_000


def test_xp_query_targets_the_xp_table() -> None:
    assert "xp_continuous_mean_spectrum" in XP_QUERY_ADQL
    assert BATCH_PLACEHOLDER in XP_QUERY_ADQL


# ---- §6.4 step 1 (Ye+2024 signature / schema contract) ----------------------


def test_ye2024_rejects_missing_source_id() -> None:
    xp = pd.DataFrame({"bp_coefficients": [[0.0] * XP_COEFF_LEN]})
    coords = pd.DataFrame({"source_id": [1], "ra": [0.0], "dec": [0.0]})
    with pytest.raises(KeyError, match="source_id"):
        apply_ye2024_correction(xp, coords)


def test_ye2024_rejects_missing_coord_columns() -> None:
    xp = pd.DataFrame({"source_id": [1]})
    coords = pd.DataFrame({"source_id": [1], "ra": [0.0]})  # no dec
    with pytest.raises(KeyError, match="dec"):
        apply_ye2024_correction(xp, coords)


def test_ye2024_rejects_wrong_sampling_length() -> None:
    xp = pd.DataFrame({"source_id": [1]})
    coords = pd.DataFrame({"source_id": [1], "ra": [0.0], "dec": [0.0]})
    with pytest.raises(ValueError, match="sampling_nm"):
        apply_ye2024_correction(xp, coords, sampling_nm=np.linspace(400, 900, 10))


def test_align_to_batch_restores_dropped_rows_as_nan() -> None:
    """_align_to_batch restores row count + order when gaiaxpy drops rows."""
    from arqueogal.data.gaia_xp import _align_to_batch

    batch = pd.DataFrame({"source_id": np.array([10, 20, 30, 40], dtype=np.int64)})
    # Simulate gaiaxpy dropping rows 20 and 40 and reordering the rest.
    syn = pd.DataFrame(
        {
            "source_id": np.array([30, 10], dtype=np.int64),
            "SkyMapper_mag_u": [15.0, 14.0],
            "SkyMapper_mag_g": [14.5, 13.5],
        }
    )
    out = _align_to_batch(batch, syn)
    assert len(out) == len(batch)
    assert out["source_id"].tolist() == [10, 20, 30, 40]
    u = out["SkyMapper_mag_u"].to_numpy()
    assert np.isclose(u[0], 14.0)
    assert np.isnan(u[1])
    assert np.isclose(u[2], 15.0)
    assert np.isnan(u[3])


def test_align_to_batch_tolerates_na_source_id() -> None:
    """_align_to_batch drops gaiaxpy's Int64 NA source_id rows before merge."""
    from arqueogal.data.gaia_xp import _align_to_batch

    batch = pd.DataFrame({"source_id": np.array([100, 200, 300], dtype=np.int64)})
    syn = pd.DataFrame(
        {
            "source_id": pd.array([100, pd.NA, 300], dtype="Int64"),
            "GaiaDr3Vega_mag_G": [11.0, 99.0, 12.0],  # NA row has junk photometry
        }
    )
    out = _align_to_batch(batch, syn)
    assert len(out) == 3
    assert out["source_id"].tolist() == [100, 200, 300]
    assert np.isclose(out["GaiaDr3Vega_mag_G"].iloc[0], 11.0)
    # The NA-source_id row was dropped → batch row 200 gets NaN photometry.
    assert np.isnan(out["GaiaDr3Vega_mag_G"].iloc[1])
    assert np.isclose(out["GaiaDr3Vega_mag_G"].iloc[2], 12.0)


# ---- §6.4 step 2 + 4: normalisation and error propagation -------------------


def test_normalise_coefficient_ratios() -> None:
    df = _fake_xp_df([42])
    norm = normalise_xp(df)

    raw = df["bp_coefficients"].iloc[0]
    c0 = raw[0]
    expected_ratios = raw[1:] / c0
    assert np.allclose(norm["bp_coeffs_norm"].iloc[0][1:], expected_ratios, rtol=1e-5)
    # Index 0 carries log10(c0) — not the ratio 1.0.
    assert np.isclose(norm["bp_coeffs_norm"].iloc[0][0], np.log10(c0), rtol=1e-5)


def test_normalise_error_propagation_exact() -> None:
    """Closed-form: σ_norm_i = sqrt((σ_i/c0)² + (c_i·σ_0/c0²)²)."""
    df = _fake_xp_df([1])
    raw = df["bp_coefficients"].iloc[0]
    sig = df["bp_coefficient_errors"].iloc[0]
    c0 = raw[0]
    sig0 = sig[0]

    norm = normalise_xp(df)
    errs_norm = norm["bp_coeff_errs_norm"].iloc[0]

    # Index 0 — error on log10(c0).
    expected_log_err = sig0 / (c0 * math.log(10))
    assert np.isclose(errs_norm[0], expected_log_err, rtol=1e-5)

    # Index i ≥ 1.
    expected_ratio_errs = np.sqrt((sig[1:] / c0) ** 2 + (raw[1:] * sig0 / c0**2) ** 2)
    assert np.allclose(errs_norm[1:], expected_ratio_errs, rtol=1e-5)


def test_normalise_drops_raw_coefficient_columns() -> None:
    df = _fake_xp_df([1, 2])
    norm = normalise_xp(df)
    assert "bp_coefficients" not in norm.columns
    assert "rp_coefficient_errors" not in norm.columns
    # Analysis-ready columns present.
    for col in (
        "bp_coeffs_norm",
        "bp_coeff_errs_norm",
        "rp_coeffs_norm",
        "rp_coeff_errs_norm",
        "bp_c0_log",
        "rp_c0_log",
    ):
        assert col in norm.columns


def test_normalise_output_dtype_is_float32() -> None:
    norm = normalise_xp(_fake_xp_df([1]))
    assert norm["bp_coeffs_norm"].iloc[0].dtype == np.float32
    assert norm["rp_coeff_errs_norm"].iloc[0].dtype == np.float32


def test_normalise_rejects_wrong_length_array() -> None:
    df = _fake_xp_df([1])
    # Truncate to 54 coefficients.
    df.at[0, "bp_coefficients"] = df["bp_coefficients"].iloc[0][:54]
    with pytest.raises(ValueError, match=r"\(\d+, 54\)"):
        normalise_xp(df)


def test_normalise_rejects_nan_coefficients() -> None:
    df = _fake_xp_df([1])
    bad = df["bp_coefficients"].iloc[0].copy()
    bad[10] = np.nan
    df.at[0, "bp_coefficients"] = bad
    with pytest.raises(ValueError, match="NaN"):
        normalise_xp(df)


def test_normalise_rejects_nonpositive_c0() -> None:
    df = _fake_xp_df([1, 2])
    bad = df["bp_coefficients"].iloc[1].copy()
    bad[0] = -1e-16
    df.at[1, "bp_coefficients"] = bad
    with pytest.raises(ValueError, match="c_0 <= 0"):
        normalise_xp(df)


def test_normalise_requires_expected_columns() -> None:
    df = pd.DataFrame({"source_id": [1], "bp_coefficients": [np.zeros(55)]})
    with pytest.raises(KeyError, match="rp_coefficients"):
        normalise_xp(df)


# ---- §6.4 step 3: z-score ----------------------------------------------------


def test_zscore_fit_and_apply() -> None:
    df = normalise_xp(_fake_xp_df([1, 2, 3, 4]))
    # All c0s equal → std is 0 → z-score fit raises.
    with pytest.raises(ValueError, match="z-score std is zero"):
        zscore_c0(df)


def test_zscore_with_varied_c0() -> None:
    rows = [
        _fake_xp_row(sid, c0_bp=10 ** (-15 + sid * 0.1), c0_rp=10 ** (-14 + sid * 0.05))
        for sid in range(1, 6)
    ]
    df = normalise_xp(pd.DataFrame(rows))
    out, stats = zscore_c0(df)

    assert "bp_c0_z" in out.columns
    # z-score output has mean ≈ 0, std ≈ 1.
    assert np.isclose(out["bp_c0_z"].mean(), 0.0, atol=1e-5)
    assert np.isclose(out["bp_c0_z"].std(ddof=0), 1.0, atol=1e-5)
    assert isinstance(stats, XpC0Stats)
    assert stats.bp_c0_log_std > 0


def test_zscore_applied_with_fixed_stats_is_not_refit() -> None:
    """Inference path: saved stats must not shift when re-applied."""
    rows = [
        _fake_xp_row(sid, c0_bp=10 ** (-15 + sid * 0.1), c0_rp=10 ** (-14 + sid * 0.05))
        for sid in range(1, 6)
    ]
    train = normalise_xp(pd.DataFrame(rows))
    _train_z, stats = zscore_c0(train)

    # Simulate inference on a shifted dataset.
    rows2 = [
        _fake_xp_row(sid, c0_bp=10 ** (-10 + sid * 0.1), c0_rp=10 ** (-14 + sid * 0.05))
        for sid in range(1, 6)
    ]
    inf = normalise_xp(pd.DataFrame(rows2))
    inf_z, _ = zscore_c0(inf, stats=stats)

    # Inference-data bp_c0_log values are offset by +5 dex from training; so
    # the z-scored inference values should not be centred at 0.
    assert inf_z["bp_c0_z"].mean() > 10  # many sigma away from training centre


def test_zscore_requires_normalise_first() -> None:
    df = pd.DataFrame({"source_id": [1]})  # no bp_c0_log column
    with pytest.raises(KeyError, match="bp_c0_log"):
        zscore_c0(df)


def test_xp_c0_stats_to_dict_roundtrips_fields() -> None:
    s = XpC0Stats(bp_c0_log_mean=-15.0, bp_c0_log_std=0.5, rp_c0_log_mean=-14.8, rp_c0_log_std=0.4)
    d = s.to_dict()
    assert d["bp_c0_log_mean"] == -15.0
    assert d["rp_c0_log_std"] == 0.4


# ---- §6.5 sanity check -------------------------------------------------------


def test_sanity_check_passes_on_clean_frame() -> None:
    counts = xp_sanity_check(_fake_xp_df([1, 2, 3]))
    assert counts["bp_coefficients_nan_rows"] == 0
    assert counts["rp_coefficients_nonpos_c0"] == 0


def test_sanity_check_flags_nan() -> None:
    df = _fake_xp_df([1])
    bad = df["bp_coefficients"].iloc[0].copy()
    bad[5] = np.nan
    df.at[0, "bp_coefficients"] = bad
    with pytest.raises(ValueError, match="nan_rows"):
        xp_sanity_check(df)


def test_sanity_check_flags_nonpositive_c0() -> None:
    df = _fake_xp_df([1])
    bad = df["rp_coefficients"].iloc[0].copy()
    bad[0] = 0.0
    df.at[0, "rp_coefficients"] = bad
    with pytest.raises(ValueError, match="nonpos_c0"):
        xp_sanity_check(df)


def test_sanity_check_flags_has_xp_continuous_false() -> None:
    df = _fake_xp_df([1])
    df["has_xp_continuous"] = [False]
    with pytest.raises(ValueError, match="has_xp_continuous_false"):
        xp_sanity_check(df)


# ---- Hermite reprojection (§6.4 step 2) --------------------------------------


def test_hermite_basis_is_orthonormal_under_trapezoidal_inner_product() -> None:
    """ΨᵀWΨ = I to tight tolerance for each band (QR with grid-weighted G)."""
    basis = _build_hermite_basis()
    for band in ("bp", "rp"):
        b = basis[band]
        gram = b["psi"].T @ (b["w"][:, None] * b["psi"])
        eye = np.eye(HERMITE_N_BASIS)
        # QR gives machine-precision orthonormality wrt the given inner product.
        np.testing.assert_allclose(gram, eye, atol=1e-8, rtol=0)


def test_hermite_basis_grid_hard_split_at_660_nm() -> None:
    """BP is [360, 660), RP is [660, 990]; 660 nm must appear in RP only."""
    basis = _build_hermite_basis()
    lam_bp, lam_rp = basis["bp"]["lam"], basis["rp"]["lam"]
    assert lam_bp.max() < 660.0
    assert lam_rp.min() >= 660.0
    # Point counts are fixed by the geomspace grid — regression-test them.
    assert lam_bp.size + lam_rp.size == YE2024_N_OUTPUT
    assert lam_bp.size == 198
    assert lam_rp.size == 132


def test_hermite_basis_fingerprint_is_deterministic() -> None:
    """lru_cache + deterministic constants → identical fingerprint across calls."""
    fp1 = _build_hermite_basis()["fingerprint_sha256"]
    fp2 = _build_hermite_basis()["fingerprint_sha256"]
    assert fp1 == fp2
    assert len(fp1) == 64  # sha256 hex length


def test_reproject_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=f"N, {YE2024_N_OUTPUT}"):
        reproject_ye_to_hermite(np.zeros((3, YE2024_N_OUTPUT - 1), dtype=np.float32))
    with pytest.raises(ValueError, match=f"N, {YE2024_N_OUTPUT}"):
        reproject_ye_to_hermite(np.zeros(YE2024_N_OUTPUT, dtype=np.float32))


def test_reproject_reproduces_pure_hermite_input_exactly() -> None:
    """If f is an exact member of span(Ψ) on one band, residual RMS ≈ 0 there.

    Build a flux that is the 3rd orthonormal BP basis function on the BP grid
    and zero on the RP grid. Then c_bp[3] should be 1 (others ≈ 0), the BP
    residual should vanish to numerical precision, and the RP residual is
    exactly zero since f_rp = 0 has zero RMS.
    """
    basis = _build_hermite_basis()
    f = np.zeros((1, YE2024_N_OUTPUT), dtype=np.float64)
    f[0, basis["bp"]["mask"]] = basis["bp"]["psi"][:, 3]
    out = reproject_ye_to_hermite(f)
    bp = out["bp_coeffs"][0]
    assert abs(bp[3] - 1.0) < 1e-5
    mask_other = np.ones(HERMITE_N_BASIS, dtype=bool)
    mask_other[3] = False
    assert np.max(np.abs(bp[mask_other])) < 1e-5
    assert out["reprojection_residual_rms_bp"][0] < 1e-5
    assert out["reprojection_residual_rms_rp"][0] == 0.0


def test_reproject_residual_rms_is_nonnegative_and_float32() -> None:
    rng = np.random.default_rng(42)
    f = rng.standard_normal((5, YE2024_N_OUTPUT)).astype(np.float32)
    out = reproject_ye_to_hermite(f)
    assert out["bp_coeffs"].dtype == np.float32
    assert out["rp_coeffs"].dtype == np.float32
    assert out["reprojection_residual_rms"].dtype == np.float32
    assert out["bp_coeffs"].shape == (5, HERMITE_N_BASIS)
    assert out["rp_coeffs"].shape == (5, HERMITE_N_BASIS)
    assert np.all(out["reprojection_residual_rms"] >= 0.0)
    assert np.all(out["reprojection_residual_rms_bp"] >= 0.0)
    assert np.all(out["reprojection_residual_rms_rp"] >= 0.0)


def test_reproject_propagates_basis_version_and_fingerprint() -> None:
    f = np.zeros((1, YE2024_N_OUTPUT), dtype=np.float32)
    out = reproject_ye_to_hermite(f)
    assert out["basis_version"] == HERMITE_REPROJECTION_VERSION
    assert out["basis_fingerprint_sha256"] == _build_hermite_basis()["fingerprint_sha256"]


def test_hermite_basis_has_positive_c0_for_positive_flux() -> None:
    """Sign convention: a constant positive flux must give a positive c_0.

    QR is column-sign-ambiguous; we enforce positive-diagonal R so that
    ψ_0 > 0 and c_0 tracks integrated flux with the physical sign.
    """
    basis = _build_hermite_basis()
    f = np.ones((1, YE2024_N_OUTPUT), dtype=np.float32)  # flat positive flux
    out = reproject_ye_to_hermite(f)
    assert out["bp_coeffs"][0, 0] > 0
    assert out["rp_coeffs"][0, 0] > 0
    # And ψ_0 itself should be non-negative on both band grids.
    assert (basis["bp"]["psi"][:, 0] >= 0).all()
    assert (basis["rp"]["psi"][:, 0] >= 0).all()


def test_reproject_bands_cover_advertised_ranges() -> None:
    """Sanity check: declared band ranges agree with grid masks."""
    basis = _build_hermite_basis()
    bp_lo, bp_hi = HERMITE_BP_RANGE_NM
    rp_lo, rp_hi = HERMITE_RP_RANGE_NM
    assert basis["bp"]["lam"].min() >= bp_lo
    assert basis["bp"]["lam"].max() < bp_hi  # right-open
    assert basis["rp"]["lam"].min() >= rp_lo
    assert basis["rp"]["lam"].max() <= rp_hi
    # And the first grid point is the geomspace origin 360 nm.
    assert math.isclose(YE2024_SAMPLING_NM[0], 360.0)
    assert math.isclose(YE2024_SAMPLING_NM[-1], 990.0)
