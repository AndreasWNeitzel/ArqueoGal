"""Tests for arqueogal.data.frozen_stats.

Covers the Stream-3 inference path: load frozen stats from the real Stream-1
provenance JSON, verify the basis-fingerprint integrity check, and apply the
z-score transform to synthetic coefficient arrays without refitting.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from arqueogal.data.frozen_stats import (
    XP_COEFF_LEN,
    FrozenStatsMismatchError,
    FrozenZScoreStats,
    apply_frozen_zscore,
    load_frozen_zscore_stats,
    verify_basis_fingerprint,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_PROVENANCE = (
    REPO_ROOT / "data" / "processed" / "pipeline1_features_stream1.provenance.json"
)
REAL_FINGERPRINT = (
    "0d34b5659e97e5891b57005215a59b0b70fc56f23d8ffb22f442c4ad5101eab7"
)


# ---- fixture: minimal provenance JSON mirroring the real layout --------------


def _write_minimal_provenance(  # noqa: PLR0913 — keyword-only fixture knobs
    path: Path,
    *,
    fingerprint: str = REAL_FINGERPRINT,
    sigma_floor: float = 1e-30,
    bp_mu: dict[int, float] | None = None,
    bp_sigma: dict[int, float] | None = None,
    rp_mu: dict[int, float] | None = None,
    rp_sigma: dict[int, float] | None = None,
    c0_bp_mu: float = -14.4,
    c0_bp_sigma: float = 0.57,
    c0_rp_mu: float = -14.44,
    c0_rp_sigma: float = 0.53,
) -> None:
    """Emit a provenance JSON with the same structure as the real Stream-1 sidecar."""
    bp_mu = bp_mu or {i: 0.1 * i for i in range(1, XP_COEFF_LEN)}
    bp_sigma = bp_sigma or {i: 0.2 for i in range(1, XP_COEFF_LEN)}
    rp_mu = rp_mu or {i: -0.05 * i for i in range(1, XP_COEFF_LEN)}
    rp_sigma = rp_sigma or {i: 0.15 for i in range(1, XP_COEFF_LEN)}

    payload = {
        "output_file": "fake.parquet",
        "extra": {
            "basis_fingerprint_sha256": fingerprint,
            "c0_zscore_frozen": {
                "bp": {"mu_log10": c0_bp_mu, "sigma_log10": c0_bp_sigma},
                "rp": {"mu_log10": c0_rp_mu, "sigma_log10": c0_rp_sigma},
                "n_reference_population": 10_000,
                "log_base": 10,
                "reference_population": "synthetic fixture",
            },
            "coef_norm_zscore_frozen": {
                "bp": {
                    str(i): {"mu": bp_mu[i], "sigma": bp_sigma[i]}
                    for i in range(1, XP_COEFF_LEN)
                },
                "rp": {
                    str(i): {"mu": rp_mu[i], "sigma": rp_sigma[i]}
                    for i in range(1, XP_COEFF_LEN)
                },
                "n_reference_population": 10_000,
                "sigma_floor": sigma_floor,
                "tiny_sigma_substituted_bp": [],
                "tiny_sigma_substituted_rp": [],
                "reference_population": "synthetic fixture",
                "stored_column_semantic": "synthetic",
            },
        },
    }
    path.write_text(json.dumps(payload))


# ---- loading from real provenance --------------------------------------------


@pytest.mark.skipif(
    not REAL_PROVENANCE.exists(),
    reason="Real Stream-1 provenance JSON not present; fixture-only tests still cover format.",
)
def test_load_from_real_provenance_has_expected_fingerprint() -> None:
    stats = load_frozen_zscore_stats(REAL_PROVENANCE)
    assert stats.basis_fingerprint == REAL_FINGERPRINT
    # Sanity: c0 stats are negative log10-flux values, sigma positive, O(0.5).
    assert -20 < stats.c0_bp_mean_log10 < -10
    assert 0.1 < stats.c0_bp_sigma_log10 < 2.0
    assert -20 < stats.c0_rp_mean_log10 < -10
    assert 0.1 < stats.c0_rp_sigma_log10 < 2.0
    # Ratio stats arrays have length XP_COEFF_LEN - 1.
    assert stats.coef_norm_bp_mean.shape == (XP_COEFF_LEN - 1,)
    assert stats.coef_norm_bp_sigma.shape == (XP_COEFF_LEN - 1,)
    assert stats.coef_norm_rp_mean.shape == (XP_COEFF_LEN - 1,)
    assert stats.coef_norm_rp_sigma.shape == (XP_COEFF_LEN - 1,)
    # No NaNs after load.
    assert np.isfinite(stats.coef_norm_bp_mean).all()
    assert np.isfinite(stats.coef_norm_bp_sigma).all()
    assert np.isfinite(stats.coef_norm_rp_mean).all()
    assert np.isfinite(stats.coef_norm_rp_sigma).all()
    # sigma_floor is carried through.
    assert stats.sigma_floor == pytest.approx(1e-30)


def test_load_from_fixture_provenance(tmp_path: Path) -> None:
    p = tmp_path / "prov.json"
    _write_minimal_provenance(p)
    stats = load_frozen_zscore_stats(p)
    assert isinstance(stats, FrozenZScoreStats)
    assert stats.basis_fingerprint == REAL_FINGERPRINT
    # Spot-check first coefficient values.
    assert stats.coef_norm_bp_mean[0] == pytest.approx(0.1)
    assert stats.coef_norm_bp_sigma[0] == pytest.approx(0.2)
    assert stats.coef_norm_rp_mean[0] == pytest.approx(-0.05)
    assert stats.coef_norm_rp_sigma[0] == pytest.approx(0.15)


# ---- integrity check ---------------------------------------------------------


def test_verify_basis_fingerprint_passes_on_match(tmp_path: Path) -> None:
    p = tmp_path / "prov.json"
    _write_minimal_provenance(p, fingerprint="a" * 64)
    stats = load_frozen_zscore_stats(p)
    # Must not raise.
    verify_basis_fingerprint("a" * 64, stats)


def test_verify_basis_fingerprint_raises_on_mismatch(tmp_path: Path) -> None:
    p = tmp_path / "prov.json"
    _write_minimal_provenance(p, fingerprint="a" * 64)
    stats = load_frozen_zscore_stats(p)
    with pytest.raises(FrozenStatsMismatchError, match="basis fingerprint mismatch"):
        verify_basis_fingerprint("b" * 64, stats)


def test_verify_basis_fingerprint_raises_on_truncated_hex(tmp_path: Path) -> None:
    p = tmp_path / "prov.json"
    _write_minimal_provenance(p, fingerprint="deadbeef" * 8)
    stats = load_frozen_zscore_stats(p)
    # Truncated / wrong-case versions of the real SHA must also fail.
    with pytest.raises(FrozenStatsMismatchError):
        verify_basis_fingerprint(("deadbeef" * 8)[:-1], stats)


# ---- apply_frozen_zscore -----------------------------------------------------


def _make_stats(  # noqa: PLR0913 — keyword-only fixture knobs
    bp_mu_val: float = 0.5, bp_sigma_val: float = 0.2,
    rp_mu_val: float = -0.3, rp_sigma_val: float = 0.1,
    c0_bp_mu: float = -14.0, c0_bp_sigma: float = 0.6,
    c0_rp_mu: float = -14.5, c0_rp_sigma: float = 0.5,
) -> FrozenZScoreStats:
    return FrozenZScoreStats(
        basis_fingerprint="x" * 64,
        c0_bp_mean_log10=c0_bp_mu, c0_bp_sigma_log10=c0_bp_sigma,
        c0_rp_mean_log10=c0_rp_mu, c0_rp_sigma_log10=c0_rp_sigma,
        coef_norm_bp_mean=np.full(XP_COEFF_LEN - 1, bp_mu_val, dtype=np.float64),
        coef_norm_bp_sigma=np.full(XP_COEFF_LEN - 1, bp_sigma_val, dtype=np.float64),
        coef_norm_rp_mean=np.full(XP_COEFF_LEN - 1, rp_mu_val, dtype=np.float64),
        coef_norm_rp_sigma=np.full(XP_COEFF_LEN - 1, rp_sigma_val, dtype=np.float64),
        sigma_floor=1e-30,
        n_reference_population=1000,
        reference_population_description="test",
    )


def test_apply_frozen_zscore_matches_manual_transform() -> None:
    stats = _make_stats()
    n = 7
    bp = np.linspace(0.0, 2.0, n * (XP_COEFF_LEN - 1)).reshape(n, XP_COEFF_LEN - 1)
    rp = np.linspace(-2.0, 0.0, n * (XP_COEFF_LEN - 1)).reshape(n, XP_COEFF_LEN - 1)
    bp_c0 = np.linspace(-15.0, -13.5, n)
    rp_c0 = np.linspace(-15.5, -13.5, n)

    bp_z, rp_z, bp_c0_z, rp_c0_z = apply_frozen_zscore(bp, rp, bp_c0, rp_c0, stats)

    expected_bp = (bp - stats.coef_norm_bp_mean) / stats.coef_norm_bp_sigma
    expected_rp = (rp - stats.coef_norm_rp_mean) / stats.coef_norm_rp_sigma
    expected_bp_c0 = (bp_c0 - stats.c0_bp_mean_log10) / stats.c0_bp_sigma_log10
    expected_rp_c0 = (rp_c0 - stats.c0_rp_mean_log10) / stats.c0_rp_sigma_log10
    np.testing.assert_allclose(bp_z, expected_bp)
    np.testing.assert_allclose(rp_z, expected_rp)
    np.testing.assert_allclose(bp_c0_z, expected_bp_c0)
    np.testing.assert_allclose(rp_c0_z, expected_rp_c0)


def test_apply_frozen_zscore_mean_shifts_to_zero_on_match() -> None:
    """If the input equals the frozen mean, the z-score output is zero."""
    stats = _make_stats(bp_mu_val=0.5, bp_sigma_val=0.2)
    n = 5
    bp_at_mean = np.full((n, XP_COEFF_LEN - 1), 0.5, dtype=np.float64)
    rp_at_mean = np.full((n, XP_COEFF_LEN - 1), -0.3, dtype=np.float64)
    bp_c0_at_mean = np.full(n, -14.0, dtype=np.float64)
    rp_c0_at_mean = np.full(n, -14.5, dtype=np.float64)
    bp_z, rp_z, bp_c0_z, rp_c0_z = apply_frozen_zscore(
        bp_at_mean, rp_at_mean, bp_c0_at_mean, rp_c0_at_mean, stats,
    )
    np.testing.assert_allclose(bp_z, 0.0, atol=1e-12)
    np.testing.assert_allclose(rp_z, 0.0, atol=1e-12)
    np.testing.assert_allclose(bp_c0_z, 0.0, atol=1e-12)
    np.testing.assert_allclose(rp_c0_z, 0.0, atol=1e-12)


def test_apply_frozen_zscore_does_not_refit() -> None:
    """The function must not recompute mean/sigma from input.

    We prove it by feeding an input whose own mean/sigma differ from the frozen
    stats and confirming that:
      (a) the output is the frozen transform (not demeaned by input mean),
      (b) repeated calls with different inputs return stats with identical
          frozen mu/sigma attributes — i.e. stats is not mutated.
    """
    stats = _make_stats(bp_mu_val=0.5, bp_sigma_val=0.2)
    frozen_bp_mean = stats.coef_norm_bp_mean.copy()
    frozen_bp_sigma = stats.coef_norm_bp_sigma.copy()

    # Input whose empirical mean is nowhere near 0.5 and sigma is not 0.2.
    bp_input = np.full((3, XP_COEFF_LEN - 1), 10.0, dtype=np.float64)
    rp_input = np.full((3, XP_COEFF_LEN - 1), -7.0, dtype=np.float64)
    bp_c0 = np.array([-16.0, -14.0, -12.0])
    rp_c0 = np.array([-16.5, -14.5, -12.5])
    bp_z, _, _, _ = apply_frozen_zscore(bp_input, rp_input, bp_c0, rp_c0, stats)

    # Frozen transform: (10 - 0.5) / 0.2 = 47.5 everywhere.
    np.testing.assert_allclose(bp_z, 47.5)

    # Second call with a totally different input: frozen stats unchanged.
    bp_input_2 = np.full((5, XP_COEFF_LEN - 1), -3.0, dtype=np.float64)
    rp_input_2 = np.full((5, XP_COEFF_LEN - 1), 2.0, dtype=np.float64)
    bp_c0_2 = np.full(5, -13.0)
    rp_c0_2 = np.full(5, -13.0)
    apply_frozen_zscore(bp_input_2, rp_input_2, bp_c0_2, rp_c0_2, stats)

    np.testing.assert_array_equal(stats.coef_norm_bp_mean, frozen_bp_mean)
    np.testing.assert_array_equal(stats.coef_norm_bp_sigma, frozen_bp_sigma)


def test_apply_frozen_zscore_supports_55_wide_input_preserving_column_0() -> None:
    """Some emit paths pass a (N, 55) array with log10(c_0) in column 0."""
    stats = _make_stats(bp_mu_val=0.5, bp_sigma_val=0.2)
    n = 4
    ratios = np.zeros((n, XP_COEFF_LEN), dtype=np.float64)
    # Fill the c_0 column with a distinctive value that must survive unchanged.
    ratios[:, 0] = -14.123
    # Fill ratio columns with 0.5 (the frozen mean).
    ratios[:, 1:] = 0.5
    rp = np.zeros((n, XP_COEFF_LEN), dtype=np.float64)
    rp[:, 0] = -14.456
    rp[:, 1:] = -0.3
    bp_c0 = np.full(n, -14.0)
    rp_c0 = np.full(n, -14.5)

    bp_z, rp_z, _, _ = apply_frozen_zscore(ratios, rp, bp_c0, rp_c0, stats)
    # Column 0 passed through.
    np.testing.assert_allclose(bp_z[:, 0], -14.123)
    np.testing.assert_allclose(rp_z[:, 0], -14.456)
    # Columns 1..54 are the z-scored ratios.
    np.testing.assert_allclose(bp_z[:, 1:], 0.0, atol=1e-12)


def test_apply_frozen_zscore_rejects_bad_shape() -> None:
    stats = _make_stats()
    bad = np.zeros((3, 10), dtype=np.float64)
    rp_ok = np.zeros((3, XP_COEFF_LEN - 1), dtype=np.float64)
    with pytest.raises(ValueError, match="coef_norm"):
        apply_frozen_zscore(bad, rp_ok, np.zeros(3), np.zeros(3), stats)


def test_apply_scalar_zscore_rejects_zero_sigma() -> None:
    stats = FrozenZScoreStats(
        basis_fingerprint="x" * 64,
        c0_bp_mean_log10=0.0, c0_bp_sigma_log10=0.0,  # pathological
        c0_rp_mean_log10=0.0, c0_rp_sigma_log10=1.0,
        coef_norm_bp_mean=np.zeros(XP_COEFF_LEN - 1),
        coef_norm_bp_sigma=np.ones(XP_COEFF_LEN - 1),
        coef_norm_rp_mean=np.zeros(XP_COEFF_LEN - 1),
        coef_norm_rp_sigma=np.ones(XP_COEFF_LEN - 1),
        sigma_floor=1e-30, n_reference_population=1,
        reference_population_description="",
    )
    bp = np.zeros((2, XP_COEFF_LEN - 1))
    rp = np.zeros((2, XP_COEFF_LEN - 1))
    with pytest.raises(ValueError, match="sigma must be positive"):
        apply_frozen_zscore(bp, rp, np.zeros(2), np.zeros(2), stats)


# ---- sigma_floor behaviour ---------------------------------------------------


def test_sigma_floor_is_preserved_on_load(tmp_path: Path) -> None:
    p = tmp_path / "prov.json"
    _write_minimal_provenance(p, sigma_floor=1e-20)
    stats = load_frozen_zscore_stats(p)
    assert stats.sigma_floor == pytest.approx(1e-20)


def test_apply_works_when_sigma_near_floor() -> None:
    """A near-floor (but > 0) sigma should still produce a finite z-score.

    The emit step substitutes sigma=1.0 when the raw sigma < sigma_floor, so
    in practice the frozen stats never carry a sub-floor sigma. This test
    pins the guarantee: sigma ~ floor still works.
    """
    stats = _make_stats(bp_sigma_val=1e-5)
    bp = np.full((2, XP_COEFF_LEN - 1), stats.coef_norm_bp_mean[0] + 1e-5)
    rp = np.full((2, XP_COEFF_LEN - 1), stats.coef_norm_rp_mean[0])
    bp_c0 = np.array([-14.0, -14.0])
    rp_c0 = np.array([-14.5, -14.5])
    bp_z, _, _, _ = apply_frozen_zscore(bp, rp, bp_c0, rp_c0, stats)
    np.testing.assert_allclose(bp_z, 1.0, atol=1e-8)


# ---- load-time validation ----------------------------------------------------


def test_load_raises_on_missing_extra_block(tmp_path: Path) -> None:
    p = tmp_path / "prov.json"
    p.write_text(json.dumps({"output_file": "foo"}))
    with pytest.raises(KeyError, match="missing 'extra' block"):
        load_frozen_zscore_stats(p)


def test_load_raises_on_missing_fingerprint(tmp_path: Path) -> None:
    p = tmp_path / "prov.json"
    p.write_text(json.dumps({"extra": {"c0_zscore_frozen": {}}}))
    with pytest.raises(KeyError, match="basis_fingerprint_sha256"):
        load_frozen_zscore_stats(p)


def test_load_raises_on_missing_coefficient(tmp_path: Path) -> None:
    """Truncated coef_norm_zscore_frozen.bp should raise a clear KeyError."""
    p = tmp_path / "prov.json"
    payload = {
        "extra": {
            "basis_fingerprint_sha256": "a" * 64,
            "c0_zscore_frozen": {
                "bp": {"mu_log10": -14.0, "sigma_log10": 0.5},
                "rp": {"mu_log10": -14.5, "sigma_log10": 0.5},
            },
            "coef_norm_zscore_frozen": {
                "bp": {str(i): {"mu": 0.0, "sigma": 0.1} for i in range(1, 30)},  # truncated
                "rp": {
                    str(i): {"mu": 0.0, "sigma": 0.1}
                    for i in range(1, XP_COEFF_LEN)
                },
            },
        },
    }
    p.write_text(json.dumps(payload))
    with pytest.raises(KeyError, match="missing coefficient"):
        load_frozen_zscore_stats(p)
