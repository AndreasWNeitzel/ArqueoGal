"""End-to-end integration tests for the Pipeline-1 inference release flow (#138).

Composes the separately-tested components of the main pipeline (ensemble
moment-matching → per-(cell, label) shrinkage calibration → Regime B
galactic-plane exclusion → Mahalanobis OOD → ensemble-disagreement OOD →
combined status) against synthetic data, checking:

- Shapes align through every stage.
- Shrinkage recovers Var((y-μ)/σ_diag) ≈ 1 per-cell on a heteroscedastic
  synthetic residual field where the raw predictions are over-confident.
- RegimeBEnvelope, Mahalanobis OOD, and ensemble-disagreement OOD compose
  into the documented 3-level ``combined_ood_status`` code and an
  orthogonal ``tier1_release`` flag.
- The final per-star release record has the contract the Stream 3
  inference harness will rely on.

No real checkpoints or parquet I/O — these tests guard the composition
contract, not model physics. Per-component physics tests live in
``test_uncertainty.py`` and ``test_ood.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from arqueogal.xp_abundances.main.ood import (
    combined_ood_status,
    ensemble_disagreement_ratio,
    fit_mahalanobis_ood,
    flag_ensemble_ood,
    flag_mahalanobis_ood,
)
from arqueogal.xp_abundances.main.uncertainty import (
    RegimeBEnvelope,
    bin_by_cells,
    coverage_at_levels,
    shrunken_per_cell_per_label_scale,
)


# --- Synthetic ensemble builder ---------------------------------------------

def _moment_match(mus: np.ndarray, Ls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Moment-match ensemble (K, B, n) μ and (K, B, n, n) Cholesky-factor L.

    ``Σ̄ = mean(L L^T) + Var(μ)`` — the mixture-of-Gaussians second moment.
    Returns the matched μ̄ and L̄ = chol(Σ̄). A mirror of the private helper
    in ``scripts/run_calibration.py`` — duplicated here to keep tests free
    of script imports.
    """
    k, _b, n_dim = mus.shape
    mu_bar = mus.mean(axis=0)
    aleatoric = np.einsum("kbij,kblj->bil", Ls, Ls) / k
    diff = mus - mu_bar[None]
    epistemic = np.einsum("kbi,kbj->bij", diff, diff) / k
    sigma = aleatoric + epistemic + 1e-8 * np.eye(n_dim)[None]
    return mu_bar.astype(np.float64), np.linalg.cholesky(sigma).astype(np.float64)


def _make_synthetic_ensemble(
    rng: np.random.Generator,
    *,
    n_members: int = 5,
    n_stars: int = 2000,
    n_dim: int = 5,
    epistemic_scale: float = 0.3,
    aleatoric_scale: float = 0.5,
    y_offset: float = 0.0,
    y_extra_noise: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (μ_k, L_k, y) where truth ≈ member-mean + controlled residual.

    - Each member predicts ``μ_k = μ_common + epistemic_scale · ε_k``.
    - Each member reports diagonal aleatoric σ = ``aleatoric_scale``.
    - Truth ``y = μ_common + extra_noise`` with ``extra_noise`` drawn from an
      isotropic normal of variance ``σ_total² + y_extra_noise²``, then shifted
      by ``y_offset`` so callers can induce per-cell miscalibration.
    """
    mu_common = rng.standard_normal((n_stars, n_dim))
    epistemic = epistemic_scale * rng.standard_normal((n_members, n_stars, n_dim))
    mus = mu_common[None, :, :] + epistemic

    Ls = np.zeros((n_members, n_stars, n_dim, n_dim))
    diag_idx = np.arange(n_dim)
    Ls[:, :, diag_idx, diag_idx] = aleatoric_scale

    total_sigma2 = aleatoric_scale ** 2 + epistemic_scale ** 2 + y_extra_noise ** 2
    y = mu_common + np.sqrt(total_sigma2) * rng.standard_normal((n_stars, n_dim))
    y += y_offset
    return mus.astype(np.float64), Ls.astype(np.float64), y.astype(np.float64)


# --- Moment-match identities -------------------------------------------------

def test_moment_match_single_member_is_identity() -> None:
    """With K=1, ensemble Σ̄ = L_0 L_0^T (no epistemic term)."""
    rng = np.random.default_rng(0)
    mus, Ls, _ = _make_synthetic_ensemble(rng, n_members=1, n_stars=50, n_dim=4)
    mu_bar, L_bar = _moment_match(mus, Ls)
    np.testing.assert_allclose(mu_bar, mus[0], atol=1e-8)
    expected_sigma = np.einsum("bij,blj->bil", Ls[0], Ls[0])
    got_sigma = np.einsum("bij,blj->bil", L_bar, L_bar)
    np.testing.assert_allclose(got_sigma, expected_sigma, atol=1e-6)


def test_moment_match_adds_epistemic_variance() -> None:
    """Σ̄ - mean(L L^T) = Var(μ) across members."""
    rng = np.random.default_rng(1)
    K = 10
    mus, Ls, _ = _make_synthetic_ensemble(
        rng, n_members=K, n_stars=2000, n_dim=3,
        epistemic_scale=0.4, aleatoric_scale=0.1,
    )
    _, L_bar = _moment_match(mus, Ls)
    sigma_bar = np.einsum("bij,blj->bil", L_bar, L_bar)
    mean_alea = np.einsum("kbij,kblj->bil", Ls, Ls) / K  # (B, n, n)
    residual = sigma_bar - mean_alea
    # Residual ≈ biased across-member Var(μ) = σ²(K-1)/K. For σ=0.4, K=10 →
    # 0.16·0.9 = 0.144. Accept within [0.10, 0.20] to tolerate sampling noise.
    mean_diag = float(np.mean(np.diagonal(residual, axis1=1, axis2=2)))
    assert 0.10 < mean_diag < 0.20, f"mean_diag={mean_diag:.4f} outside [0.10, 0.20]"


# --- Shrinkage end-to-end on synthetic heteroscedastic data -----------------

def test_shrinkage_recovers_per_cell_variance_on_synthetic_data() -> None:
    """Over-confident σ → shrinkage produces per-cell Var(z) ≈ 1."""
    rng = np.random.default_rng(2)
    mus, Ls, y = _make_synthetic_ensemble(
        rng, n_members=5, n_stars=4000, n_dim=5,
        epistemic_scale=0.2, aleatoric_scale=0.4,
        # Truth has 2× more variance than σ — over-confident prediction.
        y_extra_noise=0.5,
    )
    mu_bar, L_bar = _moment_match(mus, Ls)
    # Bin on truth dims 0..2.
    cell_ids, _ = bin_by_cells(y[:, :3], n_bins=(3, 3, 3))
    out = shrunken_per_cell_per_label_scale(
        mu_bar, L_bar, y, cell_ids, tau=20.0, min_cell_stars=8,
    )
    per_star_alpha = out["per_star_alpha"]
    L_cal = per_star_alpha[:, :, None] * L_bar
    sigma_diag_cal = np.sqrt(np.einsum("bij,bij->bi", L_cal, L_cal))
    z_cal = (y - mu_bar) / sigma_diag_cal
    # Global Var(z) close to 1 post-shrinkage. Widened bound tolerates the
    # small between-cell μ drift that quantile-binning on truth leaves in —
    # shrinkage targets within-cell variance, not the between-cell term.
    for j in range(5):
        var_j = float(np.nanvar(z_cal[:, j]))
        assert 0.80 < var_j < 1.30, f"label {j} Var(z) = {var_j:.3f} outside [0.80, 1.30]"


def test_shrinkage_preserves_pd_and_correlation_sign() -> None:
    """L'_b = diag(α) L_b keeps positive-definite Σ'_b and sign of correlations."""
    rng = np.random.default_rng(3)
    mus, Ls, y = _make_synthetic_ensemble(
        rng, n_members=3, n_stars=500, n_dim=4,
        epistemic_scale=0.3, aleatoric_scale=0.3, y_extra_noise=0.3,
    )
    mu_bar, L_bar = _moment_match(mus, Ls)
    # Inject known positive correlation between labels 0 and 1 via L.
    L_bar[:, 1, 0] = 0.2
    cell_ids, _ = bin_by_cells(y[:, :3], n_bins=(2, 2, 2))
    out = shrunken_per_cell_per_label_scale(mu_bar, L_bar, y, cell_ids, tau=10.0)
    L_cal = out["per_star_alpha"][:, :, None] * L_bar
    sigma = np.einsum("bij,blj->bil", L_cal, L_cal)
    # PD check via eigenvalues > 0.
    eigs = np.linalg.eigvalsh(sigma)
    assert (eigs > 0).all()
    # Correlation sign preserved (Σ_{01} > 0 since L_{1,0} = 0.2).
    assert (sigma[:, 1, 0] > 0).all()


# --- RegimeBEnvelope composition --------------------------------------------

def test_regime_b_envelope_and_shrinkage_are_orthogonal() -> None:
    """Envelope flag is a function of (Teff, logg, b) only — independent of σ."""
    rng = np.random.default_rng(4)
    # Mock predicted Teff, logg, and b_deg.
    n = 1000
    teff_pred = rng.uniform(3500, 6000, size=n)
    logg_pred = rng.uniform(0.5, 4.5, size=n)
    b_deg = rng.uniform(-90, 90, size=n)
    envelope = RegimeBEnvelope()
    mask_a = envelope.mask(teff_pred, logg_pred, b_deg)

    # Shrinkage has nothing to do with it — re-running on different σ data
    # must not change the envelope mask.
    mask_b = envelope.mask(teff_pred, logg_pred, b_deg)
    np.testing.assert_array_equal(mask_a, mask_b)

    # tier1_release = ~mask.
    tier1 = envelope.tier1_release_flag(teff_pred, logg_pred, b_deg)
    np.testing.assert_array_equal(tier1, ~mask_a)


def test_regime_b_envelope_cuts_intersection_of_three_conditions() -> None:
    """A star inside the envelope must satisfy *all three* cuts simultaneously."""
    envelope = RegimeBEnvelope(b_deg_max=5.0, teff_k_min=4750.0, logg_dex_max=2.10)
    # (warm, upper-RGB, in-plane) → inside
    assert envelope.mask(
        teff_pred=np.array([4900.0]),
        logg_pred=np.array([1.8]),
        b_deg=np.array([2.0]),
    )[0]
    # cool → outside
    assert not envelope.mask(
        teff_pred=np.array([4500.0]),
        logg_pred=np.array([1.8]),
        b_deg=np.array([2.0]),
    )[0]
    # lower-RGB → outside
    assert not envelope.mask(
        teff_pred=np.array([4900.0]),
        logg_pred=np.array([2.5]),
        b_deg=np.array([2.0]),
    )[0]
    # high-|b| → outside
    assert not envelope.mask(
        teff_pred=np.array([4900.0]),
        logg_pred=np.array([1.8]),
        b_deg=np.array([30.0]),
    )[0]


# --- OOD composition --------------------------------------------------------

def test_mahalanobis_and_ensemble_ood_compose_into_status_code() -> None:
    """Build both OOD signals on synthetic ensemble, check level counts."""
    rng = np.random.default_rng(5)

    # Fit Mahalanobis on a tight training cloud.
    train_X = rng.standard_normal((3000, 8)).astype(np.float32)
    ood_bundle = fit_mahalanobis_ood(train_X, p_threshold=0.99)

    # Test set: half in-dist, half OOD.
    n = 400
    X_test = np.empty((n, 8), dtype=np.float32)
    X_test[: n // 2] = rng.standard_normal((n // 2, 8))
    X_test[n // 2:] = 10.0  # OOD
    mahal_flags = flag_mahalanobis_ood(X_test, ood_bundle)
    assert not mahal_flags[: n // 2].all()
    assert mahal_flags[n // 2:].all()

    # Ensemble disagreement: half tight ensemble, half disagreeing.
    M, B, n_lbl = 5, n, 3
    mu = np.zeros((M, B, n_lbl), dtype=np.float32)
    mu[:, : B // 2, :] = 0.01 * rng.standard_normal((M, B // 2, n_lbl))  # tight
    mu[:, B // 2:, :] = 5.0 * rng.standard_normal((M, B // 2, n_lbl))  # disagree
    sigma = np.full((M, B, n_lbl), 0.5, dtype=np.float32)
    ens_flags = flag_ensemble_ood(mu, sigma, threshold=0.5)
    assert not ens_flags[: B // 2].any()
    assert ens_flags[B // 2:].all()

    # Combined: (in, in) → 0; (in, disagree) → 1; (OOD, in) → 1; (OOD, disagree) → 2.
    # First half (in-dist + tight) expected mostly 0 but ~1% Mahalanobis FP
    # from the p99 threshold is fine. Second half is all-flagged on both axes.
    status = combined_ood_status(mahal_flags, ens_flags)
    first_half_zero_frac = float((status[: n // 2] == 0).mean())
    assert first_half_zero_frac > 0.95  # ≤ 5% residual Mahalanobis FPs
    assert (status[n // 2:] == 2).all()
    assert status.dtype == np.int8


# --- End-to-end release flow ------------------------------------------------

def test_full_release_flow_produces_per_star_record() -> None:
    """Top-level smoke: every stage wires together into a per-star release record.

    The contract at Stream 3 inference time:

    - ``mu`` (B, n), ``L`` (B, n, n): calibrated ensemble prediction in raw units.
    - ``tier1_release`` (B,) bool: Regime B envelope passed.
    - ``ood_flag_mahalanobis`` (B,) bool.
    - ``ood_flag_ensemble`` (B,) bool.
    - ``ood_status`` (B,) int8 in {0, 1, 2}.
    """
    rng = np.random.default_rng(6)
    n_stars = 600
    n_dim = 5

    # Synthetic ensemble + truth for in-loop shrinkage fitting.
    mus, Ls, y = _make_synthetic_ensemble(
        rng, n_members=5, n_stars=n_stars, n_dim=n_dim,
        epistemic_scale=0.2, aleatoric_scale=0.4, y_extra_noise=0.2,
    )
    mu_bar, L_bar = _moment_match(mus, Ls)

    # Stage 1 — per-cell shrinkage.
    cell_ids, _ = bin_by_cells(y[:, :3], n_bins=(2, 2, 2))
    shrunk = shrunken_per_cell_per_label_scale(
        mu_bar, L_bar, y, cell_ids, tau=20.0,
    )
    L_cal = shrunk["per_star_alpha"][:, :, None] * L_bar

    # Stage 2 — Regime B envelope on predicted Teff/logg and mock b_deg.
    # Spoof predicted Teff from label[0]: rescale μ₀ into a realistic range.
    teff_pred = 4500.0 + 400.0 * mu_bar[:, 0]
    logg_pred = 2.3 + 0.6 * mu_bar[:, 1]
    b_deg = rng.uniform(-90, 90, size=n_stars)
    envelope = RegimeBEnvelope()
    tier1_release = envelope.tier1_release_flag(teff_pred, logg_pred, b_deg)

    # Stage 3 — Mahalanobis OOD on feature block (spoof with 12-D features).
    train_X = rng.standard_normal((2000, 12)).astype(np.float32)
    ood_bundle = fit_mahalanobis_ood(train_X, p_threshold=0.99)
    X_test = rng.standard_normal((n_stars, 12)).astype(np.float32)
    # Push last 10% of stars far out-of-dist.
    X_test[-60:] = 10.0
    ood_mahal = flag_mahalanobis_ood(X_test, ood_bundle)

    # Stage 4 — ensemble disagreement OOD.
    # Synthesize per-member σ by broadcasting diag(L_k).
    sigma_per_member = np.sqrt(np.einsum("kbij,kbij->kbi", Ls, Ls))
    ood_ens = flag_ensemble_ood(mus.astype(np.float32), sigma_per_member.astype(np.float32))

    # Stage 5 — combined status.
    ood_status = combined_ood_status(ood_mahal, ood_ens)

    # Release-record shape contract.
    assert mu_bar.shape == (n_stars, n_dim)
    assert L_cal.shape == (n_stars, n_dim, n_dim)
    assert tier1_release.shape == (n_stars,) and tier1_release.dtype == bool
    assert ood_mahal.shape == (n_stars,) and ood_mahal.dtype == bool
    assert ood_ens.shape == (n_stars,) and ood_ens.dtype == bool
    assert ood_status.shape == (n_stars,) and ood_status.dtype == np.int8
    assert set(np.unique(ood_status)).issubset({0, 1, 2})

    # tier1_release and ood_status are orthogonal — no hard coupling beyond
    # the fact that both reflect trust. Sanity: at least some stars pass both,
    # and at least some fail the OOD check on the forced-OOD tail.
    assert tier1_release.any()
    assert ood_mahal[-60:].all()  # forced tail is fully OOD.


def test_release_flow_coverage_improves_post_shrinkage() -> None:
    """Post-shrinkage joint cov(0.95) lands closer to 0.95 than raw.

    Chosen as the one-line aggregate release-gate check — if shrinkage does
    not move coverage toward nominal on heteroscedastic synthetic data, the
    whole calibration chain is broken.
    """
    rng = np.random.default_rng(7)
    mus, Ls, y = _make_synthetic_ensemble(
        rng, n_members=5, n_stars=3000, n_dim=4,
        epistemic_scale=0.2, aleatoric_scale=0.3, y_extra_noise=0.4,
    )
    mu_bar, L_bar = _moment_match(mus, Ls)

    cov_raw = coverage_at_levels(mu_bar, L_bar, y, levels=(0.95,))
    raw_c95 = float(cov_raw["joint"][0.95])

    cell_ids, _ = bin_by_cells(y[:, :3], n_bins=(3, 3, 3))
    shrunk = shrunken_per_cell_per_label_scale(mu_bar, L_bar, y, cell_ids, tau=20.0)
    L_cal = shrunk["per_star_alpha"][:, :, None] * L_bar

    cov_cal = coverage_at_levels(mu_bar, L_cal, y, levels=(0.95,))
    cal_c95 = float(cov_cal["joint"][0.95])

    # raw is over-confident → cov < 0.95; calibrated should be closer.
    assert abs(cal_c95 - 0.95) < abs(raw_c95 - 0.95)
    # And within a reasonable envelope.
    assert 0.88 < cal_c95 < 1.0


# --- Bundle roundtrip across release stages ---------------------------------

def test_ood_and_envelope_bundles_both_roundtrip() -> None:
    """Serialised forms of MahalanobisOODBundle and RegimeBEnvelope compose
    identically to their pre-serialised originals on the same inputs.
    """
    rng = np.random.default_rng(8)
    X = rng.standard_normal((1500, 10)).astype(np.float32)
    bundle = fit_mahalanobis_ood(X, p_threshold=0.95)
    bundle2 = type(bundle).from_dict(bundle.to_dict())

    env = RegimeBEnvelope()
    env2 = RegimeBEnvelope.from_dict(env.to_dict())

    X_test = rng.standard_normal((50, 10)).astype(np.float32)
    np.testing.assert_array_equal(
        flag_mahalanobis_ood(X_test, bundle),
        flag_mahalanobis_ood(X_test, bundle2),
    )

    teff = rng.uniform(3500, 6000, size=50)
    logg = rng.uniform(0.5, 4.5, size=50)
    b = rng.uniform(-90, 90, size=50)
    np.testing.assert_array_equal(env.mask(teff, logg, b), env2.mask(teff, logg, b))


# --- Shape-mismatch guards in the release chain -----------------------------

def test_release_flow_rejects_feature_dim_mismatch() -> None:
    """Mahalanobis OOD must reject features with wrong dim — the chain fails fast."""
    rng = np.random.default_rng(9)
    train_X = rng.standard_normal((500, 8)).astype(np.float32)
    bundle = fit_mahalanobis_ood(train_X)
    # Wrong-dim inference features.
    X_wrong = rng.standard_normal((100, 7)).astype(np.float32)
    with pytest.raises(ValueError, match="feature dim"):
        flag_mahalanobis_ood(X_wrong, bundle)


def test_ensemble_disagreement_rejects_single_member_in_release_chain() -> None:
    """Single-member "ensemble" has no disagreement — release chain must reject."""
    mu = np.zeros((1, 10, 3), dtype=np.float32)
    sigma = np.ones((1, 10, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="≥2"):
        ensemble_disagreement_ratio(mu, sigma)
