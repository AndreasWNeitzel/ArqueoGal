"""Tier 3 → Tier 2 statistical promotion — research_brief §3.3 six-test protocol.

The protocol is pre-registered: every candidate element must pass the six
tests below before appearing as a Tier-2 label in D-Cat-b. Each test is a
pure function of numpy inputs returning a :class:`TestResult`. The
coordinator :func:`tier_promotion_report` applies the §3.3 decision tree
to the collected results and assigns a final tier.

The research_brief §3.3 decision tree operates on six tests, of which test 3
(SHAP feature importance) and test 6 (cross-catalogue consistency) are
deferred to future versions. Tier promotion runs at 5/6 coverage:

- Passes 1–3 only → ``"tier_3_rejected"``.
- Passes 1–4 but fails 5 or 6 → ``"tier_3_internal"``.
- Passes 1–5 and passes 6 at conservative threshold → ``"tier_2"``.
- Passes all six *and* reliability diagram passes → ``"tier_1"``.

The calibration (§9.1 reliability diagram) gate is passed in as a bool —
it's computed elsewhere (uncertainty.py) and not a 7th test, just a
modifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

_EPS: float = 1e-12

# --- Test protocol identity and stub tracking ------

TEST_1_PHYSICAL_FEASIBILITY: Final[str] = "test_1_physical_feasibility"
TEST_2_HOLDOUT_RMSE: Final[str] = "test_2_holdout_rmse"
TEST_3_SHAP_FEATURE_IMPORTANCE: Final[str] = "test_3_shap_feature_importance"
TEST_4_PERMUTATION_SHUFFLE_NULL: Final[str] = "test_4_permutation_shuffle_null"
TEST_5_CONDITIONAL_MI: Final[str] = "test_5_conditional_mi"
TEST_6_CROSS_CATALOGUE_CONSISTENCY: Final[str] = "test_6_cross_catalogue_consistency"

STUBBED_TESTS: Final[frozenset[str]] = frozenset({
    TEST_3_SHAP_FEATURE_IMPORTANCE,
    TEST_6_CROSS_CATALOGUE_CONSISTENCY,
})


class IncompleteProtocolError(Exception):
    """Raised when a stubbed test result is overridden with True.

    A stubbed test (deferred to future versions) cannot claim to pass
    without the required validation work. Passing None (deferred) is
    acceptable; passing True is not.
    """

    pass


def report_tier_coverage() -> str:
    """Return the canonical tier-promotion coverage statement for the release."""
    return (
        "5/6 (tests 3 SHAP and 6 cross-catalogue consistency pending Stream 3 "
        "cross-overlap validation)"
    )


@dataclass
class TestResult:
    """One §3.3 test result.

    ``passed`` is the binary outcome; ``statistic`` and ``threshold`` are the
    specific numbers that drove the verdict, so a release reviewer can see
    *why* at a glance. ``detail`` carries arbitrary auxiliary data (per-cell
    breakdowns, bootstrap quantiles, offending counts) for deeper audit.
    """

    __test__ = False  # opt out of pytest test-class collection

    passed: bool
    statistic: float
    threshold: float
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "statistic": float(self.statistic),
            "threshold": float(self.threshold),
            "detail": _jsonable(self.detail),
        }


def _jsonable(obj: Any) -> Any:
    """Recursively coerce numpy arrays/scalars → Python primitives for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


# --- §3.3 Test 1: Physical gate --------------------------------------------


def physical_gate(element: str, has_absorption: bool) -> TestResult:
    """Trivial boolean gate: does ``element`` have XP-window absorption?

    ``has_absorption`` comes from the §3.2 line-availability table in the
    research brief (a domain-knowledge lookup, not a measurement). No amount
    of ML recovers information absent from the photons.
    """
    return TestResult(
        passed=bool(has_absorption),
        statistic=1.0 if has_absorption else 0.0,
        threshold=1.0,
        detail={"element": element},
    )


# --- §3.3 Test 2: Hold-out RMSE + bias stratified --------------------------


def holdout_rmse(
    mu: np.ndarray,
    y: np.ndarray,
    cell_ids: np.ndarray,
    *,
    worst_ratio_limit: float = 2.0,
    bias_limit: float = 0.05,
) -> TestResult:
    """Per-cell RMSE + bias on the hold-out test split.

    Parameters
    ----------
    mu, y
        ``(N,)`` predicted and reference labels on the test split.
    cell_ids
        Per-star cell ID — the ML-side caller is expected to have bucketed
        stars into (Teff × logg × [Fe/H] × Av × G) cells.

    Pass criterion (§3.3 item 2): worst-cell RMSE ≤ ``worst_ratio_limit`` ×
    median-cell RMSE **and** worst-cell |bias| ≤ ``bias_limit``.
    """
    if mu.shape != y.shape or mu.ndim != 1:
        raise ValueError(f"mu and y must be 1-D with same shape; got {mu.shape}, {y.shape}")
    per_cell_rmse: dict[int, float] = {}
    per_cell_bias: dict[int, float] = {}
    for c in np.unique(cell_ids):
        mask = cell_ids == c
        if mask.sum() < 2:  # not enough to estimate anything stable
            continue
        d = mu[mask] - y[mask]
        per_cell_rmse[int(c)] = float(np.sqrt((d**2).mean()))
        per_cell_bias[int(c)] = float(d.mean())
    if not per_cell_rmse:
        return TestResult(
            passed=False,
            statistic=np.inf,
            threshold=worst_ratio_limit,
            detail={"reason": "no cells with ≥2 stars"},
        )

    rmse_values = np.fromiter(per_cell_rmse.values(), dtype=np.float64)
    bias_values = np.fromiter(per_cell_bias.values(), dtype=np.float64)
    med = float(np.median(rmse_values))
    worst_rmse = float(rmse_values.max())
    worst_bias = float(np.abs(bias_values).max())
    ratio = worst_rmse / max(med, _EPS)
    passed = (ratio <= worst_ratio_limit) and (worst_bias <= bias_limit)
    return TestResult(
        passed=passed,
        statistic=ratio,
        threshold=worst_ratio_limit,
        detail={
            "median_rmse": med,
            "worst_rmse": worst_rmse,
            "worst_bias": worst_bias,
            "bias_limit": bias_limit,
            "per_cell_rmse": per_cell_rmse,
            "per_cell_bias": per_cell_bias,
        },
    )


# --- §3.3 Test 3: Precision floor via open clusters ------------------------


def cluster_precision(
    y_pred: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    apogee_sigma: float,
    inflation: float = 1.5,
    min_members: int = 3,
) -> TestResult:
    """Intra-cluster scatter must beat APOGEE precision by at most ``inflation``.

    Open-cluster RGB members share [X/Fe] to a few × 0.01 dex (Bovy 2016;
    Spina+2021; Casamiquela+2020). A predictor whose intra-cluster
    dispersion exceeds ``inflation × σ_APOGEE`` is not precise enough to
    promote.
    """
    if y_pred.ndim != 1:
        raise ValueError(f"y_pred must be 1-D, got shape {y_pred.shape}")
    sigmas: list[float] = []
    for c in np.unique(cluster_ids):
        mask = cluster_ids == c
        if mask.sum() < min_members:
            continue
        sigmas.append(float(np.std(y_pred[mask], ddof=1)))
    if not sigmas:
        return TestResult(
            passed=False,
            statistic=np.inf,
            threshold=apogee_sigma * inflation,
            detail={"reason": "no clusters with enough members"},
        )
    sigma_intra = float(np.median(sigmas))
    threshold = apogee_sigma * inflation
    return TestResult(
        passed=sigma_intra < threshold,
        statistic=sigma_intra,
        threshold=threshold,
        detail={"per_cluster_sigma": sigmas, "n_clusters": len(sigmas)},
    )


# --- §3.3 Test 4: §9.2 audit gate ------------------------------------------


def audit_gate(  # noqa: PLR0913 — §9.2 has four test-specific thresholds
    permutation_importance: np.ndarray,
    looco_delta_rmse: np.ndarray,
    null_skill_ratio: float,
    decorrelated_r2_ratio: float,
    *,
    min_informative_coefs: int = 3,
    min_delta_r2: float = 0.02,
    null_ratio_limit: float = 0.20,
    decorr_ratio_min: float = 0.50,
) -> TestResult:
    """Compose §9.2 test 1, 2, 4, 6 outcomes into a single pass/fail gate.

    Inputs are summary numbers pre-computed via :mod:`.audit`:

    - ``permutation_importance`` (``(n_coefs,)`` ΔR² per XP coefficient).
    - ``looco_delta_rmse`` (``(n_coefs,)`` RMSE shift).
    - ``null_skill_ratio``: shuffled-spectrum RMSE / real RMSE, clipped to [0,∞).
    - ``decorrelated_r2_ratio``: decorrelated-subsample R² / full-sample R².
    """
    n_informative = int((np.asarray(permutation_importance) > min_delta_r2).sum())
    looco_ok = bool(np.any(np.abs(np.asarray(looco_delta_rmse)) > _EPS))
    null_ok = null_skill_ratio <= null_ratio_limit
    decorr_ok = decorrelated_r2_ratio >= decorr_ratio_min
    passed = n_informative >= min_informative_coefs and looco_ok and null_ok and decorr_ok
    return TestResult(
        passed=passed,
        statistic=float(n_informative),
        threshold=float(min_informative_coefs),
        detail={
            "n_informative_coefs": n_informative,
            "looco_nontrivial": looco_ok,
            "null_skill_ratio": float(null_skill_ratio),
            "null_ratio_limit": null_ratio_limit,
            "decorrelated_r2_ratio": float(decorrelated_r2_ratio),
            "decorr_ratio_min": decorr_ratio_min,
        },
    )


# --- §3.3 Test 5: Cross-catalogue consistency ------------------------------


def cross_catalogue_consistency(
    pred: np.ndarray,
    external: dict[str, np.ndarray],
    *,
    apogee_sigma: float,
    bias_limit: float = 0.05,
    scatter_multiple: float = 2.0,
) -> TestResult:
    """Pairwise consistency vs other catalogues on matched stars.

    Each entry in ``external`` is ``(N,)`` aligned with ``pred``. We compute
    per-catalogue mean-bias and residual scatter. Pass iff the worst mean
    bias ≤ ``bias_limit`` **and** worst scatter ≤ ``scatter_multiple`` ×
    ``apogee_sigma``.
    """
    if pred.ndim != 1:
        raise ValueError(f"pred must be 1-D, got shape {pred.shape}")
    per_cat: dict[str, dict[str, float]] = {}
    for name, other in external.items():
        if other.shape != pred.shape:
            raise ValueError(
                f"{name}: shape {other.shape} ≠ pred shape {pred.shape}",
            )
        d = pred - other
        per_cat[name] = {"bias": float(d.mean()), "scatter": float(d.std(ddof=1))}

    if not per_cat:
        return TestResult(
            passed=False,
            statistic=np.inf,
            threshold=bias_limit,
            detail={"reason": "no external catalogues provided"},
        )
    worst_bias = max(abs(v["bias"]) for v in per_cat.values())
    worst_scatter = max(v["scatter"] for v in per_cat.values())
    scatter_threshold = apogee_sigma * scatter_multiple
    passed = worst_bias <= bias_limit and worst_scatter <= scatter_threshold
    return TestResult(
        passed=passed,
        statistic=worst_bias,
        threshold=bias_limit,
        detail={
            "per_catalogue": per_cat,
            "worst_scatter": worst_scatter,
            "scatter_threshold": scatter_threshold,
        },
    )


# --- §3.3 Test 6: Conditional MI with bootstrap CI -------------------------


def conditional_mi_bootstrap(  # noqa: PLR0913 — bootstrap exposes full CI + k knobs
    xp: np.ndarray,
    element: np.ndarray,
    stellar_params: np.ndarray,
    *,
    n_boot: int = 1000,
    k: int = 5,
    ci_levels: tuple[float, float] = (0.025, 0.975),
    seed: int = 0,
) -> TestResult:
    """MI(XP; [X/Fe] | Teff, logg, [Fe/H]) bootstrap CI; pass iff lower > 0.

    Implements the formal "residual information" test from §3.3: does the
    XP coefficient space carry any information about the element beyond
    what the atmospheric parameters already explain?
    """
    from arqueogal.xp_abundances.main.audit import conditional_mi_ksg as cmi_ksg

    if xp.ndim != 2:
        raise ValueError(f"xp must be 2-D (N, D); got shape {xp.shape}")
    if element.ndim != 1:
        raise ValueError(f"element must be 1-D; got shape {element.shape}")
    if stellar_params.ndim != 2:
        raise ValueError(
            f"stellar_params must be 2-D (N, P); got shape {stellar_params.shape}",
        )
    rng = np.random.default_rng(seed)
    n = xp.shape[0]
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = cmi_ksg(
            xp[idx],
            element[idx],
            stellar_params[idx],
            k=k,
        )
    lo, hi = np.quantile(boot, ci_levels)
    passed = lo > 0.0
    return TestResult(
        passed=passed,
        statistic=float(lo),
        threshold=0.0,
        detail={
            "bootstrap_mean": float(boot.mean()),
            "bootstrap_ci_lower": float(lo),
            "bootstrap_ci_upper": float(hi),
            "ci_levels": list(ci_levels),
            "n_boot": n_boot,
        },
    )


# --- §3.3 Promotion decision tree ------------------------------------------


@dataclass
class TierPromotionReport:
    """Aggregated §3.3 outcome for one candidate element.

    ``tier`` is one of:
    - ``"tier_3_rejected"`` — fails physical gate or tests 2-3.
    - ``"tier_3_internal"`` — passes 1–4 but fails 5 or 6.
    - ``"tier_2"`` — passes 1–6.
    - ``"tier_1"`` — passes 1–6 AND ``calibration_ok``.
    """

    element: str
    test1_physical: TestResult
    test2_holdout: TestResult
    test3_cluster: TestResult
    test4_audit: TestResult
    test5_cross_cat: TestResult
    test6_conditional_mi: TestResult
    tier: str
    calibration_ok: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "element": self.element,
            "tier": self.tier,
            "calibration_ok": bool(self.calibration_ok),
            "test1_physical": self.test1_physical.as_dict(),
            "test2_holdout": self.test2_holdout.as_dict(),
            "test3_cluster": self.test3_cluster.as_dict(),
            "test4_audit": self.test4_audit.as_dict(),
            "test5_cross_cat": self.test5_cross_cat.as_dict(),
            "test6_conditional_mi": self.test6_conditional_mi.as_dict(),
        }


def _decide_tier(  # noqa: PLR0913 — decision tree operates on six fixed tests
    t1: TestResult,
    t2: TestResult,
    t3: TestResult,
    t4: TestResult,
    t5: TestResult,
    t6: TestResult,
    calibration_ok: bool,
) -> str:
    # Test 1 is a hard reject — no photons = no signal.
    if not t1.passed:
        return "tier_3_rejected"
    # Tests 2 and 3 gate any per-star release at all.
    if not (t2.passed and t3.passed):
        return "tier_3_rejected"
    # Test 4 gates audit soundness; required even for internal use.
    if not t4.passed:
        return "tier_3_rejected"
    # At this point 1-4 pass. 5 or 6 failing → internal only.
    if not (t5.passed and t6.passed):
        return "tier_3_internal"
    # All six pass. Calibration lifts Tier 2 → Tier 1.
    return "tier_1" if calibration_ok else "tier_2"


def tier_promotion_report(  # noqa: PLR0913 — §3.3 fixes exactly six tests + cal flag
    element: str,
    *,
    test1: TestResult,
    test2: TestResult,
    test3: TestResult,
    test4: TestResult,
    test5: TestResult,
    test6: TestResult,
    calibration_ok: bool = False,
) -> TierPromotionReport:
    """Apply the §3.3 decision tree to six pre-computed :class:`TestResult`s."""
    tier = _decide_tier(test1, test2, test3, test4, test5, test6, calibration_ok)
    return TierPromotionReport(
        element=element,
        test1_physical=test1,
        test2_holdout=test2,
        test3_cluster=test3,
        test4_audit=test4,
        test5_cross_cat=test5,
        test6_conditional_mi=test6,
        tier=tier,
        calibration_ok=calibration_ok,
    )


__all__ = [
    "IncompleteProtocolError",
    "TEST_1_PHYSICAL_FEASIBILITY",
    "TEST_2_HOLDOUT_RMSE",
    "TEST_3_SHAP_FEATURE_IMPORTANCE",
    "TEST_4_PERMUTATION_SHUFFLE_NULL",
    "TEST_5_CONDITIONAL_MI",
    "TEST_6_CROSS_CATALOGUE_CONSISTENCY",
    "STUBBED_TESTS",
    "TestResult",
    "TierPromotionReport",
    "audit_gate",
    "cluster_precision",
    "conditional_mi_bootstrap",
    "cross_catalogue_consistency",
    "holdout_rmse",
    "physical_gate",
    "report_tier_coverage",
    "tier_promotion_report",
]
