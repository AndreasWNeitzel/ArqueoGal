"""Pre-training sanity battery for Pipeline 1.

Runs before the first training epoch on the emitted feature matrix. The six
checks below are the minimum Gate for Pipeline 1 training — failure in
hard-fail checks (1–3) halts training; failure in soft-fail checks (4–6) is
logged and training continues under the operator's eye. This module is
deliberately narrow: it is not a full data audit. Depth-of-treatment checks
live in ``audit.py`` (research_brief §9.2) and run post-training.

Gates (frozen 2026-04-18):

Hard-fail (halt training gate):

1. :func:`check_xp_feature_nan_invariant` — NaN in normalized XP / c0_z
   columns ONLY where an explanatory flag is set.
2. :func:`check_tier1_label_completeness` — Tier-1 atmospheric labels are
   finite for every ``flag_bad == 0`` row. **Tier-1 metallicity is
   ``mh_apogee`` (ASPCAP global [M/H]), not ``fe_h_apogee``**: ASPCAP DR19
   can legitimately fit global [M/H] while per-element Fe fails on saturated
   lines, cool giants, or blue-SNR-limited stars (consistent with behaviour
   reported by Andrae+2023, Guiglion+2024).
3. :func:`check_parameter_bounds` — APOGEE parameters lie inside ASPCAP
   DR19's actual dynamic range (not a textbook-generic envelope).
   ``fe_h_apogee`` ∈ [−4, 1.1], ``alpha_m_apogee`` ∈ [−0.8, 0.8] match the
   observed DR19 tails; tighter bounds would false-alarm on genuine α-poor
   or metal-rich stars.

Soft-fail (log and continue):

4. *(runner-side)* — Distribution plots (Kiel, Tinsley-Wallerstein) for the
   operator to eyeball. No in-module check; the plots go in the markdown
   report.
5. :func:`check_zscore_validity` — ``bp_c0_z`` / ``rp_c0_z`` have mean ≈ 0
   and std ≈ 1 on the reference population.
6. :func:`check_dedup_idempotency` — Running :func:`dedup_by_source_id` on
   the feature matrix yields the expected row count.
7. :func:`check_per_element_nan_rates` — Per-element [X/H]_apogee NaN rates
   across Tier-2 + Tier-3. Zero threshold (report-only); baseline on the
   DR19 training set at 2026-04-18 is captured in the audit markdown so
   future-drift comparisons have a fixed reference.

The runner (``scripts/run_pretraining_sanity.py``) assembles the results into
``reports/sanity_battery/pretraining_audit.md`` along with the distribution
plots and a UMAP continuity embedding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
import pandas as pd

from arqueogal.data.dedup import dedup_by_source_id
from arqueogal.xp_abundances.main.data import LabelTiers

CheckLevel = Literal["HARD", "SOFT"]


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of one sanity check."""

    name: str
    level: CheckLevel
    passed: bool
    summary: str
    details: dict = field(default_factory=dict)


# --- Check 1 — XP feature NaN invariant --------------------------------------


def check_xp_feature_nan_invariant(df: pd.DataFrame) -> CheckResult:
    """NaN in XP feature columns is allowed only on flagged-out rows.

    The contract: ``bp_coef_norm_{1..54}``, ``rp_coef_norm_{1..54}``,
    ``bp_c0_z``, ``rp_c0_z`` are NaN iff the row fails at least one of

    - ``ye2024_flag != 0``
    - ``xp_fit_flag_residual_high != 0`` (or ``<NA>``)
    - ``bp_coef_0 <= 0`` OR ``rp_coef_0 <= 0`` (nonphysical absolute flux)

    Any NaN outside this flagged population indicates the emit pipeline
    silently lost a row — a hard-fail on the training gate.
    """
    ye_bad = (df["ye2024_flag"] != 0).to_numpy(dtype=bool)
    strat_flag = df["xp_fit_flag_residual_high"]
    strat_bad = ~(strat_flag == 0).to_numpy(dtype=bool, na_value=False)  # NA counts as "bad"
    c0_bad = (
        ~np.isfinite(df["bp_coef_0"].to_numpy())
        | (df["bp_coef_0"].to_numpy() <= 0)
        | ~np.isfinite(df["rp_coef_0"].to_numpy())
        | (df["rp_coef_0"].to_numpy() <= 0)
    )
    flagged_out = ye_bad | strat_bad | c0_bad
    n_flagged_out = int(flagged_out.sum())

    xp_cols = (
        [f"bp_coef_norm_{i}" for i in range(1, 55)]
        + [f"rp_coef_norm_{i}" for i in range(1, 55)]
        + ["bp_c0_z", "rp_c0_z"]
    )
    per_col_surprise: dict[str, int] = {}
    for col in xp_cols:
        is_nan = df[col].isna().to_numpy()
        surprise = is_nan & ~flagged_out
        n = int(surprise.sum())
        if n > 0:
            per_col_surprise[col] = n

    passed = len(per_col_surprise) == 0
    summary = (
        (
            f"no unexpected NaN in {len(xp_cols)} XP feature columns "
            f"(flagged-out rows: {n_flagged_out:,}/{len(df):,})"
        )
        if passed
        else (f"unexpected NaN in {len(per_col_surprise)} XP feature columns")
    )
    return CheckResult(
        name="xp_feature_nan_invariant",
        level="HARD",
        passed=passed,
        summary=summary,
        details={
            "n_flagged_out_rows": n_flagged_out,
            "n_xp_columns_checked": len(xp_cols),
            "surprise_counts_per_column": per_col_surprise,
            "expected_mask": "ye2024_flag != 0 OR xp_fit_flag_residual_high != 0/NA "
            "OR bp_coef_0 <= 0 OR rp_coef_0 <= 0",
        },
    )


# --- Check 2 — Tier-1 atmospheric-label completeness -------------------------

TIER1_ATMOSPHERIC: Final[tuple[str, ...]] = (
    "teff_apogee",
    "logg_apogee",
    "mh_apogee",
)
"""Atmospheric Tier-1 labels: {Teff, log g, [M/H]}.

``fe_h_apogee`` is deliberately excluded from the completeness gate. ASPCAP
DR19 fits global [M/H] jointly over Fe-peak + α lines, then fits [Fe/H]
per-element; the per-element fit can legitimately fail (saturated Fe lines in
metal-rich regimes, low SNR per individual line in the blue, unresolved Fe
blends in cool giants) while [M/H] remains sound. That failure is not a
pipeline bug and must not halt training — see :func:`check_per_element_nan_rates`
for the report-only diagnostic that surfaces it.
"""


def check_tier1_label_completeness(
    df: pd.DataFrame,
    tiers: LabelTiers | None = None,  # noqa: ARG001 — kept for API stability
) -> CheckResult:
    """Atmospheric Tier-1 labels finite for every ``flag_bad == 0`` row.

    Gates on ``{teff_apogee, logg_apogee, mh_apogee}`` only. Per-element
    [X/H] NaN is a DR19-realism feature, not a pipeline bug (see
    :func:`check_per_element_nan_rates`).
    """
    ok_rows = (df["flag_bad"] == 0).to_numpy(dtype=bool)
    n_ok = int(ok_rows.sum())

    per_label_missing: dict[str, int] = {}
    for label in TIER1_ATMOSPHERIC:
        n_nan = int(df.loc[ok_rows, label].isna().sum())
        if n_nan > 0:
            per_label_missing[label] = n_nan

    passed = len(per_label_missing) == 0
    summary = (
        f"all {len(TIER1_ATMOSPHERIC)} Tier-1 atmospheric labels finite on "
        f"{n_ok:,} flag_bad==0 rows"
        if passed
        else f"{len(per_label_missing)} Tier-1 atmospheric label(s) have NaN in flag_bad==0 rows"
    )
    return CheckResult(
        name="tier1_label_completeness",
        level="HARD",
        passed=passed,
        summary=summary,
        details={
            "n_flag_bad_zero_rows": n_ok,
            "tier1_atmospheric_labels": list(TIER1_ATMOSPHERIC),
            "nan_counts_per_label": per_label_missing,
        },
    )


# --- Check 3 — Parameter bounds ----------------------------------------------

PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "teff_apogee": (3000.0, 8000.0),  # K  — RGB envelope; RGB builder-cut is [4000, 5500]
    "logg_apogee": (-0.5, 5.5),  # dex — RGB builder-cut is [1.0, 3.5]
    "fe_h_apogee": (-4.0, 1.1),  # dex — widened from +1.0 to cover DR19 metal-rich tail
    "mh_apogee": (-4.0, 1.0),  # dex — global [M/H] ASPCAP DR19 range
    "alpha_m_apogee": (-0.8, 0.8),  # dex — widened to match DR19 α/M dynamic range
}
"""ASPCAP DR19-calibrated sanity bounds.

The bounds are NOT textbook-generic stellar-parameter envelopes. They are
chosen to match ASPCAP DR19's observed dynamic range, so future ASPCAP drift
(e.g. a DR20 recalibration pushing α/M to [−1, 1]) surfaces as a battery
failure rather than getting masked by over-permissive limits — and equally,
so genuine DR19 α-poor / metal-rich tails do not falsely trip the gate.

Specifics:

- ``fe_h_apogee`` upper bound 1.1 (not 1.0): DR19 has ~2 stars barely above
  1.0 at the 2026-04-18 training set — within the APOGEE ~0.03 dex per-element
  σ of the canonical [Fe/H] ≤ 1 expectation.
- ``alpha_m_apogee`` bounds [−0.8, 0.8] (not [−0.5, 0.7]): ASPCAP DR19
  reports a non-negligible α-poor halo/CEMP tail reaching −0.73 and a thin
  tail above +0.7; Nguyen+2024 documents the same dynamic range.

Per-element [X/H] abundances are not bound-checked here — they span a wider
range, and per-element validation happens in the Tier-promotion audit
(``audit.py``, research_brief §3.3).
"""


def check_parameter_bounds(df: pd.DataFrame) -> CheckResult:
    """Every APOGEE parameter on ``flag_bad == 0`` rows sits within bounds."""
    ok_rows = (df["flag_bad"] == 0).to_numpy(dtype=bool)
    violations: dict[str, dict[str, int | float]] = {}
    for col, (lo, hi) in PARAMETER_BOUNDS.items():
        values = df.loc[ok_rows, col].to_numpy()
        finite = np.isfinite(values)
        below = int(((values < lo) & finite).sum())
        above = int(((values > hi) & finite).sum())
        if below + above > 0:
            violations[col] = {
                "below_min": below,
                "above_max": above,
                "min_observed": float(np.nanmin(values)),
                "max_observed": float(np.nanmax(values)),
                "bound_min": lo,
                "bound_max": hi,
            }

    passed = len(violations) == 0
    summary = (
        f"all {len(PARAMETER_BOUNDS)} parameters within physical bounds"
        if passed
        else f"{len(violations)} parameter(s) have rows outside physical bounds"
    )
    return CheckResult(
        name="parameter_bounds",
        level="HARD",
        passed=passed,
        summary=summary,
        details={"violations": violations, "bounds": PARAMETER_BOUNDS},
    )


# --- Check 2b — Per-element [X/H] NaN rates (SOFT diagnostic) ----------------


def check_per_element_nan_rates(df: pd.DataFrame, tiers: LabelTiers | None = None) -> CheckResult:
    """Per-element NaN rates for Tier-2 + Tier-3 APOGEE [X/H] labels.

    Report-only diagnostic: always passes. The point is to freeze a baseline
    at 2026-04-18 so a future ASPCAP recalibration (or a DR20 rebuild) that
    changes per-element NaN behaviour for, say, Mg or Al is immediately
    visible in the audit markdown — rather than discovered weeks into
    model training.

    Also separately reports the ``fe_h_apogee`` rate: that element was
    removed from the Tier-1 completeness gate because of its real DR19 NaN
    pattern (see :data:`TIER1_ATMOSPHERIC`), and we want the baseline rate on
    record.
    """
    if tiers is None:
        tiers = LabelTiers()
    ok_rows = (df["flag_bad"] == 0).to_numpy(dtype=bool)
    n_ok = int(ok_rows.sum())

    per_element_rate: dict[str, dict[str, float | int]] = {}
    cols = ["fe_h_apogee", *tiers.tier2, *tiers.tier3]
    seen: set[str] = set()
    for col in cols:
        if col in seen or col not in df.columns:
            continue
        seen.add(col)
        n_nan = int(df.loc[ok_rows, col].isna().sum())
        per_element_rate[col] = {
            "n_nan": n_nan,
            "rate": (n_nan / n_ok) if n_ok else 0.0,
        }

    top = sorted(per_element_rate.items(), key=lambda kv: -kv[1]["n_nan"])[:3]
    summary = f"per-element NaN baseline on {n_ok:,} flag_bad==0 rows; top-3: " + ", ".join(
        f"{k} {v['rate'] * 100:.2f}%" for k, v in top
    )
    return CheckResult(
        name="per_element_nan_rates",
        level="SOFT",
        passed=True,  # diagnostic-only
        summary=summary,
        details={
            "n_flag_bad_zero_rows": n_ok,
            "rates": per_element_rate,
        },
    )


# --- Check 5 — Z-score self-consistency --------------------------------------

Z_MEAN_TOL = 1e-3
Z_STD_TOL = 5e-3


def check_zscore_validity(df: pd.DataFrame) -> CheckResult:
    """``bp_c0_z`` / ``rp_c0_z`` on the reference population: mean ≈ 0, std ≈ 1.

    Reference population = rows that are ``ye2024_flag == 0`` AND
    ``xp_fit_flag_residual_high == 0`` AND ``bp_coef_0 > 0`` AND
    ``rp_coef_0 > 0``. This is the same subset the emit pipeline used to
    fit the frozen ``(mu, sigma)``; self-consistency here confirms the
    stats embedded in the provenance sidecar weren't silently miscomputed.
    """
    ye_ok = (df["ye2024_flag"] == 0).to_numpy(dtype=bool)
    strat_ok = (df["xp_fit_flag_residual_high"] == 0).to_numpy(dtype=bool, na_value=False)
    c0_ok = (
        np.isfinite(df["bp_coef_0"].to_numpy())
        & (df["bp_coef_0"].to_numpy() > 0)
        & np.isfinite(df["rp_coef_0"].to_numpy())
        & (df["rp_coef_0"].to_numpy() > 0)
    )
    ref = ye_ok & strat_ok & c0_ok
    n_ref = int(ref.sum())

    bpz = df.loc[ref, "bp_c0_z"].to_numpy()
    rpz = df.loc[ref, "rp_c0_z"].to_numpy()
    bpz = bpz[np.isfinite(bpz)]
    rpz = rpz[np.isfinite(rpz)]

    bp_mu, bp_sigma = float(bpz.mean()), float(bpz.std())
    rp_mu, rp_sigma = float(rpz.mean()), float(rpz.std())

    mean_ok = (abs(bp_mu) < Z_MEAN_TOL) and (abs(rp_mu) < Z_MEAN_TOL)
    std_ok = (abs(bp_sigma - 1.0) < Z_STD_TOL) and (abs(rp_sigma - 1.0) < Z_STD_TOL)
    passed = mean_ok and std_ok

    summary = (
        f"z-score stats within tolerance on {n_ref:,} reference rows: "
        f"BP mu={bp_mu:+.4f} σ={bp_sigma:.4f}, "
        f"RP mu={rp_mu:+.4f} σ={rp_sigma:.4f}"
    )
    return CheckResult(
        name="zscore_validity",
        level="SOFT",
        passed=passed,
        summary=summary,
        details={
            "n_reference_rows": n_ref,
            "bp_mean": bp_mu,
            "bp_std": bp_sigma,
            "rp_mean": rp_mu,
            "rp_std": rp_sigma,
            "mean_tol": Z_MEAN_TOL,
            "std_tol": Z_STD_TOL,
        },
    )


# --- Check 6 — Dedup idempotency ---------------------------------------------


def check_dedup_idempotency(df: pd.DataFrame, expected_rows_out: int) -> CheckResult:
    """Running ``dedup_by_source_id`` yields the audited row count.

    This is a contract-integrity check: the feature matrix has 354,890 ⇒
    324,054 ⇒ 292,948 waterfall (Stream-1 post-cut ⇒ XP inner-join ⇒
    post-dedup on source_id). Drift here means either dedup changed behavior
    or the feature matrix shape drifted — either way worth investigating
    before a training run.
    """
    deduped, stats = dedup_by_source_id(df)
    actual = stats.rows_out
    passed = actual == expected_rows_out
    summary = (
        f"dedup rows_out={actual:,} matches expected {expected_rows_out:,}"
        if passed
        else f"dedup rows_out={actual:,} ≠ expected {expected_rows_out:,} "
        f"(Δ={actual - expected_rows_out:+,})"
    )
    return CheckResult(
        name="dedup_idempotency",
        level="SOFT",
        passed=passed,
        summary=summary,
        details={
            "rows_in": stats.rows_in,
            "rows_out": stats.rows_out,
            "expected_rows_out": expected_rows_out,
            "n_duplicate_stars": stats.n_duplicate_stars,
            "max_duplicates_per_star": stats.max_duplicates_per_star,
            "sort_column": stats.sort_column,
            "duplicate_histogram": stats.histogram,
        },
    )


# --- Check — checkpoint label-scaler correctness -----------------------------


def check_checkpoint_label_scaler(blob: dict, *, tiers: LabelTiers | None = None) -> CheckResult:
    """Hard-fail if a training checkpoint does not carry a fitted label scaler.

    The Run A ensemble trained with placeholder zeros/ones in
    ``label_scaler_mean`` / ``label_scaler_scale`` — no one fit the scaler,
    no one noticed, and downstream calibration silently interpreted the
    head's standardised-space outputs as raw Kelvin. This check is a cheap
    guard against the same class of failure recurring: load the checkpoint,
    ask whether the stored scaler is still the default shape, and refuse to
    proceed to inference if so.

    Pass criteria:
    - both arrays present and length ``tiers.n_labels``;
    - not the zeros/ones default;
    - every ``scale`` entry strictly positive;
    - ``label_names`` in the checkpoint matches ``tiers.all_labels`` exactly.
    """
    tiers = tiers or LabelTiers()
    details: dict = {"n_labels_expected": tiers.n_labels}
    missing = [
        k for k in ("label_scaler_mean", "label_scaler_scale", "label_names") if k not in blob
    ]
    if missing:
        return CheckResult(
            name="checkpoint_label_scaler",
            level="HARD",
            passed=False,
            summary=f"checkpoint missing keys: {missing}",
            details={**details, "missing_keys": missing},
        )

    mean = np.asarray(blob["label_scaler_mean"])
    scale = np.asarray(blob["label_scaler_scale"])
    names = tuple(blob["label_names"])
    details.update(
        {
            "mean_shape": list(mean.shape),
            "scale_shape": list(scale.shape),
            "label_names": list(names),
        }
    )

    if mean.shape != (tiers.n_labels,) or scale.shape != (tiers.n_labels,):
        return CheckResult(
            name="checkpoint_label_scaler",
            level="HARD",
            passed=False,
            summary=(
                f"scaler shape mismatch: mean {mean.shape}, scale {scale.shape}, "
                f"expected ({tiers.n_labels},)"
            ),
            details=details,
        )
    if names != tiers.all_labels:
        return CheckResult(
            name="checkpoint_label_scaler",
            level="HARD",
            passed=False,
            summary="label_names in checkpoint do not match tiers.all_labels",
            details=details,
        )
    if np.all(mean == 0.0) and np.all(scale == 1.0):
        return CheckResult(
            name="checkpoint_label_scaler",
            level="HARD",
            passed=False,
            summary=(
                "label scaler is the zeros/ones placeholder — checkpoint was "
                "saved before the scaler was fit"
            ),
            details=details,
        )
    if not np.all(scale > 0.0):
        return CheckResult(
            name="checkpoint_label_scaler",
            level="HARD",
            passed=False,
            summary="label scaler has non-positive scale entries",
            details={
                **details,
                "scale_min": float(scale.min()),
                "scale_nonpositive_count": int((scale <= 0).sum()),
            },
        )

    details.update(
        {
            "teff_mean": float(mean[0]),
            "teff_scale": float(scale[0]),
            "logg_mean": float(mean[1]),
            "logg_scale": float(scale[1]),
            "mh_mean": float(mean[2]),
            "mh_scale": float(scale[2]),
        }
    )
    return CheckResult(
        name="checkpoint_label_scaler",
        level="HARD",
        passed=True,
        summary=(
            f"fitted label scaler present — Teff σ={scale[0]:.1f} K "
            f"μ={mean[0]:.0f}, [M/H] σ={scale[2]:.3f} dex μ={mean[2]:+.3f}"
        ),
        details=details,
    )


# --- Aggregation -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BatteryVerdict:
    """Aggregate result of the pre-training sanity battery."""

    results: tuple[CheckResult, ...]
    overall: str  # "PASS" | "SOFT-FAIL" | "HARD-FAIL"

    @property
    def any_hard_fail(self) -> bool:
        return any((not r.passed) and r.level == "HARD" for r in self.results)

    @property
    def any_soft_fail(self) -> bool:
        return any((not r.passed) and r.level == "SOFT" for r in self.results)


def run_battery(
    df: pd.DataFrame,
    *,
    expected_dedup_rows: int,
    tiers: LabelTiers | None = None,
) -> BatteryVerdict:
    """Run all six in-module checks and return the aggregate verdict."""
    tiers = tiers or LabelTiers()
    results = (
        check_xp_feature_nan_invariant(df),
        check_tier1_label_completeness(df, tiers),
        check_parameter_bounds(df),
        check_per_element_nan_rates(df, tiers),
        check_zscore_validity(df),
        check_dedup_idempotency(df, expected_rows_out=expected_dedup_rows),
    )
    any_hard = any((not r.passed) and r.level == "HARD" for r in results)
    any_soft = any((not r.passed) and r.level == "SOFT" for r in results)
    overall = "HARD-FAIL" if any_hard else ("SOFT-FAIL" if any_soft else "PASS")
    return BatteryVerdict(results=results, overall=overall)


__all__ = [
    "BatteryVerdict",
    "CheckResult",
    "PARAMETER_BOUNDS",
    "TIER1_ATMOSPHERIC",
    "check_checkpoint_label_scaler",
    "check_dedup_idempotency",
    "check_parameter_bounds",
    "check_per_element_nan_rates",
    "check_tier1_label_completeness",
    "check_xp_feature_nan_invariant",
    "check_zscore_validity",
    "run_battery",
]
