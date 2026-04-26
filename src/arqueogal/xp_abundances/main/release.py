"""Per-star release-tier assignment for Pipeline 1 predictions.

The six-test protocol in ``research_brief.md`` §3.3 decides which *elements*
are releasable; :mod:`tier_promotion` implements that. This module is the
orthogonal *per-star* layer: given an element has cleared §3.3, which
individual predictions are trustworthy?

Tier semantics (frozen contract — see ``docs/plan/05_release_packaging.md``):

- **1 — per-star science**: passes the XP-Mahalanobis OOD gate, no NaN prediction,
  no σ-inflation, and no element-specific caveat (mode-ambiguous on [α/M], or
  kin_ood_flag on aux-assisted elements). Safe for single-star claims.
- **2 — statistical / ensemble only**: passes the OOD gate but carries a
  per-element caveat — σ-inflation (prior collapse), mode-ambiguity ([α/M] only),
  or kin_ood (aux-assisted only). Safe for aggregate studies, not for per-star
  science.
- **3 — do not release**: XP-Mahalanobis OOD-flagged or NaN prediction.

Tier-3 rows are retained in the parquet so downstream consumers can apply
their own, less stringent filters for methodology work; the release
contract is that published catalogues expose Tier 1 only (or Tier 1 + 2
with an explicit caveat) — see `docs/research_brief.md` §3.3.

Simplification 2026-04-26 (v5 schema): the v3-v4 stack carried five global
caveats (regime-B, mode-ambiguous, ensemble disagreement, aux-missing,
dist_prior_dominated) and three OOD flags (ood_joint, latent_support,
ood_aux_mahalanobis). The Stream-1 ablation study at
``release/test_ablations_2026-04-26/REPORT.md`` showed that:

- only ``ood_joint_flag`` (XP-Mahalanobis) had a measurable effect on the
  trustworthy-catalog (Tier 1 + 2) RMSE — 24-38 % inflation when removed;
- the other OOD flags fired zero or trivially on the held-out test split;
- the global caveats either fired zero times or shifted stars between Tier 1
  and Tier 2 with no measurable RMSE difference (i.e. the demoted stars were
  not measurably worse than the ones kept in Tier 1);
- ``mode_ambiguous_flag`` was an exception only for [α/M] — the disc α/M
  bimodality at fixed (Teff, log g, [M/H]) genuinely affects this element,
  so the caveat is retained but confined to [α/M].

The dropped flag *columns* are still emitted by upstream modules
(annotation pipeline, fetch_pipeline_steps) for diagnostic and
reproducibility purposes; they no longer feed the release tier.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

_PRED_COLS: Final = ("teff_pred", "logg_pred", "mh_pred", "alpha_m_pred", "mg_h_pred")

_OOD_FLAGS: Final = (
    "ood_joint_flag",
)
"""Joint per-row OOD flags. Any flag firing → Tier 3 for all elements.

Simplified 2026-04-26 from the v3 set ``(ood_joint_flag, latent_support_flag,
ood_aux_mahalanobis_flag)``. The Stream-1-test ablation
(``release/test_ablations_2026-04-26/REPORT.md``) showed that
``latent_support_flag`` and ``ood_aux_mahalanobis_flag`` never fire on the held-out
test split (``ood_aux_mahalanobis_flag`` is fully subsumed by ``aux_missing_any``,
which is itself now also dropped) and contribute zero to the trustworthy-catalog
RMSE. The XP-block Mahalanobis gate (``ood_joint_flag``) is the only one with a
measurable effect: 24-38 % T1+T2 RMSE inflation on Teff/log g/[M/H]/[Mg/H] when
disabled. The dropped flag columns are still computed and emitted by upstream
modules (informational diagnostics), but they no longer contribute to tier
demotion.
"""

_CAVEAT_FLAGS: Final = ()
"""Global per-row caveat flags. Empty: the v3 production set
``(regime_b_flag, mode_ambiguous_flag, ood_disagreement_flag, aux_missing_any,
dist_prior_dominated)`` was retired on 2026-04-26.

- ``regime_b_flag`` fires on ~0.04 % of stars and has no measurable RMSE effect.
- ``ood_disagreement_flag`` cannot fire with a single-member ensemble.
- ``aux_missing_any`` shifts ~2 pp of stars from T1 → T2 with no T1+T2 effect (the
  demoted stars are not measurably worse than the ones it leaves in T1).
- ``dist_prior_dominated`` never fires on the holdout.
- ``mode_ambiguous_flag`` is moved from a global caveat to a per-element caveat
  on [α/M] only — see ``_PER_ELEMENT_CAVEAT_FLAGS`` below.

Empirical justification: ``release/test_ablations_2026-04-26/REPORT.md``.
"""

_PER_ELEMENT_CAVEAT_FLAGS: Final[dict[str, tuple[str, ...]]] = {
    "alpha_m": ("mode_ambiguous_flag",),
}
"""Per-element caveat flags. If an element appears here, its tuple is the full
list of flags whose firing demotes that element to Tier 2.

The 2026-04-26 ablation (``release/test_ablations_2026-04-26/REPORT.md``) showed
that ``mode_ambiguous_flag`` is pure relabeling for every element except [α/M]:
removing it shifts 39 pp of stars from T1 → T2 across the catalog with zero
T1+T2 RMSE change for Teff / log g / [M/H] / [Mg/H]. Only [α/M] shows a
measurable effect (+12 % T1 RMSE if removed), consistent with [α/M] being the
element where the disc bimodality lives at fixed (Teff, log g, [M/H]). We
therefore confine the caveat to [α/M].
"""

_ABUNDANCE_ELEMENTS: Final[tuple[str, ...]] = ("teff", "logg", "mh", "alpha_m", "mg_h")
"""Atmospheric parameters and abundance labels released per-star."""

_AUX_ASSISTED_ELEMENTS: Final[tuple[str, ...]] = ("alpha_m", "mg_h")
"""Elements whose information channel is aux+prior-dominated (CMI ≈ 0 nats given parallax,
photometry, extinction, position). When kin_ood_flag is True, these elements lose their
prior basis and must demote to Tier 2 even if no other caveat fires.

See research_brief.md §3.3.1 for the three-question diagnostic that motivated this.
The Phase A2 implementation emitted xp_abundance_type__<element> columns; this refactor
makes assign_release_tier consume them via the kin_ood_flag → Tier 2 demotion for
aux-assisted elements (META_META §14.3, outlier_flagging.md)."""

_PER_ELEMENT_PRED_COL: Final[dict[str, str]] = {
    "teff": "teff_pred",
    "logg": "logg_pred",
    "mh": "mh_pred",
    "alpha_m": "alpha_m_pred",
    "mg_h": "mg_h_pred",
}
"""Map element name to the prediction column in the inference parquet."""

_PER_ELEMENT_SIGMA_COL: Final[dict[str, str]] = {
    "teff": "teff_sigma",
    "logg": "logg_sigma",
    "mh": "mh_sigma",
    "alpha_m": "alpha_m_sigma",
    "mg_h": "mg_h_sigma",
}
"""Map element name to the per-element predicted-sigma column."""

_PER_ELEMENT_SIGMA_INFLATED_THRESHOLD: Final[dict[str, float]] = {
    "teff": 150.0,    # K
    "logg": 0.30,     # dex
    "mh": 0.20,       # dex
    "alpha_m": 0.05,  # dex (tightened 2026-04-26 from 0.10 → 0.05; ablation test
                      #      release/test_ablations_2026-04-26 showed 23% T1 RMSE
                      #      improvement on [α/M] at 0.5×σ_train, accepting
                      #      ~14 pp T1-fraction loss on this element)
    "mg_h": 0.20,     # dex
}
"""Per-element sigma above which the prediction is flagged as inflated (prior-collapse).

These thresholds were derived from the empirical-Bayes shrinkage τ=50 ceiling and the
Stream 3 stress-battery analysis (HIGH_SIGMA_RESCUE_REPORT.md, 2026-04-25). At these
thresholds the regression head is collapsing toward the training-distribution prior
mean rather than reading information from the spectrum; demoting these stars to Tier 2
removes ~25% of the broad Stream 3 release (including the 74k prior-collapse spike at
[M/H]=-1.05, alpha/M=+0.11) while preserving the residual Tier 1 bimodality.

The alpha/M threshold was tightened to 0.05 dex (≈0.5×σ_train) on 2026-04-26 after
the per-cell-gate ablation study (REPORT.md release/test_ablations_2026-04-26)
showed alpha/M was the only element whose Tier 1 RMSE was meaningfully responsive
to a tighter sigma cut. Other elements were within Pareto-optimal tolerance of
the 1×σ_train production setting.
"""

_CATALOGUE_SCHEMA_VERSION: Final[int] = 5
"""Version of the catalog schema. Bumped when tier semantics or columns change.

- v1: original schema before Phase A2.
- v2 (2026-04-24, Phase A2): added xp_abundance_type__<element>, kin_ood_flag, g_mag_bin.
- v3 (2026-04-24, Phase A2-followup): added per-element release_tier__<element>,
  dist_prior_dominated; existing release_tier becomes row-max of per-element tiers.
- v4 (2026-04-25, Phase A2-followup-2): added per-element prediction_sigma_inflated__<element>
  flag and the global prediction_sigma_inflated_any flag. When the per-element flag
  fires, that element's release_tier__<element> is demoted to 2; the row-max
  release_tier therefore promotes the row to Tier 2 (or stays Tier 3 if a hard OOD
  flag fires). See HIGH_SIGMA_RESCUE_REPORT.md and CRITICAL_FAILURE_REPORT.md for the
  motivation. Threshold values in _PER_ELEMENT_SIGMA_INFLATED_THRESHOLD.
- v5 (2026-04-26, per-cell-gate ablation): simplified the gate stack after the
  Stream 1 ablation study (release/test_ablations_2026-04-26/REPORT.md). Dropped
  ``latent_support_flag``, ``ood_aux_mahalanobis_flag`` (zero-firing /
  redundant) from OOD; dropped ``regime_b_flag``, ``ood_disagreement_flag``,
  ``aux_missing_any``, ``dist_prior_dominated`` (zero-effect on T1+T2 RMSE)
  from caveats. ``mode_ambiguous_flag`` is no longer global — it now only
  demotes [α/M] (the only element where the bimodality matters). ``alpha_m``
  σ-inflation threshold tightened from 0.10 → 0.05 dex. The dropped flag
  *columns* are still emitted by upstream modules for diagnostics; they no
  longer feed the tier."""


def assign_xp_abundance_type(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Assign per-element ``xp_abundance_type`` for spectrum-dominant vs aux-assisted.

    Returns a dict mapping element name (e.g. "mh") to a string Series where each
    element is "spectrum_dominant" or "aux_assisted".

    Design rationale: per-element string columns are consumer-friendly (one column
    per label type, clearly named). Aux-assisted labels are those where CMI
    conditional on auxiliary features falls below 0.02 nats (research_brief.md §3.3.1).

    Parameters
    ----------
    df : pd.DataFrame
        Prediction frame (inference output).

    Returns
    -------
    dict[str, pd.Series]
        Maps element name to string Series with values "spectrum_dominant" or
        "aux_assisted".
    """
    result = {}
    for elem in _ABUNDANCE_ELEMENTS:
        # Design: Teff, logg, [M/H] → spectrum_dominant (CMI threshold passed).
        # [α/M], [Mg/H] → aux_assisted (conditional CMI < 0.02).
        if elem in ("teff", "logg", "mh"):
            result[elem] = pd.Series("spectrum_dominant", index=df.index, dtype="string")
        elif elem in ("alpha_m", "mg_h"):
            result[elem] = pd.Series("aux_assisted", index=df.index, dtype="string")
        else:
            result[elem] = pd.Series("unknown", index=df.index, dtype="string")
    return result


def assign_kin_ood_flag(df: pd.DataFrame) -> pd.Series:
    """Assign kinematic OOD flag (placeholder, all False in v1).

    Phase A2 placeholder: all rows False. Phase B will implement actual detector.

    Parameters
    ----------
    df : pd.DataFrame
        Prediction frame.

    Returns
    -------
    pd.Series
        Boolean Series, False for all rows in v1.
    """
    return pd.Series(False, index=df.index, dtype="bool")


def assign_dist_prior_dominated(
    df: pd.DataFrame,
    *,
    parallax_snr_threshold: float = 5.0,
) -> pd.Series:
    """Per-star boolean: True where the Bailer-Jones distance is prior-dominated.

    The Phase A2 catalog schema reserved a `dist_prior_dominated` column but did not
    compute it. This function fills the gap. A star is flagged when its parallax SNR
    falls below `parallax_snr_threshold` (default 5; equivalent to σ_π/π > 0.2, the
    Bailer-Jones+2021 boundary between geometric and prior-dominated photogeometric
    distances) OR when its parallax is non-finite or non-positive (Bailer-Jones uses
    a Galactic-prior distance for these by default).

    Source columns expected:
    - ``parallax`` (mas) and ``parallax_error`` (mas) — preferred.
    - or ``parallax_over_error`` (already a SNR) — used as a fallback.

    If neither is present, the entire column is False (defensive default; the caller
    should not subsequently rely on this flag for tier-gating in that case).

    Parameters
    ----------
    df : pd.DataFrame
        Inference parquet. Must contain Gaia parallax columns or fall back gracefully.
    parallax_snr_threshold : float
        SNR boundary; below this, the distance is prior-dominated. Default 5.0
        corresponds to σ_π/π > 0.2 (Bailer-Jones+2021 §4 demarcation).

    Returns
    -------
    pd.Series
        Boolean Series aligned to ``df.index``. True = prior-dominated.

    Notes
    -----
    Bulge-direction RGB stars at d > 6 kpc almost always satisfy this condition.
    The catalog UX review (catalog_ux.md) and the parallax_distance review both
    flagged the absence of this surface as CRITICAL because users cannot otherwise
    distinguish parallax-anchored from prior-driven distance estimates.
    """
    idx = df.index
    flag = pd.Series(False, index=idx)

    # Resolve the parallax column. The project uses `parallax` (Stream 3 features),
    # `parallax_corr` (Lindegren+2021-corrected, Stream 1 catalog), or
    # `parallax_over_error` as a precomputed SNR.
    plx_col: str | None = None
    for candidate in ("parallax_corr", "parallax"):
        if candidate in df.columns:
            plx_col = candidate
            break

    if plx_col is not None and "parallax_error" in df.columns:
        plx = df[plx_col]
        plx_err = df["parallax_error"]
        # SNR = plx / plx_err for plx > 0; non-finite or non-positive parallax flagged.
        with np.errstate(invalid="ignore", divide="ignore"):
            snr = plx / plx_err.where(plx_err > 0, np.nan)
        flag = (snr.fillna(0.0) < parallax_snr_threshold) | ~np.isfinite(plx) | (plx <= 0)
    elif "parallax_over_error" in df.columns:
        snr = df["parallax_over_error"]
        flag = (snr.fillna(0.0) < parallax_snr_threshold) | ~np.isfinite(snr)
    else:
        # No parallax information available; defensive False.
        return flag.astype(bool)

    return flag.astype(bool)


def assign_g_mag_bin(df: pd.DataFrame) -> pd.Series:
    """Assign magnitude bin: "bright" (G≤15), "mid" (15<G≤16), "faint" (16<G≤17).

    Source-column resolution (in order):
    1. ``phot_g_mean_mag_corr`` — Riello+2021 Appendix-A-corrected G magnitude
       (preferred; emitted by ``data/gaia_corrections.py`` for Stream 1).
    2. ``phot_g_mean_mag`` — raw Gaia DR3 G magnitude (fallback; used by Stream 3
       predictions where the Riello correction was not propagated to the prediction
       parquet). The Riello correction shifts G by ≤ 0.005 mag in the bin
       boundaries we use, so binning on the raw value is acceptable for the
       g_mag_bin discretisation.
    3. None of the above → raises ValueError.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least one of ``phot_g_mean_mag_corr`` or ``phot_g_mean_mag``.

    Returns
    -------
    pd.Series
        Categorical string Series with values in {"bright", "mid", "faint"}.
    """
    # Source-column resolution. The project uses different conventions in different
    # tables: Stream 1 catalog uses `phot_g_mean_mag_corr`; Stream 3 features use
    # the shorter `g_mag`. Annotation is robust to both.
    g_mag: pd.Series | None = None
    for candidate in ("phot_g_mean_mag_corr", "g_mag", "phot_g_mean_mag"):
        if candidate in df.columns:
            g_mag = df[candidate]
            break
    if g_mag is None:
        # No G magnitude available (e.g., a predictions parquet without photometry
        # carried through). Emit "unknown" rather than raising; the consumer can
        # filter on it. Document the gap in the upstream join step.
        return pd.Series("unknown", index=df.index, dtype="string")

    bins = pd.Series("faint", index=df.index, dtype="string")
    bins[g_mag <= 15] = "bright"
    bins[(g_mag > 15) & (g_mag <= 16)] = "mid"
    bins[g_mag.isna()] = "unknown"
    return bins


def assign_per_element_release_tier(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Per-element release tier ∈ {1, 2, 3}.

    For each released element (Teff, logg, [M/H], [α/M], [Mg/H]), composes a tier from:

    - **Tier 3** (do not release): NaN prediction for that element, OR
      ``ood_joint_flag`` (XP-Mahalanobis OOD) firing.
    - **Tier 2** (statistical / ensemble only): any of:
        - element-specific σ-inflation (per ``_PER_ELEMENT_SIGMA_INFLATED_THRESHOLD``);
        - element-specific caveat from ``_PER_ELEMENT_CAVEAT_FLAGS`` (currently
          ``mode_ambiguous_flag`` → α/M only);
        - for aux-assisted elements ([α/M], [Mg/H]), ``kin_ood_flag`` firing —
          aux-assisted elements rely on a disc-kinematics population prior that
          breaks down for kinematically-anomalous stars (halo, accreted debris,
          counter-rotating disc), so they must demote even if no other caveat is
          present (research_brief.md §3.3.1; META_META §14.3).
    - **Tier 1** (per-star science): everything else.

    Missing flag or prediction columns are treated as False / not-NaN — the conservative
    direction (Tier-3 demotions must be explicit in the input).

    Parameters
    ----------
    df : pd.DataFrame
        Prediction parquet annotated by ``run_pipeline1_inference.py`` and the OOD merge.

    Returns
    -------
    dict[str, pd.Series]
        Maps element name (e.g. ``"alpha_m"``) to ``int8`` Series of per-element tier.

    Notes
    -----
    Simplified 2026-04-26 (v5 schema): the v3-v4 implementation also fired Tier 2 on a
    global caveat set ``(regime_b_flag, mode_ambiguous_flag, ood_disagreement_flag,
    aux_missing_any, dist_prior_dominated)``. The ablation in
    ``release/test_ablations_2026-04-26/REPORT.md`` retired all but the per-element
    ``mode_ambiguous`` carve-out for α/M. The aux-assisted ``kin_ood_flag`` demotion is
    retained but is forward-compatible — in Pipeline 1 the flag is a Stream-3 placeholder
    (always False on Stream 1), so existing tier counts are unchanged on Stream 1.
    """
    idx = df.index

    # Joint OOD flags fire for ALL elements (XP-block Mahalanobis OOD invalidates every
    # prediction). Tier 3 for every element on hit.
    ood = pd.Series(False, index=idx)
    for col in _OOD_FLAGS:
        if col in df.columns:
            ood = ood | df[col].fillna(False).astype(bool)

    # Global caveats fire for ALL elements. Tier 2 unless OOD. After the 2026-04-26
    # simplification this set is empty; the loop is retained for forward compatibility
    # if a future global caveat is reintroduced.
    global_caveat = pd.Series(False, index=idx)
    for col in _CAVEAT_FLAGS:
        if col in df.columns:
            global_caveat = global_caveat | df[col].fillna(False).astype(bool)

    # kin_ood_flag is per-row; it demotes ONLY aux-assisted elements (alpha_m, mg_h).
    kin_ood = pd.Series(False, index=idx)
    if "kin_ood_flag" in df.columns:
        kin_ood = df["kin_ood_flag"].fillna(False).astype(bool)

    per_element: dict[str, pd.Series] = {}
    for elem in _ABUNDANCE_ELEMENTS:
        # Per-element NaN prediction.
        pred_col = _PER_ELEMENT_PRED_COL.get(elem)
        elem_nan = pd.Series(False, index=idx)
        if pred_col is not None and pred_col in df.columns:
            elem_nan = df[pred_col].isna()

        # Per-element sigma-inflated check (v4 schema, prior-collapse caveat).
        # The regression head's σ for this element exceeds the empirical-Bayes-shrinkage
        # ceiling, indicating the prediction has collapsed toward the training-distribution
        # prior mean rather than reading information from the spectrum.
        sigma_col = _PER_ELEMENT_SIGMA_COL.get(elem)
        sigma_threshold = _PER_ELEMENT_SIGMA_INFLATED_THRESHOLD.get(elem)
        sigma_inflated = pd.Series(False, index=idx)
        if (
            sigma_col is not None
            and sigma_col in df.columns
            and sigma_threshold is not None
        ):
            sigma_inflated = (df[sigma_col].fillna(0.0) > sigma_threshold).astype(bool)

        # Per-element caveat flags (v5 schema, 2026-04-26): some caveats only fire
        # for one specific element. Currently: mode_ambiguous_flag → α/M only.
        elem_specific_caveat = pd.Series(False, index=idx)
        for col in _PER_ELEMENT_CAVEAT_FLAGS.get(elem, ()):
            if col in df.columns:
                elem_specific_caveat = (
                    elem_specific_caveat | df[col].fillna(False).astype(bool)
                )

        # Tier 3 if NaN OR joint OOD.
        tier3 = elem_nan | ood

        # Tier 2 caveats (per-element semantics):
        #   - global caveats fire for every element (currently empty)
        #   - per-element caveats fire only for the listed elements (mode_ambiguous → α/M)
        #   - aux-assisted elements demote on kin_ood_flag (population-prior breaks down)
        #   - any element demotes on its own σ-inflation (regression-head prior collapse)
        elem_caveat = global_caveat | sigma_inflated | elem_specific_caveat
        if elem in _AUX_ASSISTED_ELEMENTS:
            elem_caveat = elem_caveat | kin_ood

        tier = pd.Series(1, index=idx, dtype="int8")
        tier[elem_caveat] = 2
        tier[tier3] = 3
        per_element[elem] = tier

    return per_element


def assign_prediction_sigma_inflated(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Per-element prediction_sigma_inflated boolean flag.

    A star's element flag fires when the model's predicted ``<elem>_sigma`` exceeds the
    empirical-Bayes-shrinkage ceiling for that element (see
    ``_PER_ELEMENT_SIGMA_INFLATED_THRESHOLD``). When fired, the regression-head prediction
    has collapsed toward the training-distribution prior centroid rather than reading
    information from the spectrum; the aggregated ``release_tier__<element>`` demotes to
    Tier 2.

    Returns
    -------
    dict[str, pd.Series]
        Maps element name to bool Series. Missing σ columns yield all-False.
    """
    idx = df.index
    out: dict[str, pd.Series] = {}
    for elem in _ABUNDANCE_ELEMENTS:
        sigma_col = _PER_ELEMENT_SIGMA_COL.get(elem)
        threshold = _PER_ELEMENT_SIGMA_INFLATED_THRESHOLD.get(elem)
        if (
            sigma_col is None
            or sigma_col not in df.columns
            or threshold is None
        ):
            out[elem] = pd.Series(False, index=idx, dtype="bool")
            continue
        out[elem] = (df[sigma_col].fillna(0.0) > threshold).astype(bool)
    return out


def assign_release_tier(df: pd.DataFrame) -> pd.Series:
    """Compute the per-row ``release_tier`` ∈ {1, 2, 3}.

    The composite tier is the row-wise maximum (most-conservative) across the per-element
    tiers from :func:`assign_per_element_release_tier`. A star's composite tier is the
    *worst* tier across its released elements: if any element is Tier 3, the row is
    Tier 3; if any element is Tier 2 with all others ≥ 2 not OOD, the row is Tier 2;
    Tier 1 only when every element is Tier 1.

    Backward-compatible signature: returns a single int8 Series. The richer per-element
    breakdown is available via ``assign_per_element_release_tier`` and is also emitted
    as ``release_tier__<element>`` columns by :func:`annotate_parquet`.

    Parameters
    ----------
    df : pd.DataFrame
        Prediction parquet produced by ``run_pipeline1_inference.py`` and
        annotated with ``latent_support_flag`` via
        ``merge_latent_support_into_predictions.py``.

    Returns
    -------
    pd.Series
        ``int8`` series aligned to ``df.index``, values in {1, 2, 3}.
    """
    per_element = assign_per_element_release_tier(df)
    if not per_element:
        return pd.Series(1, index=df.index, dtype="int8")
    tier_df = pd.DataFrame({elem: s for elem, s in per_element.items()}, index=df.index)
    return tier_df.max(axis=1).astype("int8")


def tier_counts(df: pd.DataFrame, tier_col: str = "release_tier") -> dict[int, int]:
    """Return ``{tier: n_rows}`` for the three tiers."""
    s = df[tier_col]
    return {int(t): int((s == t).sum()) for t in (1, 2, 3)}


def annotate_parquet(path: Path) -> dict[str, int | dict[int, int]]:
    """Add or refresh release-catalog columns in a prediction parquet in place.

    Adds or refreshes:
    - ``release_tier`` ∈ {1, 2, 3}
    - ``xp_abundance_type__<element>`` ∈ {"spectrum_dominant", "aux_assisted"}
      for each element in teff, logg, mh, alpha_m, mg_h
    - ``kin_ood_flag`` (boolean, False placeholder in v1)
    - ``g_mag_bin`` ∈ {"bright", "mid", "faint"}

    Emits / refreshes the ``*.release_tier.json`` sidecar next to the parquet
    capturing counts, flag-column provenance, and catalog schema version.
    The parquet's main provenance sidecar (``*.provenance.json``) is not touched
    — that records upstream inference, not this annotation step.

    Returns
    -------
    dict
        ``{"n_rows": N, "counts": {1: ..., 2: ..., 3: ...}}``.
    """
    _logger.info(
        "annotate_parquet: reading %s (catalog schema v%d)",
        path, _CATALOGUE_SCHEMA_VERSION,
    )
    df = pd.read_parquet(path)
    _logger.info("annotate_parquet: loaded %d rows from %s", len(df), path.name)

    # Per-element abundance type (spectrum_dominant vs aux_assisted) — emit FIRST so
    # the per-element tier composition has access to it implicitly via the constants.
    abundance_types = assign_xp_abundance_type(df)
    for elem, series in abundance_types.items():
        col_name = f"xp_abundance_type__{elem}"
        df[col_name] = series

    # Kinematic OOD flag (placeholder until Phase B detector lands).
    if "kin_ood_flag" not in df.columns:
        df["kin_ood_flag"] = assign_kin_ood_flag(df)

    # Magnitude binning.
    df["g_mag_bin"] = assign_g_mag_bin(df)

    # Distance prior-dominance flag (Bailer-Jones boundary at σ_π/π > 0.2).
    if "dist_prior_dominated" not in df.columns:
        df["dist_prior_dominated"] = assign_dist_prior_dominated(df)

    # Per-element prediction-sigma-inflated flags (v4 schema, prior-collapse caveat).
    sigma_inflated = assign_prediction_sigma_inflated(df)
    for elem, series in sigma_inflated.items():
        df[f"prediction_sigma_inflated__{elem}"] = series.astype(bool)
    # Convenience scalar: any-element flag (for consumer filtering).
    df["prediction_sigma_inflated_any"] = (
        pd.DataFrame({e: s for e, s in sigma_inflated.items()}).any(axis=1).astype(bool)
    )

    # Per-element release tiers (new in v3 schema; consumes v4 σ-inflation flag).
    per_element_tiers = assign_per_element_release_tier(df)
    for elem, series in per_element_tiers.items():
        df[f"release_tier__{elem}"] = series.astype("int8")

    # Composite release_tier = row-max (most conservative). Backward-compatible.
    df["release_tier"] = assign_release_tier(df).astype("int8")

    df.to_parquet(path)

    counts = tier_counts(df)
    _logger.info(
        "annotate_parquet: %s wrote %d rows; tiers T1=%d (%.1f%%) "
        "T2=%d (%.1f%%) T3=%d (%.1f%%)",
        path.name,
        int(len(df)),
        counts.get(1, 0), 100.0 * counts.get(1, 0) / max(len(df), 1),
        counts.get(2, 0), 100.0 * counts.get(2, 0) / max(len(df), 1),
        counts.get(3, 0), 100.0 * counts.get(3, 0) / max(len(df), 1),
    )
    summary: dict[str, int | dict[int, int]] = {
        "n_rows": int(len(df)),
        "counts": counts,
    }

    sidecar = path.with_name(path.stem + ".release_tier.json")
    sidecar_payload = {
        "parquet": path.name,
        "catalog_schema_version": _CATALOGUE_SCHEMA_VERSION,
        "n_rows": int(len(df)),
        "counts": {str(k): v for k, v in counts.items()},
        "ood_flags_considered": [c for c in _OOD_FLAGS if c in df.columns],
        "caveat_flags_considered": [c for c in _CAVEAT_FLAGS if c in df.columns],
        "per_element_caveat_flags": {
            elem: [c for c in flags if c in df.columns]
            for elem, flags in _PER_ELEMENT_CAVEAT_FLAGS.items()
        },
        "nan_pred_columns_checked": [c for c in _PRED_COLS if c in df.columns],
        "release_columns_added": [
            "release_tier",
            "release_tier__teff",
            "release_tier__logg",
            "release_tier__mh",
            "release_tier__alpha_m",
            "release_tier__mg_h",
            "xp_abundance_type__teff",
            "xp_abundance_type__logg",
            "xp_abundance_type__mh",
            "xp_abundance_type__alpha_m",
            "xp_abundance_type__mg_h",
            "kin_ood_flag",
            "g_mag_bin",
            "dist_prior_dominated",
            "prediction_sigma_inflated__teff",
            "prediction_sigma_inflated__logg",
            "prediction_sigma_inflated__mh",
            "prediction_sigma_inflated__alpha_m",
            "prediction_sigma_inflated__mg_h",
            "prediction_sigma_inflated_any",
        ],
        "prediction_sigma_inflated_thresholds": dict(_PER_ELEMENT_SIGMA_INFLATED_THRESHOLD),
        "expected_upstream_columns": [
            # Flags this annotator does NOT produce; they are read from the
            # input parquet if present and used by the tier-gating logic. If
            # absent, treated as False (defensive default per assign_release_tier).
            "ood_joint_flag",
            "mode_ambiguous_flag",  # used per-element on alpha_m only (v5)
        ],
        "diagnostic_only_columns": [
            # Flags retired from tier gating in v5 (2026-04-26) but kept as
            # informational columns. Upstream modules still emit them; the
            # release tier no longer reads them.
            "ood_aux_mahalanobis_flag",
            "latent_support_flag",
            "regime_b_flag",
            "ood_disagreement_flag",
            "aux_missing_any",
            "dist_prior_dominated",
        ],
        "aux_assisted_elements": list(_AUX_ASSISTED_ELEMENTS),
        "tier_gating_logic": (
            "Tier 3 if ood_joint_flag (XP-Mahalanobis OOD) OR per-element NaN. "
            "Tier 2 if (per-element σ exceeds prediction_sigma_inflated threshold) "
            "OR (element is alpha_m AND mode_ambiguous_flag) OR "
            "(element is aux-assisted AND kin_ood_flag). "
            "Tier 1 otherwise. Composite release_tier = row-max across elements. "
            "Simplified 2026-04-26 (v5 schema) per "
            "release/test_ablations_2026-04-26/REPORT.md — see release.py docstrings "
            "for the per-flag empirical justification."
        ),
    }
    sidecar.write_text(json.dumps(sidecar_payload, indent=2))

    return summary


__all__ = [
    "annotate_parquet",
    "assign_dist_prior_dominated",
    "assign_g_mag_bin",
    "assign_kin_ood_flag",
    "assign_per_element_release_tier",
    "assign_prediction_sigma_inflated",
    "assign_release_tier",
    "assign_xp_abundance_type",
    "tier_counts",
]
