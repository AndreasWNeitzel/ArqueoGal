"""Per-star release-tier assignment for Pipeline 1 predictions.

The six-test protocol in ``research_brief.md`` §3.3 decides which *elements*
are releasable; :mod:`tier_promotion` implements that. This module is the
orthogonal *per-star* layer: given an element has cleared §3.3, which
individual predictions are trustworthy?

Tier semantics (v6 contract, 2026-05-03, see ``docs/plan/05_release_packaging.md``
and ``docs/decisions/ADR-0016_tier_v6_mahalanobis_redesign.md``):

- **1, per-star science**: prediction is finite, ``ood_joint_flag`` is False
  (passes the XP-block Mahalanobis input-OOD gate), and
  ``label_extrapolation_flag`` is False (passes the 5-D Mahalanobis output-OOD
  gate fit on APOGEE-truth at the 99th percentile). Safe for single-star claims.
- **2, statistical / ensemble only**: prediction is finite and passes the
  input-OOD gate, but ``label_extrapolation_flag`` fires (predicted label tuple
  lies outside the APOGEE-truth training envelope) or an element-specific
  caveat fires (``_PER_ELEMENT_CAVEAT_FLAGS``, currently empty). Safe for
  aggregate studies, not per-star science.
- **3, do not release**: NaN prediction OR ``ood_joint_flag`` fires.

Tier-3 rows are retained in the parquet so downstream consumers can apply
their own, less stringent filters for methodology work; the release
contract is that published catalogues expose Tier 1 only (or Tier 1 + 2
with an explicit caveat), see ``docs/research_brief.md`` §3.3.

Diagnostic-only columns (emitted, not gating)
---------------------------------------------
These columns are still computed and persisted to the parquet for downstream
diagnostic / informational use, but they DO NOT drive tier assignment as of
the 2026-05-03 redesign:

- ``prediction_sigma_inflated__<element>``, per-element σ-inflation flag.
  Was an active T2 demoter through v5 (2026-04-26). Retired here because
  σ-threshold gating couples "model is uncertain" with "prediction is
  unreliable", which conflates calibrated uncertainty with extrapolation,
  and was perceived as cherry-picking the low-σ core. The columns are kept
  so downstream users can apply their own σ filter if they want.
- ``kin_ood_flag``, disc-kinematics envelope flag. Was an aux-assisted-element
  T2 demoter through v5. Retired because halo / accreted-debris stars are
  exactly the science target for users who want them, and demoting them by
  default was the wrong move. The column is read from upstream parquets if
  present (informational only); release.py no longer generates it.
- ``mode_ambiguous_flag``, α/M bimodality boundary flag. Was a per-element
  T2 demoter for α/M. Retired because the flag fires on ~46 % of the cohort
  (the disc is genuinely bimodal at fixed Teff/log g/[M/H]); demoting half
  the catalog was not justified.
- ``regime_b_flag``, ``ood_disagreement_flag``, ``aux_missing_any``,
  ``dist_prior_dominated``, ``ood_aux_mahalanobis_flag``, ``latent_support_flag``
 , all retired in the v5 ablation (2026-04-26).

The dropped flag *columns* may still be emitted by upstream modules
(annotation pipeline, fetch_pipeline_steps) for diagnostic and reproducibility
purposes; they no longer feed the release tier.

Selection-bias caveat (still applies, in modified form)
-------------------------------------------------------
The new ``label_extrapolation_flag`` gate fits a 5-D Mahalanobis envelope on
APOGEE-truth labels (Teff, log g, [M/H], [α/M], [Mg/H]) at the 99th percentile.
APOGEE DR19 covers a restricted observational window (Teff ∈ [4000, 5500] K,
log g ∈ [1.0, 3.5], [M/H] ∈ [-2.0, +0.5], G ≲ 13.5 for the bright sample).
Stream-3 stars predicted into label-space regions that are sparse or absent
in APOGEE (cool dwarfs, metal-poor halo, faint giants) will fall into Tier 2
*by construction*, this is selection-bias of the training reference, not
a defect in the predictions themselves. Users targeting halo / accreted-debris
populations should treat T2 demotions as conservative-by-design and inspect
the ``label_mahalanobis_percentile`` column for continuous OOD severity.
See ``docs/decisions/ADR-0016_tier_v6_mahalanobis_redesign.md``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

_PRED_COLS: Final = (
    "teff_pred",
    "logg_pred",
    "mh_pred",
    "fe_h_pred",
    "alpha_m_pred",
    "mg_h_pred",
    "c_h_pred",
    "n_h_pred",
    "o_h_pred",
    "na_h_pred",
    "al_h_pred",
    "si_h_pred",
    "s_h_pred",
    "k_h_pred",
    "ca_h_pred",
    "ti_h_pred",
    "v_h_pred",
    "cr_h_pred",
    "mn_h_pred",
    "ni_h_pred",
    "ce_h_pred",
)
"""All 21 predicted elements, ordered to match LabelTiers.all_labels."""


def _coerce_flag_series(s: pd.Series) -> pd.Series:
    """Coerce a flag column to a clean boolean Series.

    Upstream pipelines emit flag columns under several physical dtypes:
    ``bool`` and ``BooleanDtype`` from arrow round-trips, plain integer 0/1
    from boolean ORs across numeric masks, and ``object`` Series mixing Python
    ``bool``, ``int``, and ``None`` from heterogeneous joins. The naïve idiom
    ``s.fillna(False).astype(bool)`` triggers a ``FutureWarning`` in
    pandas ≥ 2.1 (deprecated implicit object-dtype downcast inside
    ``fillna``) and will hard-break in pandas ≥ 2.2; the obvious replacement
    ``s.astype("boolean").fillna(False).astype(bool)`` rejects mixed
    ``bool/int`` object Series with ``TypeError: Need to pass bool-like
    values``. We instead route through ``float64`` (which accepts every flag
    representation we have ever seen, bool, int, and ``NaN`` for missing),
    fill the missing slots with ``0.0``, and cast to ``bool`` (any non-zero
    value becomes ``True``). The two intermediate allocations are O(N), still
    negligible against the rest of the release pipeline.
    """
    return s.astype("float64").fillna(0.0).astype(bool)


_OOD_FLAGS: Final = ("ood_joint_flag",)
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
  on [α/M] only, see ``_PER_ELEMENT_CAVEAT_FLAGS`` below.

Empirical justification: ``release/test_ablations_2026-04-26/REPORT.md``.
"""

_PER_ELEMENT_CAVEAT_FLAGS: Final[dict[str, tuple[str, ...]]] = {}
"""Per-element caveat flags. Empty as of 2026-05-03 redesign.

mode_ambiguous_flag was retired here because it fires on ~46% of the cohort
(the disc is genuinely bimodal at fixed Teff/log g/[M/H]), that's a
property of the data, not a property worth demoting half the catalog over.
The flag may still be present as an informational column upstream; tier
demotion no longer uses it.
"""

_ABUNDANCE_ELEMENTS: Final[tuple[str, ...]] = (
    "teff",
    "logg",
    "mh",
    "fe_h",
    "alpha_m",
    "mg_h",
    "c_h",
    "n_h",
    "o_h",
    "na_h",
    "al_h",
    "si_h",
    "s_h",
    "k_h",
    "ca_h",
    "ti_h",
    "v_h",
    "cr_h",
    "mn_h",
    "ni_h",
    "ce_h",
)
"""All 21 atmospheric parameters and abundance labels released per-star.

Ordered to match LabelTiers tier1 + tier2 + tier3 sequence as of 2026-04-28
(the full 21-element model). See data.py for tier assignments."""

_AUX_ASSISTED_ELEMENTS: Final[tuple[str, ...]] = ("alpha_m", "mg_h")
"""Elements whose information channel is aux+prior-dominated (CMI ≈ 0 nats given parallax,
photometry, extinction, position).

Historically (v5, through 2026-04-26) these elements were demoted to Tier 2 whenever
kin_ood_flag fired, on the reasoning that aux-assisted predictions lose their prior
basis for kinematically anomalous stars. The 2026-05-03 redesign retired that demotion
because halo / accreted-debris stars are exactly the science target for users who want
them; the new symmetric label-Mahalanobis output-OOD gate replaces it. This tuple is
retained for the xp_abundance_type__<element> column (informational) and the sidecar
provenance, not for tier gating.

Note: the full 21-element ensemble has not yet been audited via the §3.3 promotion
protocol. Only Teff, logg, [M/H], [α/M], [Mg/H] have been promoted in Stream 1.
This set will grow after the audit completes on the 21-label model."""

_PER_ELEMENT_PRED_COL: Final[dict[str, str]] = {
    "teff": "teff_pred",
    "logg": "logg_pred",
    "mh": "mh_pred",
    "fe_h": "fe_h_pred",
    "alpha_m": "alpha_m_pred",
    "mg_h": "mg_h_pred",
    "c_h": "c_h_pred",
    "n_h": "n_h_pred",
    "o_h": "o_h_pred",
    "na_h": "na_h_pred",
    "al_h": "al_h_pred",
    "si_h": "si_h_pred",
    "s_h": "s_h_pred",
    "k_h": "k_h_pred",
    "ca_h": "ca_h_pred",
    "ti_h": "ti_h_pred",
    "v_h": "v_h_pred",
    "cr_h": "cr_h_pred",
    "mn_h": "mn_h_pred",
    "ni_h": "ni_h_pred",
    "ce_h": "ce_h_pred",
}
"""Map element name to the prediction column in the inference parquet."""

_PER_ELEMENT_SIGMA_COL: Final[dict[str, str]] = {
    "teff": "teff_sigma",
    "logg": "logg_sigma",
    "mh": "mh_sigma",
    "fe_h": "fe_h_sigma",
    "alpha_m": "alpha_m_sigma",
    "mg_h": "mg_h_sigma",
    "c_h": "c_h_sigma",
    "n_h": "n_h_sigma",
    "o_h": "o_h_sigma",
    "na_h": "na_h_sigma",
    "al_h": "al_h_sigma",
    "si_h": "si_h_sigma",
    "s_h": "s_h_sigma",
    "k_h": "k_h_sigma",
    "ca_h": "ca_h_sigma",
    "ti_h": "ti_h_sigma",
    "v_h": "v_h_sigma",
    "cr_h": "cr_h_sigma",
    "mn_h": "mn_h_sigma",
    "ni_h": "ni_h_sigma",
    "ce_h": "ce_h_sigma",
}
"""Map element name to the per-element predicted-sigma column."""

_PER_ELEMENT_SIGMA_INFLATED_THRESHOLD: Final[dict[str, float]] = {
    "teff": 150.0,  # K
    "logg": 0.30,  # dex
    "mh": 0.20,  # dex
    "fe_h": 0.15,  # dex; v1.1 placeholder (empirical-Bayes ceiling ≈ 1·σ_train)
    "alpha_m": 0.05,  # dex (tightened 2026-04-26 from 0.10 → 0.05; ablation test
    #      release/test_ablations_2026-04-26 showed 23% T1 RMSE
    #      improvement on [α/M] at 0.5×σ_train, accepting
    #      ~14 pp T1-fraction loss on this element)
    "mg_h": 0.20,  # dex
    "c_h": 0.15,  # dex; v1.1 placeholder
    "n_h": 0.15,  # dex; v1.1 placeholder
    "o_h": 0.15,  # dex; v1.1 placeholder
    "na_h": 0.15,  # dex; v1.1 placeholder
    "al_h": 0.15,  # dex; v1.1 placeholder
    "si_h": 0.15,  # dex; v1.1 placeholder
    "s_h": 0.15,  # dex; v1.1 placeholder
    "k_h": 0.15,  # dex; v1.1 placeholder
    "ca_h": 0.15,  # dex; v1.1 placeholder
    "ti_h": 0.15,  # dex; v1.1 placeholder
    "v_h": 0.15,  # dex; v1.1 placeholder
    "cr_h": 0.15,  # dex; v1.1 placeholder
    "mn_h": 0.15,  # dex; v1.1 placeholder
    "ni_h": 0.15,  # dex; v1.1 placeholder
    "ce_h": 0.15,  # dex; v1.1 placeholder
}
"""Per-element sigma above which the prediction is flagged as inflated (prior-collapse).

Provenance of the values
------------------------
The five Stream-1 tuned entries (Teff 150 K, log g 0.30 dex, [M/H] 0.20 dex,
[α/M] 0.05 dex, [Mg/H] 0.20 dex) are calibrated on the Stream-1 holdout split via
empirical-Bayes shrinkage prior (τ=50). Concretely:

- σ_train per element is the per-cell standard deviation of the residuals
  (μ_pred − y_true) on the training partition after frozen-stats z-score (see
  ``training.py:fit_label_scaler``).
- The empirical-Bayes shrinkage prior with τ=50 caps σ_pred at the σ_train scale,
  beyond which the regressor is necessarily collapsing onto the prior mean rather
  than reading information from the spectrum. Mathematically: a posterior σ that
  exceeds σ_train means CMI(spectrum; label | aux) ≈ 0 nats for that star.

The alpha/M threshold was tightened to 0.05 dex ≈ 0.5·σ_train on 2026-04-26
after the per-cell-gate ablation study showed alpha/M was the only element whose
Tier 1 RMSE was meaningfully responsive to a tighter sigma cut (23 % T1 RMSE
improvement at 0.5·σ_train, accepting ~14 pp T1-fraction loss on this element).
The other elements were within Pareto-optimal tolerance of the 1·σ_train ceiling.

v1.1 placeholders for the new 16 elements (Fe/H through Ce/H)
-------------------------------------------------------------
The 16 new abundances have not yet undergone §3.3 promotion audit on the
21-label ensemble. Their σ-inflation thresholds are conservative placeholders
set to 0.15 dex (between the αM tight-threshold and the broader 3-element
ceiling), derived from typical APOGEE DR19 per-element uncertainties for
mid-tier elements. After the 21-label audit completes, these values will be
re-fit using the same empirical-Bayes methodology as Stream 1 tuning. See
docs/plan/04_pipeline2_main.md for the audit timeline.

Demotion semantics
------------------
Stars whose σ_pred for a given element exceeds the threshold have that element's
``release_tier__<element>`` demoted to 2 (statistical-only); the row-max
``release_tier`` then promotes the row to Tier 2 (or remains Tier 3 if a hard
OOD flag also fires). This removes ~25 % of the broad Stream 3 release,
including the 74 k prior-collapse spike at ([M/H], [α/M]) = (−1.05, +0.11),
while preserving the residual Tier 1 disc bimodality.

References
----------
- HIGH_SIGMA_RESCUE_REPORT.md (2026-04-25): empirical-Bayes derivation, σ_train
  histograms, prior-collapse spike analysis. Repo-internal; not committed.
- release/test_ablations_2026-04-26/REPORT.md: per-cell-gate ablation justifying
  the alpha/M tightening. Also repo-internal.
- AGENTS.md invariant 16 (frozen v1 stats fingerprint): σ_train values are
  bound to the v1 Hermite z-score basis; refitting requires a re-derivation.
"""

_CATALOGUE_SCHEMA_VERSION: Final[int] = 6
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
  from caveats. ``mode_ambiguous_flag`` is no longer global, it now only
  demotes [α/M] (the only element where the bimodality matters). ``alpha_m``
  σ-inflation threshold tightened from 0.10 → 0.05 dex. The dropped flag
  *columns* are still emitted by upstream modules for diagnostics; they no
  longer feed the tier.
- v6 (2026-04-28, 21-element expansion): expanded _PRED_COLS, _ABUNDANCE_ELEMENTS,
  _PER_ELEMENT_PRED_COL, _PER_ELEMENT_SIGMA_COL, and _PER_ELEMENT_SIGMA_INFLATED_THRESHOLD
  to cover all 21 elements (Teff, log g, [M/H], [Fe/H], [α/M], [Mg/H], [C/H], [N/H],
  [O/H], [Na/H], [Al/H], [Si/H], [S/H], [K/H], [Ca/H], [Ti/H], [V/H], [Cr/H], [Mn/H],
  [Ni/H], [Ce/H]). New elements use placeholder σ-thresholds (0.15 dex) pending §3.3
  audit completion on the 21-label ensemble. Tier assignments still match Stream-1
  (only Teff/logg/mh promoted; others await audit)."""


def assign_xp_abundance_type(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Assign per-element ``xp_abundance_type`` for spectrum-dominant vs aux-assisted.

    Returns a dict mapping element name (e.g. "mh") to a string Series where each
    element is "spectrum_dominant" or "aux_assisted".

    Design rationale: per-element string columns are consumer-friendly (one column
    per label type, clearly named). Aux-assisted labels are those where CMI
    conditional on auxiliary features falls below 0.02 nats (research_brief.md §3.3.1).

    v1.1 note: The 16 new elements (Fe/H through Ce/H) lack full audit data and are
    currently assigned "spectrum_dominant" as a placeholder. After the 21-label audit
    (docs/plan/04_pipeline2_main.md), some may be reclassified to "aux_assisted" if
    their CMI conditional on parallax/photometry/extinction drops below 0.02 nats.

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
        # Stream-1 tuned (audited, promoted):
        #   Teff, logg, [M/H] → spectrum_dominant
        #   [α/M], [Mg/H] → aux_assisted (conditional CMI < 0.02)
        # New elements (v1.1 placeholders):
        #   [Fe/H], [C/H], [N/H], [O/H], [Na/H], [Al/H], [Si/H], [S/H], [K/H],
        #   [Ca/H], [Ti/H], [V/H], [Cr/H], [Mn/H], [Ni/H], [Ce/H] → spectrum_dominant
        #   (pending audit classification; some may move to aux_assisted)
        if elem in ("teff", "logg", "mh"):
            result[elem] = pd.Series("spectrum_dominant", index=df.index, dtype="string")
        elif elem in ("alpha_m", "mg_h"):
            result[elem] = pd.Series("aux_assisted", index=df.index, dtype="string")
        elif elem in (
            "fe_h",
            "c_h",
            "n_h",
            "o_h",
            "na_h",
            "al_h",
            "si_h",
            "s_h",
            "k_h",
            "ca_h",
            "ti_h",
            "v_h",
            "cr_h",
            "mn_h",
            "ni_h",
            "ce_h",
        ):
            result[elem] = pd.Series("spectrum_dominant", index=df.index, dtype="string")
        else:
            result[elem] = pd.Series("unknown", index=df.index, dtype="string")
    return result


def assign_kin_ood_flag(df: pd.DataFrame) -> pd.Series:
    """Assign kinematic OOD flag, placeholder, all False (v6, 2026-05-03).

    Retained as an all-False placeholder for backward-compatible parquet schemas.
    The kinematic-OOD demotion of aux-assisted elements was retired on
    2026-05-03 in favor of the symmetric label-Mahalanobis output-OOD gate
    (``label_extrapolation_flag``). See ``docs/decisions/ADR-0016_tier_v6_
    mahalanobis_redesign.md`` for the migration rationale.

    Parameters
    ----------
    df : pd.DataFrame
        Prediction frame.

    Returns
    -------
    pd.Series
        Boolean Series, all False.
    """
    return pd.Series(False, index=df.index, dtype="bool")


def assign_dist_prior_dominated(
    df: pd.DataFrame, *, parallax_snr_threshold: float = 5.0
) -> pd.Series:
    """Per-star boolean: True where the Bailer-Jones distance is prior-dominated.

    The Phase A2 catalog schema reserved a `dist_prior_dominated` column but did not
    compute it. This function fills the gap. A star is flagged when its parallax SNR
    falls below `parallax_snr_threshold` (default 5; equivalent to σ_π/π > 0.2, the
    Bailer-Jones+2021 boundary between geometric and prior-dominated photogeometric
    distances) OR when its parallax is non-finite or non-positive (Bailer-Jones uses
    a Galactic-prior distance for these by default).

    Source columns expected:
    - ``parallax`` (mas) and ``parallax_error`` (mas), preferred.
    - or ``parallax_over_error`` (already a SNR), used as a fallback.

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
    1. ``phot_g_mean_mag_corr``, Riello+2021 Appendix-A-corrected G magnitude
       (preferred; emitted by ``data/gaia_corrections.py`` for Stream 1).
    2. ``phot_g_mean_mag``, raw Gaia DR3 G magnitude (fallback; used by Stream 3
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
    """Per-element release tier ∈ {1, 2, 3} (v6 schema, 2026-05-03).

    For each released element, composes a tier from:

    - **Tier 3** (do not release): NaN prediction for that element, OR
      ``ood_joint_flag`` (XP-block Mahalanobis input-OOD) firing.
    - **Tier 2** (statistical / ensemble only): ``label_extrapolation_flag``
      firing (predicted label tuple outside the APOGEE-truth Mahalanobis envelope
      at the 99th percentile, fit by
      ``run_pipeline1_inference._fit_label_mahalanobis_bundle``), OR an
      element-specific caveat from ``_PER_ELEMENT_CAVEAT_FLAGS`` (currently empty).
    - **Tier 1** (per-star science): everything else.

    The ``label_extrapolation_flag`` column MUST be present in ``df``; absence is
    treated as a contract violation and raises ``ValueError`` to prevent silent
    T2 → T1 promotions on stale parquets. NaN predictions and missing
    ``ood_joint_flag`` are tolerated (NaN → T3 per element; missing ``ood_joint_flag``
    is conservatively treated as False).

    Diagnostic-only columns NOT consumed by this function (see module docstring):
    ``prediction_sigma_inflated__<element>``, ``kin_ood_flag``,
    ``mode_ambiguous_flag``, ``regime_b_flag``, ``aux_missing_any``,
    ``dist_prior_dominated``. These were active T2/T3 gates through v5
    (2026-04-26) and were retired in the 2026-05-03 redesign in favor of the
    symmetric input/output Mahalanobis pair.

    Parameters
    ----------
    df : pd.DataFrame
        Prediction parquet annotated by ``run_pipeline1_inference.py``.
        MUST contain ``label_extrapolation_flag``.

    Returns
    -------
    dict[str, pd.Series]
        Maps element name (e.g. ``"alpha_m"``) to ``int8`` Series of per-element tier.

    Raises
    ------
    ValueError
        If ``label_extrapolation_flag`` is not present in ``df``. This guards
        against silently demoting T2 candidates to T1 when an upstream inference
        run failed to fit the label-Mahalanobis bundle, or when the parquet
        predates the 2026-05-03 redesign.

    Notes
    -----
    History: v3-v4 used five global caveats + three OOD flags. The 2026-04-26
    ablation (``release/test_ablations_2026-04-26/REPORT.md``) retired all but
    ``ood_joint_flag``, the per-element ``mode_ambiguous`` carve-out for α/M,
    and the aux-assisted ``kin_ood_flag`` demotion. The 2026-05-03 redesign
    further retired ``mode_ambiguous_flag`` (~46 % fire rate, not justified),
    ``kin_ood_flag`` (demoted halo science targets), and the σ-inflation
    thresholds (perceived as σ-tail cherry-picking), and replaced them with
    the symmetric ``label_extrapolation_flag``. See
    ``docs/decisions/ADR-0016_tier_v6_mahalanobis_redesign.md``.
    """
    if "label_extrapolation_flag" not in df.columns:
        raise ValueError(
            "assign_per_element_release_tier: 'label_extrapolation_flag' column is "
            "required (v6 schema, 2026-05-03) but not present in the input DataFrame. "
            "This column is emitted by run_pipeline1_inference._fit_label_mahalanobis_bundle "
            "and must be plumbed through any upstream join. Without it, Tier-2 "
            "candidates would be silently promoted to Tier 1. See "
            "docs/decisions/ADR-0016_tier_v6_mahalanobis_redesign.md for the "
            "v5 → v6 migration notes."
        )

    idx = df.index

    # Tier 3, INPUT out-of-domain.
    #   Joint OOD flags fire for ALL elements (XP-block Mahalanobis OOD
    #   invalidates every prediction). Tier 3 for every element on hit.
    ood = pd.Series(False, index=idx)
    for col in _OOD_FLAGS:
        if col in df.columns:
            ood = ood | _coerce_flag_series(df[col])

    # Tier 2, OUTPUT out-of-domain (v6, 2026-05-03).
    #   The 5-D Mahalanobis distance of the predicted label vector against
    #   the APOGEE-truth training-label envelope (fit by
    #   run_pipeline1_inference._fit_label_mahalanobis_bundle, threshold at
    #   the 99th training-truth percentile). A True flag means the model
    #   produced a label tuple outside the APOGEE training envelope, either
    #   a rare-but-real object or a model extrapolation, both of which
    #   warrant scrutiny. The presence of the column is asserted at the top
    #   of this function; here we only need to coerce its dtype.
    label_extrap = _coerce_flag_series(df["label_extrapolation_flag"])

    per_element: dict[str, pd.Series] = {}
    for elem in _ABUNDANCE_ELEMENTS:
        # Per-element NaN prediction → Tier 3.
        pred_col = _PER_ELEMENT_PRED_COL.get(elem)
        elem_nan = pd.Series(False, index=idx)
        if pred_col is not None and pred_col in df.columns:
            elem_nan = df[pred_col].isna()

        # Per-element caveat flags: empty as of 2026-05-03. Retained as a
        # forward-compatible hook so future per-element demoters (e.g. a
        # restored mode_ambiguous narrowing for α/M) can be added without
        # touching the tier-composition logic below.
        elem_specific_caveat = pd.Series(False, index=idx)
        for col in _PER_ELEMENT_CAVEAT_FLAGS.get(elem, ()):
            if col in df.columns:
                elem_specific_caveat = elem_specific_caveat | _coerce_flag_series(df[col])

        tier3 = elem_nan | ood
        tier2 = label_extrap | elem_specific_caveat

        tier = pd.Series(1, index=idx, dtype="int8")
        tier[tier2] = 2
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
        if sigma_col is None or sigma_col not in df.columns or threshold is None:
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
     - ``release_tier__<element>`` (per-element tier) for all 21 elements
     - ``xp_abundance_type__<element>`` ∈ {"spectrum_dominant", "aux_assisted"}
       for all 21 elements (Teff, logg, [M/H], [Fe/H], [α/M], [Mg/H], and 16 others)
     - ``prediction_sigma_inflated__<element>`` (per-element σ-tail flag,
       diagnostic-only as of v6 / 2026-05-03; not a tier driver)
     - ``kin_ood_flag`` (placeholder, False; diagnostic-only as of v6, not a tier driver)
     - ``g_mag_bin`` ∈ {"bright", "mid", "faint"}
     - ``dist_prior_dominated`` (Bailer-Jones parallax-SNR boundary)
     - ``prediction_sigma_inflated_any`` (convenience: any element flag)

     Emits / refreshes the ``*.release_tier.json`` sidecar next to the parquet
     capturing counts, flag-column provenance, and catalog schema version.
     The parquet's main provenance sidecar (``*.provenance.json``) is not touched
    , that records upstream inference, not this annotation step.

     Returns
     -------
     dict
         ``{"n_rows": N, "counts": {1: ..., 2: ..., 3: ...}}``.
    """
    _logger.info(
        "annotate_parquet: reading %s (catalog schema v%d)", path, _CATALOGUE_SCHEMA_VERSION
    )
    df = pd.read_parquet(path)
    _logger.info("annotate_parquet: loaded %d rows from %s", len(df), path.name)

    # Per-element abundance type (spectrum_dominant vs aux_assisted), emit FIRST so
    # the per-element tier composition has access to it implicitly via the constants.
    abundance_types = assign_xp_abundance_type(df)
    for elem, series in abundance_types.items():
        col_name = f"xp_abundance_type__{elem}"
        df[col_name] = series

    # kin_ood_flag retired 2026-05-03, was demoting aux-assisted elements
    # to T2 based on a disc-kinematics envelope. Per the redesign discussion,
    # halo / accreted-debris stars are *exactly* the science target for users
    # who want them; flagging them as suspect by default was the wrong move.
    # If the column is present in input data it is left alone (informational);
    # release.py no longer generates or reads it.

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
        "annotate_parquet: %s wrote %d rows; tiers T1=%d (%.1f%%) T2=%d (%.1f%%) T3=%d (%.1f%%)",
        path.name,
        int(len(df)),
        counts.get(1, 0),
        100.0 * counts.get(1, 0) / max(len(df), 1),
        counts.get(2, 0),
        100.0 * counts.get(2, 0) / max(len(df), 1),
        counts.get(3, 0),
        100.0 * counts.get(3, 0) / max(len(df), 1),
    )
    summary: dict[str, int | dict[int, int]] = {
        "n_rows": int(len(df)),
        "counts": counts,
    }

    sidecar = path.with_name(path.stem + ".release_tier.json")
    # Dynamically build release_columns_added for all 21 elements.
    release_cols = ["release_tier"]
    for elem in _ABUNDANCE_ELEMENTS:
        release_cols.append(f"release_tier__{elem}")
    for elem in _ABUNDANCE_ELEMENTS:
        release_cols.append(f"xp_abundance_type__{elem}")
    # kin_ood_flag is NOT in this list as of v6 (2026-05-03): annotate_parquet
    # no longer auto-emits it. If upstream joined it in, it survives unchanged.
    release_cols.extend(
        [
            "g_mag_bin",
            "dist_prior_dominated",
        ]
    )
    for elem in _ABUNDANCE_ELEMENTS:
        release_cols.append(f"prediction_sigma_inflated__{elem}")
    release_cols.append("prediction_sigma_inflated_any")

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
        "release_columns_added": release_cols,
        "prediction_sigma_inflated_thresholds": dict(_PER_ELEMENT_SIGMA_INFLATED_THRESHOLD),
        "expected_upstream_columns": [
            # Required for v6 tier assignment; assign_per_element_release_tier
            # raises ValueError if label_extrapolation_flag is missing.
            "label_extrapolation_flag",
            # Read if present (defensive default False); fires Tier 3 on hit.
            "ood_joint_flag",
        ],
        "diagnostic_only_columns": [
            # Flags retired from tier gating but kept as informational columns
            # in the published parquet. Upstream modules may still emit them;
            # the release tier does not read them.
            #
            # Retired 2026-05-03 (v6 redesign):
            "prediction_sigma_inflated__<element>",  # σ-tail flag, was T2 demoter
            "kin_ood_flag",  # disc-kinematics envelope, was aux-T2
            "mode_ambiguous_flag",  # disc-bimodality boundary, was α/M T2
            # Retired 2026-04-26 (v5 ablation):
            "ood_aux_mahalanobis_flag",
            "latent_support_flag",
            "regime_b_flag",
            "ood_disagreement_flag",
            "aux_missing_any",
            "dist_prior_dominated",
        ],
        "aux_assisted_elements": list(_AUX_ASSISTED_ELEMENTS),
        "tier_gating_logic": (
            "v6 schema (2026-05-03). "
            "Tier 3 if ood_joint_flag (XP-block Mahalanobis input-OOD) OR per-element "
            "NaN prediction. "
            "Tier 2 if label_extrapolation_flag (5-D Mahalanobis on predicted "
            "(Teff, log g, [M/H], [α/M], [Mg/H]) outside the APOGEE-truth p99 "
            "envelope) OR per-element caveat in _PER_ELEMENT_CAVEAT_FLAGS "
            "(currently empty). "
            "Tier 1 otherwise. Composite release_tier = row-max across elements. "
            "σ-inflation thresholds, kin_ood_flag, and mode_ambiguous_flag are "
            "DIAGNOSTIC-ONLY in v6, see docs/decisions/ADR-0016_tier_v6_"
            "mahalanobis_redesign.md for the v5 → v6 migration rationale."
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
