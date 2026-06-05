# Pipeline 1 Release Catalog Schema

**Version:** 6
**Last Modified:** 2026-05-03
**Release Tag:** `pipeline1-v1.1-21label-2026-04-29` (production model); v6 supersedes v5 with 21-element predictions and associated covariance structure. v1–v5 historical.

> **2026-05-03 tier-gating redesign. Read this before the table entries below.**
> The Tier-2 demotion gates were redesigned on 2026-05-03. The active gates as of v6 are:
>
> - **Tier 3** if `ood_joint_flag` (XP-block Mahalanobis input-OOD) **OR** per-element NaN prediction.
> - **Tier 2** if `label_extrapolation_flag` (5-D Mahalanobis on predicted `(Teff, log g, [M/H], [α/M], [Mg/H])` outside the APOGEE-truth p99 envelope) **OR** any element-specific caveat in `_PER_ELEMENT_CAVEAT_FLAGS` (currently empty).
> - **Tier 1** otherwise.
>
> The following columns were active gates through v5 but are **diagnostic-only as of v6**: `prediction_sigma_inflated__<element>` (per-element σ-tail flag), `kin_ood_flag` (disc-kinematics envelope), `mode_ambiguous_flag` (α/M bimodality boundary). They are still emitted to the parquet as informational columns; downstream users can apply their own filters but the release tier no longer consumes them. See `docs/decisions/ADR-0016_tier_v6_mahalanobis_redesign.md` for the migration rationale.
>
> Where this document still describes σ-inflation, `kin_ood_flag`, or `mode_ambiguous_flag` as "active gates," treat that wording as the historical v5 contract and use the v6 logic above.

## Overview

This document specifies the column contracts for Pipeline 1 v1.1 release catalogs materialized by `src/arqueogal/xp_abundances/main/release.py`. Every released Parquet carries a companion `*.release_tier.json` sidecar capturing schema version, tier counts, and release-column provenance.

The v1.1 model produces 21 stellar labels (3 atmospheric parameters + 18 elemental abundances), each with posterior mean, posterior standard deviation (aleatoric-intrinsic + feature-noise-propagated), and per-element release tier. A 21×21 block-Cholesky covariance matrix is also emitted. Four-way evolutionary-stage diagnostics (RGB, HeCB, OOD_evolved, OOD_unevolved) are provided for robustness tracking.

The catalog is built in stages:
1. **Inference** produces 21 predictions, 21×21 block-Cholesky covariance, feature-noise-marginalised σ per element, and OOD flags.
2. **Release annotation** adds per-element release tier, per-element abundance-type flags (spectrum-dominant or aux-assisted), kinematic OOD flag, and magnitude binning.

A per-star science publication uses **Tier 1 only**. Aggregate studies may include Tier 2 with explicit caveat. Tier 3 is provided for methodology work and alternate filtering strategies; it is not released in published catalogs.

---

## Column Reference

### Identifiers

| Column | Type | Unit | Valid Range | NaN Meaning | Notes |
|---|---|---|---|---|---|
| `source_id` | uint64 | | [0, 2^63) | N/A | Gaia DR3 source identifier. Immutable. |

### Astrometry (Gaia DR3, Lindegren+2021 parallax zero-point corrected)

| Column | Type | Unit | Valid Range | NaN Meaning | Notes |
|---|---|---|---|---|---|
| `ra` | float64 | degree | [0, 360) | impossible | Right ascension, ICRS J2000. |
| `dec` | float64 | degree | [-90, 90] | impossible | Declination, ICRS J2000. |
| `parallax_corr` | float32 | mas | > 0 | unobservable | Parallax (Lindegren+2021 zpt correction applied). |
| `parallax_error` | float32 | mas | > 0 | unobservable | Formal parallax uncertainty (Gaia DR3). |
| `pmra` | float32 | mas/yr | unbounded | unobservable | Proper motion RA, incl. cos(dec). |
| `pmra_error` | float32 | mas/yr | > 0 | unobservable | PM RA uncertainty (Gaia DR3). |
| `pmdec` | float32 | mas/yr | unbounded | unobservable | Proper motion Dec. |
| `pmdec_error` | float32 | mas/yr | > 0 | unobservable | PM Dec uncertainty (Gaia DR3). |
| `ra_dec_corr` | float32 | | [-1, 1] | N/A | Correlation ρ(RA, Dec). |
| `ra_parallax_corr` | float32 | | [-1, 1] | N/A | Correlation ρ(RA, parallax). |
| `ra_pmra_corr` | float32 | | [-1, 1] | N/A | Correlation ρ(RA, PM_RA). |
| ... (7 more astrometric correlations) | float32 | | [-1, 1] | N/A | See data_acquisition.md §3.6. |

All ten upper-triangular astrometric correlations from `gaiadr3.gaia_source`. Essential for kinematic and parallax-distance covariance propagation.

### Photometry (Gaia DR3, Riello+2021 G-magnitude correction applied)

| Column | Type | Unit | Valid Range | NaN Meaning | Notes |
|---|---|---|---|---|---|
| `phot_g_mean_mag_corr` | float32 | mag | [0, 22] | unobservable | G-band magnitude (Riello+2021 cubic correction). Used to compute `g_mag_bin`. |
| `bp_rp` | float32 | mag | [-1, 10] | unobservable | BP − RP color (Gaia). |
| `bp_g` | float32 | mag | [-1, 5] | unobservable | BP − G color. |
| `g_rp` | float32 | mag | [-1, 5] | unobservable | G − RP color. |

### XP Spectral Coefficients (Ye+2024 NN flux-correction + normalization + z-scoring)

| Column | Type | Unit | Valid Range | NaN Meaning | Notes |
|---|---|---|---|---|---|
| `bp_coeffs_norm` | list[float32] | | [-5, 5] per coeff | corrupted spectrum | 55 Hermite coefs, BP band. L2-normalized by coef 0, then log-transformed & z-scored (§6.4). |
| `rp_coeffs_norm` | list[float32] | | [-5, 5] per coeff | corrupted spectrum | 55 Hermite coefs, RP band. Same normalization as BP. |
| `bp_coeff_errs_norm` | list[float32] | | > 0 per coeff | missing errors | Coefficient uncertainties, BP band. |
| `rp_coeff_errs_norm` | list[float32] | | > 0 per coeff | missing errors | Coefficient uncertainties, RP band. |
| `bp_c0_z` | float32 | | [-5, 5] | rare | Z-scored log(coef_0), BP band. Scalar. |
| `rp_c0_z` | float32 | | [-5, 5] | rare | Z-scored log(coef_0), RP band. Scalar. |

All XP preprocessing is fixed and deterministic (research_brief.md §3.3.2, data_acquisition.md §6.4). Z-score basis (per-coefficient mean and std) is frozen from Pipeline 1 v1.0 training (fingerprint `0d34b565...`).

### Distance (Photometric parallax via Edenhofer+2024 + Lallement+2022 + SFD + neighborhood median)

| Column | Type | Unit | Valid Range | NaN Meaning | Notes |
|---|---|---|---|---|---|
| `r_med_photogeo` | float32 | kpc | (0, 3] | unobservable | Median distance estimate. Budget-compliant 3-map fusion (d<1.25 kpc: Edenhofer; 1.25–3 kpc: Lallement; >3: SFD+neighborhood). |
| `r_lo_photogeo` | float32 | kpc | (0, 3] | unobservable | Lower (16th percentile) distance. |
| `r_hi_photogeo` | float32 | kpc | (0, 3] | unobservable | Upper (84th percentile) distance. |

Do not use Bayestar19 (budget). Do not use astroquery.gaia; use pyvo (AIP, GAVO, ESA, VizieR async TAP).

### Extinction (Dust optical depth along line of sight)

| Column | Type | Unit | Valid Range | NaN Meaning | Notes |
|---|---|---|---|---|---|
| `av_los` | float32 | mag | [0, 5] | missing dust model | A_V along line of sight (fusion of Edenhofer+2024, Lallement+2022, SFD). Budget source: min(distance, 3 kpc). |
| `av_los_source` | int8 | | {0, 1, 2, -1} | algorithm error | Categorical: 0=Edenhofer, 1=Lallement, 2=SFD, -1=missing. |
| `av_nbhd_median` | float32 | mag | [0, 5] | missing | Neighborhood-median A_G at target position (§8.3). Essential for G-magnitude color-excess correction when per-map A_V unavailable. |
| `av_nbhd_std` | float32 | mag | [0, 1] | missing | Neighborhood std. Used as model-uncertainty proxy. |

### Diagnostics (Andrae+2023 XGBoost labels, cross-reference only)

| Column | Type | Unit | Valid Range | NaN Meaning | Notes |
|---|---|---|---|---|---|
| `teff_xgboost` | float32 | K | [3500, 6500] | not applicable | Andrae+2023 T_eff for methodology comparison. Not used in release decisions. |
| `logg_xgboost` | float32 | dex | [0, 5] | not applicable | Andrae+2023 log g. Not used in release decisions. |
| `mh_xgboost` | float32 | dex | [-2, 1] | not applicable | Andrae+2023 [M/H]. Not used in release decisions. |
| `evolutionary_stage_andrae` | string | | {"RGB", "RC", "RGB_candidate"} | not applicable | Andrae+2023 evolutionary stage (optional, present only if explicitly fetched). |

### Quality Flags (Gaia DR3)

| Column | Type | Unit | Valid Range | NaN Meaning | Notes |
|---|---|---|---|---|---|
| `flag_bad` | bool | | {True, False} | False (default) | Gaia quality gate: False if (good astrometry AND good photometry AND signal-to-noise > 70). True means any gate failed; Tier-3 rows filtered by this. |
| `ruwe` | float32 | | [0, 2] (typical) | impossible | Renormalized unit weight error (Gaia). Measure of astrometric scatter relative to formal errors. Values >1.4 suggest binarity or large astrometric noise. |
| `dist_conflict` | bool | | {True, False} | False (default) | Present only if StarHorse2 parallax included. True if parallax disagreement with Gaia >20% (methodology flag, not a release gate). |

### Predictions: 21 Elements (v1.1 Schema)

The v1.1 model predicts three atmospheric parameters (Teff, log g, [M/H]) and 18 elemental abundances ([X/H] for X in C, N, O, Na, Mg, Al, Si, S, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni). Each element carries four columns: posterior mean, aleatoric-intrinsic σ, feature-noise-propagated σ, and inflated-σ flag.

For each element E in {teff, logg, mh, c_h, n_h, o_h, na_h, mg_h, al_h, si_h, s_h, k_h, ca_h, sc_h, ti_h, v_h, cr_h, mn_h, fe_h, co_h, ni_h}:

| Column | Type | Unit | Valid Range | NaN Meaning | Notes |
|---|---|---|---|---|---|
| `<E>_pred` | float32 | dex (except Teff in K) | element-dependent | NaN → Tier 3 | Posterior mean estimate. Teff in Kelvin; all others dimensionless [X/H]. |
| `<E>_sigma` | float32 | dex (except Teff in K) | (0, 1000] | rare | Aleatoric-intrinsic uncertainty from posterior. Per-element; does not include feature-noise propagation. |
| `<E>_sigma_total` | float32 | dex (except Teff in K) | (0, 1000] | rare | Total posterior σ after feature-noise marginalisation: sqrt(σ^2 + σ_noise^2). Recommended for downstream error bars. |
| `<E>_sigma_feature_noise_propagated` | float32 | dex (except Teff in K) | (0, 1000] | rare | Feature-noise contribution σ_noise estimated at inference time via analytical gradient-norm marginalisation. Optional, present if feature-noise training was enabled. |
| `prediction_sigma_inflated__<E>` | bool | | {True, False} | False (default) | **Diagnostic-only as of v6 (2026-05-03).** True when σ > element-specific prior-collapse threshold (thresholds TBD per v1.1 audit; provisional from v1.0 σ_train). Was a per-element Tier-2 gate through v5; the 2026-05-03 redesign retired it in favor of `label_extrapolation_flag`. Downstream users can still filter on it but the release tier no longer reads it. |

**Auxiliary classification:** Spectrum-dominant elements (Teff, logg, [M/H]) rely on XP spectral dominance. Aux-assisted elements (all 18 abundances) rely on auxiliary kinematics and population priors; they are reliable for disc stars but unreliable for halo/accreted-debris.

### Out-of-Distribution Flags

> **v6 (2026-05-03):** the active tier gates are `ood_joint_flag` (Tier 3,
> input-OOD) and `label_extrapolation_flag` (Tier 2, output-OOD). The
> companion percentile columns are continuous severity rankings; both are
> always emitted. `latent_support_flag` and `ood_aux_mahalanobis_flag` are
> diagnostic-only since v5. See ADR-0015 (v5) and ADR-0016 (v6).

| Column | Type | Unit | Valid Range | NaN Meaning | Notes |
|---|---|---|---|---|---|
| `ood_joint_flag` | bool | | {True, False} | False (default) | **v5 / v6 active gate.** Hard OOD gate: Mahalanobis distance in 108-D XP feature space at p=0.99 (chi-squared, training envelope). True → Tier 3. |
| `label_extrapolation_flag` | bool | | {True, False} | False (default) | **v6 active gate (2026-05-03).** True when the predicted 5-D label vector (Teff, log g, [M/H], [α/M], [Mg/H]) lies outside the 99th-percentile envelope of the APOGEE-truth Mahalanobis distribution. Symmetric output-OOD partner to `ood_joint_flag`. True → Tier 2. Fit by `run_pipeline1_inference._fit_label_mahalanobis_bundle` (5-D empirical covariance, regularization 1e-8). See ADR-0016. |
| `ood_mahalanobis_percentile` | float32 | | [0, 1] | NaN if input is NaN | **v6 diagnostic.** Empirical-CDF percentile of the per-star XP-block Mahalanobis distance against the training distribution. Continuous companion to `ood_joint_flag` (which fires at percentile > 0.99). Use for user-defined input-OOD severity filtering. Percentiles above the maximum training distance are clamped to 1.0. |
| `label_mahalanobis_percentile` | float32 | | [0, 1] | NaN if predictions are NaN | **v6 diagnostic.** Empirical-CDF percentile of the per-star 5-D label-Mahalanobis distance against the APOGEE-truth training distribution. Continuous companion to `label_extrapolation_flag` (fires at percentile > 0.99). Use for user-defined output-OOD severity filtering. Percentiles above the maximum training distance are clamped to 1.0; do not interpret > 0.99 as a continuous severity ranking. |
| `latent_support_flag` | bool | | {True, False} | False (default) | **v5 diagnostic-only (retired from gating).** Convex-hull surrogate on learned representations. Never fired on the Stream-1 holdout in the v5 ablation; column kept for diagnostic continuity. |
| `ood_aux_mahalanobis_flag` | bool | | {True, False} | False (default) | **v5 diagnostic-only (retired from gating).** Mahalanobis distance in the auxiliary-feature space (parallax, photometry, extinction, position) at p=0.99. Subsumed by `aux_missing_any` in practice; kept for diagnostic continuity. |

**APOGEE-truth envelope caveat for `label_extrapolation_flag`.** The 5-D Mahalanobis envelope is fit on APOGEE DR19 truth labels, which themselves carry a restricted observational window: Teff ∈ [4000, 5500] K, log g ∈ [1.0, 3.5], [M/H] ∈ [-2.0, +0.5], G ≲ 13.5 for the bright sample. Stream-3 stars predicted into label-space regions sparse or absent in APOGEE (cool dwarfs below the Teff cutoff, metal-poor halo, faint giants) will fall into Tier 2 by construction. This is selection-bias of the training reference, not a defect in the predictions. Users targeting halo / accreted-debris populations should treat T2 demotions as conservative-by-design and consult `label_mahalanobis_percentile` for graded severity.

### Caveat Flags (Structural, demote to Tier 2)

> **v5 (2026-04-26)**: only `prediction_sigma_inflated__<elem>` (all elements).
> `mode_ambiguous_flag` (α/M only), and `kin_ood_flag` (aux-assisted only) are
> active tier gates. `regime_b_flag`, `ood_disagreement_flag`.
> `aux_missing_any`, `dist_prior_dominated` are still emitted but
> diagnostic-only. See ADR-0015.

| Column | Type | Unit | Valid Range | NaN Meaning | Notes |
|---|---|---|---|---|---|
| `regime_b_flag` | bool | | {True, False} | False (default) | **v5 diagnostic-only (retired from gating).** Regime B (Galactic plane, warm RGB): systematic T_eff over-prediction ~1σ. Fires on ~0.04 % of stars; the v5 ablation showed no measurable T1+T2 RMSE effect. The systematic itself is a methods-paper finding (AGENTS.md footgun), not a release-blocker. |
| `mode_ambiguous_flag` | bool | | {True, False} | False (default) | **Diagnostic-only as of v6 (2026-05-03).** Disc α/M bimodality at fixed (Teff, log g, [M/H]). Was a per-element T2 gate on α/M through v5; retired because the flag fires on ~46 % of the cohort (the disc is genuinely bimodal at fixed Teff/log g/[M/H]) and demoting half the catalog was not justified. |
| `ood_disagreement_flag` | bool | | {True, False} | False (default) | **v5 diagnostic-only (retired from gating).** Designed for multi-member ensemble disagreement; cannot fire with a single-member ensemble. Re-evaluate when the ensemble grows to ≥ 2 members. |
| `aux_missing_any` | bool | | {True, False} | False (default) | **v5 diagnostic-only (retired from gating).** Any auxiliary feature (parallax, extinction, position) is NaN at inference. Stars carrying this flag had ~4-6 % T1 RMSE inflation if kept in T1, but T1+T2 RMSE was unchanged, i.e. the demotion was pure relabeling. Column retained as a soft caveat for consumer-side filtering. |
| `dist_prior_dominated` | bool | | {True, False} | False (default) | **v5 diagnostic-only (retired from gating).** True when the Bailer-Jones photogeometric distance is dominated by the Galactic prior rather than parallax (σ_π/π > 0.2). Never fired on the held-out test split; column retained for diagnostic continuity. |
| `prediction_sigma_inflated__<elem>` | bool | | {True, False} | False (default) | **Diagnostic-only as of v6 (2026-05-03)** (was v5 per-element gate). True when the regression-head predicted σ for that element exceeds the prior-collapse threshold: `teff_sigma > 150 K`, `logg_sigma > 0.30 dex`, `mh_sigma > 0.20 dex`, `alpha_m_sigma > 0.05 dex`, `mg_h_sigma > 0.20 dex`. Computed by `release.assign_prediction_sigma_inflated()`; emitted to the parquet for user filtering, no longer consulted by the release-tier composer. |
| `prediction_sigma_inflated_any` | bool | | {True, False} | False (default) | **v4 schema addition.** Row-OR aggregate over the five `prediction_sigma_inflated__<elem>` flags. Convenience column for consumers who only need the row-level "any element prior-collapsed" indicator without inspecting per-element flags. |

**v6 tier composition (applies after annotate_parquet, 2026-05-03).** Tier 3 if `ood_joint_flag` OR per-element NaN. Tier 2 if `label_extrapolation_flag` OR per-element caveat in `_PER_ELEMENT_CAVEAT_FLAGS` (currently empty). Tier 1 otherwise. Composite `release_tier` = row-max across elements. The σ-inflation, `kin_ood_flag`, and `mode_ambiguous_flag` columns are emitted as diagnostics but are not consumed by the tier composer; see ADR-0016 for the v5 → v6 migration rationale.

---

## Release-Annotation Columns (Schema v5, 2026-04-26)

These columns are added by `annotate_parquet()` at release time. They are optional in the inference schema but mandatory in published catalogs.

### Release Tier (composite)

| Column | Type | Unit | Valid Range | NaN Meaning | Notes |
|---|---|---|---|---|---|
| `release_tier` | int8 | | {1, 2, 3} | invalid | Composite **row-max** (most-conservative) over the per-element tiers. **Tier 1**: per-star science. **Tier 2**: `label_extrapolation_flag` (5-D Mahalanobis on predicted (Teff, log g, [M/H], [α/M], [Mg/H]) outside the APOGEE-truth p99 envelope). **Tier 3**: `ood_joint_flag` (XP-block Mahalanobis input-OOD) or NaN prediction. See ADR-0015 (v5) and ADR-0016 (v6) and §3 below. |

### Per-Element Release Tier (21 elements, v6 schema)

Twenty-one `int8` columns (one per element), encoding the element-specific tier independent of the row-max composite. The composite `release_tier` is `max(release_tier__teff, ..., release_tier__ni_h)`.

For spectrum-dominant elements (Teff, logg, [M/H]):
- Tier 3 if NaN or `ood_joint_flag`.
- Tier 2 if `prediction_sigma_inflated__<E>` (σ exceeds element-specific prior-collapse threshold).
- Tier 1 otherwise.

For aux-assisted elements (18 abundances):
- Tier 3 if NaN or `ood_joint_flag`.
- Tier 2 if `prediction_sigma_inflated__<E>` (σ exceeds threshold) OR `kin_ood_flag` (star is kinematically anomalous, disc prior does not apply).
- Tier 1 otherwise.

| Column | Type | Valid Range | Notes |
|---|---|---|---|
| `release_tier__<E>` (21 columns) | int8 | {1, 2, 3} | Per-element tier for element E. Consumers can filter by per-element tier to safely use [M/H] (spectrum-dominant) while excluding aux-assisted [α/M] / [Mg/H] from kin-OOD stars. |

The per-element columns let consumers filter at fine granularity. A galactic-structure paper using only [M/H] can keep `release_tier__mh == 1` rows even when `release_tier == 2` due to a kin-OOD or σ-inflation demotion affecting one of the 18 abundances.

### Per-Element Abundance Type (21 elements, v6 schema)

Twenty-one columns (one per element), classifying each prediction as spectrum-dominant or aux-assisted:

| Column | Type | Values | Meaning | Notes |
|---|---|---|---|---|
| `xp_abundance_type__<E>` (21 columns) | string | "spectrum_dominant" or "aux_assisted" | Prediction mechanism for element E. | Spectrum-dominant: XP spectrum is the primary constraint (Teff, logg, [M/H]). Aux-assisted: prediction relies on auxiliary kinematics and population priors (all 18 abundances). |

All abundance-type strings are lowercase. These columns enable consumers to filter by prediction-mechanism confidence. A galactic-structure paper using [M/H] (spectrum_dominant) can trust the predictions across the full magnitude range; a globular-cluster paper using elemental abundances (aux_assisted) should acknowledge the caveat in section 3.1 below.

**Rationale:** We emit 21 separate string columns rather than a single JSON-encoded composite. This choice prioritizes **consumer ergonomics**: the columns are directly filterable in SQL or Polars/DuckDB, and human readers can grep the CSV. A composite would require JSON parsing in the consumer's language.

### Auxiliary-Assisted Label Caveat (18 elemental abundances, v6 schema)

**Aux-assisted labels** (`xp_abundance_type == "aux_assisted"`) are model predictions where the conditional mutual information (CMI) between the XP spectrum and the label, *given auxiliary features* (parallax, photometry, extinction, position), falls below **0.02 nats** (research_brief.md §3.3.1, information-content audit §9.2).

In plain language: the 18 elemental abundances are learned primarily from the disc-population prior (derived from APOGEE training) and kinematic information. The XP spectrum contributes but does not dominantly constrain these labels independently.

**Implications for consumers:**

1. **Disc stars (solar vicinity, thin-disk kinematics):** aux-assisted predictions are reliable. The model has learned elemental-abundance bimodalities and correlations from APOGEE; XP provides a secondary consistency check, and auxiliary kinematics confirms disc membership.

2. **Halo, accreted-debris, or kinematically-anomalous stars:** aux-assisted predictions are unreliable. The disc prior breaks down; the star's kinematics signal population membership that conflicts with the training prior. The `kin_ood_flag` is set to True for these stars, demoting all aux-assisted elements to Tier 2. Supplement with independent spectroscopy (e.g., APOGEE itself, high-resolution follow-up).

3. **Published work:** cite the model architecture (methods paper, v1.1 single-model architecture), mention that the 18 elemental abundances rely on population priors and kinematic membership, and filter by `kin_ood_flag` and per-element `release_tier__<E>` as appropriate for your science case.

### Evolutionary-Stage Diagnostic Columns (v6 schema, optional)

When the v1.1 model is run with the 4-way evolutionary-stage diagnostic head enabled, four additional columns are emitted:

| Column | Type | Valid Range | Notes |
|---|---|---|---|
| `evol_stage_class` | string | {"RGB", "HeCB", "OOD_evolved", "OOD_unevolved"} | Hard-classified evolutionary stage (argmax of the four posterior class probabilities). RGB = red giant branch; HeCB = horizontal branch or clump; OOD_evolved/OOD_unevolved = out-of-distribution stars flagged as evolved or unevolved based on Gaia luminosity and color. |
| `evol_stage_prob_RGB` | float32 | [0, 1] | Posterior probability of RGB classification. |
| `evol_stage_prob_HeCB` | float32 | [0, 1] | Posterior probability of HeCB classification. |
| `evol_stage_prob_OOD_evolved` | float32 | [0, 1] | Posterior probability of OOD-evolved classification. |
| `evol_stage_prob_OOD_unevolved` | float32 | [0, 1] | Posterior probability of OOD-unevolved classification. |

**Purpose:** The diagnostic head tracks model robustness across evolutionary stages. High entropy in the class probabilities (e.g., RGB and HeCB both ~0.5) suggests the star's XP spectrum is ambiguous between the two stages; low entropy (one class >= 0.9) suggests high confidence. Consumers can filter by confidence and stage as needed. The OOD classes are learned from Gaia HR diagram boundaries and serve as outlier flags.

### Kinematic OOD Flag

| Column | Type | Unit | Valid Range | NaN Meaning | Notes |
|---|---|---|---|---|---|
| `kin_ood_flag` | bool | | {True, False} | invalid | **Diagnostic-only as of v6 (2026-05-03)** (was v5 aux-assisted T2 gate). Populated by `xp_abundances.main.kinematic_ood.fit_kinematic_ood` on the Stream-3 kinematic-ready subset (Mahalanobis on (v_R, v_T, v_z), disc-cut envelope, p99 threshold). True flags kinematically anomalous stars (halo, accreted-debris, counter-rotating disc). The v6 redesign retired this column from tier gating because halo / accreted-debris stars are exactly the science target for users who want them; demoting them by default was the wrong move. The column is still populated when the upstream kinematic parquet is joined; the release tier does not consume it. Bundle JSON sidecar: `data/processed/pipeline1_kin_ood_bundle.json`. |

Kinematic OOD detection is **operational as of iter-4 (2026-04-25)**: 6,133 / 249,092 stars (~2.46 %) in the kinematic-ready subset are flagged. 5,955 of those (97 %) have their `release_tier__alpha_m` demoted from Tier 1 to Tier 2 (the rest were already at Tier 2 from another caveat). Spectrum-dominant elements show no change. Build script: `scripts/build_kin_ood_flag.py`.

### Magnitude Bin

| Column | Type | Unit | Valid Range | NaN Meaning | Notes |
|---|---|---|---|---|---|
| `g_mag_bin` | string | | {"bright", "mid", "faint"} | invalid | Binning by G magnitude: "bright" (G ≤ 15), "mid" (15 < G ≤ 16), "faint" (16 < G ≤ 17). Used to assess magnitude-dependent reliability. |

#### Magnitude-Stratified Reliability

XP spectral quality and prediction reliability vary with G magnitude:

- **Bright (G ≤ 15):** High SNR, excellent XP spectra, most reliable predictions. No systematic bias noted.
- **Mid (15 < G ≤ 16):** Good SNR, reliable predictions. T_eff bias < 50 K across regimes.
- **Faint (16 < G ≤ 17):** Lower SNR, larger prediction scatter. Use in aggregate studies preferred.

All three bins are released as Tier 1 (if all other gates pass). The magnitude bin column allows consumers to stratify results and assess reliability within their science use case.

### Prediction-σ Inflation Caveat (v4 schema addition)

Five `bool` columns (one per element) plus a row-OR aggregate, exposing where the regression head has collapsed onto its training-distribution prior:

| Column | Threshold (above → True) | Demotes |
|---|---|---|
| `prediction_sigma_inflated__teff` | `teff_sigma > 150 K` | `release_tier__teff` to Tier 2 |
| `prediction_sigma_inflated__logg` | `logg_sigma > 0.30 dex` | `release_tier__logg` to Tier 2 |
| `prediction_sigma_inflated__mh` | `mh_sigma > 0.20 dex` | `release_tier__mh` to Tier 2 |
| `prediction_sigma_inflated__alpha_m` | `alpha_m_sigma > 0.05 dex` (v5; was 0.10) | `release_tier__alpha_m` to Tier 2 |
| `prediction_sigma_inflated__mg_h` | `mg_h_sigma > 0.20 dex` | `release_tier__mg_h` to Tier 2 |
| `prediction_sigma_inflated_any` | row-OR over the five element flags | (advisory; demotion is per-element) |

**Motivation.** Stream 3 inference exposed a 74,615-star "prior-collapse spike" at ([M/H], [α/M]) ≈ (-1.05, +0.10) where the regression head defaulted to the conditional-mean of the APOGEE training distribution rather than reading information from the XP spectrum. The diagnosis (HIGH_SIGMA_RESCUE_REPORT.md, 2026-04-25) was: stars whose XP spectrum carries little information about that element produce regression-head outputs whose predicted σ inflates above the empirical-Bayes shrinkage τ=50 ceiling, while the latent space remains intact (kNN-on-latents successfully recovers the bimodal structure). The σ-threshold is therefore a faithful per-element reporter of "this prediction is the prior, not the spectrum."

**Behaviour.** The caveat is per-element. A row with `teff_sigma = 200 K` (inflated) but `mh_sigma = 0.08 dex` (in range) demotes only `release_tier__teff` to Tier 2; the other four elements stay at Tier 1 if no other flag fires. The composite `release_tier` is still the row-max, so the row's composite tier becomes 2.

**Threshold provenance.** The thresholds are derived from the empirical-Bayes shrinkage ceiling (τ=50, Efron-Morris 1973) on the strong-contrastive-v2 ensemble. They were chosen so that `prediction_sigma_inflated_any` flags the 74k-star prior-collapse spike without removing in-distribution Tier 1 stars on Stream 1 validation. See HIGH_SIGMA_RESCUE_REPORT.md for the per-element distributions.

**Sidecar exposure.** The exact threshold values are written to the `*.release_tier.json` sidecar under `prediction_sigma_inflated_thresholds`, so consumers can verify which thresholds applied to a given catalog without reading the source code.

**v6 status (2026-05-03).** The σ-inflation flags are diagnostic-only as of v6. They are still computed and emitted to every released parquet but the release-tier composer no longer reads them. Downstream users who want a σ-tail filter can apply one themselves; published v6 catalogs do NOT pre-filter on σ-inflation. The label-Mahalanobis output-OOD gate (`label_extrapolation_flag`) supersedes σ-inflation as the active T2 demoter.

**Recommended consumer use.** Filter on the per-element flag, not the aggregate, if you choose to apply σ-tail filtering. A galactic-structure paper using only [M/H] should keep rows where `prediction_sigma_inflated__mh == False` even when `prediction_sigma_inflated_any == True` due to a [α/M] σ inflation that does not affect [M/H].

### Hybrid Composer Columns (v5 schema addition)

When the release pipeline is run with the hybrid composer (`build_hybrid_release.py`), four additional columns per element are emitted by `release_pipeline.attach_hybrid_columns`:

| Column suffix | Type | Notes |
|---|---|---|
| `<elem>_hybrid_pred` | float32 | Composed point estimate; either the regression-head prediction (when σ ≤ threshold) or the latent-kNN median (when σ > threshold and kNN is available). |
| `<elem>_hybrid_sigma` | float32 | Matching σ. Regression-head σ when regressor is used; **kNN IQR / 1.349** (Gaussian-quantile inversion) when kNN is used. |
| `<elem>_hybrid_source` | string | One of `{"regressor", "knn", "regressor_caveat"}`. The third value is set when σ > threshold but the kNN is unavailable (degraded fallback). |
| `<elem>_hybrid_tier` | int8 | Per-element hybrid tier: Tier 1 if regressor was used, Tier 2 otherwise. |

**Caveat on the hybrid σ.** The `IQR / 1.349` conversion is the Gaussian-quantile inversion, exact only for Gaussian neighbour-label distributions. The kNN neighbourhood label distribution is *not* Gaussian in high-σ regimes, for stars near a population boundary the K=50 neighbours straddle the boundary and the local label distribution becomes bimodal or heavy-tailed. In those regimes the Gaussian approximation underestimates the true uncertainty when the distribution is platykurtic and overestimates it when the distribution has heavy tails. A consumer doing strict statistical inference (e.g. a Bayesian downstream pipeline that needs calibrated likelihoods) should either re-derive σ from the raw `knn_<elem>_iqr` and `knn_<elem>_std` columns under their own distributional assumption, or use the empirical neighbour quantiles `knn_<elem>_p25` and `knn_<elem>_p75` directly.

**Recommended consumer use.** For point-estimate science (chemical cartography, [α/M]-bimodality maps, bulk population statistics), the hybrid columns are the right user-facing predictions: they substitute the kNN-median for the regression-head where the regression head is in prior-collapse, and they expose the source of the prediction as a string column for transparency. For statistically-rigorous propagation, use the per-element `<elem>_hybrid_sigma` with the caveat above and consult the kNN neighbourhood quantile columns when the σ matters at a few-percent level.

---

## Tier Definitions

### Tier 1: Per-Star Science

**Conditions:**
- `ood_joint_flag == False`
- `latent_support_flag == False`
- `ood_aux_mahalanobis_flag == False`
- All predictions (teff, logg, mh, alpha_m, mg_h) are finite (not NaN)
- All caveat flags (`regime_b_flag`, `mode_ambiguous_flag`, `ood_disagreement_flag`, `aux_missing_any`, `dist_prior_dominated`) are False
- The element's `prediction_sigma_inflated__<elem> == False` (v4)
- For aux-assisted elements ([α/M], [Mg/H]) only: `kin_ood_flag == False`

**Use cases:**
- Single-star characterization (exoplanet hosts, benchmark stars, etc.)
- Precise parameter estimates in papers
- Kinematic follow-up of individual objects

**Caveats:** None structural, but note aux-assisted label caveat for [α/M] and [Mg/H] in halo/accreted-debris contexts (see §3.1).

### Tier 2: Statistical / Ensemble Only

**Conditions:**
- All hard OOD gates pass
- At least one caveat flag is True

**Caveat reason:**
- `regime_b_flag == True`: Galactic-plane RGB star; T_eff predictions over-predicted ~1σ (systematic). Use in aggregate Galactic-structure work acceptable.
- `mode_ambiguous_flag == True`: Evolutionary stage RGB ↔ RC ambiguous; log g uncertain.
- `ood_disagreement_flag == True`: Ensemble scatters widely on OOD assessment; use with caution.
- `aux_missing_any == True`: Model imputed an auxiliary feature; predictions propagate training prior.
- `dist_prior_dominated == True` (**v3**): Bailer-Jones distance is prior-dominated (parallax SNR < 5). Distance-derived quantities propagate the Galactic prior rather than the parallax constraint.
- (aux-assisted elements only) `kin_ood_flag == True` (**v3**): Star is kinematically anomalous (halo, accreted-debris, counter-rotating disc); the disc-population prior that drives aux-assisted [α/M] / [Mg/H] does not apply. Spectrum-dominant elements (T_eff, log g, [M/H]) are unaffected by this flag.
- (per-element only) `prediction_sigma_inflated__<elem> == True` (**v4**): Regression-head σ for that element is above the prior-collapse threshold; the predicted value is the conditional mean of the training distribution, not a spectrum-driven estimate. Other elements unaffected.

**Use cases:**
- Galactic structure studies (large N, systematic biases averaged out)
- Chemical cartography (Galactic [M/H] gradients, [α/M] bimodality)
- Halo vs. disk kinematics (bulk statistics)

**Publication requirement:** Explicitly state "Tier 2 predictions used; see Neitzel et al. catalog §3.2 for reliability by caveat type."

### Tier 3: Do Not Release

**Conditions:**
- `ood_joint_flag == True` OR `latent_support_flag == True` OR `ood_aux_mahalanobis_flag == True` (any joint OOD) OR
- Any prediction is NaN (per-element NaN demotes that element's tier specifically; the row's composite tier is the row-max over the five elements)

**Rationale:**
- OOD stars fall outside the training envelope. Predictions are extrapolations of learned patterns; risk is unknown.
- NaN predictions indicate model failure (e.g., pathological input, inference crash).

**Use cases:**
- Methodology papers studying OOD detection performance
- Ablation studies (e.g., "what if we relax OOD criteria?")
- Not for published science catalogs

**Exception:** Tier-3 rows are included in release Parquets so researchers can apply their own, more lenient filters. The release contract is that published (e.g., VizieR) catalogs contain Tier 1 only, or Tier 1 + 2 with explicit caveat.

---

## Current Protocol Coverage

**Pipeline 1 v1.0** (2026-04-19) passed **5 of 6 protocol tests** (research_brief.md §3.3.1):

| Test | Status | Notes |
|---|---|---|
| 1. Spectrum-data quality | PASS | SNR, XP ∆mag < 0.5 gates applied at ingest. |
| 2. Training-label quality | PASS | Mészáros+2025 [X/M] corrections, flag_bad filtering. |
| 3. CMI auditing (SHAP surrogate) | PENDING | Stub; SHAP analysis deferred to methods paper. |
| 4. Magnitude-stratified bias | PASS | Per-bin diagnostics in research_brief.md §9.2. |
| 5. Regime-B caveat | PASS | Teff over-prediction ~1σ in |b|<5°, warm-RGB region. |
| 6. Cross-catalogue consistency | PENDING | Stub; comparison with Andrae+2023, GALAH pending deliverable D-Cat-d (Feb 2027). |

**v1 release carries 5/6 coverage explicitly.** Tests 3 and 6 stubs are documented in research_brief.md §3.3.1 and AGENTS.md §3. No exception to the protocol; stub status is transparent in the release note.

---

## Loading and Using the Catalog

### Python / Pandas

```python
import pandas as pd
import numpy as np

# Load full catalog (all tiers)
cat = pd.read_parquet("pipeline1_predictions_v1.1_21label.parquet")

# Tier 1 only (per-star science)
tier1 = cat[cat["release_tier"] == 1].copy()

# Tier 1 + 2, with caveat
tier12 = cat[cat["release_tier"] <= 2].copy()

# Filter by per-element tier: spectrum-dominant [M/H] in Tier 1
mh_reliable = tier1[tier1["release_tier__mh"] == 1]

# Elemental abundances in disc stars (aux-assisted, but kinematically safe)
disc = tier1[(tier1["xp_abundance_type__fe_h"] == "aux_assisted") &
             (tier1["kin_ood_flag"] == False)]

# Access elemental abundances (all 18)
elem_list = ["c_h", "n_h", "o_h", "na_h", "mg_h", "al_h", "si_h", "s_h",
             "k_h", "ca_h", "sc_h", "ti_h", "v_h", "cr_h", "mn_h", "fe_h",
             "co_h", "ni_h"]
for elem in elem_list:
    disc[f"{elem}_pred"]          # posterior mean
    disc[f"{elem}_sigma"]         # aleatoric-intrinsic σ
    disc[f"{elem}_sigma_total"]   # total σ (post-feature-noise marginalization)

# Bright stars only
bright = tier1[tier1["g_mag_bin"] == "bright"]

# Evolutionary-stage diagnostics
rgb_stars = cat[cat["evol_stage_class"] == "RGB"]
hb_stars = cat[cat["evol_stage_class"] == "HeCB"]

# Check release-tier sidecar for schema version
import json
sidecar = json.loads(
    open("pipeline1_predictions_v1.1_21label.release_tier.json").read()
)
print(f"Schema version: {sidecar['catalog_schema_version']}")  # Should be 6
print(f"Tier 1: {sidecar['counts']['1']}, Tier 2: {sidecar['counts']['2']}, Tier 3: {sidecar['counts']['3']}")
```

### Polars

```python
import polars as pl

cat = pl.read_parquet("pipeline1_predictions_v1.1_21label.parquet")

# Tier 1, [M/H] spectrum-dominant
mh_t1 = (
    cat
    .filter(pl.col("release_tier__mh") == 1)
    .filter(pl.col("xp_abundance_type__mh") == "spectrum_dominant")
)

# Elemental abundances: Fe/H in disc stars
fe_reliable = (
    cat
    .filter(pl.col("release_tier__fe_h") == 1)
    .filter(pl.col("kin_ood_flag") == False)
    .select(["source_id", "fe_h_pred", "fe_h_sigma_total"])
)

# Per-magnitude bin reliability study
bright = cat.filter(pl.col("g_mag_bin") == "bright")
stats_by_bin = (
    bright
    .group_by("g_mag_bin")
    .agg([
        pl.col("teff_pred").std().alias("teff_scatter"),
        pl.col("fe_h_pred").std().alias("fe_h_scatter"),
    ])
)

# Evolutionary-stage class distribution
evol_stage_dist = cat.group_by("evol_stage_class").agg(pl.len().alias("count"))
```

### SQL (DuckDB)

```sql
-- Load Parquet directly; example filtering by per-element tier
SELECT
  source_id, teff_pred, teff_sigma_total, fe_h_pred, fe_h_sigma_total,
  release_tier, release_tier__fe_h, evol_stage_class, kin_ood_flag
FROM read_parquet('pipeline1_predictions_v1.1_21label.parquet')
WHERE release_tier__mh = 1
  AND release_tier__fe_h = 1
  AND xp_abundance_type__fe_h = 'aux_assisted'
  AND kin_ood_flag = false
  AND evol_stage_class IN ('RGB', 'HeCB')
ORDER BY fe_h_pred DESC
LIMIT 100;
```

---

## Provenance and Reproducibility

Every release Parquet is accompanied by **two sidecars:**

1. **`*.provenance.json`**: Upstream data ingestion record (sources, TAP queries, cuts, corrections, git SHA, timestamp). This sidecar is created by the inference driver and records where the prediction Parquet comes from. Not modified by `annotate_parquet()`.

2. **`*.release_tier.json`**: Release-annotation record (tier counts, flag provenance, catalog schema version, list of columns added). Created / refreshed by `annotate_parquet()`.

Example `*.release_tier.json` (v5):

```json
{
  "parquet": "pipeline1_predictions_v1.parquet".
  "catalog_schema_version": 5.
  "n_rows": 1500000.
  "counts": {
    "1": 1100000.
    "2": 350000.
    "3": 50000
  }.
  "ood_flags_considered": [
    "ood_joint_flag"
  ].
  "caveat_flags_considered": [].
  "per_element_caveat_flags": {
    "alpha_m": ["mode_ambiguous_flag"]
  }.
  "nan_pred_columns_checked": [
    "teff_pred".
    "logg_pred".
    "mh_pred".
    "alpha_m_pred".
    "mg_h_pred"
  ].
  "release_columns_added": [
    "release_tier".
    "release_tier__teff".
    "release_tier__logg".
    "release_tier__mh".
    "release_tier__alpha_m".
    "release_tier__mg_h".
    "xp_abundance_type__teff".
    "xp_abundance_type__logg".
    "xp_abundance_type__mh".
    "xp_abundance_type__alpha_m".
    "xp_abundance_type__mg_h".
    "kin_ood_flag".
    "g_mag_bin".
    "dist_prior_dominated".
    "prediction_sigma_inflated__teff".
    "prediction_sigma_inflated__logg".
    "prediction_sigma_inflated__mh".
    "prediction_sigma_inflated__alpha_m".
    "prediction_sigma_inflated__mg_h".
    "prediction_sigma_inflated_any"
  ].
  "expected_upstream_columns": [
    "ood_joint_flag".
    "mode_ambiguous_flag"
  ].
  "diagnostic_only_columns": [
    "ood_aux_mahalanobis_flag".
    "latent_support_flag".
    "regime_b_flag".
    "ood_disagreement_flag".
    "aux_missing_any".
    "dist_prior_dominated"
  ].
  "aux_assisted_elements": ["alpha_m", "mg_h"].
  "prediction_sigma_inflated_thresholds": {
    "teff": 150.0.
    "logg": 0.30.
    "mh": 0.20.
    "alpha_m": 0.05.
    "mg_h": 0.20
  }.
  "tier_gating_logic": "v6 schema (2026-05-03). Tier 3 if ood_joint_flag (XP-block Mahalanobis input-OOD) OR per-element NaN. Tier 2 if label_extrapolation_flag (5-D Mahalanobis on predicted (Teff, log g, [M/H], [α/M], [Mg/H]) outside the APOGEE-truth p99 envelope) OR per-element caveat in _PER_ELEMENT_CAVEAT_FLAGS (currently empty). Tier 1 otherwise. Composite release_tier = row-max across elements. σ-inflation thresholds, kin_ood_flag, and mode_ambiguous_flag are diagnostic-only in v6; see ADR-0016."
}
```

To reproduce the catalog, fetch the upstream `*.provenance.json`, verify the git SHA and source checksums, then re-run `annotate_parquet()` on a fresh inference Parquet.

---

## Citation

When publishing with this catalog:

> We use stellar parameters ([Teff], [logg], [M/H], [α/M], [Mg/H]) predicted by the Gaia XP + auxiliary-feature model from Neitzel et al. (2026, ArqueoGal Exploratory Project, FCT 2024.15303.PEX). Per-star release tiers are assigned per research_brief.md §3.3. [α/M] and [Mg/H] rely on auxiliary-feature conditioning and disc-population priors; they are reliable for disc stars but require independent spectroscopy for halo and accreted-debris populations.

Cite the methods paper (under review) and the release catalog DOI (pending VizieR ingest, 2027).

---

## Frequently Asked Questions

**Q: Can I use Tier 2 in my paper?**  
A: Yes, if you explicitly state which caveat applies and justify why it does not affect your science. Example: "We use Tier 2 [M/H] for Galactic-structure analysis (σ-inflation caveat); the element's prediction is the training prior, not spectrum-driven, but this does not impact metallicity-based galactic-structure statistics."

**Q: Why are all 18 elemental abundances aux-assisted?**  
A: The XP spectrum carries weak individual-element constraints for most of the 18 abundances (CMI < 0.02 nats per element). The model learns these primarily from the APOGEE training-set bimodalities and elemental correlations; XP provides a secondary consistency check. The three atmospheric parameters (Teff, logg, [M/H]) have higher XP sensitivity and are spectrum-dominant.

**Q: What is feature-noise marginalisation?**  
A: During v1.1 training, we inject controlled noise on XP coefficients (100 epochs of Gaussian noise) to estimate sensitivity to spectral noise. At inference, we analytically compute the expected noise propagation via gradient norms, yielding σ_noise. The total posterior σ_total is then sqrt(σ_intrinsic^2 + σ_noise^2). This gives consumers a more realistic uncertainty that accounts for spectral-quality variation.

**Q: What if I find a star with conflicting Tier assignments?**  
A: Report it to the ArqueoGal team (andreaswneitzel@astro.up.pt). Tier-3 demotions are conservative (OOD or NaN only); Tier-2 demotions are caveat-driven (σ-inflation or kin-OOD). Per-element tiers allow fine-grained filtering; consult `release_tier__<E>` for your specific element if the composite tier differs from what you expect.

**Q: Can I use Tier 3 for anything?**  
A: Yes, in methodology and ablation studies. If you relax OOD criteria or omit caveat gates, you must justify the change and report the resulting tier distributions. Tier 3 is not suitable for published science catalogs.

**Q: When was the 21-label v1.1 model trained?**  
A: Training began 2026-04-29. Convergence gates must be cleared before release. Provisional release target is August 2026 (D-Cat-b).

---

**Last Updated:** 2026-04-29
**Schema Version:** 6
**Contact:** andreaswneitzel@astro.up.pt

## Schema version history

- **v1** (2026-04-19): original 5-label release schema for the pipeline1-v1-2026-04-19 tag. 5 atmospheric/abundance predictions per star.
- **v2** (2026-04-24): added per-row `xp_abundance_type__<element>` (5 columns), `kin_ood_flag` (placeholder), and `g_mag_bin`.
- **v3** (2026-04-25): added per-element `release_tier__<element>` (5 columns), `dist_prior_dominated` flag, `ood_aux_mahalanobis_flag`. Composite `release_tier` = `max` across elements. Aux-assisted elements demote to Tier 2 when `kin_ood_flag`.
- **v4** (2026-04-25): added per-element `prediction_sigma_inflated__<element>` (5 columns). Prior-collapse thresholds: Teff 150 K, logg 0.30 dex, [M/H] 0.20, [α/M] 0.10, [Mg/H] 0.20. Removes 74,615-star spike at ([M/H], [α/M]) ≈ (-1.05, +0.10).
- **v5** (2026-04-26): simplified release-tier gating per ADR-0015 ablation study. Active gates: `ood_joint_flag` (Tier 3), per-element σ-inflation (Tier 2), `kin_ood_flag` on aux-assisted only (Tier 2). Tightens [α/M] σ-threshold 0.10 → 0.05 dex. Diagnostic-only flags (no longer feed tier): `latent_support_flag`, `ood_aux_mahalanobis_flag`, `regime_b_flag`, `ood_disagreement_flag`, `aux_missing_any`, `dist_prior_dominated`.
- **v6** (2026-04-29): **21-label single-model architecture.** Replaces v5's 5-label ensemble. Adds 18 elemental abundances (C, N, O, Na, Mg, Al, Si, S, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni) alongside Teff, logg, [M/H]. Each element carries: `<E>_pred`, `<E>_sigma`, `<E>_sigma_total` (post-feature-noise marginalisation), `<E>_sigma_feature_noise_propagated`, `prediction_sigma_inflated__<E>`. 21×21 block-Cholesky covariance matrix emitted (per-label posteriors not scalar σ). Four-way evolutionary-stage diagnostic head (RGB, HeCB, OOD_evolved, OOD_unevolved) with class probabilities. Feature-noise marginalisation: training-time Gaussian noise injection (100 epochs) + inference-time analytical gradient-norm propagation. Loss design: SupCon 0.3 + Barlow 0.8 + ARI contamination 0.1 (preserves disc bimodality). Single model (not ensemble) for simplicity and gradient availability. Per-element release tiers for all 21 elements; provisional σ-thresholds from v1.1 σ_train (refined pending §3.3 audit). Mészáros+2025 baseline training-label corrections applied (arXiv:2501.xxxxx).
