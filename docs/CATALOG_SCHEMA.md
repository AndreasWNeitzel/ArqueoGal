# Pipeline 1 Release Catalog Schema

**Version:** 5
**Last Modified:** 2026-04-26
**Release Tag:** `pipeline1-v1-2026-04-19` (base predictions); v2 columns added in Stream 3 Phase A2; v3 columns added in Phase A2-followup (per-element release tiers, dist_prior_dominated, ood_aux_mahalanobis_flag); v4 columns added in Phase A2-followup-2 (per-element `prediction_sigma_inflated__<elem>` σ-threshold caveat to demote prior-collapse stars); v5 ablation-driven simplification (ADR-0015): retires `latent_support_flag`, `ood_aux_mahalanobis_flag`, `regime_b_flag`, `ood_disagreement_flag`, `aux_missing_any`, `dist_prior_dominated` from tier gating (kept as diagnostic columns), confines `mode_ambiguous_flag` to `[α/M]` only, tightens `[α/M]` σ-threshold from 0.10 to 0.05 dex.

## Overview

This document specifies the column contracts for Pipeline 1 release catalogs materialized by `src/arqueogal/xp_abundances/main/release.py`. Every released Parquet carries a companion `*.release_tier.json` sidecar capturing schema version, tier counts, and release-column provenance.

The catalog is built in stages:
1. **Inference** produces predictions and OOD flags (§10.2, data_acquisition.md).
2. **Release annotation** adds release tier, per-label abundance-type flags, kinematic OOD flag, and magnitude binning (this module).

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

### Predictions (Pipeline 1, all XP+auxiliary-feature conditioned)

| Column | Type | Unit | Valid Range | NaN Meaning | Notes |
|---|---|---|---|---|---|
| `teff_pred` | float32 | K | [3500, 6500] | NaN → Tier 3 | Predicted effective temperature (spectrum-dominant). |
| `teff_err` | float32 | K | (0, 1000] | rare | Prediction uncertainty (ensemble standard deviation or posterior credible interval width). |
| `logg_pred` | float32 | dex | [0.5, 5] | NaN → Tier 3 | Predicted surface gravity (spectrum-dominant). |
| `logg_err` | float32 | dex | (0, 2] | rare | Prediction uncertainty. |
| `mh_pred` | float32 | dex | [-2.5, 1] | NaN → Tier 3 | Predicted [M/H] (spectrum-dominant). |
| `mh_err` | float32 | dex | (0, 0.5] | rare | Prediction uncertainty. |
| `alpha_m_pred` | float32 | dex | [-0.5, 1] | NaN → Tier 3 | Predicted [α/M] (aux-assisted). See §3 for caveat. |
| `alpha_m_err` | float32 | dex | (0, 0.4] | rare | Prediction uncertainty. |
| `mg_h_pred` | float32 | dex | [-1, 1] | NaN → Tier 3 | Predicted [Mg/H] (aux-assisted). See §3 for caveat. |
| `mg_h_err` | float32 | dex | (0, 0.3] | rare | Prediction uncertainty. |

All predictions are posterior means (Bayesian NNs with BayesByBackprop or ensemble). Uncertainties are ensemble standard deviations or credible interval widths, depending on model architecture.

### Out-of-Distribution Flags

> **v5 (2026-04-26)**: only `ood_joint_flag` is an active tier gate.
> `latent_support_flag` and `ood_aux_mahalanobis_flag` are still emitted but
> are diagnostic-only, they do not feed `release_tier`. See ADR-0015 and
> `release/test_ablations_2026-04-26/REPORT.md`.

| Column | Type | Unit | Valid Range | NaN Meaning | Notes |
|---|---|---|---|---|---|
| `ood_joint_flag` | bool | | {True, False} | False (default) | **v5 active gate.** Hard OOD gate: Mahalanobis distance in 108-D XP feature space at p=0.99 (chi-squared, training envelope). True → Tier 3. |
| `latent_support_flag` | bool | | {True, False} | False (default) | **v5 diagnostic-only (retired from gating).** Convex-hull surrogate on learned representations. Never fired on the Stream-1 holdout in the v5 ablation; column kept for diagnostic continuity. |
| `ood_aux_mahalanobis_flag` | bool | | {True, False} | False (default) | **v5 diagnostic-only (retired from gating).** Mahalanobis distance in the auxiliary-feature space (parallax, photometry, extinction, position) at p=0.99. Subsumed by `aux_missing_any` in practice; kept for diagnostic continuity. |

### Caveat Flags (Structural, demote to Tier 2)

> **v5 (2026-04-26)**: only `prediction_sigma_inflated__<elem>` (all elements).
> `mode_ambiguous_flag` (α/M only), and `kin_ood_flag` (aux-assisted only) are
> active tier gates. `regime_b_flag`, `ood_disagreement_flag`.
> `aux_missing_any`, `dist_prior_dominated` are still emitted but
> diagnostic-only. See ADR-0015.

| Column | Type | Unit | Valid Range | NaN Meaning | Notes |
|---|---|---|---|---|---|
| `regime_b_flag` | bool | | {True, False} | False (default) | **v5 diagnostic-only (retired from gating).** Regime B (Galactic plane, warm RGB): systematic T_eff over-prediction ~1σ. Fires on ~0.04 % of stars; the v5 ablation showed no measurable T1+T2 RMSE effect. The systematic itself is a methods-paper finding (CLAUDE.md footgun), not a release-blocker. |
| `mode_ambiguous_flag` | bool | | {True, False} | False (default) | **v5 active gate (per-element, [α/M] only).** Disc α/M bimodality at fixed (Teff, log g, [M/H]). Gaussian-NLL μ-collapse risk. Demotes only `release_tier__alpha_m` to Tier 2. Other elements unaffected. The v5 ablation showed +12 % α/M T1 RMSE inflation when this gate is removed; zero effect for the other elements. |
| `ood_disagreement_flag` | bool | | {True, False} | False (default) | **v5 diagnostic-only (retired from gating).** Designed for multi-member ensemble disagreement; cannot fire with a single-member ensemble. Re-evaluate when the ensemble grows to ≥ 2 members. |
| `aux_missing_any` | bool | | {True, False} | False (default) | **v5 diagnostic-only (retired from gating).** Any auxiliary feature (parallax, extinction, position) is NaN at inference. Stars carrying this flag had ~4-6 % T1 RMSE inflation if kept in T1, but T1+T2 RMSE was unchanged, i.e. the demotion was pure relabeling. Column retained as a soft caveat for consumer-side filtering. |
| `dist_prior_dominated` | bool | | {True, False} | False (default) | **v5 diagnostic-only (retired from gating).** True when the Bailer-Jones photogeometric distance is dominated by the Galactic prior rather than parallax (σ_π/π > 0.2). Never fired on the held-out test split; column retained for diagnostic continuity. |
| `prediction_sigma_inflated__<elem>` | bool | | {True, False} | False (default) | **v4 schema addition (per-element); v5 active gate.** True when the regression-head predicted σ for that element exceeds the prior-collapse threshold: `teff_sigma > 150 K`, `logg_sigma > 0.30 dex`, `mh_sigma > 0.20 dex`, **`alpha_m_sigma > 0.05 dex` (tightened from 0.10 in v5)**, `mg_h_sigma > 0.20 dex`. True → that element's tier demotes to Tier 2. Other elements unaffected. Computed by `release.assign_prediction_sigma_inflated()`. |
| `prediction_sigma_inflated_any` | bool | | {True, False} | False (default) | **v4 schema addition.** Row-OR aggregate over the five `prediction_sigma_inflated__<elem>` flags. Convenience column for consumers who only need the row-level "any element prior-collapsed" indicator without inspecting per-element flags. |

**v5 tier composition (applies after annotate_parquet).** Tier 3 if `ood_joint_flag` OR per-element NaN. Tier 2 if per-element σ-inflation OR (element is α/M AND `mode_ambiguous_flag`) OR (element is aux-assisted AND `kin_ood_flag`). Tier 1 otherwise. Composite `release_tier` = row-max across elements. Diagnostic-only flags do not affect tier.

---

## Release-Annotation Columns (Schema v5, 2026-04-26)

These columns are added by `annotate_parquet()` at release time. They are optional in the inference schema but mandatory in published catalogs.

### Release Tier (composite)

| Column | Type | Unit | Valid Range | NaN Meaning | Notes |
|---|---|---|---|---|---|
| `release_tier` | int8 | | {1, 2, 3} | invalid | Composite **row-max** (most-conservative) over the per-element tiers. **Tier 1**: per-star science. **Tier 2**: σ-inflation (any element), `mode_ambiguous_flag` (α/M only), or `kin_ood_flag` (aux-assisted elements only). **Tier 3**: `ood_joint_flag` (XP-Mahalanobis OOD) or NaN prediction. See ADR-0015 and §3 below. |

### Per-Element Release Tier (v3 schema addition)

Five `int8` columns, one per atmospheric parameter / abundance, encoding the element-specific tier independent of the row-max composite. The composite `release_tier` is `max(release_tier__teff, ..., release_tier__mg_h)`.

| Column | Type | Unit | Valid Range | NaN Meaning | Notes |
|---|---|---|---|---|---|
| `release_tier__teff` | int8 | | {1, 2, 3} | invalid | Per-element tier for T_eff. Tier 3 if NaN or `ood_joint_flag`; Tier 2 if `prediction_sigma_inflated__teff`; else Tier 1. T_eff is spectrum-dominant so neither `mode_ambiguous_flag` nor `kin_ood_flag` demote. |
| `release_tier__logg` | int8 | | {1, 2, 3} | invalid | As for T_eff (spectrum-dominant). |
| `release_tier__mh` | int8 | | {1, 2, 3} | invalid | As for T_eff (spectrum-dominant). |
| `release_tier__alpha_m` | int8 | | {1, 2, 3} | invalid | Per-element tier for [α/M]. Tier 3 if NaN or `ood_joint_flag`; Tier 2 if `prediction_sigma_inflated__alpha_m` (threshold tightened to 0.05 dex in v5) OR `mode_ambiguous_flag` (per-element caveat, α/M-only) OR `kin_ood_flag` (aux-assisted demotion); else Tier 1. |
| `release_tier__mg_h` | int8 | | {1, 2, 3} | invalid | Per-element tier for [Mg/H]. Tier 3 if NaN or `ood_joint_flag`; Tier 2 if `prediction_sigma_inflated__mg_h` OR `kin_ood_flag` (aux-assisted demotion); else Tier 1. |

The per-element columns let consumers safely filter at finer granularity. A galactic-structure paper using only [M/H] can keep `release_tier__mh == 1` rows even when `release_tier == 2` due to a kin-OOD-driven [α/M] demotion that does not affect [M/H].

### Per-Element Abundance Type

Five columns, one per atmospheric parameter / abundance:

| Column | Type | Values | Meaning | Notes |
|---|---|---|---|---|
| `xp_abundance_type__teff` | string | "spectrum_dominant" | T_eff prediction is XP-constrained (CMI > threshold). | No uncertainty penalty; trust XP dominance. |
| `xp_abundance_type__logg` | string | "spectrum_dominant" | log g prediction is XP-constrained. | As for Teff. |
| `xp_abundance_type__mh` | string | "spectrum_dominant" | [M/H] prediction is XP-constrained (CMI > 0.02). | As for Teff. |
| `xp_abundance_type__alpha_m` | string | "aux_assisted" | [α/M] prediction conditional on auxiliary. | See §3 caveat. |
| `xp_abundance_type__mg_h` | string | "aux_assisted" | [Mg/H] prediction conditional on auxiliary. | See §3 caveat. |

All abundance-type strings are lowercase. These columns enable consumers to filter by prediction-mechanism confidence. For example, a galactic-structure paper using [M/H] can trust the spectrum_dominant mechanism across the full magnitude range; a globular-cluster paper using [α/M] should acknowledge the aux_assisted caveat.

#### Design Rationale: Per-Label Columns vs. Composite

We emit five separate string columns (`xp_abundance_type__<element>`) rather than a single JSON-encoded composite. This choice prioritizes **consumer ergonomics**: the columns are directly filterable in SQL or Polars/DuckDB, and human readers can grep the CSV. A composite would require JSON parsing in the consumer's language.

### Auxiliary-Assisted Label Caveat

**Aux-assisted labels** (`xp_abundance_type == "aux_assisted"`) are model predictions where the conditional mutual information (CMI) between the XP spectrum and the label, *given auxiliary features* (parallax, photometry, extinction, position), falls below **0.02 nats** (research_brief.md §3.3.1, information-content audit §9.2).

In plain language: the model learns these labels primarily from the disc-population prior (derived from APOGEE training) and auxiliary data. The XP spectrum contributes but does not dominantly constrain the label.

**Implications for consumers:**

1. **Disc stars (solar vicinity, thin disk kinematics):** aux-assisted predictions are reliable. The model has learned the disc [α/M]–[M/H] relation and [Mg/H]–[Fe/H] bimodality from APOGEE; XP provides a secondary consistency check.

2. **Halo, accreted-debris, or kinematically-anomalous stars:** aux-assisted predictions are unreliable. The disc prior breaks down. Supplement with independent spectroscopy (e.g., APOGEE itself, high-resolution follow-up).

3. **Published work:** cite the model architecture (methods paper, BayesByBackprop or ensemble) and acknowledge that [α/M] and [Mg/H] rely on population priors. Do not claim XP-derived abundances for stars where the disc prior is violated.

### Kinematic OOD Flag

| Column | Type | Unit | Valid Range | NaN Meaning | Notes |
|---|---|---|---|---|---|
| `kin_ood_flag` | bool | | {True, False} | invalid | **Phase B implementation (2026-04-25):** populated by `xp_abundances.main.kinematic_ood.fit_kinematic_ood` on the Stream-3 kinematic-ready subset. Mahalanobis-on-velocity envelope fit on the disc-cut subset (\|v_z\|<80, v_T>100 km/s); per-star Mahalanobis distance to disc mean compared to the 99th-percentile threshold. True ⇒ star is kinematically anomalous (halo, accreted-debris, counter-rotating disc), aux-assisted elements ([α/M], [Mg/H]) demote to Tier 2 because the disc-population prior they rely on does not apply. Spectrum-dominant elements (Teff, logg, [M/H]) are unaffected. Stars without kinematics (Stream-3 outside the kinematic-ready subset) default to False (conservative, keeps them in Tier 1; the user can join the kinematic parquet themselves for finer per-star analysis). Bundle JSON sidecar: `data/processed/pipeline1_kin_ood_bundle.json`. |

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

**Recommended consumer use.** Filter on the per-element flag, not the aggregate. A galactic-structure paper using only [M/H] should keep rows where `prediction_sigma_inflated__mh == False` even when `prediction_sigma_inflated_any == True` due to a [α/M] σ inflation that does not affect [M/H].

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

**v1 release carries 5/6 coverage explicitly.** Tests 3 and 6 stubs are documented in research_brief.md §3.3.1 and CLAUDE.md §3. No exception to the protocol; stub status is transparent in the release note.

---

## Loading and Using the Catalog

### Python / Pandas

```python
import pandas as pd

# Load full catalog (all tiers)
cat = pd.read_parquet("pipeline1_predictions_v1.parquet")

# Tier 1 only (per-star science)
tier1 = cat[cat["release_tier"] == 1].copy()

# Tier 1 + 2, with caveat
tier12 = cat[cat["release_tier"] <= 2].copy()

# Filter by abundance type (trust spectrum-dominant [M/H])
mh_reliable = tier1[tier1["xp_abundance_type__mh"] == "spectrum_dominant"]

# [α/M] in disk (aux-assisted OK; caveat noted)
disk = tier1[(tier1["xp_abundance_type__alpha_m"] == "aux_assisted") &
             (tier1["kin_ood_flag"] == False)]

# Bright stars only
bright = tier1[tier1["g_mag_bin"] == "bright"]

# Check release-tier sidecar for schema version
import json
sidecar = json.loads(
    open("pipeline1_predictions_v1.release_tier.json").read()
)
print(f"Schema version: {sidecar['catalog_schema_version']}")
print(f"Tier 1: {sidecar['counts']['1']}, Tier 2: {sidecar['counts']['2']}, Tier 3: {sidecar['counts']['3']}")
```

### Polars

```python
import polars as pl

cat = pl.read_parquet("pipeline1_predictions_v1.parquet")

# Tier 1, [M/H] spectrum-dominant
mh_t1 = (
    cat
    .filter(pl.col("release_tier") == 1)
    .filter(pl.col("xp_abundance_type__mh") == "spectrum_dominant")
)

# Per-magnitude bin reliability study
bright = cat.filter(pl.col("g_mag_bin") == "bright")
stats_by_bin = (
    cat
    .group_by("g_mag_bin")
    .agg([
        pl.col("teff_pred").std().alias("teff_scatter").
        pl.col("mh_pred").std().alias("mh_scatter").
    ])
)
```

### SQL (DuckDB)

```sql
-- Load Parquet directly
SELECT
  source_id, teff_pred, mh_pred, alpha_m_pred, release_tier, g_mag_bin.
  xp_abundance_type__mh, xp_abundance_type__alpha_m
FROM read_parquet('pipeline1_predictions_v1.parquet')
WHERE release_tier = 1
  AND xp_abundance_type__mh = 'spectrum_dominant'
ORDER BY mh_pred DESC
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
  "tier_gating_logic": "Tier 3 if ood_joint_flag (XP-Mahalanobis OOD) OR per-element NaN. Tier 2 if (per-element σ exceeds prediction_sigma_inflated threshold) OR (element is alpha_m AND mode_ambiguous_flag) OR (element is aux-assisted AND kin_ood_flag). Tier 1 otherwise. Composite release_tier = row-max across elements. Simplified 2026-04-26 (v5 schema) per release/test_ablations_2026-04-26/REPORT.md and ADR-0015."
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
A: Yes, if you explicitly state which caveat applies and justify why it does not affect your science. Example: "We use Tier 2 [M/H] for Galactic-structure analysis (regime-B caveat); systematic T_eff bias does not impact metallicity inference."

**Q: Why are [α/M] and [Mg/H] aux-assisted but [M/H] is not?**  
A: The XP spectrum is poor at constraining [α/M] and [Mg/H] independently (CMI < 0.02 nats). The model learns these from APOGEE training and disc kinematics; XP provides a consistency check. [M/H] has higher CMI due to XP's sensitivity to iron-group absorption lines.

**Q: What if I find a star with conflicting Tier assignments?**  
A: Report it to the ArqueoGal team (andreaswneitzel@astro.up.pt). Tier-3 demotions are conservative; re-running `assign_release_tier()` with updated flag logic should justify any reclassification.

**Q: Can I use Tier 3 for anything?**  
A: Yes, in methodology and ablation studies. If you relax OOD criteria or omit caveat gates, you must justify the change and report the resulting tier distributions. Tier 3 is not suitable for published science catalogs.

**Q: When will kinetic OOD flag be populated?**  
A: Populated as of iter-4 (2026-04-25). 6,133 / 249,092 (2.46 %) stars in the kinematic-ready subset are flagged; the other 365k stars (no kinematics) default to False.

---

**Last Updated:** 2026-04-26
**Schema Version:** 5
**Contact:** andreaswneitzel@astro.up.pt

## Schema version history

- **v1** (2026-04-19): original release schema for the pipeline1-v1-2026-04-19 tag.
- **v2** (2026-04-24, Phase A2): added per-row `xp_abundance_type__<element>`, `kin_ood_flag` (placeholder), and `g_mag_bin` columns.
- **v3** (2026-04-25, Phase A2-followup): added per-element `release_tier__<element>` (5 columns), `dist_prior_dominated` caveat flag, `ood_aux_mahalanobis_flag` joint OOD flag. The composite `release_tier` is now `max` across the five `release_tier__<element>` columns. Aux-assisted elements ([α/M], [Mg/H]) demote to Tier 2 when `kin_ood_flag == True` (population-prior reliability gate); spectrum-dominant elements are unaffected.
- **v4** (2026-04-25, Phase A2-followup-2): added per-element `prediction_sigma_inflated__<element>` (5 columns) and the `prediction_sigma_inflated_any` aggregate. When the regression-head σ for an element exceeds the prior-collapse threshold (Teff: 150 K; logg: 0.30 dex; [M/H], [Mg/H]: 0.20 dex; [α/M]: 0.10 dex), that element demotes to Tier 2. The thresholds are exposed in the sidecar under `prediction_sigma_inflated_thresholds`. Motivation and threshold provenance: HIGH_SIGMA_RESCUE_REPORT.md (2026-04-25); the v4 caveat removes the 74,615-star prior-collapse spike at ([M/H], [α/M]) ≈ (-1.05, +0.10) without affecting in-distribution Tier 1 stars.
- **v5** (2026-04-26, per-cell-gate ablation): simplified release-tier gating after the Stream-1 ablation study (`release/test_ablations_2026-04-26/REPORT.md`). Active gates reduced to `ood_joint_flag` (Tier 3); per-element σ-inflation (Tier 2); `mode_ambiguous_flag` on [α/M] only (Tier 2, per-element caveat); `kin_ood_flag` on aux-assisted elements (Tier 2). The flags `latent_support_flag`, `ood_aux_mahalanobis_flag`, `regime_b_flag`, `ood_disagreement_flag`, `aux_missing_any`, `dist_prior_dominated` are retired from gating and reclassified as diagnostic-only, still emitted by `annotate_parquet`, no longer feed `release_tier`. The [α/M] σ-threshold tightens 0.10 → 0.05 dex (≈ 0.5×σ_train). Stream 1 holdout: T1 fraction goes 47.6 % → 92.6 % on Teff/log g/[M/H]/[Mg/H] with T1+T2 RMSE preserved; α/M T1 RMSE goes 0.043 → 0.034 dex with T1 fraction 47.9 % → 36.1 %. Rationale: ADR-0015.

### Iter-3 (2026-04-25), bimodality grid edge shift + convergence retrain

The mode-ambiguous grid edges in `xp_abundances/main/bimodality.py:fit_bimodality_grid`
were shifted on the [M/H] axis from `np.arange(-3.0, 0.501, 0.20)` to
`np.arange(-2.9, 0.401, 0.20)`. The original placement put a cell boundary
at exactly [M/H] = 0.0 dex, which sits at the disc peak of the APOGEE
training distribution; per-cell bimodal-coverage statistics differ sharply
across that boundary (data-rich cells just below 0 vs sparse cells just above).
which produced a tier-filter cliff: 65 % of stars at [M/H] = -0.005 dex were
flagged `mode_ambiguous_flag`, vs 0 % at [M/H] = +0.005 dex. The shift puts
0.0 dex in the middle of the [-0.1, +0.1] cell; post-fix the rate is 24.2 %
just below 0 vs 24.5 % just above (smooth). The Teff and logg axes were
audited for analogous artefacts; their visible cell-edge effects track the
underlying physical α-sequence transitions in the training data and are
expected (not artefacts).

The strong-contrastive-v2 ensemble was retrained with convergence-tuned
loss weights (SupCon 1.0 → 0.3, Barlow 0.5 → 0.8, τ_init 0.10 → 0.15) to
close a widening train/val gap on the SupCon component. The
2026-04-26 long-train rerun (30 epochs) produced
`models/main/xp_abundances/20260425_6b96c06_cd1cbb9_ensemble_5label/`.
which is the current production checkpoint pointed to by
`scripts/run_pipeline1_inference.py:DEFAULT_ENSEMBLE_DIR` and the gallery
plots. The earlier `models/main/xp_abundances/strong_contrastive_2026-04-25/`
directory is retained for methodology comparison; do not consume it for
release artefacts. Best val_loss 2.715 → 0.983 (2.8× lower) on the v2
recipe, then 0.888 best at epoch 29/30 on the convergence rerun.
Stress battery 7/7 PASSED on the converged model.

Tier-1 count went from 212,656 to 209,917 (−1.3 %); the full trustworthy
catalog (Tier 1 + Tier 2) is unchanged at 491,756 stars. Structure-
preservation metrics regressed marginally (ARI 0.566 → 0.558, macro-F1
0.819 → 0.815, both within the production gate); centroid drift on the
test split improved 19 % (0.033 → 0.027 dex).
