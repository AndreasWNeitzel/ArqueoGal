# 3 Methods

This section is the methodological core of the methods paper. It describes the
end-to-end Pipeline 1 architecture, training procedure, calibration, out-of-
distribution detection, and, the project's distinguishing methodological
contribution, the per-label information-content audit that gates tier
promotion. Drafted 2026-04-24 from the metas + specialist reports; expected
to undergo co-author revision before submission.

## 3.1 Input space and preprocessing

Pipeline 1 ingests Gaia DR3 BP/RP "XP" coefficients (De Angeli et al. 2023;
Carrasco et al. 2021) projected onto an internally-calibrated Hermite basis,
giving 55 BP plus 55 RP coefficients per source. The preprocessing chain is
a fixed contract (research_brief §6.4, CLAUDE.md invariant 12) applied
identically at training and inference:

1. Mandatory NN-based flux correction of Ye et al. 2025 (A&A 695, A75;
   arXiv:2411.19105), reducing systematic flux errors from 3.2–3.7 % to
   1.2–2.4 %.
2. Coefficients 1–54 of each band are normalised by coefficient 0
   (per-band amplitude removal).
3. Coefficient 0 itself is log-transformed and z-scored using fixed
   per-band statistics.
4. The remaining 108 normalised coefficients are z-scored using fixed
   per-coefficient statistics.

The fixed statistics are pinned via a SHA-256 fingerprint of the Hermite basis
(`0d34b5659e97...`) computed at training time. Inference verifies the
fingerprint with a halting assertion (`assert_frozen_stats_match`); any
mismatch raises `FrozenStatsMismatchError` (research_brief §3.6,
data_acquisition §6.4). A second defensive supplement,
`verify_frozen_stats_match_parquet`, additionally re-computes the c0 mean and
sigma on a 50 k-row subsample of the training Parquet and rejects
multiplicative drift exceeding 10 % in σ or 0.05 in mean.

Auxiliary features supplied to the model alongside the 108 XP coefficients
are: Gaia DR3 parallax with the Lindegren et al. 2021 zero-point applied
(A&A 649, A4); Gaia G-band magnitudes with the Riello et al. 2021 Appendix A
cubic correction applied (A&A 649, A3); 2MASS J/H/K and WISE W1/W2 magnitudes;
the composed extinction prior A_V from the Edenhofer et al. 2024 (d < 1.25
kpc) + Lallement et al. 2022 (1.25 < d < 3 kpc) + SFD (d > 3 kpc) +
GSP-Phot neighbourhood-median Av stack; and Bailer-Jones et al. 2021
photogeometric distances. Aux-feature NaN values are sanitised at the
data-loader boundary via `np.nan_to_num(..., nan=0.0)` at training; the
inference driver mirrors this and additionally asserts finite output via
`np.isfinite(features).all()` before the first model forward pass (ADR-0012,
META_META §14.1).

The training labels are the per-element [X/M] APOGEE DR19 ASPCAP values with
the mandatory Mészáros et al. 2025 Teff-dependent corrections applied
(arXiv:2506.07845, AJ in press). The corrections are populated for 14
elements (alpha, O, Na, Mg, Al, Si, S, K, Ca, Ti, Cr, Mn, Ni, Ce) with the
RGB-validity guard (logg < 3.8); excluded are C and N (first-dredge-up
mixing invalidates cluster-based detrending), Fe (reference), V and Cu
(not published in Mészáros+2025).

## 3.2 Architecture: shared-encoder ensemble with block-Cholesky covariance

The model is a shared-encoder ensemble (ADR-0010): a single MLP encoder
maps the 108-D XP block to a 32-D latent representation; per-element
heads then predict the mean μ_i and lower-triangular Cholesky factor L_i
of the per-star covariance matrix Σ_i = L_i L_i^T over the five released
labels (Teff, logg, [M/H], [α/M], [Mg/H]). The block-Cholesky
parameterisation (Pourahmadi 1999, Biometrika 86) enforces positive-
definiteness at the parameter level and reduces capacity from 25 entries
of a full 5×5 covariance to 15 lower-triangular entries plus a diagonal
positive transform (10+5 = 15 effective parameters per head).

Ensembling: M = 10 members trained from different random seeds, sharing
the encoder pretraining checkpoint per ADR-0010. Sequential training
fits the RTX 3060's 6 GB VRAM budget; AMP bfloat16 forward passes plus
gradient accumulation over an effective batch of 512 keep peak memory
below 4 GB. Per-batch finite gradients are asserted via gradient
clipping at 1.0 (line 586–598 of training.py). Best-model checkpointing
on validation loss with early stopping prevents over-training.

The loss is the heteroscedastic β-NLL of Seitzer et al. 2022 (arXiv:
2203.09168) with β = 0.5 (ADR-0011), evaluated through the four-block
Cholesky parameterisation (ADR-0008). Per-label missingness is handled
via a binary mask in the NLL accumulator: missing labels do not enter
the residual sum but the Cholesky covariance still couples them through
the off-diagonal structure, preserving label correlations. Per-element
NaN rates in the training set are V ~5.3 %, Mg/Fe ~1.6 %, α/M 0 %.

## 3.3 Calibration: empirical-Bayes per-cell shrinkage

After training, the raw heteroscedastic σ predicted per star is calibrated
per regime cell on the validation set using Efron-Morris empirical-Bayes
shrinkage (Efron and Morris 1973, JASA 68). Regime cells are the joint
binning of (Teff × logg × [M/H]) used by research_brief §10.5. Per-cell
σ is shrunk toward a global mean with τ = 50: σ_shrunk = (n_cell σ_cell +
τ σ_global) / (n_cell + τ), where n_cell is the per-cell sample count.

The choice of τ = 50 is the production setting (ADR-0003). A Gaussian-
process smoothing alternative (`gp_smoothed_per_cell_per_label_scale`) was
tested and rejected for production: it added complexity without improving
expected calibration error (ECE) measurably and is retained only as a
methodology comparison toggle (ADR-0003). The GP path is documented as a
negative result in §6.

A τ-sensitivity sweep (τ ∈ {10, 20, 30, 50, 100, 150}) is recommended
before final submission (META_META §14.4 P1; protocol document
`docs/protocols/tau_sweep.md`). At time of writing, the production τ = 50
is empirically validated against per-cell coverage but not against a full
ECE-vs-τ curve.

## 3.4 Out-of-distribution detection

Three flags surface in the released catalog:

- `ood_joint_flag`: Mahalanobis distance on the 108-D XP block exceeds
  the training-set 99th percentile (Lee et al. 2018,
  arXiv:1807.03888). Captures stars whose XP coefficients are unlike
  the training distribution.
- `ood_aux_mahalanobis_flag` (added in v1.0.1, schema v3): a parallel
  Mahalanobis detector fit on the auxiliary feature space (parallax,
  photometry, extinction, distance). Captures stars whose aux features
  are extreme. Fitted with looser regularisation (1e-4 vs 1e-6) than
  the XP detector to reflect wider per-feature dynamic range.
- `latent_support_flag`: a convex-hull surrogate on the encoder's
  latent space.

Any of the three firing demotes the star to Tier 3 (do not release per-
star). Population-level science can still use the predictions with
appropriate caveats.

A separate `kin_ood_flag` is computed in the kinematic-OOD module
(`src/arqueogal/xp_abundances/main/kinematic_ood.py`, added in v1.0.1):
a Mahalanobis detector on 3D Galactocentric velocity (v_R, v_φ, v_z),
fit on the disc-only subset of Stream 1. Stars exceeding the disc 99th-
percentile distance are kinematically out-of-distribution relative to the
disc training prior. This flag does not on its own demote any element to
Tier 3, it acts conditionally on aux-assisted abundances; see §3.5.

## 3.5 The information-content audit and tier promotion

This subsection describes the project's distinguishing methodological
contribution. It addresses Fallows and Sanders 2024's open question
(MNRAS 531, 2126; arXiv:2405.10699): "where is the spectrum information
actually coming from?", the question that motivates a per-label audit
of spectrum-driven vs auxiliary-prior-driven contributions.

### 3.5.1 Per-label conditional mutual information

For every released abundance label, we compute the conditional mutual
information CMI(XP ; label | aux), where aux = (parallax, photometry,
A_V, position). The estimator uses 7-component PCA summary of the 108-D
XP block (≥ 95 % variance retention; ADR-0009) plugged into a Kraskov-
Stögbauer-Grassberger (KSG) k-nearest-neighbour MI estimator with
k = 5. The 2-D KSG estimator used in earlier project versions inflated
Teff CMI by ~5× and was deprecated in favour of the PCA-summary
estimator (ADR-0009; research_brief §9.2.1).

### 3.5.2 Three-question diagnostic for zero-CMI labels

Some labels pass the shuffled-spectrum null test (Test 4 of the §3.3
protocol), meaning the model loses ≥ 20 % of its predictive skill when
XP is permuted, yet show CMI ≈ 0 with the 7-component PCA summary. The
[α/M] case at Pipeline 1 v1 was the canonical example: shuffled-spectrum
skill_ratio of −0.257 confirmed XP carries information, but PCA-CMI = 0
nats with full aux conditioning. This apparent contradiction is resolved
via the three-question diagnostic (research_brief §3.3.1):

1. Does an aux-only baseline match the full-model RMSE? For [α/M], the
   answer was approximately yes (~ 90 % skill recovery from aux alone),
   indicating the disc-population prior absorbs most predictive power.
2. Does increasing PCA components (15 components, ~ 99 % variance)
   restore CMI? For [α/M], no, the high-order modes do not carry
   independent signal.
3. Does conditioning only on parallax (excluding photometry, extinction,
   position) restore CMI? For [α/M], yes, CMI(XP ; [α/M] | parallax
   only) = 0.1125 nats, 56× the 0.02 nats threshold.

Interpretation: XP carries weak independent signal for [α/M], but
photometry + extinction + position absorb the kinematic-population
correlation that the model uses dominantly. This is honest information
theory, not a model defect: at low resolution (R ~ 30–100), [α/M] is
genuinely information-poor in XP. The model uses the disc-population
prior implicit in the APOGEE training set as the dominant constraint.

### 3.5.3 The spectrum-dominant vs aux-assisted dichotomy

Per-element classification follows from the information-content audit:

- **Spectrum-dominant** (CMI > 0.02 nats with full aux conditioning, or
  diagnosed as such by the three-question protocol): the XP coefficients
  carry independent signal and the prediction reflects spectroscopic
  measurement of the stellar atmosphere. Teff, logg, [M/H] qualify in
  Pipeline 1 v1.
- **Aux-assisted** (CMI < 0.02 nats with full aux conditioning, with the
  three-question diagnostic resolving the apparent contradiction by
  identifying aux-feature absorption of the disc prior): the prediction
  is dominated by the disc-population prior learned in training.
  [α/M], [Mg/H] qualify in Pipeline 1 v1.

The catalog emits these classifications as per-star columns
(`xp_abundance_type__teff`, ..., `xp_abundance_type__mg_h`) so consumers
can filter without reading the audit framework.

### 3.5.4 Tier promotion as a release gate

The six-test §3.3 promotion protocol gates per-element release. Pipeline 1
v1 ships at 5/6 coverage: tests 3 (SHAP feature attribution) and 6
(cross-catalogue consistency) are documented stubs awaiting Stream 3
overlap with GALAH DR4 / AspGap / Guiglion+2024 / SHBoost / Fallows-
Sanders 2024. Tier-promotion code (`tier_promotion.py`) refuses to report
6/6 coverage in the absence of those test results: any caller passing a
True value for a stubbed test result raises `IncompleteProtocolError`.

The composite `release_tier` is the per-row maximum (most conservative)
of per-element tiers `release_tier__<element>`. The per-element logic is:

- Tier 3 if NaN prediction OR any joint OOD flag firing
  (Mahalanobis-XP, Mahalanobis-aux, latent-support).
- Tier 2 if any global caveat firing (regime_b_flag, mode_ambiguous_flag,
  ood_disagreement_flag, aux_missing_any, dist_prior_dominated) OR, for
  aux-assisted elements, `kin_ood_flag` firing.
- Tier 1 otherwise.

The aux-assisted-with-kinematic-OOD demotion is the operationalisation of
the [α/M] reframing. The aux-assisted prediction relies on the disc-
population prior; for kinematically-anomalous stars (halo, accreted-
debris, counter-rotating disc) where that prior breaks down, the
prediction is unreliable and is correctly demoted to Tier 2 even when no
other caveat fires. Spectrum-dominant elements (Teff, logg, [M/H]) are
unaffected by `kin_ood_flag`.

## 3.6 Three-criteria evaluation

A model that minimises per-element RMSE alone is not sufficient for a
release-grade abundance catalog. The thin disc dominates the APOGEE
training distribution, so a regressor that collapses to the conditional
mean of the disc (low [α/M], near-solar [M/H]) achieves competitive RMSE
while erasing the chemically-defined population structure that downstream
science requires. We therefore evaluate the kNN+strong-contrastive-v2
hybrid on three independent criteria, each producing a number we can
quote in the catalog release:

1. **Per-element RMSE / bias / σ-coverage.** Standard regression metrics
   on a 70/15/15 train/val/test split with leave-one-out kNN on train.
   The acceptance condition is RMSE within 10 % of the per-fold mean
   across 5-fold CV (test 1 of the stress battery, §3.5.4) and
   σ-coverage within ±5 percentage points of the IQR target
   (50 %) and 1σ target (68.3 %) (test 4).

2. **Structure preservation, three-component GMM tracking.** A 3-component
   Gaussian mixture is fit on the *truth* chemistry plane ([M/H], [α/M])
   and the assignments are followed into the predicted plane. We report
   per-component centroid drift and the adjusted Rand index (ARI) between
   the truth-derived clustering and an independent GMM re-fit on the
   prediction plane, Hungarian-aligned to the truth components by
   centroid distance. ARI is 0 for random clusterings and 1 for perfect
   recovery. On the kNN+strong-contrastive-v2 hybrid (this work,
   2026-04-25, sidecar
   ``reports/figures/real_data_plots/comprehensive/37_gmm_cluster_tracking_metrics.json``)
   we measure ARI = 0.566 ± 0.001 across train / val / test splits and
   centroid drift = 0.033 ± 0.005 dex RMS (per-component drifts
   ≈ 0.057, 0.014, 0.013 dex on train, weakly larger for the metal-poor
   component and tighter for the disc components). Membership purity
   (fraction of stars whose Hungarian-aligned cluster is preserved when
   re-fit blind on the prediction plane) is 0.870 ± 0.001. The bimodality
   is preserved enough that the metal-poor halo and α-rich disc components
   are recovered as distinct clusters by an independent re-fit.

3. **Per-class contamination, macro-F1 and Hellinger.** Beyond ARI we
   quantify per-class purity (precision), completeness (recall), F1, and
   the macro-F1 across the three GMM components, plus Hellinger and
   Total Variation distances between truth-Gi and pred-Gi 2-D densities
   on a 64 × 64 chemistry grid (sidecar
   ``reports/figures/real_data_plots/comprehensive/38_contamination_metrics.json``).
   The dominant contamination route is G2 (mid disc) → G3 (thin disc)
   at ≈ 33.7 % flow on every split, consistent with the fact that the
   [α/M]-poor disc occupies the high-density region the regression head
   finds easiest to default to. Macro-F1 = 0.819 ± 0.001 across splits;
   per-class F1 = (0.82, 0.72, 0.92) for (G1, G2, G3); per-class
   completeness = (0.79, 0.64, 0.95). Per-class Hellinger distances
   between the truth-Gi 2-D density and the pred-Gi 2-D density on the
   64 × 64 chemistry grid are ≈ 0.59 (G1, metal-poor), 0.32 (G2, mid),
   0.20 (G3, thin) on the test split: the metal-poor component is the
   most diffuse and therefore the most morphology-sensitive under
   prediction. The G3 → G3 self-flow is 95 %, the G1 → G1 self-flow is
   79 %, the G2 → G2 self-flow is 64 % (G2 is the dominant donor of
   contamination because it neighbours both endpoints).

These three criteria together gate model promotion. RMSE alone fails
silently on prior-collapse; the structure-preservation and contamination
criteria catch precisely the failure mode that motivated the
strong-contrastive-v2 ensemble in the first place
(HIGH_SIGMA_RESCUE_REPORT, 2026-04-25). Quantitative gates on each criterion
are wired into ``tests/integration/test_hybrid_stress_battery.py`` and into
the ``scripts/gallery/plot_19_gmm_cluster_tracking.py`` and
``scripts/gallery/plot_20_contamination_analysis.py`` diagnostic stages, so
the criteria reproduce on every release rebuild rather than being
documentation-only.

## 3.7 Compound selection function

The Pipeline 1 v1 release scope is defined by a compound selection
function (ADR-0013). Sources are included if (i) Gaia DR3 source_id
present with valid 5- or 6-parameter astrometric solution; (ii) BP/RP
coefficients available and post-Ye+2025 corrected without
catastrophic failures; (iii) G < 17 (XP-native regime); (iv) auxiliary
features sufficiently complete (per-family fallback to imputation
flagged via `aux_missing_any`); (v) for training-set inclusion, APOGEE-
Gaia cross-match passes the dr2_neighbourhood many-to-one tie-break
on smallest |Δmag| within 300 mas / 0.1 mag.

## 3.8 Code and data availability

All code is released under the MIT license at the project's GitHub
repository. The trained model checkpoint, the frozen v1 Hermite
statistics with basis fingerprint `0d34b5659e97...`, and the catalog
release artefacts (FITS, VOTable, CDS-standard ReadMe) are deposited at
Zenodo with DOI assigned post-acceptance. The CDS-VizieR submission is
in preparation alongside this manuscript.
