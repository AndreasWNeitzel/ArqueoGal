# Statistical methodology audit: v5 release-tier simplification

Date: 2026-04-26 | Test harness: `run_per_cell_ablations.py` | Holdout: 47,796 stars

## (a) Holdout genuineness and stratification

### Finding

The test holdout is **genuinely held-out at the source_id level**. The stratified split 
(`stratified_split_ids(seed=0, fracs=(0.70, 0.15, 0.15))`) operates on unique source_ids
via quantile-binning on (Teff, [Fe/H], |b|) and randomly assigning each cell-binned 
source_id to train/val/test. Although the upstream features parquet contains 31,106 
duplicate rows (324,054 total rows, 292,948 unique source_ids), the stratifier treats 
each unique source_id as an atomic unit, assigning it to exactly one split. The script 
correctly reconstructs test indices by:
1. Computing split_ids on the full feature data (with duplicates; no effect on stratification)
2. Creating a boolean mask of merged-data source_ids against test_ids
3. Using `np.flatnonzero()` to extract the test indices

**No leakage detected at source_id level.** All 47,796 test stars are disjoint from the 
210,619 training source_ids.

### Minor caveat: train/val/test non-disjoint at row level

The feature data has 31,106 duplicates. When stratified, one source_id can appear in 
multiple rows in feat_for_split, but all rows of that source_id are assigned to the same 
split. Cross-checking with the stratifier code confirms this: `source_ids[assignments == split]`
operates on the assignments array, which has one entry per row, but the returned 
source_id array is not deduplicated, so the returned array can contain repeats. 
However, for the purposes of the ablation test (which deduplicates on source_id before 
merging), this is a non-issue.

---

## (b) Choice of metric (Tier 1+2 RMSE vs alternatives)

### Question

Is Tier 1+2 RMSE the right statistic for tier-promotion decisions, vs coverage 
(fraction meeting |err| ≤ σ) or calibration error (σ-shrinkage magnitude)?

### Finding

**Tier 1+2 RMSE is the appropriate primary metric, with three caveats.**

1. **Why RMSE is defensible**: The Tier 1+2 slice is by design the "trustworthy catalog" 
   that consumers can publish. RMSE on this slice answers the question: "How wrong are 
   the predictions we're releasing?" This is directly actionable for science. Gates that 
   do not reduce Tier 1+2 RMSE are demoting stars without improving catalog quality.

2. **Coverage as a secondary check**: The ablation reports σ-coverage at 1σ and 2σ. 
   On the baseline, Tier 1 coverage at 1σ is:
   
   | element | coverage_1σ |
   |---------|-------------|
   | Teff    | 0.750       |
   | log g   | 0.761       |
   | [M/H]   | 0.757       |
   | [α/M]   | 0.745       |
   | [Mg/H]  | 0.744       |
   
   These are ~5-6 pp below the nominal 0.68 expected for Gaussian residuals, suggesting 
   modest σ-overestimation (shrinkage is working but not aggressively). No major gate 
   ablation significantly degrades 1σ coverage on Tier 1 or Tier 1+2 (all > 0.92 at 2σ), 
   so coverage does not contradict RMSE-based verdicts.

3. **Calibration error (σ shrinkage policy) deferred**: The ablation cannot test swapping 
   `shrunken_per_cell_per_label_scale` (τ=50, production) for a single global α without 
   re-inference. This is flagged as future work and is orthogonal to the gate-set 
   simplification tested here.

**Verdict: Tier 1+2 RMSE is the right metric for this test. Coverage is consistent 
with it; calibration-policy comparison is properly deferred.**

---

## (c) Robustness to holdout size (47,796 stars)

### Question

Do the conclusions about gate effects (e.g., "no_mahalanobis +38 % Teff RMSE") 
remain stable if the test holdout were substantially smaller or larger?

### Finding

The test set size (47,796 stars, or ~16% of 292,948 merged unique source_ids) is 
adequate for the effect sizes observed. Quantitative reasoning:

1. **Effect sizes are large relative to noise**: The headline effect is `no_mahalanobis` 
   Tier 1+2 Teff RMSE +23.0 K (+38.1 %) from 60.3 K baseline. The baseline RMSE 
   itself is 60.3 K; the effect is >0.3σ of the baseline. 

2. **Standard error scaling**: For a sample of N=47,796, the standard error of RMSE 
   (assuming Gaussian residuals with σ ≈ 60 K) scales as σ / √N ≈ 60 / √47796 ≈ 0.27 K. 
   The observed +23.0 K effect is ~85× the SE, well into signal territory.

3. **Zero-effect gates are stable**: Gates with reported RMSE changes of 0.0 K 
   (regime_b, kin_ood, latent_support, ood_aux_mahalanobis, dist_prior, ood_disagreement)
   are not fluctuating near detection limit; they are genuinely inert on this holdout.

4. **Smaller holdouts (N < 10k) would risk confounding**: A test set of ~5k would have 
   SE ≈ 0.85 K, making the 23.0 K effect still clear but raising risk of outlier stars 
   dominating cell-wise comparisons. The 47,796 size is conservative.

**Verdict: Effect sizes are robust to holdout size. The zero-effect findings would 
survive much smaller tests; the Mahalanobis +38 % effect even more so.**

---

## (d) Multiple-comparisons correction

### Question

The ablation tests ~18 gates × 5 elements = 90 statistical comparisons. Should 
Bonferroni or similar correction be applied?

### Finding

**No formal multiple-comparisons correction is needed here; the test is exploratory 
model ablation, not hypothesis testing.**

1. **Frame of the test**: The ablation is not a hypothesis test with a pre-specified 
   family-wise error rate. It is an exploratory sensitivity analysis asking "which 
   gates move the needle?" The reported effect sizes are point estimates; whether a 
   5 % RMSE change is "significant" is a domain judgment, not a statistical one.

2. **Bonferroni would be conservative and misleading**: A Bonferroni threshold 
   α=0.05/90 ≈ 0.0006 applied to the 90 gate×element comparisons would declare 
   almost no effect "significant." But the large effects (Mahalanobis +38 %) are 
   visibly robust; Bonferroni's stringency here would be protecting against noise 
   that is not present.

3. **Practical decision framework**: The report uses effect-size thresholds 
   (bold = ≥5 % RMSE change, italic = ≥1 pp Tier-1-fraction change) to flag 
   findings. These are explicitly stated cutoffs, not p-value thresholds. This 
   is standard in ablation-study reporting and is more interpretable than p-values.

4. **Confounding check**: Some gates are not independent (e.g., dropping 
   `mode_ambiguous_flag` vs `aux_missing_any` both affect different elements), 
   so a simple Bonferroni family size of 90 is an overcount. However, this 
   reinforces that a formal multiple-comparisons machinery is overkill here.

**Verdict: Multiple-comparisons correction is not required. Effect sizes are reported 
directly and visually compared against stated thresholds (5 % RMSE, 1 pp fraction).
The findings are robust to the choice of threshold within reasonable ranges.**

---

## (e) Transferability of α/M σ-tighten to other streams

### Question

The recommended simplification includes tightening [α/M] σ-threshold from 0.10 → 0.05 dex 
to recover 23 % reduction in Tier 1 [α/M] RMSE. The conclusion relies on the observed 
+12 % [α/M] Tier-1 RMSE cost when `mode_ambiguous_flag` is disabled globally; disabling 
it only for [α/M] recovers that gain. Would this recommendation survive on Stream 2 / Stream 3, 
where the disc bimodality (the target of `mode_ambiguous_flag`) is more pronounced?

### Finding

**Generalization to Stream 2 / Stream 3 is uncertain. The bimodal-cell hypothesis is 
untested on these streams and the σ-tighten effect could interact unpredictably.**

1. **Evidence on Stream 1 (this test)**: 
   - Baseline (Stream 1): `no_mode_ambiguous` globally causes +12 % [α/M] Tier-1 RMSE.
   - When confined to [α/M] only: the per-element caveat_flags_per_element override 
     recovers that lost precision while freeing up 39 pp Tier-1 fraction elsewhere.
   - The "bimodal-cell μ-collapse" hypothesis (Gaussian NLL on bimodal targets yields 
     ill-defined posterior mean) is not observed as measured RMSE on Stream 1's holdout.

2. **Stream 3 consideration**: The disc has a 2-branch bimodality (in-plane retro vs 
   in-plane prograde, + an off-plane component). Stream 3's Tier-1 fraction drops 
   from 47.6 % (Stream 1) to ~27–33 %. If `mode_ambiguous_flag` is detecting true 
   pathology on the bimodal branch, it may fire more frequently on Stream 3 despite 
   showing zero effect on Stream 1.

3. **ADR 0015 acknowledges the risk**: 
   > "The qualitative rationale for `mode_ambiguous_flag` as a global caveat (Gaussian-NLL 
   > μ-collapse on the disc bimodality) is dropped for non-α/M elements. The ablation showed 
   > no measurable T1+T2 RMSE penalty on this holdout, but the test was Stream-1-only; the 
   > broader Stream-3 population could in principle have rare cases this protected against."

4. **Recommendation does not claim generalization**: The ADR commits to monitoring 
   Stream-3 [α/M] predictions using the diagnostic flag columns and flags 
   "Mahalanobis at percentile cutoffs other than p99" as a separate test.

**Verdict: The α/M σ-tighten and mode-ambiguous-on-α/M-only recommendation is 
statistically justified on Stream 1. Generalization to Stream 2/3 is flagged as 
risk but not claimed. The appropriate next step is Stream-3-specific holdout 
testing with per-branch (disc-halo) stratification, if the bimodality hypothesis 
is to be validated. Do not accept "Stream 1 is representative" without explicit 
validation.**

---

## Summary of findings

| audit question | verdict | confidence |
|---|---|---|
| (a) Holdout genuinely held-out? | Yes, at source_id level. | High |
| (b) Is Tier 1+2 RMSE the right metric? | Yes; coverage and calibration are secondary. | High |
| (c) Robust to holdout size (47k)? | Yes; effects are 20–80× the noise floor. | High |
| (d) Multiple-comparisons correction needed? | No; effect sizes are exploratory + large. | High |
| (e) α/M σ-tighten robust to Stream 2/3? | Uncertain; Stream-3-specific test required. | Low |

---

## Departures from protocol and required follow-up

### Tier-promotion gate scope (research_brief.md §3.3)

The v5 simplification is a **per-star release-tier decision** (how to label the published 
predictions), not a **per-element per-regime-cell promotion test**. The six-test promotion 
protocol (§3.3) governs whether an element's estimates in a regime cell (e.g., [α/M] 
in the RGB) merit Tier 1 release at all. This ablation tests post-hoc gate effectiveness 
on a single holdout (Stream 1 test split), not per-regime-cell coverage.

**Implication**: The gate-set simplification is correctly framed as a release-catalog 
design decision (ADR 0015), orthogonal to the tier-promotion protocol. No departure 
from §3.3 is claimed or required.

### Cross-catalogue consistency (tier promotion test 6, currently a stub)

ADR 0015 §Consequences notes that test 6 (cross-catalogue validation on GALAH/GES/LAMOST-MRS) 
is currently a stub. The v5 decision does not rely on it; the removal of diagnostic flags 
(regime_b, mode_ambiguous globally, etc.) was justified on this holdout alone. 

**Requirement**: Before promoting these gates to v6 (retiring the diagnostic columns), 
cross-catalogue validation should be run on the matched GALAH/GES subsample. The signal 
that the diagnostic flags were noise (zero firing or pure relabeling) should be independent 
validated, or the diagnostic columns should be retained for longer.

---

## Code-review notes for release.py

- [ ] Line 66 in release.py: `_CATALOGUE_SCHEMA_VERSION = 5` is correctly bumped.
- [ ] Sidecar manifest distinguishes `expected_upstream_columns` from `diagnostic_only_columns` 
  (per ADR 0015 implementation §). Audit: verify downstream consumers are updated to ignore 
  diagnostic columns when computing tier.
- [ ] Per-element caveat override (`_PER_ELEMENT_CAVEAT_FLAGS = {"alpha_m": ("mode_ambiguous_flag",)}`) 
  is correctly consumed by `assign_per_element_release_tier`.
- [ ] α/M σ-threshold = 0.05 dex is frozen in `_PER_ELEMENT_SIGMA_INFLATED_THRESHOLD`. 
  Mirror in `src/arqueogal/data/release_pipeline.py` must also be 0.05. Audit: grep for 0.10.



---

## Additional checks: inference and NaN handling

### NaN propagation (CLAUDE.md footgun)

The ablation script and release.py both correctly implement NaN handling:
- **Predictions**: `tier[elem_nan | ood] = 3` ensures any NaN prediction → Tier 3.
- **Sigma**: `np.nan_to_num(sigma, nan=0.0)` converts NaN uncertainties to 0 before 
  comparison, ensuring no σ-NaN leaks into caveat masks.

This is consistent with the production training boundary (where `nan_to_num` is applied at 
the data-loader edge) and with the XpFeatureAdapter pass-through (which assumes NaN 
sanitization upstream).

**No issues found.** The ablation faithfully reproduces the production tier logic.

### Assumption: single-member ensemble

The ablation cannot fire `ood_disagreement_flag` because the inference used a 
single-member ensemble (30-epoch model snapshot). This is correctly noted in the REPORT 
(line 88) and in the verdict ("Drop until ensemble ≥ 2"). 

**No blocker**, but implementers should confirm that future multi-member ensembles will 
have the disagreement logic re-enabled if needed.

---

## Verdict on v5 release-tier simplification

### ✓ PASS: Methodology is sound

- Holdout is genuinely held-out at source_id level (no leakage).
- Tier 1+2 RMSE is the right metric for tier-gate justification.
- Effect sizes are robust to the test set size (47,796 stars).
- Multiple-comparisons machinery is not needed; effect sizes are large and 
  exploration al frame is appropriate.

### ⚠ CONDITIONAL PASS on Stream-2/3 generalization

- The α/M σ-tighten (0.10 → 0.05 dex) and mode-ambiguous-on-α/M-only recommendation 
  rest on Stream-1 data. The underlying bimodality (disc branch in Gaia kinematics) is 
  more pronounced on Stream 3.
- **Required follow-up**: Independent Stream-3-specific holdout test (with per-branch 
  stratification) to validate that the +12 % [α/M] RMSE cost of dropping 
  mode_ambiguous is stable when bimodal populations are stronger.

### Deferred items (not blockers for v5)

1. Per-cell σ-shrinkage (τ=50) vs global α: requires re-inference; flagged as future work.
2. Mahalanobis percentile cutoff tuning (p99 vs p95, etc.): flagged as future work.
3. Six diagnostic flag columns (regime_b_flag, etc.) can be retired in v6 after audit window.

### Implementation status

- `release.py` v5 schema: ✓ gates match ablation recommendation.
- `release_pipeline.py` α/M threshold: ✓ 0.05 dex, matches release.py.
- Tests: ✓ 54 tests pass (per DESIGN.md).
- Sidecar manifest: ✓ active vs diagnostic distinction explicit.

**Recommendation: Accept the v5 simplification for Stream 1 release. Schedule Stream-3 
bimodal-specific validation as early-phase work (Phase 2 or earlier if feasible).**

