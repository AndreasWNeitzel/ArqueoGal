# Protocol: cross-catalogue consistency (test 6 of §3.3)

**Status:** **Implemented (2026-04-28).** Framework lives in
`src/arqueogal/xp_abundances/main/cross_catalogue.py` (statistics) and
`cross_catalogue_plots.py` (seven diagnostic-plot families); CLI driver is
`scripts/run_cross_catalogue_validation.py`. Outstanding work is the
catalogue-cross-match step (each external catalogue requires a different
TAP/VizieR ingestion before this driver can be run end-to-end against the
production release).
**Authoring trigger:** external_peer_review.md CRITICAL #3; META_META §8 P1-9;
domain-reviewer 2026-04-28 SOTA-viability finding 1.

## 1 Why this protocol

Test 6 of the §3.3 promotion protocol, cross-catalogue consistency.
is the second of two stubbed tests in Pipeline 1 v1. It is the test
that says: when our predictions overlap with another XP-abundance
pipeline (AspGap, SHBoost, Fallows-Sanders, Guiglion, etc.) or with
a high-resolution catalog (GALAH DR4), do we agree within tolerance?

Without this test, the methods paper cannot quantify Pipeline 1's
performance against the published state of the art. Hostile referees
will ask for it.

## 2 Comparison catalogues

The five overlap pools, sized roughly:

- **GALAH DR4** (Buder et al. 2024, arXiv:2410.12272). High-resolution
  optical spectroscopy. Overlap with Stream 3 estimated at 100–500k
  stars. The gold-standard comparison for Mg, Fe, [α/Fe].
- **AspGap** (Li et al. 2024, ApJ 974, 42; arXiv:2309.14294). XP-NN
  pipeline with a similar information channel. Comparable architecture;
  comparison gives apples-to-apples performance.
- **SHBoost** (Anders et al. 2024, A&A 691, A127; arXiv:2407.06963).
  Boosting on XP. Different methodology but same input and similar
  per-element outputs.
- **Guiglion et al. 2024** (A&A 682, A9; arXiv:2306.05086). GSP-Spec
  cross-validation pipeline.
- **Fallows and Sanders 2024** (MNRAS 531, 2126; arXiv:2405.10699).
  XP-NN pipeline that explicitly raises the spectrum-vs-prior question
  Pipeline 1 answers via the information-content audit.

## 3 Plan

### Step A: cross-match

Per comparison catalog, cross-match Stream 3 source_ids. For each pair.
construct the overlap subset.

### Step B: per-element Bland-Altman analysis

For each element X in the project's release set ∩ the comparison
catalog's release set:

- Compute mean offset: <Pipeline 1 X − comparison X>.
- Compute scatter: stddev of (Pipeline 1 X − comparison X).
- Compute the σ-ratio: stddev_of_difference / sqrt(σ_pipeline² +
  σ_comparison²). If σ-ratio is far from 1.0, the joint uncertainty
  is mis-estimated.

Per-magnitude-bin breakdown (G ≤ 15, 15 < G ≤ 16, G > 16) so the
methods paper can quote magnitude-dependent agreement.

### Step C: methods-paper figure

A 5-panel Bland-Altman plot per element, with one panel per comparison
catalog (or a single overview figure). Figure 7 in the manifest.

## 4 Acceptance criteria (per element × catalogue × magnitude bin)

Test 6 PASS:
- |bias| ≤ 0.05 dex (or ≤ 50 K for Teff, ≤ 0.10 dex for logg).
- σ-ratio in [0.7, 1.4] (joint σ correctly within 30 % of the
  empirical scatter).
- ≥ 100 paired stars per (element × catalog × magnitude bin)
  combination.

Test 6 FAIL: any of the above violated. Document with explicit per-
element-per-catalog notes.

## 5 Implementation map (2026-04-28)

The framework is shipped; only the cross-match step is per-catalogue work.

- `arqueogal.xp_abundances.main.cross_catalogue.compute_cross_catalogue_report`
  takes the Pipeline-1 release frame plus a dict of already-cross-matched
  reference frames and returns a :class:`CrossCatalogueReport` carrying:
  per-(label, catalogue, mag-bin) Bland-Altman cells (bias, scatter,
  σ-ratio, MAD-robust scatter, 68/95/99 % coverage, Pearson, n);
  per-(label, catalogue) trend curves vs ArqueoGal [M/H] and Teff;
  per-(label, catalogue) Teff×log g cell heatmaps; pass/fail map under the
  §3.3 acceptance gate.
- `cross_catalogue_plots.render_all` emits the seven plot families
  (Bland-Altman scatter facets, residual histogram + N(0,1) overlay,
  metallicity-dependent bias trend, Teff-dependent bias trend, Teff×log g
  heatmap, coverage curve, rank-summary heatmap) as PDF (paper-grade) +
  PNG (gallery) into a per-slice subdirectory.
- `scripts/run_cross_catalogue_validation.py` is the CLI driver. It also
  provides the matched-σ-subsample diagnostic (`--matched-sigma-quantile`)
  that controls for the σ-inflation Tier-2 demotion selection bias
  documented in `release.py`.
- Bindings between ArqueoGal label keys and external-catalogue column
  names live in `configs/main/cross_catalogue_bindings.yaml`. Five
  catalogues are pre-bound (AspGap, SHBoost, Guiglion+2024, Andrae+2023,
  Zhang+2023, GALAH DR4); add new entries as needed.

The remaining work is the cross-match itself: each reference catalogue
needs a TAP/VizieR ingestion script that emits a parquet of overlap rows
1:1-aligned with the Pipeline-1 release. That work is per-catalogue and
not in scope for this protocol document.

## 6 Effort estimate (revised 2026-04-28)

- Framework: shipped.
- Per-catalogue cross-match: 1-2 days each (5 catalogues × 1.5 days ≈
  2 weeks).
- Methods-paper figure consolidation from the seven plot families: 2-3
  days.

## 7 Outcome contract

When the cross-matches are done, the framework moves test 6 from
STUBBED_TESTS in `tier_promotion.py` to PASSED, the methods-paper §3.3
promotion narrative becomes 6/6 coverage, and the catalogue release
publishes the seven plot families plus the long-form CSV as a
supplementary appendix.
