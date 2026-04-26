# Protocol: cross-catalogue consistency (test 6 of §3.3)

**Status:** Scoped. Deferred to D-Cat-d (February 2027).
**Authoring trigger:** external_peer_review.md CRITICAL #3; META_META §8 P1-9.

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

## 5 Why this is deferred

Two reasons:

1. **Stream 3 inference must complete first.** Until Stream 3 is run
   at full scale (1.5 M sources), the overlap subsets with each
   comparison catalog are too small for statistically meaningful
   per-magnitude-bin breakdowns.

2. **The compute is non-trivial.** Cross-matching Stream 3 against
   five comparison catalogs at full scale is a multi-day pyvo / ADQL
   operation. Bland-Altman per element × catalog × magnitude bin
   produces ~150 figures; aggregating to a small Methods-paper
   figure requires careful summary.

Defer to D-Cat-d (February 2027) per META_META §10 user decision 2.
The methods paper at submission cites test 6 as "deferred to D-Cat-d
release per the timeline in `docs/plan/03_stream3_inference.md`".

## 6 Effort estimate

2–3 weeks once Stream 3 inference is complete.

## 7 Outcome contract

When complete, test 6 moves from STUBBED_TESTS in `tier_promotion.py`.
the methods-paper §3.3 promotion narrative becomes 6/6 coverage, and
the catalog v1.2 release publishes the cross-catalogue Bland-Altman
results as a supplementary appendix.
