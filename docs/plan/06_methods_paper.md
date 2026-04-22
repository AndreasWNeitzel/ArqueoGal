# Phase 06 — Methods paper(s)

**Status: Tracked, not actively drafted. Content spine identified.**

## Findings to publish (from 2026-02–04 sprint)

The seven methodology findings identified during the Pipeline 1 port sprint, any subset
or all of which could form a methods paper:

1. Noise-floor + PCA result: 43 effective modes of abundance-relevant information
   remain after Ye+2024 correction; PC1 carries 73.7% on the 43-D truncation.
2. Label scaling is necessary for multi-task NLL training — previously missing in
   published XP pipelines.
3. β-NLL at β=0.5 absorbs per-cell μ bias into inflated σ. Publishable negative
   result about a widely-used loss choice. β=0 exposes the bias as explicit mean
   error.
4. Per-coefficient z-scoring of orthonormalised Hermite coefs is the dominant fix for
   per-cell μ bias (78–79% reduction on Teff and [M/H]). Published XP pipelines
   normalise but do not per-coefficient z-score.
5. Reducing 21 → 5 labels improves joint-tail coverage without changing per-label
   marginals. Quantifies the cost of Tier-3 gradients in joint-covariance training.
6. GP smoothing of calibration α fails at physically-structured parameter boundaries
   (cool-giant corners). Negative methodology result.
7. Empirical-Bayes per-(cell, label) shrinkage beats GP smoothing for this calibration
   problem. Preserves per-cell semantics GP breaks.

Additional from §9.2 audit:

8. 2-D KSG CMI summaries are biased (inflated upward for Teff by ~5×, collapsed to
   ~zero for [M/H], [α/M], [Mg/H]) relative to PCA summaries. Methodology note for
   §9.2-style audits.
9. [α/M]'s zero PCA-CMI despite unambiguous shuffle-null and joint-shuffle signals
   — H2 finding that aux features absorb [α/M]'s variance more than other chemistry
   labels.
10. The three-diagnostic triage (CMI / permutation / aux-only baseline) resolves
    §9.2 shuffle-null failures into either label-specific per-star-release decisions
    or model-wide problems. Publishable as "when to run auxiliary diagnostics on
    §9.2 failures".

## Design goal

Write up the **full diagnostic sequence**, not just the final calibrated model. Each
step falsified a plausible hypothesis; the falsification chain is the content. No
published XP-abundance pipeline has reported calibration at this rigor.

## Acceptance criteria

**Not yet defined.** No target journal, no outline, no author list, no timeline.

## Needs clarification

- Venue(s): A&A methodology section? A&A-S (Supplement)? NeurIPS ML4PS? MNRAS?
- One paper or multiple? Pipeline 1 methodology vs §9.2 audit methodology could be
  one or two papers. Population-classification methodology is a Starfold concern
  and will be written up separately by that repo.
- Author list and order. Standard for ArqueoGal team is lead = first, PI = last.
- Timeline relative to D-Cat-b (Aug 2026) release. Pre-release (which helps peer
  review of the catalogue) vs post-release (which lets the paper cite the catalogue
  DOI).
