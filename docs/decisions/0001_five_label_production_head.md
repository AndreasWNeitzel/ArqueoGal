# ADR-0001: 21-label → 5-label production head for Pipeline 1 v1

**Date**: 2026-04-19 · **Status**: Accepted, shipped

> **Note (2026-04-22):** references to "Pipeline 2" below describe the
> original in-repo population-classification module, which has since been spun
> out into the **Starfold** repository. The consequence recorded here (5-D
> covariance footprint for downstream consumers) still applies, it now
> applies to Starfold rather than an in-tree Pipeline 2. The historical
> reasoning is preserved as-is.

## Context

Original Pipeline 1 design (per `research_brief.md §3.2`) targeted 21 labels spanning
Tier 1 (Teff, log g, [M/H]), Tier 2 (5 per-element), Tier 3 (13 audit-only). The
block-Cholesky covariance head was designed with a 4-block physics-motivated structure
(atmospheric / α-process / Fe-peak / light) plus diagonal-only tail.

## Decision

**Pipeline 1 main production model for D-Cat-b Tier 1 is the 5-label variant**
(`LabelTiers.five_label()`) with a single 5×5 full-Cholesky block: {Teff, log g, [M/H],
[α/M], [Mg/H]}. 21-label checkpoint retained as a methods-paper comparator; do NOT
promote to main.

## Rationale

Empirical comparison at 2026-04-19 showed:

- Per-label marginal calibration essentially unchanged between 21-label and 5-label
  on the shared labels (~1-2 pp coverage deviations in both).
- Joint-tail coverage: 21-label cov95 was −13.3 pp below nominal; 5-label cov95 was
  −5.9 pp below nominal. Fewer covariance cross-terms = less capacity to misestimate
  distant off-diagonals.
- Smoothness of calibration: 21-label tripped the adjacent-cell α-ratio > 2 flag;
  5-label did not (max 1.78).

The original hypothesis that "Tier-3 gradients corrupt the Tier-1 representation" was
NOT supported, per-cell bias on Teff and [M/H] is ~unchanged between label counts.
But the joint-tail improvement is real, and it matters for scientific use of the full
covariance matrix.

Scientific reframing: D-Cat-b's scope is "XP abundances for stars without APOGEE,
scientifically defensible per §3.3 tier-promotion protocol". NOT "all 21 labels".
Uncalibrated per-star predictions for 16 additional elements are liability, not
surface area. Future additional elements go through **separate specialist heads**
(AspGap hydra pattern) on the shared pretrained encoder, each §3.3-audited
independently before release.

## Alternatives rejected

- **Ship 21-label**: better "surface area" nominally but worse calibration per label
  reliability in the release-gate sense. Liability, not value.
- **Drop to 3 labels (Tier 1 only)**: considered and rejected because α-process
  chemistry is scientifically load-bearing for Galactic archaeology and [Mg/H] is
  the strongest individual α-tracer in XP (Mg b + MgH).

## Consequences

- 21-label head stays trainable for methods-paper comparisons; never promoted to main.
- Any future additional element gets a specialist head + §3.3 audit, not a joint
  retrain.
- Pipeline 2's MC ensemble over Pipeline 1 posteriors operates on 5-D covariance, not
  21-D. Smaller computational footprint downstream.

## Methodology note

Joint comparative analysis produced unambiguous evidence for 5-label winning on the
gating metric (joint-tail coverage). Decision is robust.
