# ADR-0011: β-NLL β=0.5 retained; β=0 canary ruled it out as miscalibration cause

**Date**: 2026-04-19 · **Status**: Accepted

## Context

During the calibration-diagnostic sequence, one hypothesis was that β-NLL at β=0.5
(Seitzer et al.) absorbs per-cell μ-bias into inflated σ, masking structural model
limitations. β=0 (pure Gaussian NLL) would expose the bias directly as explicit
mean error. A β=0 canary retrain tested this.

## Decision

**Retain β=0.5 for production**. The β=0 canary confirmed that β-NLL's variance-
inflation trick is NOT the dominant cause of the per-cell μ-bias. Both β=0 and β=0.5
show the same bias structure; β=0.5 just makes the σ wider to absorb it, which is a
separate problem (under-calibration, not mis-localisation).

## Rationale

The β=0 retrain showed:
- Total E[z²] ≈ 1 per label → σ globally well-calibrated at β=0 (consistent with
  β=0.5 post-scaling).
- E[Mahal²] pre-shrink = 21.01 (exactly n_dims) → joint Gaussian perfectly calibrated
  marginally.
- **But**: ~50–60% of per-label σ² remained between-cell μ-bias variance, same as
  β=0.5. Both loss choices split the total budget the same way.

Interpretation: β=0.5's variance-inflation was a symptom, not the cause. The cause
was input-side (ADR-0002 per-coefficient z-scoring) and head-capacity (ADR-0001
5-label simplification). Once those were fixed, β=0.5 is fine, and β=0 offers no
additional advantage.

## Alternatives rejected

- **Switch to β=0 production**: no calibration benefit; slightly less robust to
  outliers during training.
- **Switch to β=1 (full variance weighting)**: more extreme than β=0.5, not
  empirically tested, no reason to favour.

## Consequences

- β=0.5 is the production loss-weighting in `beta_nll_block_cholesky`.
- The β=0 canary result is itself a methods-paper contribution, β-NLL's
  variance-inflation is not the mechanism behind observed miscalibration in XP
  pipelines (at least not for this architecture).

## Methodology note

This ADR documents a hypothesis that was ruled out by evidence. The negative result
is itself valuable: it redirected the diagnostic sequence toward the input side,
where the real cause lived. Methods paper should feature the full diagnostic
sequence honestly, including this falsified hypothesis.
