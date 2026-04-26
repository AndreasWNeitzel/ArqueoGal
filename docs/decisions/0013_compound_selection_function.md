# ADR-0013: Compound selection function (|b|×G×Teff×log g + IR completeness)

**Date**: 2026-04-19 · **Status**: Accepted, Phase 2 complete

## Context

Initial selection-function design (#100, v1.0) was a 2-D grid of |b| × G capturing
the Ye+2024 `NO_SYNTH_PHOT` flag rate across the sky. Stream 1 diagnostic showed
134× ratio between low-|b| (10.48% flag rate) and high-|b| (0.08%), substantial
enough that the selection function is a first-class scientific feature of the
catalogue, not a footnote.

Pre-Phase-3 scope discussion (2026-04-19) expanded this:
- IR completeness rate (~99.46% Stream 3) couples to per-star inference reliability
  (ADR-0007). Missing IR → zero-impute → rare-pattern regime → lower per-star
  confidence.
- Parallax and extinction-map completeness should also contribute.

## Decision

**Compound selection function v1.1**: per-star `selection_prob` scalar combining:
1. Ye+2024 retention probability from 4-D |b|×G×Teff×log g grid (5×5×5×5).
2. IR completeness probability from training-distribution IR availability.
3. Parallax/extinction availability gates (0/1, not soft).

Computed at feature-matrix build time using Andrae+2023 Teff/log g estimates
pre-inference (not the model's own predictions, avoids circular dependency).

## Rationale

- Ye retention alone is insufficient: at low |b| the retention is ~90% for bright
  giants but drops to ~60% for faint/complex stars.
- IR completeness adds a second selection dimension: some regions of sky have
  materially lower IR coverage, and missing IR converts to reduced per-star
  reliability.
- Volume-complete population analysis downstream (density estimates, population
  fractions) requires the compound weight to be accurate.
- Pre-inference computation avoids bootstrapping the selection weight on the
  model's own output, which would create reasoning circularity.

## Alternatives rejected

- **Simple |b| step function**: under-represents the magnitude dependence.
- **2-D |b| × G grid**: initial v1.0; misses Teff/log g dependence.
- **Post-inference weighting using model predictions**: circularity.
- **Omit selection function from release**: makes volume-complete analysis
  impossible for downstream users.

## Consequences

- Per-star `selection_prob` is a first-class column in D-Cat-b.
- Release documentation section dedicated to the compound selection function:
  how it's computed, what downstream users should do with it, how to invert-weight
  for volume-complete analyses.
- Stream 1 training stats used to compute the IR-completeness probability are
  themselves a provenance artefact (`reports/selection_function/`).

## Methodology note

The decision to use Andrae+2023 Teff/log g estimates (rather than the pipeline's own
model predictions) for the pre-inference selection weight is the right
circularity-prevention choice, a subtle point worth documenting.
