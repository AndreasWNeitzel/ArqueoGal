# ADR-0007: IR photometry is load-bearing, 2MASS + AllWISE fetch mandatory

**Date**: 2026-04-19 · **Status**: Accepted, Phase 2 complete

## Context

Pipeline 1 v1 was trained on Stream 1 (APOGEE × Gaia DR3) with per-star 2MASS J/H/K +
AllWISE W1/W2 photometry populated (these are in APOGEE DR19 HDU 2). Stream 3 is
non-APOGEE by construction; those stars do NOT have IR photometry from APOGEE.

Pre-flight halt before Stream 3 inference surfaced this gap. A diagnostic on
the Stream 1 val set tested three IR-completeness conditions:

| Condition | Teff Δ% | log g Δ% | [M/H] Δ% | [α/M] Δ% | [Mg/H] Δ% |
|---|---|---|---|---|---|
| Baseline (all IR) | 0 | 0 | 0 | 0 | 0 |
| Zero-imputed IR | +154% | +211% | +131% | +28% | +146% |
| NaN-imputed IR | 100% NaN | 100% NaN | 100% NaN | 100% NaN | 100% NaN |

## Decision

**Mandatory**: 2MASS + AllWISE fetch for all Stream 3 stars in Ye-OK subset. Module at
`src/arqueogal/data/ir_photometry.py`; fetch via Gaia
`gaiadr3.tmass_psc_xsc_best_neighbour` + `gaiadr3.allwise_best_neighbour` joins. ~200 MB
for ~1.3 M stars.

**For residual missing-IR stars** (~0.5% based on Stream 1 × Stream 3 measurement):
zero-impute + `ir_missing_flag = True` (matches training behaviour on the 0.1% of
Stream 1 without 2MASS counterpart). Downstream users treat flagged stars with
additional caution or exclude from volume-complete analyses.

**Broader scope**: aux-missingness flag system covering IR, parallax, extinction 
not just IR. Each gets its own boolean column in Pipeline 1 inference output.

## Rationale

- Zero-imputation at full-sample scale is a ~8σ OOD event relative to the IR
  magnitude distribution. Training's ~0.1% rate of IR = 0 was a "rare pattern"
  the model learned to recognise; pushing 100% of stars into that pattern is a
  massive distribution shift.
- NaN propagates through the trunk (XpFeatureAdapter is a pass-through) to NaN
  predictions, and the current OOD module (108-D XP Mahalanobis) does not cover
  aux features. Silent failure without a flag.
- All 5 labels show substantial degradation without IR. [α/M]'s +28% is the mildest;
  the others at +131% to +211% are catastrophic. IR is load-bearing for v1 as
  trained.

## Alternatives rejected

- **Retrain without IR as separate ensemble**: weeks of work, out of D-Cat-b scope.
- **Impute IR from Gaia color-magnitude relations**: would be systematic, needs
  its own validation step. Not feasible within v1.
- **Zero-impute without flag**: silent data-quality degradation downstream. Violates
  honesty-under-uncertainty.

## Consequences

- Phase 2 implemented `data/ir_photometry.py` with tests and provenance.
- Inference-driver fix (ADR-0012) mirrors training's `nan_to_num` boundary and emits
  aux-missingness flags.
- Selection function v1.1 (ADR-0013) incorporates IR-completeness probability into
  the compound selection function.
- Residual missing-IR stars (~0.5%) flagged and documented as "rare-pattern regime"
  in release notes.

## Methodology note

The IR question was initially open: IR could have been decorative. The diagnostic
resolved it decisively. Three of five labels (Teff +154%, log g +211%, [M/H] +131%)
are worse off without IR than the §9.2 permutation analysis alone suggested. IR
carries more absolute value for atmospheric parameter recovery than that analysis
indicated. Worth remembering: permutation importance within a set of correlated
features can underestimate the group's load-bearing contribution.
