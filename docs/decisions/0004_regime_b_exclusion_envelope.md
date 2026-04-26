# ADR-0004: Regime B exclusion envelope (low-|b| warm-upper-RGB population-only)

**Date**: 2026-04-19 · **Status**: Accepted, in production

## Context

Pipeline 1 v1 halt-cell diagnosis identified 2 cells (34, 49) at |b|<3°, Teff>4820 K,
log g<2.05 showing structural miscalibration: mean-bias signature E[z_Teff] = +1.0 to
+1.4, indicating systematic over-prediction of Teff by ~1σ. Not fixable within
Pipeline 1's dust-map budget (no better 3D maps available at |b|<5°).

## Decision

Per-star boolean exclusion flag `pipeline1_tier1_release = False` when (|b|<5° AND
predicted Teff > 4750 K AND predicted log g < 2.1), with small buffer around cell
boundaries. Excluded stars are carried in D-Cat-b as **population-level only**: the
predictions exist and can support population statistics, but per-star reliability is
not claimed.

## Rationale

- Regime B stars are physically problematic, not a calibration failure to fix.
  Galactic-plane crowding, high extinction, and the A_V uncertainty at low |b| are
  real confounders that post-hoc calibration cannot remove.
- The mean-bias direction is itself informative: over-prediction of Teff (not
  under-prediction as under-dereddening would produce) suggests over-correction or
  a training-set correlation the model learned wrongly. Flagged for methods paper.
- Exclusion is cheap (~30 val stars, 0.07%; expected ~1-5% of Stream 3 based on
  |b| distribution).
- Exclusion is honest. Shipping per-star predictions known to be biased by ~1σ in a
  parameter region would contradict D-Cat-b's differentiator (calibrated honesty
  about where the model works).

## Alternatives rejected

- **Fix the extinction handling with Bayestar19 + 3D dust at low |b|**: budget-busting
  (~3 GB) and out of scope for v1.
- **Post-hoc μ correction per cell**: rejected at the earlier calibration iteration
  (ADR-0003 sibling) because μ correction is label redefinition disguised as
  calibration; violates scientific honesty of the release.
- **Per-star retention with warning flag only**: tempting but pushes the burden of
  interpretation onto downstream users who may not apply it consistently. Exclusion
  is cleaner.

## Consequences

- ~30 val stars excluded from Tier 1 per-star release in v1 testing.
- Expected 1–5 % of Stream 3 inference output will carry the Regime B flag.
- Methods paper documents the Teff-bias direction puzzle as a specific empirical
  finding; invites future investigation (3D dust, multi-spectrum cross-validation).

## Methodology note

Direction-of-bias puzzle flagged for the methods paper as unresolved but not
release-blocking.
