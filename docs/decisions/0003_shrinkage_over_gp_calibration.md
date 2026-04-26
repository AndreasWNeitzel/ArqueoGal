# ADR-0003: Empirical-Bayes shrinkage over GP smoothing for per-cell calibration

**Date**: 2026-04-19 · **Status**: Accepted, in production. GP retained-but-rejected.

## Context

After per-coefficient z-scoring (ADR-0002) fixed the dominant per-cell μ-bias, residual
calibration issues concentrated in 5 halt cells (4, 15, 28, 34, 49). Two regimes:

- **Regime A** (cells 4, 15, 28): sparse cells at Teff<4441 K, mid-high |b|, with
  Var(z) 2–5 indicating σ under-predicted by 1.4–2.2×.
- **Regime B** (cells 34, 49): low-|b| warm-upper-RGB, systematic Teff over-prediction.

For Regime A, two calibration approaches were tested:
- **Gaussian-process α-smoothing** across (Teff, log g, [M/H]) grid, borrowing strength
  from well-populated neighbours.
- **Empirical-Bayes per-(cell, label) shrinkage** toward per-label globals (τ=50).

## Decision

**Empirical-Bayes shrinkage is the production calibrator** (`shrunken_per_cell_per_label_scale`
in `uncertainty.py`). **GP smoothing is retained as code but rejected as production**
(`gp_smoothed_per_cell_per_label_scale`; largest single function in `src/` by fan-out,
117 edges, but not the production path). `run_calibration.py --apply-gp-smoothing`
toggles GP for methodology comparison only.

Regime B is handled separately by **explicit exclusion envelope** (`RegimeBEnvelope`:
|b|<5° ∧ Teff>4750 K ∧ log g<2.1 flagged population-only).

## Rationale

GP smoothing was the initial approach, framed as a principled Bayesian update with
shrinkage toward well-populated neighbours. Evidence rejected it:

- GP over-smoothed at cool-giant corners. Regime A needs α = 1.3–1.5× inflation; GP
  pulled those cells toward 0.9× because neighbours (warmer giants) have smaller
  calibration corrections.
- Global reliability error: shrinkage 0.080 vs GP 0.136. Halt cells: shrinkage 5 vs
  GP 6. cov95: shrinkage 0.970 vs GP 0.885.
- Physical interpretation: the (Teff, log g, [M/H]) parameter space is not smooth at
  cool-giant corners. TiO/MgH molecular bands introduce opacity discontinuities,
  Ye+2024 training density drops at the low-Teff edge. A smoothness prior is
  physically wrong there.

Shrinkage handles sparse cells by pulling their α estimates toward label-wise globals
(τ=50 means ~50 stars' worth of prior; λ_c = n_c/(n_c+50)). In sparse cells the
shrinkage dominates; in well-populated cells the data dominates. No smoothness
assumption across physical regime boundaries.

## Alternatives rejected

- **GP smoothing** (as above).
- **Single global α per label**: too aggressive; loses per-cell structure entirely.
- **Smoother GP with longer length-scale**: still wrong at physical boundaries.
- **Training-data expansion at cool edge (SNR 70 → 50)**: would help but doesn't
  remove the structural regime difference. Deferred to a hypothetical v2.

## Consequences

- GP-smoothing code stays in `uncertainty.py` because it's the largest single function
  and removing it would create a noticeable gap in the module. Also, `run_calibration.py`
  documents GP as the rejected comparator in methodology output.
- A scanner examining code size or fan-out will mis-identify GP smoothing as the
  primary path. It is not. Do not propose GP as default.
- Cool giants stay in per-star D-Cat-b Tier 1 release (with shrinkage-inflated σ
  documented). Regime B (low-|b| warm RGB) does not, by explicit envelope.
- Methods paper should feature the GP rejection as a negative methodology result 
  "smoothness prior fails at physically-structured parameter boundaries".

## Methodology note

The initial plan was GP smoothing, on the (correct in isolation) argument that it is
the principled Bayesian update for sparse-cell calibration. That turned out to be
wrong: the smoothness assumption is the problem, not the shrinkage. The direct
shrinkage-vs-GP comparison on halt-cell reliability provided the decisive evidence.
Lesson: when picking a prior, check that the prior's structural assumptions match
the problem's physical structure. GP's smoothness works when neighbours are
physically interchangeable; it fails when they are not.
