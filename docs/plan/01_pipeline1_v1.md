# Phase 01 — Pipeline 1 v1 production model

**Status: SHIPPED 2026-04-19. Tagged `pipeline1-v1-2026-04-19`. Do not re-litigate.**

## What shipped

- 5-member ensemble, 5-label block-Cholesky head (Teff, log g, [M/H], [α/M], [Mg/H]).
- Shared contrastive-pretrained encoder
  (`models/main/xp_abundances/20260419_nogit_1ca1ddf/...contrastive_seed0_best.pt`),
  varied head seeds only.
- Empirical-Bayes shrunken per-(cell, label) α calibration (τ = 50).
- Mahalanobis (108-D XP) + ensemble-disagreement OOD flags, serializable bundles.
- Regime B Galactic-plane exclusion envelope (|b|<5° ∧ Teff>4750 K ∧ log g<2.1),
  released population-only.
- 213/213 main test suite green.

## Artefacts of record

- Release report: `reports/pipeline1/run_a/final_release_report_5label.md`
- Ensemble checkpoints: `models/main/xp_abundances/20260419_nogit_a0e10aa_ensemble_5label/`
- Technical survey: `reports/codebase_survey_2026-04-19.md`
- Git tag `pipeline1-v1-2026-04-19` — annotated with architecture, calibration, OOD,
  known stubs, known caveats, next phases.

## Known caveats (shipped-as-is)

1. **Non-uniform σ by parameter region.** Cool-giant σ inflation up to 1.5× relative
   to warmer giants at the cool edge of training coverage. Honest statement of model
   uncertainty, not a bug. Documented per-cell in calibration report.
2. **Between-cell μ drift in regime A cells** (sparse cool-giant corners). Structural,
   not statistical. Shrinkage handles it; GP smoothing fails on it (see ADR-0003).
3. **Regime B population-only.** Low-|b| warm-upper-RGB stars excluded from per-star
   Tier 1 release by envelope flag. ~30 val stars, 0.07%.
4. **Three audit/tier-promotion subtests are stubs** (ADR-0009):
   - `audit.py` test 3 (SHAP) — external lib not in pinset.
   - `audit.py` test 6 (decorrelated subsample) — incomplete scaffolding.
   - `tier_promotion.py` test 6 (cross-catalogue consistency) — stub; unblocks once
     Stream 3 inference runs.

## Acceptance criteria (all met)

- Global reliability err ≤ 10 % — met at 7.95 %.
- Joint cov95 within 5 pp of nominal — met at −5.8 pp.
- All 5 labels' per-label marginals within 1–2 pp at 68/95/99 % — met.
- Integration tests pass end-to-end — 213/213 green.

## What NOT to do with Pipeline 1 v1

- Don't re-open the 21-label vs 5-label decision (ADR-0001). 21-label checkpoint retained
  as a methods-paper comparator; do not promote to main.
- Don't propose GP smoothing as the calibration default (ADR-0003). It's retained code
  but rejected methodology.
- Don't claim full §3.3/§9.2 compliance — three subtests are stubs (ADR-0009).
- Don't silently refit Hermite z-score stats on new data. v1's frozen stats
  (basis fingerprint `0d34b565...`) are the contract.
