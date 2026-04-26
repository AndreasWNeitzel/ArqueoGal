# Changelog Audit — ArqueoGal

**Audit date:** 2026-04-26  
**Repo state:** 6b96c06 (2026-04-24) with unstaged changes to 5 DESIGN.md files, release code, and infrastructure  
**Release tag:** `pipeline1-v1-2026-04-19` with 2-paragraph release notes

## Findings

### 1. No top-level CHANGELOG.md exists

The repository has no Keep a Changelog-formatted file at the root. Version history is documented across five per-module DESIGN.md files (`archaeogal/{data,utils,xp_abundances,xp_abundances/main,xp_abundances/experimental}/DESIGN.md`), each with a "Change log" section at the foot (where present). This distributes release narrative across modules and defeats single-source-of-truth retrieval.

### 2. Per-DESIGN.md change logs are comprehensive but unstaged

The v5 schema changes (2026-04-26: gate simplification, α/M σ-threshold tighten, hybrid composer columns) are **fully documented in unstaged diffs** to `src/arqueogal/xp_abundances/main/DESIGN.md` and `src/arqueogal/data/DESIGN.md`. The main DESIGN.md already lists the v5 changelog entry (new, 15 lines); data/DESIGN.md acquired new sections for five new modules (`frozen_stats.py`, `release_artefacts.py`, `release_pipeline.py`, `master_schema.py`) and a detailed Stream 1 deduplication contract. These changes have not been committed, meaning v5 exists only in working-directory state.

### 3. Release tag notes are terse and buried

The `pipeline1-v1-2026-04-19` tag carries annotation text (accessed via `git tag -l -n 10`) that is 21 lines of prose covering architecture, calibration, OOD rejection, and known stubs. The annotation is **not** a GitHub Release note (no GitHub UI, not pushed). A standard GitHub release would render this as machine-readable release assets, link to commits, and show in the Releases tab. Currently it exists only in raw `git tag` form.

### 4. Drift: v1 release tag vs current codebase state

The v1 tag (2026-04-19) documents a 5-label ensemble with three OOD flags and six caveat flags. The unstaged changes introduce v4 (2026-04-25, kNN rescue + σ-inflation caveat) and v5 (2026-04-26, gate simplification, flag retirement) changes. None of this is represented in a release note for v4 or v5. The tag is stale relative to the current working state.

## GitHub-ready release scaffold

A minimal top-level CHANGELOG.md following [Keep a Changelog](https://keepachangelog.com/) format would be:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **v5 schema (2026-04-26):** Simplified release-tier gates, per-element σ-inflation flags
- **v5 hybrid columns:** Latent-kNN rescue surface (regressor + kNN-median composition)
- Frozen z-score statistics container (`frozen_stats.py`) for Stream-3 inference consistency
- Release artefacts builder (`release_artefacts.py`): HRD, kinematic, Tier-1-only subsets

### Changed
- Release-tier logic: reduced 6 caveat flags to 4 (ood_joint_flag, σ-inflation per-element, mode_ambiguous on α/M only, kin_ood on aux-assisted labels)
- [α/M] σ-threshold tightened from 0.10 to 0.05 dex (0.5 × σ_train)
- Stream-1 Tier-1 fractions: Teff 47.6% → 92.6%, log g 47.5% → 92.3%, [M/H] 46.4% → 89.5%
- α/M Tier-1 precision improved 23%, fraction reduced ~14 pp (threshold trade-off)

### Removed
- Six OOD/caveat flags retired from active gating (moved to diagnostic-only): latent_support_flag, ood_aux_mahalanobis_flag, regime_b_flag, ood_disagreement_flag, aux_missing_any, dist_prior_dominated

## [1.0.0] - 2026-04-19

### Pipeline 1 v1 Release

**Architecture:**
- 5-member ensemble, shared contrastive encoder
- 5-label block-Cholesky head: Teff, log g, [M/H], [α/M], [Mg/H]
- Ye+2024 NN flux correction on XP coefficients
- Mészáros+2025 [X/M]/Teff corrections on APOGEE DR19 labels

**Calibration:**
- Empirical-Bayes per-cell per-label α shrinkage (τ=50)
- Global reliability error 7.95%, joint coverage 95% within 5.8 pp
- Regime B exclusion envelope: low-latitude warm giants (|b|<5°, Teff>4750 K, log g<2.1) flagged population-only

**OOD Rejection:**
- Mahalanobis distance on 108-D XP shape features
- Ensemble-disagreement epistemic flag (epistemic/total σ > 0.5)

**Known limitations (v1):**
- Cool-giant σ inflation up to 1.5x (structural, documented)
- 7/62 (Teff, log g, [M/H]) cells exceed 15% error (cool corner, documented)
- Audit tests 3 (SHAP) and 6 (decorrelated subsample) incomplete

**Artifacts:**
- Checkpoint: `models/main/xp_abundances/20260419_nogit_a0e10aa_ensemble_5label/`
- Report: `reports/pipeline1/run_a/final_release_report_5label.md`

[Unreleased]: https://github.com/user/arqueogal/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/user/arqueogal/releases/tag/v1.0.0
```

**Schema version mapping:** v1 (2026-04-19, initial 5-label release) → v4 (2026-04-25, σ-inflation caveats) → v5 (2026-04-26, gate simplification).

## v5 release entry (minimal for the CHANGELOG or GitHub Release)

```markdown
## [Unreleased] or [2.0.0-alpha.1] - 2026-04-26

### Schema v5: Release-Tier Simplification

After per-cell-gate ablation testing, the OOD/caveat flag stack was streamlined:

#### Added
- `_PIPELINE1_KNN_RESCUE_COLS` (27 columns): latent-kNN median/quantile/distance per element
- Hybrid-column composer: `<elem>_hybrid_pred`, `_hybrid_sigma`, `_hybrid_source`, `_hybrid_tier`
- ADR-0015: Decision provenance for v4→v5 gate simplification

#### Changed
- [α/M] σ-threshold: 0.10 dex → 0.05 dex
- Release-tier logic: 6 gates → 4 active gates (ood_joint, σ-inflation, mode_ambiguous, kin_ood)
- Tier-1 promotion (Stream 1 holdout, 47,796 stars):
  - Teff: 47.6% → 92.6%
  - log g: 47.5% → 92.3%
  - [M/H]: 46.4% → 89.5%
  - [Mg/H]: 47.0% → 91.0%
  - [α/M] precision +23%, fraction −14 pp

#### Removed
- Six flags retired from active gating (emitted for diagnostics only):
  `latent_support_flag`, `ood_aux_mahalanobis_flag`, `regime_b_flag`,
  `ood_disagreement_flag`, `aux_missing_any`, `dist_prior_dominated`

#### Caveats
- Hybrid column assignment requires both regressor σ estimate and kNN-median surface;
  rows with missing kNN data fall back to `regressor_caveat` source
- α/M Tier-1 fraction drops due to threshold tightening (trade-off for RMSE gain)
```

## Summary

The repository lacks a consolidated changelog following Keep a Changelog. The v1 release (2026-04-19) was tagged with terse annotation but no GitHub Release artefact. Interim schema versions v4 (2026-04-25) and v5 (2026-04-26) are documented in five unstaged DESIGN.md diffs, creating a documentation-git divergence. A top-level CHANGELOG.md and GitHub Release notes are needed before public D-Cat-b release.
