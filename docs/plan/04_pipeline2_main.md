# Phase 04. Population classification moved to Starfold

**Status (2026-04-22):** the stellar-population classification pipeline
originally scoped as "Pipeline 2" in this repository has been spun out into
a separate open-source tool, **Starfold**, now published at
<https://github.com/AndreasWNeitzel/Starfold>. This repository's scope now
ends at Pipeline 1 predictions.

## What changed

The in-repo `src/arqueogal/population_classifier/` tree, its tests, its
driver scripts, and its configs have been removed. The two gallery entries
(`15_pipeline2_features`, `16_pipeline2_classification`) have been moved to
`reports/gallery/archive/`. Historical development reports are under
`reports/archive/pipeline2/`.

## Why

- ArqueoGal's remit in the FCT 2024.15303.PEX project is the XP →
  abundances pipeline. Population classification is a reusable downstream
  tool that deserves its own release cycle and user base.
- Starfold's release schedule and versioning should not be coupled to this
  repository's Pipeline 1 cadence.
- Keeping them separate lets each repository have a focused README,
  installation story, and citation.

## Integration contract

Starfold consumes the Pipeline 1 prediction parquets produced by this
repository. The relevant files are:

- `data/processed/pipeline1_predictions_stream3_joint.parquet` (union) and
  the arm-specific `*_volume.parquet` / `*_uniform.parquet` variants.
- The `release_tier` column (added by the Phase B work below) labels each
  row Tier 1 / Tier 2 / Tier 3 per the tier-promotion protocol in
  `docs/research_brief.md` §3.3. Starfold should consume Tier 1 (and
  optionally Tier 2 with appropriate caveats) rows only.
- The OOD-gate columns (`ood_joint_flag`, `latent_support_flag`), the
  per-label σ columns, and the regime-B flag are carried through for any
  Starfold-side filtering decisions.

The kinematics module (`src/arqueogal/data/kinematics.py`) may be exposed as
an importable utility if Starfold wants to reuse galpy McMillan17 Staeckel
action computation on the same input catalogues, or it may be duplicated
inside Starfold. This choice is deferred to Starfold's own architecture
review.

## FIRE-2 method validation

The FIRE-2 hare-and-hounds validation (Subtask 5.1) follows Starfold, not
this repo. Production Pipeline 1 consumes only real observational data;
validation is conducted on real hold-out splits and external catalogue
cross-matches. Historical: synthetic-data method-validation in early prototypes
has been discontinued in favour of real-data validation.

## What stays in this repo

- Pipeline 1 (Stream-1 training, Stream-3 inference, all supporting data
  acquisition and preprocessing).
- The release-tier quality-flag machinery that sits on top of Pipeline 1
  predictions (Phase B of the Phase-A/B/C plan).
- All ArqueoGal-internal documentation and ADRs referencing the overall
  project scope.
