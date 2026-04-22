# Archive

Gallery entries moved here when the corresponding code path is no longer
part of this repository's scope.

## Pipeline 2 → Starfold

As of 2026-04-22 the stellar-population classification pipeline (originally
Pipeline 2 of this repo) has been spun out into a separate repository,
**Starfold** (forthcoming). Starfold consumes the Pipeline 1 prediction
parquets produced by ArqueoGal (with the `release_tier` quality-flag column)
as an input and produces cluster assignments via parametric UMAP + HDBSCAN.

The two entries archived here document the abandoned in-repo scaffolding:

- `15_pipeline2_features/` — feature-matrix assembly for the 10–11D
  chrono-chemo-kinematic vector.
- `16_pipeline2_classification/` — HDBSCAN clustering, DBCV grid search,
  MC-ensemble soft memberships.

Neither was ever run end-to-end inside this repo. The designs are preserved
here for historical context; the active implementation lives in Starfold.
