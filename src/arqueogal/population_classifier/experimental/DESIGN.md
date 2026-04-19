# `arqueogal.population_classifier.experimental` — Design

## Status

**Segregated exploration arm**. Does NOT contribute to D5.1 / D-Cat-d unless promoted to
main via the bar below. Does NOT import from `main/` and is not imported by it. Shared code
lives in `arqueogal.utils`.

## Planned subdirectories

See research_brief §11 for full rationale.

```
arqueogal.population_classifier.experimental
├── mahalanobis_umap/    — per-star feature covariance in the UMAP distance metric. Natural
│                           uncertainty propagation without MC ensembling. Expensive.
├── aligned_umap/        — McInnes 2020 Aligned UMAP. Shared embedding across multiple
│                           FIRE-2 galaxies or survey partitions (north/south).
├── deep_clustering/     — DEC (Xie+2016), IDEC (Guo+2017), VaDE (Jiang+2017). End-to-end
│                           embedding + clustering. Less transparent; harder to audit.
├── tda/                 — Topological Data Analysis: ToMATo (Chazal+2013), persistent
│                           homology via giotto-tda, SigMA (Ratzenböck+2023). Persistence
│                           complementary to HDBSCAN stability.
├── diffusion_maps/      — Coifman & Lafon 2006. Spectral alternative to UMAP.
├── dp_gmm/              — Dirichlet-process GMM (Rasmussen 2000). Adaptive component count.
└── hierarchical/        — Hierarchical clustering on the UMAP embedding, separate from
                            HDBSCAN's internal hierarchy. Nested structure (disc →
                            α-rich/α-poor → finer chemo-dynamical).
```

## Promotion rule (research_brief §11)

An experimental method is promoted to `main/` only when all hold:

1. Matches or exceeds `main/` DBCV on the real catalogue hold-out.
2. Matches or exceeds `main/` bootstrap-ARI stability per cluster (N=500).
3. Matches or exceeds `main/` on FIRE-2 hare-and-hounds informedness (ARI, AMI, Youden J,
   MCC).
4. Passes the held-out-feature-consistency test.
5. Provides soft memberships (or an equivalent continuous membership product) at calibrated
   reliability.

Promotion is a PR with the comparison table, not a refactor commit.

## Hard rules

- No cross-imports with `main/`. Shared helpers → `arqueogal.utils`.
- Experimental configs → `configs/experimental/`. Notebooks →
  `notebooks/population_classifier/experimental/`. Tests →
  `tests/population_classifier/experimental/`. Models → `models/experimental/`.
- Each non-trivial subdir owns its own `README.md` or sub-`DESIGN.md`.
- Pre-register hypothesis, success criteria, and thresholds before running an experimental
  arm.
- **The collaborator HPC sweep** (Optuna / persistence on FIRE-2): not automatically
  promoted. Integrate outputs only via an explicit `hyperparameter_prior_<sweep-id>.yaml`
  under `configs/experimental/` with documented provenance, and only if the sweep has been
  augmented with DBCV + ground-truth metrics (research_brief §10.7).
