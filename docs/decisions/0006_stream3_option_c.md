# ADR-0006: Stream 3 Option C, dual samples (uniform + volume-limited)

**Date**: 2026-04-19 · **Status**: Accepted, Phase 3 pending execution

> **Note (2026-04-22):** references to "Pipeline 2" below describe the
> original in-repo population-classification consumer of the volume-limited
> sample. Population classification has since moved to the **Starfold**
> repository; the volume-limited parquet is now produced for Starfold (or any
> other downstream density-based analysis) to consume. The dual-sample design
> and its rationale are unchanged.

## Context

Stream 3 inference feeds two downstream purposes: Pipeline 1 release-catalogue audit
(§9.2 test 6 + per-star release) and Pipeline 2 HDBSCAN clustering. These have
conflicting sample-design requirements:

- **Pipeline 1 audit needs uniform stratification** across parameter space to get
  equal statistical power in every (Teff, log g, [M/H], G) cell.
- **Pipeline 2 clustering needs natural density** so HDBSCAN finds real structure
  rather than stratification artifacts.

A single sample forces a compromise that hurts one pipeline.

## Decision

**Dual samples**:
- **~400 k uniform-stratified** (revised from 800 k under 10 GB budget) →
  `pipeline1_inference_uniform.parquet`. Audit + per-star release input.
- **~250 k volume-limited** at d ≤ 2.5 kpc (revised from 500 k) →
  `pipeline2_features_volume.parquet`. Pipeline 2 HDBSCAN input. Random 250 k from the
  ~4 M candidate pool, no oversampling.

Union of source_ids for XP fetch (avoids duplicate compute). Separate provenance
sidecars per matrix documenting sampling method.

## Rationale

- Preserves each pipeline's ideal sample design without compromise.
- Storage cost is modest: ~1.6 GB combined vs ~1.0 GB for a single sample.
- Option B (stratified oversampling from the 16× over-provisioned volume-limited
  pool) was considered and rejected for v1 because it complicates Pipeline 2's
  density-based analysis. Pure random sampling from the 4M-candidate pool is cleaner.
- The over-provisioning (4 M candidates at d ≤ 2.5 kpc for a 250 k target) means
  we can afford random sampling without depleting rare-population coverage.

## Alternatives rejected

- **Single sample (uniform)**: Pipeline 2 HDBSCAN sees stratified rather than natural
  density. Clusters HDBSCAN finds may be binning artifacts.
- **Single sample (volume-limited)**: Pipeline 1 audit has uneven statistical power;
  rare cells in parameter space are undersampled.
- **Two samples with full overlap**: inefficient, inflates storage without purpose.
- **Option B (oversample rare strata in volume-limited)**: better for population
  discovery but complicates natural-density analysis. Deferred as potential v2 work.

## Consequences

- XP fetch is ~1.2–1.3 M unique source_ids, not 650 k (~2× the storage for raw XP
  intermediates). Budget check: 5.1 + 1.6 = 6.7 GB vs 10 GB ceiling. OK.
- Pipeline 1 inference runs twice (once per matrix). Fast; not a bottleneck.
- Pipeline 2's HDBSCAN receives natural-density structure; population discovery
  depends on natural abundance of stars in each population. Rare populations may be
  marginal in a 250 k random sample.

## Methodology note

Sample size reductions (800 k → 400 k uniform; 500 k → 250 k volume-limited) were
budget-driven rather than science-driven but do not change the statistical argument
(both sample sizes are adequate for their respective purposes).
