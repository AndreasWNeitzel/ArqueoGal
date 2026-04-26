# ADR 0015, v5 release_tier simplification

**Status:** accepted
**Date:** 2026-04-26
**Supersedes:** the gate composition aspects of ADRs 0004 (regime-B exclusion envelope), 0005 (Teff/log g tier decisions). The σ-inflation caveat introduced in v4 of `release.py` (HIGH_SIGMA_RESCUE_REPORT.md, 2026-04-25) is preserved and extended.
**Related:** `release/test_ablations_2026-04-26/REPORT.md` (empirical justification), `src/arqueogal/xp_abundances/main/DESIGN.md` (contract), `src/arqueogal/xp_abundances/main/release.py:_CATALOGUE_SCHEMA_VERSION = 5`.

## Context

Through v3-v4 of `release.py` (Phase A2 and the 2026-04-25 σ-inflation follow-up) the per-star release-tier composition had grown to:

- **Three OOD flags** (Tier 3): `ood_joint_flag` (XP-Mahalanobis), `latent_support_flag` (convex-hull surrogate), `ood_aux_mahalanobis_flag`.
- **Five global caveats** (Tier 2): `regime_b_flag`, `mode_ambiguous_flag`, `ood_disagreement_flag`, `aux_missing_any`, `dist_prior_dominated`.
- **Per-element σ-inflation caveat** (Tier 2) on Teff (150 K), log g (0.30 dex), [M/H] / [Mg/H] (0.20 dex), [α/M] (0.10 dex).
- **Aux-assisted demotion** via `kin_ood_flag` for [α/M] / [Mg/H].

This stack was the cumulative result of multiple incident responses: regime-B was a 2026-04 systematic bias finding, mode-ambiguous addressed Gaussian-NLL μ-collapse on the disc bimodality, ensemble disagreement was a v3 artefact carried forward from a multi-member design, σ-inflation came in after the prior-collapse incident on Stream 3.

The cumulative effect was visually obvious: the Stream 1 / Stream 3 Tier 1 Kiel diagrams showed pronounced "Swiss-cheese" chunking. Teff bands missing entire log g ranges, the disc giant branch fragmented. The user identified this as a deliberate-but-untested architectural choice and asked for empirical justification. None existed; the gates had been added on incident-response logic, never adversarially tested against the trustworthy-catalog (Tier 1 + Tier 2) RMSE.

## Empirical evidence

A segregated ablation harness (`scripts/test_ablations/run_per_cell_ablations.py`) was built. It re-runs the tier-assignment logic post-hoc on the saved 30-epoch ensemble inference of Stream 1 (324,054 rows merging to 292,948 unique source_ids). Metrics computed on the held-out 47,796-star test split (`stratified_split_ids(seed=0, fracs=(.70, .15, .15))`).

The ablation iterated over each gate individually, then over composite configurations. The full report is at `release/test_ablations_2026-04-26/REPORT.md`. Headline findings:

| gate | T1+T2 RMSE inflation if disabled | T1 fraction shift | verdict |
|---|---|---|---|
| `ood_joint_flag` (XP-Mahalanobis) | **+24-38 %** on Teff/log g/[M/H]/[Mg/H], +2 % α/M | +1.6 pp | **Keep** |
| `latent_support_flag` | 0 (never fires) | 0 | **Drop** |
| `ood_aux_mahalanobis_flag` | 0 (subsumed by `aux_missing_any`) | 0 | **Drop** |
| `regime_b_flag` | 0 (113 / 324k stars fire, 0.04 % rate) | 0 | **Drop** |
| `ood_disagreement_flag` | 0 (single-member ensemble can't fire) | 0 | **Drop until ensemble ≥ 2** |
| `mode_ambiguous_flag` (global) | 0 on Teff/log g/[M/H]/[Mg/H]; **+12 %** on α/M | -39 pp T1 | **Confine to α/M only** |
| `aux_missing_any` | 0 (4-6 % T1 shift, no T1+T2 effect) | -2 pp T1 | **Drop** |
| `dist_prior_dominated` | 0 (never fires) | 0 | **Drop** |
| `kin_ood_flag` (Stream-1 layer) | 0 on Stream 1 (designed for Stream 3 disc cuts) | 0 | **Keep at Stream-3 layer only; passive on Stream 1** |
| Per-element σ-thresholds (1×σ_train) | mostly Pareto-optimal | varies | **Tighten α/M to 0.5×σ_train (0.05 dex)** for 23 % α/M T1 RMSE improvement |

**Headline conclusion**: of the 8 per-cell caveat / OOD gates, only one (`ood_joint_flag`) provides a quantifiable T1+T2 RMSE benefit. `mode_ambiguous_flag` is justifiable for [α/M] only. The other six were either zero-firing or pure relabeling.

## Decision

Adopt the simplified gate set as schema v5 of `release.py`:

**Tier 3** (do not release):
- `ood_joint_flag` fires (XP-Mahalanobis OOD).
- OR per-element NaN prediction.

**Tier 2** (statistical / ensemble only):
- Per-element σ-inflation per `_PER_ELEMENT_SIGMA_INFLATED_THRESHOLD` (Teff: 150 K; log g: 0.30 dex; [M/H], [Mg/H]: 0.20 dex; **[α/M]: 0.05 dex**, tightened from 0.10).
- OR `mode_ambiguous_flag` fires (per-element caveat, [α/M] only).
- OR aux-assisted element ([α/M] / [Mg/H]) AND `kin_ood_flag` fires.

**Tier 1**: everything else.

The retired flag *columns* (`latent_support_flag`, `ood_aux_mahalanobis_flag`, `regime_b_flag`, `ood_disagreement_flag`, `aux_missing_any`, `dist_prior_dominated`) are still emitted by upstream annotation for diagnostics and reproducibility. They no longer feed `release_tier`. The sidecar manifest now distinguishes `expected_upstream_columns` (active gates) from `diagnostic_only_columns` (retired but emitted).

`_CATALOGUE_SCHEMA_VERSION` is bumped 4 → 5.

## Implementation

`src/arqueogal/xp_abundances/main/release.py`:
- `_OOD_FLAGS` reduced to `("ood_joint_flag")`.
- `_CAVEAT_FLAGS = ()` (the global-caveat tuple is now empty).
- New `_PER_ELEMENT_CAVEAT_FLAGS = {"alpha_m": ("mode_ambiguous_flag")}` consumed by `assign_per_element_release_tier`.
- `_PER_ELEMENT_SIGMA_INFLATED_THRESHOLD["alpha_m"]` = 0.05 (was 0.10).
- `_CATALOGUE_SCHEMA_VERSION = 5`.
- Module + function docstrings rewritten.
- Sidecar manifest split into `expected_upstream_columns` / `diagnostic_only_columns`; `tier_gating_logic` prose rewritten.

`src/arqueogal/data/release_pipeline.py`:
- Mirror α/M threshold updated to 0.05.

Tests (`tests/xp_abundances/main/test_release.py`, `tests/data/test_release_pipeline.py`):
- `test_single_caveat_demotes_to_tier_2` parametrize set replaced by:
  - `test_diagnostic_only_flags_do_not_change_tier` (asserts the retired flags do NOT demote);
  - `test_mode_ambiguous_demotes_only_alpha_m_to_tier_2` (asserts the per-element carve-out).
- `test_aux_mahalanobis_ood_flag_demotes_all_to_tier_3` and `test_dist_prior_dominated_demotes_all_to_tier_2` inverted to assert *no* demotion.
- Schema version assertion bumped to 5; new sidecar fields asserted.
- All 54 release-tier tests pass.

Live parquets re-annotated under v5 (no inference re-run; tier flags are computed post-hoc on saved predictions):
- `data/processed/pipeline1_predictions_stream2.parquet`
- `data/processed/pipeline1_predictions_stream3.parquet`
- `release/D-Cat-b/hybrid_pipeline_run/predictions_with_features.parquet`
- `release/D-Cat-b/hybrid_pipeline_run_stream2/predictions_with_features.parquet`

The hybrid composition (`attach_hybrid_columns`) was rerun on the two D-Cat-b parquets so that the regressor↔kNN-median choice for [α/M] reflects the new 0.05 dex threshold (Stream 3: ~365k of 614k α/M predictions now use the kNN surface).

Gallery scripts updated where the v4 → v5 transition changes the narrative:
- `plot_18_ood_gates.py`: bold-blue marks active gates, grey marks diagnostic-only.
- `plot_20_hybrid_inference_planes.py`: docstring carries the new T1/T2/T3 percentages.
- `plot_21_release_tier_regime.py`: docstring updated.
- `plot_21b_flag_coloured_chemistry.py`: priority order rewritten with active gates first, diagnostic-only set tagged.
- `plot_26_stream2_inference_summary.py`: Mahalanobis panel title clarifies which flags are active vs diagnostic.

DESIGN.md cross-references in `src/arqueogal/xp_abundances/main/DESIGN.md` and `src/arqueogal/data/DESIGN.md` updated in the same commit (invariant 15).

## Consequences

### Positive

- The Tier 1 Kiel and chemistry planes regain the natural giant-branch shape on Stream 1 (T1 fraction goes 47.6 % → 92.6 % on Teff/log g/[M/H]/[Mg/H], with T1+T2 RMSE preserved).
- [α/M] precision is now tunable through one explicit knob (the σ-threshold) rather than a dispersed cocktail of caveats.
- The release-tier contract is now defensible: every active gate has a measured T1+T2 RMSE justification on the held-out test split.
- The sidecar manifest now makes the active-vs-diagnostic distinction explicit, simplifying downstream-consumer filter design.

### Negative / risks

- The qualitative rationale for `mode_ambiguous_flag` as a global caveat (Gaussian-NLL μ-collapse on the disc bimodality) is dropped for non-α/M elements. The ablation showed no measurable T1+T2 RMSE penalty on this holdout, but the test was Stream-1-only; the broader Stream-3 population could in principle have rare cases this protected against. Followup: monitor Stream-3 hybrid α/M predictions for outliers using the diagnostic columns.
- On Stream 2 / Stream 3 the simplification is dominated by the α/M σ-tighten, which decreases T1 fraction overall (S2: 59.2 % → 52.2 %; S3: 33 % → 27.6 %). This inverts the "simplified ⇒ more T1 stars" intuition for these streams. The compensating signal is sharper [α/M] on the T1 stars that remain.
- Six diagnostic flags are still computed and emitted, contributing nothing to the tier and adding ~6 boolean columns × 614k rows ≈ 30 MB of overhead per Stream-3 release. Acceptable for diagnostic auditing of the v5 decision; could be retired entirely in v6 if no v5-era audit needs them.

### Followup work flagged but deferred

1. **Per-cell σ-shrinkage vs single-global α**: the ablation could not test this post-hoc, it requires a re-inference with the alternative calibration applied to the saved checkpoint. Build a modified `apply_calibration` that takes a single global α per label, run inference, compare σ-coverage vs the production per-cell version. Prior: the τ=50 empirical-Bayes shrinkage already collapses heavily to the global prior, so the per-cell signal may already be washed out. (See HIGH_SIGMA_RESCUE_REPORT.md.)
2. **Mahalanobis at percentile cutoffs other than p99**: test p95, p97, p99.5 to see if the Stream-3 Tier 3 fraction (currently 19.9 %) is appropriately calibrated or over-tight.
3. **Six retired flag columns retired entirely**: in v6 (post-v5-audit window) drop the six diagnostic flag columns from `annotate_parquet` if no audit consumed them.

## References

- `release/test_ablations_2026-04-26/REPORT.md`
- `release/test_ablations_2026-04-26/ablations.json`
- `reports/test_ablations_2026-04-26/*.png` (5 plots: per_gate_effect, rmse_vs_tier1_fraction, per_element_breakdown, kiel_per_ablation, recommended_vs_production)
- `src/arqueogal/xp_abundances/main/release.py` (canonical implementation)
- `src/arqueogal/xp_abundances/main/DESIGN.md` (contract)
- `docs/research_brief.md` §3.3 (six-test promotion protocol, orthogonal layer; this ADR is the per-star layer)
- ADR 0004 (regime-B exclusion envelope; superseded for tier purposes, the envelope is preserved as a methods-paper systematic but no longer demotes)
- ADR 0005 (Teff/log g tier decisions; refined here)
