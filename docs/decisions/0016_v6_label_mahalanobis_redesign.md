# ADR 0016, v6 release_tier label-Mahalanobis redesign

**Status:** accepted
**Date:** 2026-05-03
**Supersedes:** the σ-inflation, `kin_ood_flag`, and `mode_ambiguous_flag` tier-gate provisions of ADR 0015 (v5 release_tier simplification, 2026-04-26). ADR 0015's promise that "the σ-inflation caveat introduced in v4 of `release.py` (HIGH_SIGMA_RESCUE_REPORT.md, 2026-04-25) is preserved and extended" is reversed: the σ-inflation gate (and the per-element caveat carve-outs that survived ADR 0015) are retired. The σ-inflation columns themselves are still emitted as diagnostic-only.
**Related:** `src/arqueogal/xp_abundances/main/release.py` (v6 implementation, lines 560-694), `src/arqueogal/xp_abundances/main/run_pipeline1_inference.py:_fit_label_mahalanobis_bundle` (the new bundle), `src/arqueogal/xp_abundances/main/ood.py:percentile_mahalanobis_ood` (percentile-column emission), `docs/CATALOG_SCHEMA.md` v6 banner block, `reports/reviews/github_readiness/{data_scientist,bayesian_rigor_reviewer,galactic_archaeology_reviewer}.md` (the contemporary reviews that frame the trade-off).

## Context

ADR 0015 (2026-04-26) consolidated the v3-v4 gate stack to a v5 schema: `ood_joint_flag` for Tier 3, plus three Tier-2 demoters: per-element σ-inflation thresholds (Teff 150 K, log g 0.30 dex, [M/H] 0.20 dex, [α/M] 0.05 dex, [Mg/H] 0.20 dex), `mode_ambiguous_flag` confined to α/M, and `kin_ood_flag` confined to aux-assisted elements. That simplification was empirically defended by the per-cell ablation in `release/test_ablations_2026-04-26/REPORT.md`.

Through April 28 - May 2, 2026, three independent failure modes of the v5 gates surfaced under usage:

1. **σ-inflation reads as cherry-picking.** The σ-threshold gate sits on the high-σ tail of the predicted-uncertainty distribution. Removing it inflates the published Tier-1-only RMSE because the demoted stars are the model's own self-flagged uncertain ones. Both the user and the bayesian-rigor reviewer flagged this as conflating "model is uncertain" with "prediction is unreliable" and as an optics problem at minimum, a methodological one at worst.
2. **`kin_ood_flag` demotes science targets.** The disc-kinematics envelope gate (Mahalanobis on (v_R, v_T, v_z), p99) flags halo and accreted-debris stars as Tier 2 by construction. These stars are exactly the science target for downstream users (Starfold population separation, halo metallicity histograms, Gaia-Enceladus chemistry). The gate makes them harder to find, not easier.
3. **`mode_ambiguous_flag` fires on ~46 % of the cohort.** The disc α/M bimodality is genuinely present at fixed (Teff, log g, [M/H]); the model's posterior reflects that real-world ambiguity. Demoting half the catalog as a "caveat" against the data's own structure is not justified.

Concurrently, the user asked: is there a smarter T2 gate that mirrors the symmetry of the T3 input-OOD gate? The natural answer is an *output*-Mahalanobis: fit the empirical 5-D label-space envelope on the APOGEE truth used for training, and flag predictions that lie outside it. The two gates then form a clean pair, input-OOD (T3) and output-OOD (T2), both Mahalanobis on a chi-squared p99 cut, no σ-tail cherry-picking, no demotion of populations the model was never trained to identify (halo) or asked to distinguish (disc bimodality).

## Decision

The v6 release-tier composition (effective 2026-05-03) is:

- **Tier 3** if `ood_joint_flag` (XP-block Mahalanobis input-OOD, p99) OR per-element NaN prediction.
- **Tier 2** if `label_extrapolation_flag` (5-D Mahalanobis on the predicted (Teff, log g, [M/H], [α/M], [Mg/H]) tuple, fit on APOGEE-truth labels at p99, regularization 1e-8) OR any per-element caveat in `_PER_ELEMENT_CAVEAT_FLAGS`. The per-element caveat dict is empty as of v6 but retained as a forward-compatible hook.
- **Tier 1** otherwise.

The gate is implemented in `release.py:assign_per_element_release_tier` (lines 560-694). The label-Mahalanobis bundle is fit at inference time by `run_pipeline1_inference._fit_label_mahalanobis_bundle` (lines 832-866) and applied via `ood.percentile_mahalanobis_ood` to emit both the binary flag and a continuous percentile column.

The σ-inflation, `kin_ood_flag`, and `mode_ambiguous_flag` columns are **diagnostic-only**: still computed and persisted to the parquet for downstream user filtering, but no longer consulted by the tier composer.

Two new continuous columns join the schema:

- `ood_mahalanobis_percentile` (float32, [0, 1]): empirical-CDF percentile of the per-star XP-block Mahalanobis distance against the training distribution. Continuous companion to `ood_joint_flag`.
- `label_mahalanobis_percentile` (float32, [0, 1]): empirical-CDF percentile of the per-star 5-D label-Mahalanobis distance against the APOGEE-truth training distribution. Continuous companion to `label_extrapolation_flag`.

Both percentiles are computed via `np.searchsorted(side="right")` on the sorted training distances; the maximum training distance maps to 1.0, and any inference distance beyond that is also clamped to 1.0. This is acceptable for the binary gate (p99 cut) but the percentile columns must not be interpreted as a continuous severity ranking above 0.99.

## Consequences

### Reversed claims from ADR 0015

The "σ-inflation caveat is preserved and extended" promise of ADR 0015 is reversed. The Tier-1 fractions reported in ADR 0015 will not match v6 outputs; downstream consumers must consult the v6 sidecar JSON and the `*.release_tier.json` provenance for the actual tier composition.

The aux-assisted `kin_ood_flag` demotion path described in ADR 0015 §4 is retired. The `_AUX_ASSISTED_ELEMENTS = ("alpha_m", "mg_h")` tuple in release.py is retained for the `xp_abundance_type__<element>` informational column but no longer drives tier assignment.

The α/M `mode_ambiguous_flag` carve-out described in ADR 0015 §3 is retired. `_PER_ELEMENT_CAVEAT_FLAGS` is now empty.

### Tier fractions on the canonical 2026-05-03 inference

| stream | Tier 1 | Tier 2 | Tier 3 | n |
|---|---|---|---|---|
| Stream 1 (APOGEE × XP, dedup ~293k) | 94.30 % | 0.37 % | 5.33 % | (canonical parquet) |
| Stream 2 (TESS asteroseismic × XP, ~72k) | 96.90 % | 0.02 % | 3.08 % | (canonical parquet) |
| Stream 3 (Andrae+23 RGB × XP, ~614k) | 78.90 % | 1.46 % | 19.64 % | (canonical parquet) |

The Stream 3 T3 fraction (19.64 %) is dominated by `ood_joint_flag` (the XP-block input-OOD gate), not by the new `label_extrapolation_flag`, which fires on only the ~1-2 % of stars whose predicted label vector sits outside the APOGEE-truth p99 envelope. This confirms the v6 redesign is dominantly driven by the input-OOD gate already in place; the new output-OOD gate is a tighter, more conservative top-up.

### Trade-off vs the §3.3 six-test promotion protocol

The data-scientist review (`reports/reviews/github_readiness/data_scientist.md`) correctly notes that the v6 redesign **structurally departs from the pre-registered §3.3 six-test promotion protocol** in `docs/research_brief.md` §3.3. The §3.3 decision tree pre-registers six tests (physical feasibility, holdout RMSE per cell, open-cluster precision floor, information-content audit, conditional MI, cross-catalogue consistency) as the gate for tier promotion. The v6 redesign replaces this decision tree with a binary Mahalanobis gate.

This trade-off is explicit and accepted:

- **Operational gain.** A single, fully reproducible, computationally cheap gate that scales to the full 21-element label space without per-element calibration. The existing 5/6 §3.3 audit coverage (test 6 is the cross-catalogue check, deferred to Phase D) is mechanically supplemented by the Mahalanobis envelope.
- **Coverage cost.** Only Teff, log g, [M/H], [α/M], [Mg/H] have passed the §3.3 six-test promotion. The other 16 elements (Fe/H through Ce/H) inherit Tier 1 by default if `label_extrapolation_flag` does not fire, even though their per-cell RMSE / CMI / cross-catalogue checks are pending. This is documented in `release.py` (`_AUX_ASSISTED_ELEMENTS` docstring + `_ABUNDANCE_ELEMENTS` provenance comment) as "audit pending" and marked in the `xp_abundance_type__<element>` column.
- **Bias risk.** The label-Mahalanobis envelope is fit on APOGEE-truth, which is heavily disc-dominated. Stream-3 stars predicted into label-space regions sparse or absent in APOGEE (cool dwarfs below the Teff cutoff, metal-poor halo, faint giants) will fall into Tier 2 by construction. The CATALOG_SCHEMA banner block (v6 caveat near line 152) makes this explicit. Users targeting accreted-debris populations should treat T2 demotions as conservative-by-design and inspect `label_mahalanobis_percentile` for graded severity.

### Phase 5 follow-ups required

The following items are tracked as v1.1 follow-ups, not blockers for the 2026-05-03 commit:

1. **Test 2 contingency analysis.** Compute the (§3.3 Test 2 pass/fail, `label_extrapolation_flag` pass/fail) 2×2 contingency on the Stream-1 holdout for the 5 §3.3-promoted elements. Confirm whether the gates flag the same stars or orthogonal populations.
2. **Per-regime T2-demotion audit.** For Stream 3, fraction of `label_extrapolation_flag` fires by [α/M] bin, Galactic latitude, and halo-likelihood proxy. Tests whether the gate biases against accreted/halo populations.
3. **§3.3 audit schedule for the 16 un-audited elements** (C, N, O, Na, Al, Si, S, K, Ca, Ti, V, Cr, Mn, Fe, Ni, Ce). Tests 2 (per-cell RMSE), 5 (CMI), 6 (cross-catalogue) for each. Until completion, these elements are released at the same default Tier 1 the v5 schema would have given them, but the methods paper must explicitly mark them "preliminary".
4. **Loss-recipe ablation.** β-NLL-only training run on Stream-1 holdout, side-by-side Y16/Y17 calibration vs the canonical SupCon=1.0 + Barlow=0.5 + β-NLL=1.0 + ARI=0 recipe. Quantifies whether the contrastive components are load-bearing for calibration.

### Empirical validation status

The v6 schema's empirical calibration is verified by:

- **Y16 pull distributions.** Robust pull σ ∈ [0.95, 1.12] across all 5 promoted labels. Within the ±20 % acceptance window for calibrated σ.
- **Y17 reliability diagrams.** ⟨RMSE/σ⟩ ∈ [0.92, 1.12] across all 5 labels. Calibrated, with conservative pessimism in [α/M] high-σ tail.
- **H8 / H9 / H10 tier-visualization figures.** Dual-Mahalanobis decision rule is visually transparent. Tier 2 demotions concentrate near the chemistry-plane boundaries, as expected from output-OOD intuition.

### Validation guard added

`assign_per_element_release_tier` now raises `ValueError` if `label_extrapolation_flag` is missing from the input DataFrame (release.py:626-635). This prevents the silent T2 → T1 promotion that would occur on stale parquets predating the 2026-05-03 redesign or on parquets where upstream inference failed to fit the label-Mahalanobis bundle. The error message points to this ADR.

## Notes for downstream consumers

- The `*.release_tier.json` sidecar `tier_gating_logic` string explicitly describes the v6 logic and lists the diagnostic-only retired columns. Always read the sidecar to confirm the active gates for any given parquet.
- The CATALOG_SCHEMA.md v6 banner block at the top of the document is the authoritative tier-semantics reference. Where individual table cells in older sections still describe σ-inflation, `kin_ood_flag`, or `mode_ambiguous_flag` as active gates, treat that wording as v5 historical and use the v6 banner.
- Backup parquets from the iterative training rounds (`pipeline1_predictions_stream*.parquet.{canonical_v1,v3_backup,v4_backup,v5_p99_backup,v5_pre_labelmh}`) were deleted as part of the 2026-05-03 cleanup; the only authoritative parquet is the unsuffixed one. The provenance sidecar carries the canonical model checkpoint hash, frozen-Hermite basis fingerprint, and inference timestamp.
