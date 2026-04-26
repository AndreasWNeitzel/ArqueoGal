# Per-cell tier-promotion gate ablation report

**Date**: 2026-04-26 · **Status**: Test-only, segregated. Main release untouched.

## Setup

- **Predictions**: `release/test_ablations_2026-04-26/predictions_stream1.parquet` —
  inference output of the production 30-epoch ensemble
  (`20260425_6b96c06_cd1cbb9_ensemble_5label`) on the full Stream 1 features
  parquet (324,054 rows). Inner-joined with truth labels and deduped on
  `source_id` → 292,948 unique stars.
- **Holdout**: 47,796 stars from the test split of the
  `stratified_split_ids(seed=0, fracs=(.70, .15, .15))` partitioning used
  during training. The model never saw these stars.
- **Truth**: `teff_apogee`, `logg_apogee`, `mh_apogee`, `alpha_m_apogee`,
  `mg_h_apogee` from the post-Mészáros+2025 [X/M]-corrected APOGEE DR19
  labels.
- **Per-element training-σ marginal** (used for global-σ-threshold ablations):
  Teff = 269.2 K, log g = 0.516 dex, [M/H] = 0.344 dex, [α/M] = 0.0953 dex,
  [Mg/H] = 0.289 dex.

Each ablation toggles one gate from the production stack and re-runs the
tier-assignment logic post-hoc (no retraining, no re-inference). We report
**Tier 1** (per-star science slice) and **Tier 1+2** (full trustworthy catalog)
metrics per element.

## Production-baseline reference

| element | Tier 1 fraction | Tier 1 RMSE | Tier 1+2 RMSE |
|---|---|---|---|
| Teff | 47.6 % | 51.1 K | 60.3 K |
| log g | 47.5 % | 0.125 dex | 0.148 dex |
| [M/H] | 46.4 % | 0.084 dex | 0.106 dex |
| [α/M] | 47.9 % | 0.043 dex | 0.052 dex |
| [Mg/H] | 47.0 % | 0.072 dex | 0.097 dex |

## Per-gate ablation effect on Tier 1 RMSE

Bold = ≥ 5 % RMSE change vs production baseline. Italic = ≥ 1 percentage-point
Tier-1-fraction change.

| ablation | Teff Δ | log g Δ | [M/H] Δ | [α/M] Δ | [Mg/H] Δ | Tier 1 fraction shift |
|---|---|---|---|---|---|---|
| **`no_mahalanobis`** | +14.0 K **(+27 %)** | +0.035 **(+28 %)** | +0.016 **(+19 %)** | +0.001 (+2 %) | +0.016 **(+22 %)** | *47.6 → 49.2 %* |
| `no_aux_missing` | +2.3 K (+5 %) | +0.005 (+4 %) | +0.003 (+4 %) | +0.002 (+5 %) | +0.004 (+6 %) | *47.6 → 49.8 %* |
| `no_mode_ambiguous` | -1.3 K (-3 %) | +0.002 (+1 %) | -0.001 (-1 %) | +0.005 **(+12 %)** | +0.004 (+5 %) | *47.6 → 86.7 %* |
| `no_regime_b` | +0.013 K (0 %) | 0 | 0 | 0 | 0 | unchanged |
| `no_kin_ood` | 0 | 0 | 0 | 0 | 0 | unchanged |
| `no_latent_support` | 0 | 0 | 0 | 0 | 0 | unchanged |
| `no_aux_mahalanobis` | 0 | 0 | 0 | 0 | 0 | unchanged |
| `no_dist_prior` | 0 | 0 | 0 | 0 | 0 | unchanged |
| `no_disagreement` | 0 | 0 | 0 | 0 | 0 | unchanged |
| `sigma_global_0.5×σ_train` | -1.2 K (-2 %) | -0.005 (-4 %) | -0.002 (-2 %) | **-0.010 (-23 %)** | -0.004 (-6 %) | varied |
| `sigma_global_1.0×σ_train` | +3.3 K (+6 %) | +0.006 (+5 %) | +0.009 **(+11 %)** | 0 | +0.006 (+8 %) | varied |
| `sigma_global_2.0×σ_train` | +3.4 K (+7 %) | +0.007 (+6 %) | +0.013 **(+15 %)** | +0.001 (+2 %) | +0.013 **(+18 %)** | varied |
| `all_caveats_off` | +2.4 K (+5 %) | +0.012 **(+10 %)** | +0.004 (+5 %) | +0.007 **(+16 %)** | +0.009 **(+13 %)** | *47.6 → 92.6 %* |

## Per-gate ablation effect on Tier 1+2 RMSE (full trustworthy catalog)

| ablation | Teff RMSE | log g RMSE | [M/H] RMSE | [α/M] RMSE | [Mg/H] RMSE |
|---|---|---|---|---|---|
| baseline_prod | 60.3 K | 0.148 | 0.106 | 0.052 | 0.097 |
| `no_mahalanobis` | **83.3 K (+38 %)** | **0.183 (+24 %)** | **0.134 (+26 %)** | 0.053 (+2 %) | **0.122 (+26 %)** |
| `no_aux_missing` | 60.3 K | 0.148 | 0.106 | 0.052 | 0.097 |
| `no_mode_ambiguous` | 60.3 K | 0.148 | 0.106 | 0.052 | 0.097 |
| All other caveat ablations | identical to baseline | | | | |

**Key observation**: of all caveat ablations, **only Mahalanobis OOD changes the
Tier 1+2 (trustworthy-catalog) RMSE**. The other caveats — including
mode-ambiguous, regime-B, kin_ood, latent-support, aux-Mahalanobis,
dist-prior, disagreement, aux-missing — *only redistribute stars between
Tier 1 and Tier 2 without removing them from the trustworthy catalog*. The
stars they demote do not have measurably higher prediction error than the
ones they keep in Tier 1.

## Verdict per gate

| gate | empirical justification | recommendation |
|---|---|---|
| `ood_joint_flag` (XP-Mahalanobis) | **Strong**: 24-38 % T1+2 RMSE inflation if disabled. Catches genuinely-bad predictions on Teff / log g / [M/H] / [Mg/H]. | **Keep.** Marginal benefit only for α/M (+2 %) — could be exempted there. |
| `aux_missing_any` | **Modest**: 4-6 % T1 RMSE improvement, no T1+2 effect. | **Keep but document as soft caveat** — it's flagging stars whose aux uncertainty is higher, even if the regressor itself is fine. |
| `mode_ambiguous_flag` | **Pure relabeling**: shifts 39 percentage-points of stars from T1→T2 with **no T1+2 RMSE effect** (Tier 2 stars are not worse than Tier 1 stars). Modest +12 % α/M T1 RMSE inflation when off, but no other element changes. The bimodal-cell μ-collapse hypothesis doesn't show up in measured RMSE on the test holdout. | **Drop or relax.** The hypothesis it operationalises (Gaussian-NLL μ collapses to the valley between bimodal modes) doesn't manifest at measurable scale. The 39-pp T1 → T2 demotion has no payoff on this metric. |
| `regime_b_flag` | **Zero**: 113 / 324k stars fire. No measurable RMSE effect. | **Drop.** The systematic it documents (Teff over-prediction at warm-RGB low-\|b\|) may be real, but at 0.04 % firing rate it is not contributing as a tier gate. |
| `kin_ood_flag` | **Zero on Stream 1**: kin_ood was designed for Stream 3 disc-cut population. On the Stream-1 holdout it never fires. | **Keep at the Stream-3 layer**, but it should not be in the Stream-1 tier logic at all. |
| `latent_support_flag` | **Zero**: never fires on the holdout. | **Drop or audit when it ever fires** (currently dead code as a gate). |
| `ood_aux_mahalanobis_flag` | **Zero**: subsumed by `aux_missing_any`. | **Drop**, redundant. |
| `dist_prior_dominated` | **Zero**: never fires on the holdout. | **Audit** — either dead or a v3 caveat that was never exercised on Stream 1. |
| `ood_disagreement_flag` | **Zero**: requires ensemble disagreement, single-member ensemble means it can't fire. | **Drop until ensemble has ≥ 2 members.** |
| Per-element σ-thresholds | **Production values are roughly Pareto-optimal.** Translated to σ_train units: Teff 0.56, log g 0.58, [M/H] 0.58, [α/M] 1.05, [Mg/H] 0.69. The 0.5×σ_train alternative is slightly tighter for α/M and gives a 23 % α/M T1-RMSE improvement at the cost of T1 fraction (47.9 → 33.8 %). The 1× and 2× alternatives are uniformly worse. | **Consider tightening α/M to 0.5×σ_train (~0.05 dex)** for sharper [α/M] T1, accepting the T1-fraction drop. Other elements OK as-is. |
| Per-cell σ-shrinkage (`shrunken_per_cell_per_label_scale`) | **Not testable post-hoc**: this gate modifies σ at inference time. Comparing per-cell shrinkage vs single-global-α requires a separate inference run with the alternative calibration applied. | **Defer** — flagged as separate test (see "Future work" below). |

## Headline conclusion

**6 of the 8 per-cell caveat / OOD gates are doing nothing measurable on the
Stream 1 test holdout. Only Mahalanobis OOD provides a quantifiable improvement
to the trustworthy catalog RMSE.**

The visible "chunking" you observed in the Tier 1 Kiel diagram is therefore
mostly the work of `mode_ambiguous_flag` (39-pp demotion with no measurable
quality difference) and `ood_joint_flag` (bona fide OOD detection but
patchy in Teff because the Mahalanobis envelope is more discriminating in
some Teff bands than others, given Stream 1's non-uniform Teff distribution).

A radically simpler tier system — Mahalanobis OOD → Tier 3, σ-inflation
per element → Tier 2, otherwise Tier 1 — would deliver:
- The same Tier 1+2 trustworthy-catalog RMSE.
- A much larger Tier 1 fraction (~85-92 % vs current 47 %).
- A visually-smoother Tier 1 Kiel diagram (without the mode-ambiguous Swiss cheese).

The cost would be: stars currently demoted by mode-ambiguous would land in
Tier 1, with no documented "valley collapse" risk surfacing in the measured
RMSE, but the *qualitative* concern (Gaussian-NLL μ on bimodal targets is
ill-defined) would no longer be operationally guarded against. Whether that
qualitative concern is worth the 39-pp Tier 1 fraction loss is a science
call, not an engineering one.

## Future work (separate tests)

1. **Per-cell σ-shrinkage vs single-global-α**: needs a re-inference with the
   alternative calibration applied to the saved member checkpoint. Build a
   modified `apply_calibration` that takes a single global α per label, run
   inference, compare σ-coverage at 1σ/2σ vs the production per-cell version.
2. **Mahalanobis at percentile cutoffs other than p99**: test p95, p97, p99,
   p99.5 to see if the Stream-3 Tier 3 fraction (currently 19.9 %) is
   appropriately calibrated or over-tight.
3. **Mode-ambiguous-on-α/M-only**: the only element where dropping
   mode_ambiguous costs us measurable RMSE (+12 % T1) is α/M. Demoting
   only α/M (and keeping Teff / log g / [M/H] / [Mg/H] in Tier 1 regardless of
   mode_ambiguous) recovers ~38 percentage points of Tier 1 fraction
   without losing meaningful α/M precision.
