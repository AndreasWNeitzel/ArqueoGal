# IR-photometry dependency diagnostic — Pipeline 1 v1 5-label ensemble

_Ensemble: `20260419_nogit_a0e10aa_ensemble_5label` · Val split seed 0 · N_val = 41851_

Three-condition robustness probe of the IR photometry block (`j_mag, h_mag, k_mag, w1_mag, w2_mag`) at inference time. Pre-flight for Thread 3 Stream-3 inference — decides whether a 2MASS+AllWISE fetch is required before we can release predictions for the ~1.3 M Stream-3 application sample.

## Per-label RMSE

| label | RMSE baseline | RMSE zero | RMSE NaN | Δ% zero | Δ% NaN | verdict |
|-------|---------------|-----------|----------|---------|--------|---------|
| teff | 67.0956 | 170.4270 | NaN | +154.01 | n/a | `divergent` |
| logg | 0.1571 | 0.4879 | NaN | +210.56 | n/a | `divergent` |
| mh | 0.1154 | 0.2664 | NaN | +130.87 | n/a | `divergent` |
| alpha_m | 0.0547 | 0.0702 | NaN | +28.32 | n/a | `divergent` |
| mg_h | 0.1041 | 0.2563 | NaN | +146.28 | n/a | `divergent` |

## Baseline sanity check

Per-label drift vs `reports/pipeline1/audit/SUMMARY.md` (must be < 5%):

- **teff**: observed 67.0956 vs expected 67.0956 (drift +0.00%)
- **logg**: observed 0.1571 vs expected 0.1571 (drift -0.01%)
- **mh**: observed 0.1154 vs expected 0.1154 (drift -0.01%)
- **alpha_m**: observed 0.0547 vs expected 0.0547 (drift +0.06%)
- **mg_h**: observed 0.1041 vs expected 0.1041 (drift -0.04%)

## Adapter NaN behaviour

Source-level inspection of `src/arqueogal/xp_abundances/main/adapter.py` (`XpFeatureAdapter.forward`): the adapter is a pass-through identity apart from an optional scatter-zero over the (`bp_c0_z, rp_c0_z`) c0 scalar positions. There is no NaN sentinel handling and no imputation layer between the raw feature matrix and the encoder. Training imputes NaN -> 0 before the Dataset is built (`build_dataloaders` in `src/arqueogal/xp_abundances/main/training.py`); the inference driver `scripts/run_pipeline1_inference.py` does not replicate that step — it trusts the Stream-3 emit to deliver finite features.

Probe result (seed-0 member, first val row, NaN injected at IR positions):

```
first-member forward did NOT raise on NaN-IR input; output NaN counts per tensor: {'out_0': 5, 'out_1': 15, 'out_2': 32, 'out_3': 32}. Interpretation: adapter is a pass-through (no imputation); the XpFeatureAdapter only optionally zeroes c0 scalars. NaN in IR positions propagates through the linear trunk and contaminates every downstream activation -> μ and L_chol are NaN -> predictions are NaN.
```

## Interpretation

**The prior assumption that IR is 'nearly decorative given BJ distances do the aux work' is _not_ confirmed.** The zero-IR condition — where we set `j, h, k, w1, w2` uniformly to 0.0 for every val star, replicating the training-time NaN→0 imputation but applied to the entire val split rather than the 0.1%-0.2% of training rows that natively lacked IR — degrades RMSE by +28% to +211%. Four of five labels cross the 15% load-bearing gate. [α/M] is at +28%, still above the 15% gate and in no sense decorative. Load-bearing across the board.

**The NaN-IR condition is a separate, more severe finding.** Every one of the 41,851 × 5 output cells is NaN. The `XpFeatureAdapter` is a pure pass-through (optional scatter-zero on the 2 c0 positions, otherwise identity); the inference driver `scripts/run_pipeline1_inference.py` does not impute NaN; and the model's forward pass propagates NaN through the linear trunk to μ and L. In other words — **if Stream-3 emits any row with a NaN in `j_mag`, `h_mag`, `k_mag`, `w1_mag`, or `w2_mag`, that row's prediction silently becomes NaN**. No warning, no flag, no Mahalanobis escalation (the OOD score is computed on the 108-D XP block only, not the aux block).

The spec called for a halt on 'NaN worse than zero' divergence. That condition is satisfied: zero is bad (+28% to +211% RMSE); NaN is categorically worse (100% prediction loss).

## Thread-3 recommendation

**2MASS + AllWISE cross-match is mandatory before Stream-3 inference.** NaN-imputation is not a workable fallback given current adapter behaviour. Two follow-ups feed the same decision:

1. Cross-match 2MASS (via Gaia DR3 `tmass_best_neighbour`) and AllWISE (via Gaia DR3 `allwise_best_neighbour`) for the Stream-3 ~1.3 M source_id list. Budget: see `docs/data_acquisition.md` §5 / §7 (2MASS + AllWISE together ≲ 200 MB Parquet if we keep only J/H/K + W1/W2 + errors).

2. For the residual ~few-percent of Stream-3 stars without a 2MASS/AllWISE counterpart (optical-only, confusion-limited, close binaries, crowded fields), **do not emit raw NaN**. Either (a) add a zero-imputation + `ir_missing_flag` column in the Stream-3 emit and accept the zero-IR RMSE hit on that subset (releasable with the caveat), or (b) extend `XpFeatureAdapter.forward` with an optional imputation hook keyed on aux-column positions + a companion `feature_missing` input the head can condition on. Option (b) requires a short retrain; option (a) is zero-cost but bakes in the 28-210% degradation for the affected rows. Recommend (a) for the v1 D-Cat-b release, (b) as a v2 experimental-arm followup.

## Notes

- The spec assumed the IR columns in `pipeline1_features_stream1.parquet` are z-scored; they are **raw magnitudes** (J mean ~10.95, H ~10.28, K ~10.94, W1 ~10.03, W2 ~10.08; std ~1.3 mag each). Setting them to zero places every val star ~8 σ below the population mean in IR space — an aggressive OOD perturbation. The test is nevertheless load-bearing because `training.build_dataloaders` calls `np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)` on the _raw_ feature matrix after the XP-finite filter, so the ensemble was trained with `IR=0` acting as the 'missing IR' signal for the ~0.1%-0.2% of training rows that lacked native IR counterparts. Our zero-IR condition extrapolates that signal to 100% of val rows, which is the relevant question for Stream-3 inference where IR is missing in bulk, not in sparse singletons.

- Native NaN rates in the val IR columns are tiny ({'j_mag': 38, 'h_mag': 38, 'k_mag': 38, 'w1_mag': 60, 'w2_mag': 49}); the vast majority of val stars had real IR mags at training time, which is why the model has learned to rely on them.

- log g has the worst degradation (+211% on zero-IR). This is _consistent_ with the Option-2 tier decision documented in `SUMMARY.md` — log g is prior-augmented, with XP contributing only 30% of the error reduction over an aux-only baseline (0.225 → 0.157 dex). If you strip IR from the aux block, you strip most of what drives log g.
