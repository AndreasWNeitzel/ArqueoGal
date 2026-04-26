# 03 — Hermite reprojection + z-score

**What this shows.** Ye-corrected flux samples are reprojected onto a
110-dim Hermite basis, then per-coefficient z-scored using stats **frozen
at v1 fit** (basis fingerprint `0d34b565…`, see
`reference_frozen_v1_stats.md` memory). Stream 3 reuses v1's stats —
never refits.

## Figures

| # | file | what to look at | status |
|--:|---|---|---|
| 01 | [pca_110d.png](../../figures/hermite_smoke/pca_110d.png) | Cumulative variance explained by 110-dim projection. | existing |
| 02 | [residual_rms.png](../../figures/hermite_smoke/residual_rms.png) | Reprojection residual RMS per coef; should lie below measurement noise floor. | existing |
| 03 | [pca_compare.png](../../figures/hermite_smoke/pre_emit/pca_compare.png) | Pre-emit PCA comparison. | existing |
| 04 | [failure_population_overlay.png](../../figures/hermite_smoke/pre_emit/failure_population_overlay.png) | Which stars have bad reprojections (pre-emit diagnostic). | existing |
| 05 | hermite_zscore_per_coef.png | Per-coef hist pre and post z-score, all 110 coefs in a 10×11 panel. | batch 3 |
| 06 | frozen_stats_v1_vs_v11.png | v1 frozen stats vs v1.1 re-fit; should overlay exactly. Contract verification. | batch 3 |

## Failure modes
- Residual RMS rising above noise floor for one coef class (e.g. high-order
  RP) means the 110-dim basis truncation is too aggressive for that regime.
- Frozen-stats mismatch between v1 and Stream 3 inference means the
  inference driver refitted stats — this is the contract violation the
  "frozen Hermite z-score stats across runs" project invariant is there
  to prevent.
