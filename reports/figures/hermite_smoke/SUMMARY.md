# Hermite re-projection smoke test — SUMMARY

**Purpose.** Validate §6.4 step 2 before re-emitting `pipeline1_features_stream1.parquet` with 55+55 Hermite coefficients.

- Basis version: `v1.0`
- Basis fingerprint (SHA-256): `0d34b5659e97e5891b57005215a59b0b70fc56f23d8ffb22f442c4ad5101eab7`
- Sample size: **1490 stars** drawn from `pipeline1_features_stream1.parquet` (Ye flag=0 only).
- Catastrophic-Ye rows (residual RMS ≥ 1e-10): **35** — these are Ye+2024 NN failure modes, not a reprojection bug. They must be flagged at materialisation time (`xp_fit_flag = RESIDUAL_HIGH`), not dropped.

## Forced sub-populations

| sub-pop | N | catastrophic | normal p50 | normal p99 |
|---|---:|---:|---:|---:|
| `blue_neg` | 100 | 4 | 9.682e-18 | 4.760e-15 |
| `fe_h_lt_-1.5` | 50 | 0 | 1.368e-18 | 2.815e-17 |
| `teff_gt_6000` | 50 | 3 | 9.577e-18 | 8.854e-12 |
| `av_sfd_gt_3` | 50 | 1 | 6.569e-17 | 2.607e-11 |
| `abs_b_lt_15` | 100 | 1 | 2.787e-18 | 9.692e-14 |

## Residual RMS — NORMAL population (RMS < catastrophic threshold)

- p50: **4.568e-18**
- p90: 7.885e-17
- p95: 8.039e-16
- p99 (⇒ `XP_FIT_FLAG_RESIDUAL_HIGH` threshold candidate): **6.470e-13**

## Residual RMS — including catastrophic Ye failures

- p50: 4.779e-18
- p99: 1.065e-05  (pulled far above normal p99 by the catastrophic tail)

## PCA explained variance (110-dim standardised coeffs, catastrophic rows removed)

- PC1: 29.245%
- PC2: 19.166%
- PC3: 7.430%
- PC4: 6.213%
- PC5: 4.288%

## Noise-floor check

- BP σ_MAD median, modes 0–9:   9.201e-16
- BP σ_MAD median, modes 40–54: 3.248e-17
- RP σ_MAD median, modes 0–9:   9.115e-16
- RP σ_MAD median, modes 25–54: 2.850e-18

## Figures

- `residual_rms.png` — residual RMS histograms overall + per sub-pop, with the catastrophic cutoff and normal-p99 marked.
- `c0_vs_g.png` — signed log|c₀| vs G per band.
- `pca_110d.png` — PC1/PC2 of the 110-coefficient vector coloured by Teff, [Fe/H], log g, A_V, G (catastrophic rows excluded).
- `noise_floor.png` — per-mode |median| and σ_MAD for BP and RP.
