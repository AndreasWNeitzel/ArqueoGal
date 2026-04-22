# logg_apogee — §9.2 information-content report card

_Ensemble: 5-label main (seed 0–4) · Val split seed 0 · N_val used per test: see §tests below._

## Final tier decision (ratified 2026-04-19, Option 2): **Tier 1 with prior-augmented caveat**

The original audit verdict ("prior-augmented / tier-1-caveat") is retained.
The three-diagnostic follow-up (`three_question_diagnostic.md`) quantified
the caveat and confirmed that XP still carries signal above the release-gate
floor:

- **Aux-only RMSE 0.225 dex vs full 0.157 dex — 30% improvement** (Q3, ratio 1.44).
- **0 of the 10 top-ranked permutation features are XP coefficients** (Q2);
  the first XP feature (`bp_coef_norm_2`) ranks 13th with ΔRMSE ≈ 0.021, just
  below `parallax` at rank 12.
- **PCA-summary CMI = 0.0311 nats** (Q1), clearing the 0.02 release-gate floor.
  The 2-D-summary CMI (0.0401 nats) is only mildly inflated (1.3×) for log g
  vs Teff (4.6×) — consistent with the log-g signal being carried by
  low-order Hermite structure already captured in the 2-D sum.

Option 2 verdict: XP contributes meaningfully (> 5% caveat threshold, > 1.10
aux/full ratio), but the contribution is secondary to geometric (parallax,
distance) and photometric (magnitudes, colours) features. The caveat is
released alongside the per-star label.

**Release statement (verbatim, for D-Cat-b documentation):**

> log g predictions use Gaia XP spectra augmented by auxiliary features (parallax, magnitudes, extinction). An aux-only baseline MLP achieves RMSE 0.225 dex on the validation set; adding XP spectral information improves this to 0.157 dex (30% improvement). The spectral contribution is secondary to geometric and photometric features. Users requiring the full marginal contribution of spectra to log g predictions should note this and consider their use case accordingly.

---

## Original verdict (machine-generated, retained): **prior-augmented** → **tier-1-caveat**

No halt triggers. Tier decision above includes the prior-augmented caveat
verbatim.

## Summary

| metric | value |
|---|---|
| baseline RMSE (raw units) |   0.1571 |
| σ(y_truth) (raw units) |   0.5159 |
| real skill (1 − RMSE/σ) |   0.6955 |
| shuffled-spectrum null RMSE |   0.2038 |
| null skill |   0.6050 |
| **null / real skill ratio** | **  0.8698** (halt ≥ 0.2) |
| **XP joint shuffle ΔRMSE / σ(y)** | **  0.0906** (caveat ≥ 0.05, Tier-1 ≥ 0.20) |
| conditional MI I(XP; y | aux) [nats] |   0.0401 (release-gate ≥ 0.02) |

## Test 2 — Permutation importance by feature family

| family | ΔRMSE | ΔRMSE / σ(y_truth) |
|---|---|---|
| bp_shape |   0.0029 |   0.0056 |
| rp_shape |   0.0019 |   0.0036 |
| xp_c0 |   0.0005 |   0.0009 |
| residual |   0.0000 |   0.0000 |
| aux |   0.0583 |   0.1130 |

Top-10 single features by permutation ΔRMSE:

| rank | flat feature idx | ΔRMSE |
|---|---|---|
| 1 | 124 |   0.4616 |
| 2 | 123 |   0.3591 |
| 3 | 132 |   0.1635 |
| 4 | 125 |   0.1517 |
| 5 | 116 |   0.0707 |
| 6 | 128 |   0.0641 |
| 7 | 127 |   0.0550 |
| 8 | 129 |   0.0405 |
| 9 | 130 |   0.0272 |
| 10 | 117 |   0.0233 |

## Test 1 — LOOCO by feature family (mean ΔRMSE per coefficient)

| family | mean per-coeff ΔRMSE |
|---|---|
| bp_shape |   0.0098 |
| rp_shape |   0.0094 |
| xp_c0 |   0.0072 |

## Test 4 — Shuffled-spectrum null

Within each (Teff, log g) cell, all XP columns (110 Hermite + 2 c0) are jointly permuted across stars. The aux prior, residual, and photometric columns are untouched — so the null isolates what the model can infer *without* spectral shape.

- baseline RMSE =   0.1571
- null RMSE =   0.2038
- null / real skill ratio =   0.8698

## Test 5 — Conditional MI (KSG, k=8)

I(XP-summary; label | bp_rp, g_mag, parallax, av_sfd). A 2-D XP summary (|BP|-sum, |RP|-sum) feeds the KSG estimator on up to 8 000 finite-row samples drawn from the val set. Low-dim summary chosen to keep KSG well-conditioned at 2 + 1 + 4 = 7 joint dims; all-sky-complete conditioning columns used to avoid NaN-induced bias (Teff_gspphot and av_nbhd_median are 40–47 % NaN on DR19 and were excluded).

- CMI (2-D summary, original audit) = 0.0401 nats (release-gate ≥ 0.02)
- **CMI (PCA summary, 7 comp / 95.81% variance, methodology-consistent) = 0.0311 nats** (from `three_question_diagnostic.json`, Q1). 2-D/PCA ratio = 1.3× — log g is less affected by the KSG-on-low-D-summary bias than Teff (4.6×). PCA value is the primary estimate going forward; it clears the release-gate floor.

## Notes

- Tests 3 (SHAP) and 6 (decorrelated sub-sample) are deferred stubs at 2026-04-19; see ``docs/research_brief.md §9.2``.
