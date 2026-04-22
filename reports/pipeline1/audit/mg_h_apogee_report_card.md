# mg_h_apogee — §9.2 information-content report card

_Ensemble: 5-label main (seed 0–4) · Val split seed 0 · N_val used per test: see §tests below._

## Final tier decision (ratified 2026-04-19, Option 2): **Tier 1 (clean)**

The shuffled-spectrum null (skill_ratio +0.0619) and the XP joint shuffle
ΔRMSE/σ (0.6017) are both load-bearing evidence for Tier 1: this label is
spectrum-driven in the literal §4 sense. The 2-D-summary CMI of 0.0000 nats
is the KSG-on-low-D-summary pathology; a PCA-summary CMI (7 components,
95.81% variance) recomputed for methodology consistency is reported in
`pca_cmi_all_labels.json`.

## Verdict: **information-rich** → **tier-1**

No halt triggers.

## Summary

| metric | value |
|---|---|
| baseline RMSE (raw units) |   0.1041 |
| σ(y_truth) (raw units) |   0.2902 |
| real skill (1 − RMSE/σ) |   0.6415 |
| shuffled-spectrum null RMSE |   0.2787 |
| null skill |   0.0397 |
| **null / real skill ratio** | **  0.0619** (halt ≥ 0.2) |
| **XP joint shuffle ΔRMSE / σ(y)** | **  0.6017** (caveat ≥ 0.05, Tier-1 ≥ 0.20) |
| conditional MI I(XP; y | aux) [nats] |   0.0000 (release-gate ≥ 0.02) |

## Test 2 — Permutation importance by feature family

| family | ΔRMSE | ΔRMSE / σ(y_truth) |
|---|---|---|
| bp_shape |   0.0079 |   0.0273 |
| rp_shape |   0.0032 |   0.0110 |
| xp_c0 |   0.0003 |   0.0009 |
| residual |   0.0000 |   0.0000 |
| aux |   0.0332 |   0.1143 |

Top-10 single features by permutation ΔRMSE:

| rank | flat feature idx | ΔRMSE |
|---|---|---|
| 1 | 123 |   0.3610 |
| 2 | 124 |   0.1577 |
| 3 | 2 |   0.1297 |
| 4 | 125 |   0.1120 |
| 5 | 132 |   0.0981 |
| 6 | 1 |   0.0589 |
| 7 | 14 |   0.0395 |
| 8 | 0 |   0.0342 |
| 9 | 116 |   0.0313 |
| 10 | 3 |   0.0303 |

## Test 1 — LOOCO by feature family (mean ΔRMSE per coefficient)

| family | mean per-coeff ΔRMSE |
|---|---|
| bp_shape |   0.0116 |
| rp_shape |   0.0095 |
| xp_c0 |   0.0039 |

## Test 4 — Shuffled-spectrum null

Within each (Teff, log g) cell, all XP columns (110 Hermite + 2 c0) are jointly permuted across stars. The aux prior, residual, and photometric columns are untouched — so the null isolates what the model can infer *without* spectral shape.

- baseline RMSE =   0.1041
- null RMSE =   0.2787
- null / real skill ratio =   0.0619

## Test 5 — Conditional MI (KSG, k=8)

I(XP-summary; label | bp_rp, g_mag, parallax, av_sfd). A 2-D XP summary (|BP|-sum, |RP|-sum) feeds the KSG estimator on up to 8 000 finite-row samples drawn from the val set. Low-dim summary chosen to keep KSG well-conditioned at 2 + 1 + 4 = 7 joint dims; all-sky-complete conditioning columns used to avoid NaN-induced bias (Teff_gspphot and av_nbhd_median are 40–47 % NaN on DR19 and were excluded).

- CMI (2-D summary, original audit) = 0.0000 nats (release-gate ≥ 0.02)
- **CMI (PCA summary, 7 comp / 95.81% variance, methodology-consistent): see `pca_cmi_all_labels.json`** (produced by `scripts/run_pca_cmi_all_labels.py`). As for [α/M], the 2-D-summary value is the pathological 0.0000 — the KSG-on-low-D-summary bias collapses the estimate. Load-bearing evidence for the Tier 1 decision is the shuffled-spectrum null (skill_ratio +0.0619) and the XP joint shuffle ΔRMSE/σ = 0.6017 — both pass cleanly.

## Notes

- Tests 3 (SHAP) and 6 (decorrelated sub-sample) are deferred stubs at 2026-04-19; see ``docs/research_brief.md §9.2``.
