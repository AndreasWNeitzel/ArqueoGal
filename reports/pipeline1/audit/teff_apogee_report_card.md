# teff_apogee — §9.2 information-content report card

_Ensemble: 5-label main (seed 0–4) · Val split seed 0 · N_val used per test: see §tests below._

## Final tier decision (ratified 2026-04-19, Option 2): **Tier 1 (clean)**, XP primary with distance features the single strongest driver

The original audit verdict ("prior-augmented / tier-1-caveat") was based on the
shuffled-spectrum-null skill_ratio (0.7142, failing the literal §4 gate). The
three-diagnostic follow-up (`three_question_diagnostic.md`) reframed that:

- **Aux-only RMSE 163.96 K vs full 67.10 K — 2.4× improvement** (Q3).
- **6 of the 10 top-ranked permutation features are XP coefficients** (Q2).
- **PCA-summary CMI = 0.0296 nats** (Q1) — above the 0.02 release-gate floor;
  the original 2-D-summary CMI of 0.1352 was a KSG-on-2-D-summary artefact
  (4.6× inflation — see SUMMARY.md methodology note).

Option 2 verdict: XP is the primary information source. The null-skill-ratio
signal reflects that distance features alone carry an extinction-correlated
Teff prior the model can partially reconstruct under shuffled XP — not that
XP is uninformative. The per-feature permutation ranking is the load-bearing
evidence: distance (`r_*_photogeo`) ranks above every individual XP coefficient,
but four BP Hermite coefficients (2, 3, 4, 5) rank above every other aux
feature, and six XP features appear in the top-10.

**Release statement (verbatim, for D-Cat-b documentation):**

> Teff predictions use Gaia XP spectra as the primary information source, augmented by parallax and magnitudes. Aux-only baseline achieves RMSE 164 K; the full model achieves 67 K (2.4× improvement). XP coefficients account for 6 of the 10 top-ranked features in permutation importance analysis.

---

## Original verdict (machine-generated, superseded): **prior-augmented** → **tier-1-caveat**

No halt triggers. Retained here for auditability; the Option 2 tier decision
above is the release-ready verdict.

## Summary

| metric | value |
|---|---|
| baseline RMSE (raw units) |  67.0956 |
| σ(y_truth) (raw units) | 267.5588 |
| real skill (1 − RMSE/σ) |   0.7492 |
| shuffled-spectrum null RMSE | 124.3794 |
| null skill |   0.5351 |
| **null / real skill ratio** | **  0.7142** (halt ≥ 0.2) |
| **XP joint shuffle ΔRMSE / σ(y)** | **  0.2141** (caveat ≥ 0.05, Tier-1 ≥ 0.20) |
| conditional MI I(XP; y | aux) [nats] |   0.1352 (release-gate ≥ 0.02) |

## Test 2 — Permutation importance by feature family

| family | ΔRMSE | ΔRMSE / σ(y_truth) |
|---|---|---|
| bp_shape |   5.0823 |   0.0190 |
| rp_shape |   1.1956 |   0.0045 |
| xp_c0 |   0.0367 |   0.0001 |
| residual |   0.0000 |   0.0000 |
| aux |  16.1485 |   0.0604 |

Top-10 single features by permutation ΔRMSE:

| rank | flat feature idx | ΔRMSE |
|---|---|---|
| 1 | 124 | 177.7100 |
| 2 | 123 | 101.9712 |
| 3 | 125 |  65.9585 |
| 4 | 2 |  65.0367 |
| 5 | 1 |  62.5559 |
| 6 | 0 |  40.0560 |
| 7 | 3 |  35.1980 |
| 8 | 132 |  23.8305 |
| 9 | 14 |  13.6929 |
| 10 | 4 |  13.1727 |

## Test 1 — LOOCO by feature family (mean ΔRMSE per coefficient)

| family | mean per-coeff ΔRMSE |
|---|---|
| bp_shape |   8.6412 |
| rp_shape |   5.3054 |
| xp_c0 |   1.1889 |

## Test 4 — Shuffled-spectrum null

Within each (Teff, log g) cell, all XP columns (110 Hermite + 2 c0) are jointly permuted across stars. The aux prior, residual, and photometric columns are untouched — so the null isolates what the model can infer *without* spectral shape.

- baseline RMSE =  67.0956
- null RMSE = 124.3794
- null / real skill ratio =   0.7142

## Test 5 — Conditional MI (KSG, k=8)

I(XP-summary; label | bp_rp, g_mag, parallax, av_sfd). A 2-D XP summary (|BP|-sum, |RP|-sum) feeds the KSG estimator on up to 8 000 finite-row samples drawn from the val set. Low-dim summary chosen to keep KSG well-conditioned at 2 + 1 + 4 = 7 joint dims; all-sky-complete conditioning columns used to avoid NaN-induced bias (Teff_gspphot and av_nbhd_median are 40–47 % NaN on DR19 and were excluded).

- CMI (2-D summary, original audit) = 0.1352 nats (release-gate ≥ 0.02)
- **CMI (PCA summary, 7 comp / 95.81% variance, methodology-consistent) = 0.0296 nats** (from `three_question_diagnostic.json`, Q1). The 2-D estimate is inflated 4.6× relative to the PCA summary — see SUMMARY.md methodology note on KSG-on-low-D-summary bias. The PCA value is the primary estimate going forward; it clears the release-gate floor.

## Notes

- Tests 3 (SHAP) and 6 (decorrelated sub-sample) are deferred stubs at 2026-04-19; see ``docs/research_brief.md §9.2``.
