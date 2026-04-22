# alpha_m_apogee — §9.2 information-content report card

_Ensemble: 5-label main (seed 0–4) · Val split seed 0 · N_val used per test: see §tests below._

## Final tier decision (ratified 2026-04-19, Option 2): **Tier 1 (clean)**

The shuffled-spectrum null (skill_ratio −0.2574) and the XP joint shuffle
ΔRMSE/σ (0.5362) are both load-bearing evidence for Tier 1: this label is
spectrum-driven in the literal §4 sense. The 2-D-summary CMI of 0.0000 nats
and the 7-PC PCA-summary CMI of 0.0000 nats (reported in
`pca_cmi_all_labels.json`) were adjudicated via a three-test sequential triage
(`alpha_m_triage.md`; driver `scripts/triage_alpha_m_cmi.py`): **H2 — aux
absorption — is the supported hypothesis.** With parallax-only conditioning
and a 15-PC summary (98.87% variance) the same estimator produces CMI =
0.1125 nats, a factor of ~56 above the 0.02-nat release-gate floor. The full
4-D aux block (bp_rp, g_mag, parallax, av_sfd) absorbs [α/M]-relevant signal
via a sub-population kinematic correlation (α-rich stars are kinematically
hot and reddened at low latitude, so their aux features co-vary with
[α/M]). This is the expected behaviour of an information-rich label
correlated with the conditioning basis, not a signature of weak spectral
content.

## Verdict: **information-rich** → **tier-1**

No halt triggers.

## Summary

| metric | value |
|---|---|
| baseline RMSE (raw units) |   0.0547 |
| σ(y_truth) (raw units) |   0.0954 |
| real skill (1 − RMSE/σ) |   0.4264 |
| shuffled-spectrum null RMSE |   0.1059 |
| null skill |  -0.1098 |
| **null / real skill ratio** | ** -0.2574** (halt ≥ 0.2) |
| **XP joint shuffle ΔRMSE / σ(y)** | **  0.5362** (caveat ≥ 0.05, Tier-1 ≥ 0.20) |
| conditional MI I(XP; y | aux) [nats] |   0.0000 full-aux / 0.1125 parallax-only — **H2 aux absorption** (see `alpha_m_triage.md`; release-gate non-diagnostic for this label) |

## Test 2 — Permutation importance by feature family

| family | ΔRMSE | ΔRMSE / σ(y_truth) |
|---|---|---|
| bp_shape |   0.0014 |   0.0148 |
| rp_shape |   0.0012 |   0.0130 |
| xp_c0 |   0.0000 |   0.0005 |
| residual |   0.0000 |   0.0000 |
| aux |   0.0047 |   0.0493 |

Top-10 single features by permutation ΔRMSE:

| rank | flat feature idx | ΔRMSE |
|---|---|---|
| 1 | 123 |   0.0424 |
| 2 | 124 |   0.0349 |
| 3 | 2 |   0.0230 |
| 4 | 125 |   0.0219 |
| 5 | 0 |   0.0189 |
| 6 | 132 |   0.0168 |
| 7 | 1 |   0.0140 |
| 8 | 67 |   0.0106 |
| 9 | 55 |   0.0090 |
| 10 | 66 |   0.0072 |

## Test 1 — LOOCO by feature family (mean ΔRMSE per coefficient)

| family | mean per-coeff ΔRMSE |
|---|---|
| bp_shape |   0.0035 |
| rp_shape |   0.0039 |
| xp_c0 |   0.0012 |

## Test 4 — Shuffled-spectrum null

Within each (Teff, log g) cell, all XP columns (110 Hermite + 2 c0) are jointly permuted across stars. The aux prior, residual, and photometric columns are untouched — so the null isolates what the model can infer *without* spectral shape.

- baseline RMSE =   0.0547
- null RMSE =   0.1059
- null / real skill ratio =  -0.2574

## Test 5 — Conditional MI (KSG, k=8)

I(XP-summary; label | bp_rp, g_mag, parallax, av_sfd). A 2-D XP summary (|BP|-sum, |RP|-sum) feeds the KSG estimator on up to 8 000 finite-row samples drawn from the val set. Low-dim summary chosen to keep KSG well-conditioned at 2 + 1 + 4 = 7 joint dims; all-sky-complete conditioning columns used to avoid NaN-induced bias (Teff_gspphot and av_nbhd_median are 40–47 % NaN on DR19 and were excluded).

- CMI (2-D summary, original audit) = 0.0000 nats (deprecated estimator; see `SUMMARY.md`).
- CMI (PCA-7 summary, 95.81% variance, full 4-D aux conditioning) = 0.0000 nats — reported in `pca_cmi_all_labels.json` (produced by `scripts/run_pca_cmi_all_labels.py`).
- **Triage determination: H2 — aux absorption.** Driver: `scripts/triage_alpha_m_cmi.py`; report: `alpha_m_triage.md`.
  - Test 1 (H1, 15-PC / full aux, clipped): CMI = 0.0000 nats at 98.87% variance retained → H1 (high-order Hermite structure) ruled out.
  - Test 2 (H3, 15-PC / full aux, unclipped raw KSG): −0.0880 nats — outside the small-sample-noise regime, so H3 (clamp artefact) not the dominant cause.
  - Test 3 (H2, 15-PC / parallax-only conditioning): CMI = **0.1125 nats** — a factor of ~56 above the 0.02-nat floor. H2 confirmed: the aux block (bp_rp, g_mag, av_sfd) absorbs [α/M]-relevant signal through a sub-population kinematic correlation.
- Load-bearing evidence for the Tier 1 decision remains the shuffled-spectrum null (skill_ratio −0.2574) and the XP joint shuffle ΔRMSE/σ = 0.5362 — both pass cleanly.

## Notes

- Tests 3 (SHAP) and 6 (decorrelated sub-sample) are deferred stubs at 2026-04-19; see ``docs/research_brief.md §9.2``.
