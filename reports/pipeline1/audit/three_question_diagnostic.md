# §9.2 three-question diagnostic — Teff and log g

_Timestamp: 2026-04-19 16:06:23 UTC · Ensemble: `20260419_nogit_a0e10aa_ensemble_5label` · Val split seed 0 · N_val = 41851_

Follow-up to `SUMMARY.md`. Scope is **Teff + log g only** — the three chemistry labels already pass the shuffled-spectrum null cleanly. This document provides evidence, not tier verdicts.

## Q1 — Conditional mutual information under richer XP summary

Original audit used a 2-D XP summary (|BP|-sum, |RP|-sum). Here we recompute CMI with a PCA summary of the 108 BP+RP normalised coefficients retaining 95% variance (→ 7 components, cumulative variance 95.81%). Conditioning set, estimator (KSG k=8) and subsample cap (8000) match the production audit.

| label | CMI (original 2-D, from payload) | CMI (2-D, rerun) | CMI (PCA summary) |
|---|---|---|---|
| teff_apogee | 0.1352 | 0.1352 | 0.0296 |
| logg_apogee | 0.0401 | 0.0401 | 0.0311 |

Release-gate CMI floor: ≥ 0.02 nats. Interpretation: if the PCA-summary CMI stays comparably small, the XP block carries near-zero information about the label beyond the photometric/astrometric priors; if it is materially larger, the original 2-D estimate was a summary artefact (higher-order Hermite structure carries the residual signal).

## Q2 — Per-feature permutation importance ranking

Full per-feature permutation importance rerun on the same val split. Each feature column is shuffled across stars; reported is ΔRMSE = permuted − baseline. Feature group: `xp` covers BP/RP normalised shape coefficients (108) + c0 z-scored scalars (2) = 110 features. `aux` covers the 3 residual columns + 26 auxiliary photometric/astrometric/extinction columns = 29 features.

### teff_apogee

Full-model baseline RMSE: 67.0956

Group totals across all 139 features — Σ(ΔRMSE, xp) = 339.0778, Σ(ΔRMSE, aux) = 419.8620. Top-10 composition: 6 XP, 4 aux.

| rank | feature | group | ΔRMSE |
|---|---|---|---|
| 1 | `r_lo_photogeo` | aux | 177.7100 |
| 2 | `r_med_photogeo` | aux | 101.9712 |
| 3 | `r_hi_photogeo` | aux | 65.9585 |
| 4 | `bp_coef_norm_3` | xp | 65.0367 |
| 5 | `bp_coef_norm_2` | xp | 62.5559 |
| 6 | `bp_coef_norm_1` | xp | 40.0560 |
| 7 | `bp_coef_norm_4` | xp | 35.1980 |
| 8 | `av_sfd` | aux | 23.8305 |
| 9 | `bp_coef_norm_15` | xp | 13.6929 |
| 10 | `bp_coef_norm_5` | xp | 13.1727 |

### logg_apogee

Full-model baseline RMSE: 0.1571

Group totals across all 139 features — Σ(ΔRMSE, xp) = 0.2584, Σ(ΔRMSE, aux) = 1.5151. Top-10 composition: 0 XP, 10 aux.

| rank | feature | group | ΔRMSE |
|---|---|---|---|
| 1 | `r_lo_photogeo` | aux | 0.4616 |
| 2 | `r_med_photogeo` | aux | 0.3591 |
| 3 | `av_sfd` | aux | 0.1635 |
| 4 | `r_hi_photogeo` | aux | 0.1517 |
| 5 | `bp_rp` | aux | 0.0707 |
| 6 | `k_mag` | aux | 0.0641 |
| 7 | `h_mag` | aux | 0.0550 |
| 8 | `w1_mag` | aux | 0.0405 |
| 9 | `w2_mag` | aux | 0.0272 |
| 10 | `bp_g` | aux | 0.0233 |

## Q3 — Auxiliary-only baseline vs full 5-label ensemble

Aux-only MLP trained on the same train split (seed 0, N_train=195295, N_val=41851) using 29 input features: the 3 residual columns + 26 auxiliaries. No XP shape, no c0 scalars. 256→128 MLP, masked MSE on standardised labels, AdamW, early-stop on val loss (29 epochs actually run, final val loss 0.5672).

| label | full-model RMSE | aux-only RMSE | aux / full ratio | σ(y) |
|---|---|---|---|---|
| teff_apogee | 67.0956 | 163.9619 | 2.4437 | 267.5588 |
| logg_apogee | 0.1571 | 0.2254 | 1.4350 | 0.5159 |

Interpretation thresholds (from user): ratio ≈ 1.00 (within ~5 %) → XP contribution is noise; ratio > 1.10 → XP contributes meaningfully; intermediate → judgment call.

Baseline checkpoint: `/home/aneitzel/projects/ArqueoGal/models/main/xp_abundances/aux_only_baseline_20260419/aux_only_baseline_seed0.pt`

## Honest characterization per label

Synthesising Q1 (CMI), Q2 (permutation ranking) and Q3 (aux-only head-to-head). This is evidence framing — the tier decision remains with the user.

### teff_apogee

- Q1 CMI: 2-D = 0.1352, PCA = 0.0296 nats.
- Q2 top-10 composition: 6 XP features / 4 aux. XP family Σ(ΔRMSE) = 339.0778 (44.7% of combined total 758.9398).
- Q3 aux-only / full RMSE ratio = 2.4437 (163.9619 / 67.0956).
- Characterization: **XP contributes meaningfully.**

### logg_apogee

- Q1 CMI: 2-D = 0.0401, PCA = 0.0311 nats.
- Q2 top-10 composition: 0 XP features / 10 aux. XP family Σ(ΔRMSE) = 0.2584 (14.6% of combined total 1.7736).
- Q3 aux-only / full RMSE ratio = 1.4350 (0.2254 / 0.1571).
- Characterization: **XP contributes meaningfully.**

---
Inputs and artefacts:
- Existing audit: `reports/pipeline1/audit/audit_payload.json`
- This diagnostic (JSON): `reports/pipeline1/audit/three_question_diagnostic.json`
- Aux-only baseline: `/home/aneitzel/projects/ArqueoGal/models/main/xp_abundances/aux_only_baseline_20260419/aux_only_baseline_seed0.pt` + `.provenance.json`

## PCA-CMI across all 5 labels (methodology-consistency pass)

_Appended 2026-04-19 16:20:36 UTC · Same val split (seed 0, N_val=41851), same PCA basis (7 components, 95.81% variance), same KSG (k=8, 8000-sample cap)._

Extends Q1 to the three info-rich chemistry labels so the final audit uses the same CMI estimator across all released labels.

| label | CMI (2-D, original audit) | CMI (2-D, rerun) | CMI (PCA summary) | PCA / 2-D |
|---|---|---|---|---|
| teff_apogee | 0.1352 | 0.1352 | 0.0296 | 0.219 |
| logg_apogee | 0.0401 | 0.0401 | 0.0311 | 0.776 |
| mh_apogee | 0.0088 | 0.0088 | 0.0357 | 4.055 |
| alpha_m_apogee | 0.0000 | 0.0000 | 0.0000 | nan |
| mg_h_apogee | 0.0000 | 0.0000 | 0.0357 | nan |

Release-gate CMI floor: ≥ 0.02 nats. Interpretation per label is folded into the per-label report cards and `SUMMARY.md`.
