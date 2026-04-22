# [α/M] PCA-CMI triage — hypothesis verdict

_Timestamp: 2026-04-19 16:32:13 UTC · Ensemble: `20260419_nogit_a0e10aa_ensemble_5label` · Val split seed 0 · N_val = 41851_

Follow-up to `SUMMARY.md` and `pca_cmi_all_labels.json`. The §9.2 final audit recomputed PCA-CMI for all five labels on a 7-component PCA summary (95.8% variance). [M/H] and [Mg/H] recovered above the 0.02-nat release-gate floor; **[α/M] remained at 0.0000 nats alone**. This triage runs three targeted tests in order, short-circuiting as soon as one explanation fits.

## Verdict

**Supported hypothesis:** H2

**Short-circuited after test:** 3

**Tests actually run:** test_1_pca15_clipped, test_2_pca15_unclipped, test_3_pca15_parallax_only

**Rationale:** Parallax-only CMI = 0.1125 nats ≥ 0.01 while full-aux CMI = 0.0000 stays at/below the floor. The aux block (bp_rp, g_mag, av_sfd) is absorbing [α/M]-relevant signal — likely through a sub-population kinematic correlation.

## Test results

### Test 1 — 15-PC CMI, full 4-D aux, clipped (H1 probe)

- PCA components: 15
- Cumulative variance: 98.87%
- Conditioning: bp_rp, g_mag, parallax, av_sfd
- Samples used: 8000
- **CMI (clipped): 0.0000 nats**
- H1 trigger: CMI ≥ 0.01 nats → high-order Hermite structure carries the signal.

### Test 2 — 15-PC CMI, full 4-D aux, UNCLIPPED (H3 probe)

- Raw KSG estimator (no max(I_hat, 0) clamp): **-0.0880 nats**
- Samples used: 8000
- H3 trigger: |raw| ≤ 0.003 → small-sample KSG noise around a true value near zero.
- H1+H3 trigger: raw ≥ 0.01 → real signal masked by the clamp.

### Test 3 — 15-PC CMI, parallax-only conditioning (H2 probe)

- Conditioning: parallax (1-D)
- Samples used: 8000
- **CMI (clipped): 0.1125 nats**
- H2 trigger: parallax-only CMI ≥ 0.01 while full-aux CMI ≤ 0.01 → aux features absorbing [α/M]-relevant signal through sub-population kinematic correlation.

## Release-gate context

Load-bearing evidence for the [α/M] Tier-1 (clean) release is the shuffled-spectrum null (skill_ratio -0.2574) and the XP-joint-shuffle (ΔRMSE/σ = 0.5362). Both pass cleanly. CMI is a methodological cross-check, not a release-blocking quantity for [α/M]. This triage answers *why* the CMI specifically behaves as it does so the D-Cat-b methods paper can be precise.

## Reproducibility

- Estimator: KSG k=8
- Max samples per test: 8000
- Full aux conditioning: bp_rp, g_mag, parallax, av_sfd
- Ensemble: `/home/aneitzel/projects/ArqueoGal/models/main/xp_abundances/20260419_nogit_a0e10aa_ensemble_5label`
- Parquet: `/home/aneitzel/projects/ArqueoGal/data/processed/pipeline1_features_stream1.parquet`
- Driver: `scripts/triage_alpha_m_cmi.py` (short-circuits after the first test to answer the question).
