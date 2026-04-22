# Pipeline-1 v1.1 delta report — inverse-frequency [M/H] weighting

**Ensemble dir**: `models/main/xp_abundances/20260421_38a993e_cdc55be_ensemble_5label` (5 seeds, 10 epochs, 5-label tier)
**Predecessor**: `models/main/xp_abundances/20260419_nogit_a0e10aa_ensemble_5label` (v1, `pipeline1-v1-2026-04-19`)
**Driver change**: `scripts/run_ensemble.py --inverse-freq --inverse-freq-clip 5.0`

## Motivation

v1 exhibited metal-poor [α/M] regression-to-mean on Stream-3 deployment: predictions at
[M/H] ≈ -1 clustered near ≈ +0.10 with very tight dispersion, while the APOGEE training-
truth mean at that metallicity is ≈ +0.23. Issue #198.

## Change

`beta_nll_block_cholesky` gains an inverse-frequency label-weighting term `w_i = 1/p(bin_i)`
on a 5-bin [M/H] partition `(-∞, -1.5, -1.0, -0.5, 0.0, +∞)`, clipped at 5.0 and mean-1
normalised across the batch. The clip × mean-1 normalisation compresses the nominal 5×
clip to an observed ~3.06× metal-poor/disc weight ratio after normalisation.

No architecture change. No data change. Same pretrained encoder. Same split seed.

## Training telemetry

Per-seed best val loss (v1.1): -0.0212, -0.0224, -0.0220, -0.0215, -0.0217.
Mean = -0.0218, spread = 0.0012.

v1 reference: mean val loss -0.025 on the same partition.

The ~0.003 loss degradation is expected: the weighted loss trades aggregate fit for
metal-poor representation. Spread is tight → ensemble is coherent.

## Val-set bias by truth [M/H] bin (N=41,851)

Diagnostic: `scripts/diagnose_alpha_m_by_mh_bin.py` — stratifies by *truth* [M/H]
and compares v1 vs v1.1 ensemble mean predictions against truth.

| [M/H] bin      |    n  | v1 α_pred | v1 α_truth | v1 bias  | v1.1 α_pred | v1.1 bias | Δ\|bias\| |
|----------------|------:|----------:|-----------:|---------:|------------:|----------:|----------:|
| (-∞, -1.50)    |   509 |   +0.2214 |    +0.2420 | -0.0206  |     +0.2248 |  -0.0172  |   -0.0034 |
| [-1.50, -1.00) |   982 |   +0.2081 |    +0.2230 | -0.0149  |     +0.2121 |  -0.0108  |   -0.0041 |
| [-1.00, -0.50) | 5,743 |   +0.1780 |    +0.2000 | -0.0220  |     +0.1889 |  -0.0111  |   **-0.0109** |
| [-0.50,  0.00) |25,596 |   +0.0812 |    +0.0809 | +0.0004  |     +0.0841 |  +0.0032  |   +0.0028 |
| [ 0.00, +∞)    | 9,021 |   +0.0183 |    +0.0071 | +0.0112  |     +0.0148 |  +0.0077  |   -0.0035 |

**Reading**: all three metal-poor bins show bias reduction; biggest gain is in the
[-1.0, -0.5) bin where the v1 bias of -0.022 halves to -0.011. Small disc trade-off
(+0.003). Metal-rich tail also improves marginally.

## Stream-3 pred-space stratified by predicted [M/H] (N=613,939)

Diagnostic: `scripts/compare_stream3_alpha_m_bias.py` — joins v1 and v1.1 prediction
Parquets on `source_id`, bins by v1's predicted [M/H], reports per-bin mean/median
[α/M] for both.

| [M/H] bin (v1) |       n  | v1 α_mean | v1.1 α_mean | v1 α_med | v1.1 α_med |    Δα_mean |
|----------------|---------:|----------:|------------:|---------:|-----------:|-----------:|
| (-∞, -1.50)    |   10,611 |   +0.2301 |     +0.2320 |  +0.2336 |    +0.2360 |    +0.0019 |
| [-1.50, -1.00) |  125,560 |   +0.1205 |     +0.1260 |  +0.1060 |    +0.1114 |    +0.0055 |
| [-1.00, -0.50) |  138,898 |   +0.1513 |     +0.1648 |  +0.1255 |    +0.1471 |    **+0.0134** |
| [-0.50,  0.00) |  203,376 |   +0.0688 |     +0.0705 |  +0.0553 |    +0.0576 |    +0.0018 |

**Reading**: v1.1 is strictly ≥ v1 in every populated bin; largest movement is +0.013
at the [-1.0, -0.5) bin where the user-visible bias lives (medians lift from 0.126 →
0.147, ~+0.02).

## Honest magnitude assessment

The fix moves in the right direction but **does not fully close the Stream-3 gap**:

- Val-set bias at [-1.0, -0.5) was ≈ -0.022 (v1); v1.1 halves it to -0.011.
- Stream-3 mean at the same bin was 0.151 (v1) → 0.165 (v1.1); truth-target is ≈ 0.20
  → residual Stream-3 gap ≈ 0.035.

The val-set bias magnitude (~0.02) is much smaller than the Stream-3 deployment bias
(~0.05). That delta is **domain shift**, not label-space imbalance. Stream-3 stars
(Andrae+2023 RGB sample) differ in population mix from APOGEE training; weighted-loss
fixes the within-distribution representation, not the out-of-distribution shift.

Closing the residual requires a different class of intervention (semi-supervised,
domain-adaptive pretraining, or broader training-set coverage) — out of scope for v1.1.

## Tier-promotion impact

[α/M] was **not Tier-1** in v1 (per `research_brief.md §3.3` six-test protocol).
v1.1 does not change that. The fix narrows the metal-poor bias but does not pass
Test 2 (per-cell coverage) nor trigger any tier-promotion gate.

## Ensemble artefacts

- Checkpoints: `member_seed{0..4}/xp_abundances_main_ensemble_5label_seed*_best.pt`
- Config: `reports/pipeline1/run_a_v11/ensemble_config.json`
- Summary: `reports/pipeline1/run_a_v11/ensemble_history.json`
- Val-bias JSON: `reports/pipeline1/run_a_v11/alpha_m_bias_by_mh_bin.json`
- Stream-3-bias JSON: `reports/pipeline1/run_a_v11/stream3_alpha_m_bias.json`

## Stream-3 v1.1 inference

- Union: `data/processed/pipeline1_predictions_stream3_v11.parquet` (613,939 rows)
  - OOD-joint rate: 19.90%; Regime-B rate: 0.072%; aux-missing: 1.41%
- Volume-limited arm: `data/processed/pipeline1_predictions_stream3_volume_v11.parquet` (249,092)
- Uniform arm: `data/processed/pipeline1_predictions_stream3_uniform_v11.parquet` (364,847)

## Pipeline-2 v1.1 impact

Rebuild: features from v1.1 volume-limited predictions (230,274 stars post-gates, vs
229,970 for v1), cuML UMAP (n_neighbors=30, min_dist=0), HDBSCAN-DBCV-grid winner
(mcs=200, ms=10, eom).

**Structure preserved**: K=2, same disc/halo split.

**DBCV: +0.265 (v1) → +0.180 (v1.1).** The embedding's intrinsic cluster quality
decreased. Reading: v1 had an artificially clean chemical axis because it compressed
halo [α/M] toward the disc mean; v1.1's halo cluster now sits at higher, more accurate
[α/M], reducing the visual chemical-plane gap. The cluster assignments themselves
remain dominated by kinematics (E, L_z, ecc) and [M/H], not α.

**Cluster medians** (v1 → v1.1):

| cluster       | n (v1) | n (v1.1) | [M/H] v1 → v1.1    | [α/M] v1 → v1.1  |  L_z v1 → v1.1 |
|---------------|-------:|---------:|-------------------:|-----------------:|---------------:|
| 0 (halo-like) |  8,679 |    9,007 |   -1.050 → -1.151  |   0.107 → 0.112  |  1857 → 1866   |
| 1 (disc)      | 221,291|  221,267 |   -0.087 → -0.069  |   0.037 → 0.034  |  1717 → 1717   |

The halo-like cluster is 328 stars larger (3.77% → 3.91% minority share), with its
median [M/H] pushed metal-poorer by 0.10 dex and [α/M] pushed higher by 0.005 dex —
both movements in the scientifically correct direction, but small. The disc cluster
is essentially unchanged.

**Figure**: `reports/pipeline2/figures/pipeline2_v11_diagnostics.png` (v1 preserved
at `pipeline2_v1_diagnostics.png` for A/B).

The halo-cluster median [α/M]=0.112 is still below APOGEE-truth expectation at
[M/H]≈-1 (≈+0.20). This reflects the same Stream-3 domain-shift residual described
above — not a Pipeline-2 defect.

## Calibration (existing gates only)

`scripts/run_calibration.py --ensemble <v1.1> --report-dir reports/pipeline1/run_a_v11/`

| metric                      |    v1 |   v1.1 |
|-----------------------------|------:|-------:|
| global err ≤ 10%            |  fail | **pass** |
| per-cell err ≤ 15%          |  fail |   fail |
| cov95 ≤ 5pp                 |  fail | **pass** |
| halt cells (err > 30%)      |     9 |  **5** |
| cells > 15%                 |    18 |  **6** |
| smoothness_flag_v2          |  true | **false** |
| overall release gate        |  fail |   fail |

v1.1 passes 2/3 top-level gates where v1 passed 0/3. Halt-cell set went from
{3, 11, 15, 33, 34, 39, 49, 54, 59} (v1) to {4, 15, 28, 34, 49} (v1.1) — persistent
offenders are {15, 34, 49}; most of v1's warm-upper-RGB / metal-poor halt cells
resolved. GP α-ratio max rose slightly (1.86 → 2.11) but median dropped (1.09 → 1.07)
and p90 dropped (1.30 → 1.28).

Overall: **v1.1 is strictly better calibrated than v1.** The release gate still fails
on per-cell stringency (same semantic outcome as v1 shipped, just fewer cells over
the line). Tier-1 release remains contingent on regime-B envelope exclusion, as in v1.

Report: `reports/pipeline1/run_a_v11/calibration_report.json`.

## Open

- Per-[M/H]-bin bias gate in `run_calibration.py` (#199) — not wired; flagged for
  user design (§3.3/§9.2 gate decision, not an autonomous change).
- Release tag `pipeline1-v1.1` pending user sign-off (#206).
