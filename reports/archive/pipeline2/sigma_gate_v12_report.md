# Pipeline-2 v1.2 — σ-gate fix for unphysical halo cluster

**Trigger**: User flagged that v1.1's red halo-like cluster on the [α/M]–[M/H]
chemical plane (`reports/pipeline2/figures/pipeline2_v11_diagnostics.png`) was
physically impossible: 9,007 stars at [M/H] ≈ −1.15 with [α/M] = 0.107 ± IQR
0.006 dex. Halo α-ratios at that metallicity should scatter over ≈ 0.05–0.25.
A near-zero α dispersion in a halo-populated region is the signature of
regression-to-the-prior, not of a real population.

## Diagnosis — Pipeline-1 was telling us it didn't know

The 9,007 "halo" stars had been correctly routed by Pipeline-1 v1.1 as
"I don't know" — but Pipeline-2 ignored the `sigma` columns and used the
point predictions as if they were ground truth.

Stratifying the Stream-3 volume-limited set by predicted [M/H]:

| pred [M/H] bin    |     n   | α_pred IQR | σ_α mean  | σ_mh mean  | ood_joint_rate |
|-------------------|--------:|-----------:|----------:|-----------:|---------------:|
| [−1.5, −1.0)      |   8,858 |   0.004    |   0.174   |   0.228    |     22%        |
| [−1.0, −0.8)      |  15,700 |   0.015    |   0.122   |   0.170    |     63%        |
| [−0.8, −0.5)      |  52,990 |   0.028    |   0.082   |   0.111    |      8%        |
| ≥ −0.5            | 171,544 |   0.041    |   0.054   |   0.085    |      6%        |

The [M/H] ∈ [−1.5, −1.0) regime self-reports σ_α = 0.17 dex aleatoric +
σ_epi = 0.17 dex epistemic. α IQR is 0.004 dex. That is 40× smaller than
the self-reported σ — a textbook prior-collapse signature.

The ensemble-disagreement OOD gate fails precisely here because all 5 seed
members collapsed to the same disc-mean prior, so their member-to-member
spread is tiny even though each individual member is reporting a huge
aleatoric σ. **The OOD gate saw agreement; the individual σ̄ saw uncertainty.
Only the latter was diagnostic.**

For context, astroNN's benchmark per-star α precision on APOGEE is
0.03–0.05 dex. A reported σ_α of 0.17 dex means "I have no idea" even in
absolute terms.

## Fix — σ-gate on Pipeline-1 reported uncertainty

`scripts/build_pipeline2_features.py` gains Gate #3 (σ-gate), after the
existing OOD-joint gate and regime-B gate:

```python
sig_mask = (
    (sigma_alpha > 0.08) |
    (sigma_mh    > 0.20) |
    (sigma_mg_h  > 0.25) |
    any_sigma_nan
)
preds = preds.loc[~sig_mask]
```

Thresholds chosen to separate disc (σ_α ≈ 0.04) from prior-collapsed halo
(σ_α ≈ 0.17) with a comfortable margin above per-element NaN-rate
variation for [Mg/H].

Configurable via `--sigma-alpha-max`, `--sigma-mh-max`, `--sigma-mg-h-max`.
Provenance JSON records the thresholds and drop count; `docs/decisions/`
ADR entry pending user sign-off.

## Resulting row-count waterfall

| stage                        |     n    |  share |
|------------------------------|---------:|-------:|
| volume-limited Stream-3      | 249,092  | 1.000  |
| − OOD-joint gate             |  18,811  | 0.076  |
| − regime-B                   |       7  | 0.000  |
| **− σ-gate**                 | **18,535** | **0.074** |
| − kinematics/feature NaN     |       0  | 0.000  |
| **Pipeline-2-v1.2 input**    | **211,739** | **0.850** |

v1.1 fed 230,274 stars. v1.2 feeds 211,739. The 18,535-star σ-gate drop is
dominated by the former [−1.5, −1.0)-bin prior-collapse region (≈ 8,858
stars) plus σ-unreliable wings spread across other bins.

## Classifier output

`run_population_classifier.py` v1 grid (12 cells, `mcs ∈ {200, 500, 1000}
× ms ∈ {10, 20} × ε ∈ {0, 0.1} × eom`):

- **Winner**: `mcs=1000, ms=10, ε=0.1, eom` → K=2, noise=0.182, DBCV=−0.0038.
- v1.1 winner had DBCV=+0.180 — the σ-gate cost DBCV because the two most
  cleanly separated "clusters" (halo-like vs disc) were *driven by the
  prior-collapse pocket*. Removing that pocket leaves an intrinsically
  harder clustering problem.

**Cluster composition** (medians, n, fraction):

| cluster       |     n   | fraction | [M/H]    | [α/M]  | L_z    |      E       | Interpretation |
|---------------|--------:|---------:|---------:|-------:|-------:|-------------:|----------------|
| -1 (noise)    |  38,626 |  0.182   |  -0.110  | 0.053  | 1602   | -163,453     | boundary + sparse halo/thick-disc |
| 0 (minority)  |   1,294 |  0.006   |  -0.082  | 0.071  | 1362   | -170,288     | cold disc pocket (high-E, low-L_z) |
| 1 (majority)  | 171,819 |  0.811   |  -0.047  | 0.029  | 1738   | -159,686     | thin disc |

**The halo is no longer a cluster.** After σ-gating, there are insufficient
σ-reliable halo-metallicity stars in the Stream-3 volume-limited corpus for
HDBSCAN to resolve one. The halo stars that survive the gate sit in cluster
-1 (noise). This is the honest answer: **Pipeline-1 v1.1 does not deliver
σ-reliable halo predictions on Andrae+2023 RGB**, and v1.2's K=2 reflects
disc substructure only.

## Figures

- `reports/pipeline2/figures/pipeline2_v12_diagnostics.png` — chemical plane
  is physically plausible: monotonic [α/M]–[M/H] envelope, no tight red
  horizontal ribbon. Compare against
  `pipeline2_v11_diagnostics.png` (preserved for A/B).

## Pipeline-1 NN validation figure

Added `scripts/plot_pipeline1_v11_diagnostics.py` →
`reports/pipeline1/run_a_v11/pipeline1_v11_diagnostics.png` (4 × 5 panel):

- Row 1: pred-vs-truth hexbin per label (bias, RMSE inset).
- Row 2: residual histograms with Gaussian fit.
- Row 3: σ reliability (log-log, ideal = 1:1).
- Row 4: per-seed training curves · α by truth [M/H] bin · σ_α val-vs-Stream-3
  overlay (the key panel for the collapse diagnosis) · 68%/95% coverage
  bars · ensemble-epistemic σ distributions.

**Aggregate val calibration (5-member ensemble, N=41,851):**

| label   | bias    | RMSE   | σ̄       | cov@1σ | cov@1.96σ |
|---------|--------:|-------:|--------:|-------:|----------:|
| Teff    | -1.977 K | 67.98 K | 60.99 K | 0.753  | 0.964     |
| logg    | -0.003  | 0.159  | 0.151   | 0.769  | 0.963     |
| [M/H]   | -0.006  | 0.118  | 0.110   | 0.763  | 0.967     |
| [α/M]   | +0.002  | 0.056  | 0.054   | 0.748  | 0.956     |
| [Mg/H]  |  (n/a)* | (n/a)*  | 0.098   | 0.746  | 0.965     |

*[Mg/H] truth has NaNs on ~5% of val stars — bias/RMSE reported over
non-NaN subset only by `diagnose_alpha_m_by_mh_bin.py`.*

Coverage is slightly over-conservative (emp ≈ 0.75 vs nominal 0.68),
consistent with the ensemble between-variance term in
`_moment_match` inflating σ a touch beyond aleatoric truth. Acceptable for
σ-gate usage — a conservative σ gates more, not fewer, uncertain stars.

The σ_α val-vs-Stream-3 panel shows that val σ_α stays < 0.08 across all
five truth-[M/H] bins, while Stream-3 halo (pred [M/H] < −1) σ_α peaks at
0.17 — this is the visual mandate for the σ-gate threshold.

## What this does NOT fix

- **The halo is lost**, not classified honestly. A halo classifier needs
  Pipeline-1 predictions that are σ-reliable at [M/H] < −1. Options:
  1. **Semi-supervised / domain-adaptive pretraining** on unlabelled XP
     halo RGB stars (Stream-3 proper). Out of v1 scope.
  2. **External kinematic seed** (e.g. retrograde orbit cut first, then
     cluster within the seed). Violates §10.4's pure-unsupervised invariant
     but is a legitimate v2 path for a targeted halo-cluster driver.
  3. **Broader training sample** — metal-poor-rich APOGEE DR19 subset with
     per-star inverse-frequency weighting at much higher clip. Has
     diminishing returns: Stream 1 has only ~500 stars with [M/H] < −1.5.
- **The v1 chemical axis was artificially clean** (v1 DBCV +0.265, v1.1
  +0.180, v1.2 −0.004). Honest uncertainty-gated clustering on these
  Pipeline-1 predictions is K=2 disc-only. The "chemical-plane gap" the
  earlier Pipeline-2-v1 release claimed was an artefact of prior-collapse
  clustering; retract that finding in the methods-paper discussion.

## Open tags pending user sign-off

- `pipeline2-v1.2-2026-04-21` tag on `data/processed/pipeline2_*_v12.parquet`
- ADR entry under `docs/decisions/` formalising σ-gate thresholds and
  direction-of-bias argument (σ over-conservatism vs under-conservatism).
- Retract the v1 / v1.1 "halo-cluster recovered" claim in any
  methods-paper draft; replace with the §5.1 domain-shift caveat.

## Artefacts

- `data/processed/pipeline2_features_stream3_volume_v12.parquet`
  (211,739 × 9) + `.provenance.json`
- `data/processed/pipeline2_labels_stream3_volume_v12.parquet`
  (211,739 × {label, probability, boundary, soft_mem_0, soft_mem_1, GLOSH})
  + `.provenance.json`
- `data/processed/pipeline2_parametric_umap_stream3_volume_v12.pt`
- `reports/pipeline2/figures/pipeline2_v12_diagnostics.png`
- `reports/pipeline1/run_a_v11/pipeline1_v11_diagnostics.png`
- `reports/pipeline1/run_a_v11/val_predictions.parquet` (41,851 × 20)
- `scripts/build_pipeline1_val_predictions.py` (new, 160 lines)
- `scripts/plot_pipeline1_v11_diagnostics.py` (new, 290 lines)
