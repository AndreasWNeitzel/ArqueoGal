# `arqueogal.xp_abundances.main` — Design

## Status

This is the **production pipeline** for D-Cat-b. The main tree holds the frozen
deliverable implementation; feature work lives under `../experimental/` and is
promoted only after passing the bar in research_brief §8.6.

See `../DESIGN.md` for pipeline-level scope/rules and `docs/research_brief.md`
§§3, 7, 9 for the science.

## Module layout

```
arqueogal.xp_abundances.main
├── data.py            — FeatureLayout, LabelTiers, load_arrays, stratified split
├── model.py           — encoder (contrastive trunk) + heteroscedastic head + evol-stage head
├── training.py        — ensemble training loop, stratified splits, AMP, OneCycleLR
├── inference.py       — batched / streaming inference on data/processed/pipeline1_inference
├── uncertainty.py     — heteroscedastic covariance (Cholesky), post-hoc calibration
│                         (temperature / isotonic), coverage tests, conformal prediction
├── audit.py           — information-content audit (research_brief §9.2): LOOCO,
│                         permutation importance, SHAP, shuffled-spectrum null,
│                         conditional MI, decorrelated sub-sample test. Per-label report card.
├── sanity.py          — pre-training gate: schema, distributions, coverage
└── tier_promotion.py  — research_brief §3.3 six-test statistical promotion protocol
                          (Tier 3 → Tier 2 → Tier 1). Pre-registered, deterministic.
```

## Feature contract (frozen 2026-04-18)

All columns in `data/processed/pipeline1_features_stream1.parquet`. Order of mention
below is the canonical column order in the parquet file. Every listed column is
required; a sanity-battery check fails training if any are missing.

### Identifiers & audit (8 cols)

| Column | dtype | Source | Purpose |
|---|---|---|---|
| `source_id` | int64 | Gaia DR3 | primary key |
| `spectrum_pk` | int64 | Astra 0.6.0 | Astra spectrum primary key — **unique identifier for each duplicate row pre-dedup**; the audit handle for task-level analyses of the same physical star |
| `apogee_id` | string | APOGEE DR19 | 2MASS-style ID; cross-reference handle |
| `v_astra` | categorical | Astra | pipeline-variant; currently constant `"0.6.0"` on DR19. Shipped as categorical for future DR20/21-mix proofing |
| `n_aspcap_tasks` | int32 | derived | count of rows per `source_id` in this pre-dedup pool; diagnostic for the dedup-invariance audit |
| `snr` | float32 | APOGEE | spectrum SNR; `dedup_by_source_id` sort key |
| `ra_deg`, `dec_deg` | float64 | Gaia | ICRS coordinates |
| `b_deg` | float32 | derived | Galactic latitude via `astropy.coordinates.SkyCoord.galactic.b.deg`; drives `stratified_split_ids` |

### APOGEE labels — Tier 1 atmospheric (3 + 3 errors)

| Column | Astra source | Notes |
|---|---|---|
| `teff_apogee`, `e_teff_apogee` | `teff`, `e_teff` | effective temperature (K) |
| `logg_apogee`, `e_logg_apogee` | `logg`, `e_logg` | surface gravity |
| `mh_apogee`, `e_mh_apogee` | `m_h_atm`, `e_m_h_atm` | Mészáros+2025-corrected global [M/H] |

Tier 1 training target is **[M/H]** (ASPCAP global metallicity over Fe-peak + α
lines jointly), not per-element `[Fe/H]`. ASPCAP DR19 fits [M/H] first and
[Fe/H] per-element afterwards; the per-element fit can legitimately fail
(saturated Fe lines in metal-rich regimes, low blue-SNR per individual line,
unresolved Fe blends in cool giants). The sanity battery's Tier-1-completeness
gate therefore requires only `{teff_apogee, logg_apogee, mh_apogee}`; per-element
`fe_h_apogee` joins Tier 2. See `sanity.py::TIER1_ATMOSPHERIC` and
research_brief §3.2.

**Naming rationale**: `*_apogee` suffix makes provenance explicit. Parquet must not
carry bare `teff`/`logg`/`fe_h` — multiple Teff/logg/metallicity sources co-exist
(APOGEE, Gaia GSP-Phot, Gaia GSP-Spec), and unsuffixed names have been a
contamination risk. `*_atm` (Astra-internal nomenclature) is also ambiguous against
GSP-Phot atmospheric parameters from a different instrument at a different
resolution.

### APOGEE labels — Tier 2 (5 + 5 errors)

`fe_h_apogee`, `alpha_m_apogee`, `mg_h_apogee`, `c_h_apogee`, `n_h_apogee` +
their `e_*_apogee`. `fe_h_apogee` is the per-element iron abundance (allowed
NaN per DR19 realism; NaN-masked in training).

### APOGEE labels — Tier 3 (13 + 13 errors)

`o_h_apogee`, `na_h_apogee`, `al_h_apogee`, `si_h_apogee`, `s_h_apogee`,
`k_h_apogee`, `ca_h_apogee`, `ti_h_apogee`, `v_h_apogee`, `cr_h_apogee`,
`mn_h_apogee`, `ni_h_apogee`, `ce_h_apogee` + their `e_*_apogee`.

Tier is a **release** decision (research_brief §3.2), not a training-time exclusion.
The network learns all 21 labels jointly; the block-Cholesky covariance head uses a
**physics-motivated 4-block layout** (atmospheric / α-process / Fe-peak / light) —
see the "Block-Cholesky covariance structure" section below. Tier-based block
layout was considered and rejected because atmospheric parameters and α-process
elements covary across tiers; physics-motivated blocks capture the load-bearing
covariances while cutting trainable off-diagonals from 210 to ~30.

### Gaia astrometry & quality (4)

`parallax` (mas), `parallax_error`, `parallax_corr` (Lindegren+2021 zpt-corrected),
`ruwe`.

### Gaia photometry (6)

`g_mag` (Riello+2021-corrected), `bp_mag`, `rp_mag`, derived colours `bp_rp`,
`bp_g`, `g_rp`.

### IR photometry (10)

`j_mag`, `h_mag`, `k_mag` (2MASS) + `e_*`; `w1_mag`, `w2_mag` (WISE) + `e_*`.

### Distance (3)

`r_med_photogeo`, `r_lo_photogeo`, `r_hi_photogeo` (Bailer-Jones+2021 photogeometric,
pc, asymmetric σ).

### Extinction priors (12) — multi-column by design

The model sees all extinction sources as separate features and learns which prior
to trust per star. No single-column `av_los` abstraction.

| Column | Source | Notes |
|---|---|---|
| `av_edenhofer` | derived | `3.1 × ebv_edenhofer_2023`; Edenhofer+2024 3D map, primary for d < 1.25 kpc |
| `ebv_edenhofer_2023`, `e_ebv_edenhofer_2023` | Edenhofer+2024 | raw E(B−V); kept for audit |
| `av_sfd` | SFD | line-of-sight, far-field; renamed from `a_v_sfd` |
| `ebv_sfd`, `e_ebv_sfd` | SFD | raw E(B−V) |
| `av_lallement` | Lallement+2022 | renamed from `av_lallement_xcheck`; first-class feature, not a cross-check |
| `av_nbhd_median`, `av_nbhd_std`, `n_neighbors_75pc` | derived | GSP-Phot 75 pc-sphere neighborhood median from `ag_gspphot` |
| `ag_gspphot`, `ag_gspphot_lower`, `ag_gspphot_upper` | Gaia GSP-Phot | per-star Gaia-derived A_V with asymmetric errors |

### XP Hermite coefficients — 3-tier convention (225 cols)

**ML-input primary** (110 cols): **`bp_coef_norm_1..54`**, **`rp_coef_norm_1..54`**
— normalized shape coefficients. `bp_coef_norm_i = bp_coef_i / bp_coef_0` for rows
where `xp_fit_flag_residual_high == 0 AND ye2024_flag == 0`; NaN elsewhere. The
trivial `bp_coef_norm_0 ≡ 1` is **not stored** (constant, carries no information).

**ML-input scalars** (2 cols): **`bp_c0_z`**, **`rp_c0_z`** — `z_score(log10(bp_coef_0))`
on the normal-population subset. Population (μ, σ) **persisted in provenance**
(`extra.c0_zscore_frozen`) so Stream-3 inference applies training-set statistics,
not its own. Stream-3 drift is detectable via the comparison.

**Diagnostic/audit only, NOT ML inputs** (110 cols): `bp_coef_0..54`, `rp_coef_0..54`
— raw unnormalized Hermite coefficients. Kept for the §9.2 information-content
audit's LOOCO attribution (letting us distinguish "network uses shape coefficient n"
from "network uses c0 to scale the normalization"). The loader's `FeatureLayout`
default does NOT include these; the audit path opts in explicitly.

**Reprojection residuals** (3 cols, **ML-input features**): `reprojection_residual_rms`,
`reprojection_residual_rms_bp`, `reprojection_residual_rms_rp`. Fed to the encoder
as features (not as sample weights). The network learns to inflate predicted
uncertainty for high-residual stars; auditable via the heteroscedastic head output.

### Flags (5)

`flag_bad`, `flag_warn` (APOGEE); `ye2024_flag` (Ye+2024 correction status:
0=ok, 1=no_synth_phot, 2=calibrate_fail); `xp_fit_flag_residual_high` (Teff-
stratified p99 per-bin); `xp_fit_flag_residual_high_global` (flat global p99,
auxiliary).

### Stage-B auxiliary (1 — not an ML input)

`teff_gspphot` — the Gaia GSP-Phot Teff used as the stratification axis for
the Hermite-fit residual p99 thresholds in `pre_emit_decisions.json`. Retained
in the parquet so the stage-B emit is reproducible; NOT fed to the encoder.
The auxiliary coexists with `teff_apogee` (the training label) because the
residual-flag thresholds were frozen against GSP-Phot Teff and re-stratifying
them on APOGEE Teff would require recomputing the reference thresholds.

## Inputs to the encoder

The encoder input vector is built by `data.load_arrays`:

```
X = [ bp_coef_norm_1..54          # 54 floats, shape coefficients BP
    | rp_coef_norm_1..54          # 54 floats, shape coefficients RP
    | bp_c0_z, rp_c0_z            # 2 scalars, absolute-scale information
    | reprojection_residual_rms,  # 1, combined-band residual
      reprojection_residual_rms_bp, _rp  # 2, per-band residuals
    | <aux block>                 # Gaia + IR photometry + extinction + distance + RUWE
    ]
```

With the default `FeatureLayout.aux_cols`, the total `input_dim ≈ 140` (54+54+2+3+~27).
Update the layout's auxiliary block deliberately; any column change goes through the
same-commit DESIGN discipline.

**Preprocessing done at emit time** (not loader time): Ye+2024 flux-correction,
Hermite reprojection, c0 normalization, c0 z-scoring. **Preprocessing done at
loader time**: none — the parquet is ML-ready as shipped. This reverses the earlier
design where data_acquisition.md §6.4 steps 2–5 were loader-deferred; materializing
them at emit time (a) makes the contract explicit in the provenance, (b) keeps
Stream-3 inference consistent with Stream-1 training via frozen z-score stats,
and (c) removes one class of loader-time bugs.

## Outputs per star

Label vector + full covariance (lower-triangular Cholesky factor):

- Tier 1 atmospheric: Teff, log g, [M/H].
- Tier 2 per-element (RGB-gated where applicable): [Fe/H], [α/M], [Mg/H], [C/H], [N/H].
- Tier 3: 13 individual [X/H]. Learned jointly, not released per-star.
- Evolutionary-stage probability vector (RGB / RC / subgiant / MS-turnoff / AGB).

## Architecture

1. **Contrastive pre-training** on the full labeled training pool. Soft-positive
   SupCon with Gaussian kernel on label distances; learnable temperature τ (bounded).
   Architecture carried over from TESS_ML prototype: trunk 256→128→D, LayerNorm+GELU,
   L2-normalized projection head.
2. **Supervised fine-tune** on the same pool. Mészáros+2025 [X/M]/Teff corrections
   applied upstream (already in the parquet labels).
3. **Heteroscedastic multi-task head** (Kendall & Gal 2017) — predicts 21 means +
   block-structured Cholesky factor of covariance (4 physics-motivated blocks:
   atmospheric {Teff, log g, [M/H]}, α-process {Mg, Si, Ca, Ti}, Fe-peak {Fe, Mn,
   Ni, Cr}, light {C, N, O, Na, Al, K}). Positive-definiteness enforced via softplus
   on block diagonals; free off-diagonals within blocks; zero between blocks.
   Beta-NLL (Seitzer+2022, β=0.5). Inherits the TESS_ML `gnll` loss form,
   generalized from 2-D diagonal to 21-D block-Cholesky. Covariance is first-class.
4. **Evolutionary-stage head** trained jointly — gates Tier 2 [C/N] release.
5. **Ensemble** — 5–10 members share Run-A's pretrained encoder checkpoint and vary
   only the supervised head's initialization seed. Data splits are frozen across
   members to avoid confounding epistemic uncertainty with split-dependent label
   coverage. Epistemic σ = ensemble spread of means; aleatoric σ = mean heteroscedastic
   σ across members; total σ = √(epistemic² + aleatoric²) released.

**108-D vs 41-D and the three-run design**: research_brief §3.1 documents the σ_MAD
noise-floor plateau at BP[0:20] + RP[0:23] (41 effective shape modes post-Ye; 43 with
the two `bp_c0_z` / `rp_c0_z` scalars). Starting hypothesis: 108-D for contrastive
pretraining (self-supervised, data-abundant, let the network discover the noise
floor); 41-D for supervised fine-tuning (data-limited, don't waste capacity on
known-noise modes). Sprint #119 tests this via three runs:

- **Run A** — 108-D pretrain + 108-D fine-tune (baseline; no dimensional pruning).
- **Run B** — 108-D pretrain + 41-D fine-tune (hypothesis: fine-tuning benefits from
  pruning noise modes while contrastive pretraining tolerates them).
- **Run C** — 41-D pretrain + 41-D fine-tune (probes whether pretraining loses
  anything by discarding high-order modes).

Runs A and B share **the same saved pretrained encoder checkpoint** from Run A's
pretrain stage — divergence in the fine-tuning outcome is therefore attributable to
the fine-tuning representation, not to pretraining stochasticity. Reliability
diagrams per (Teff × log g × [M/H]) cell are the primary comparison; the run with
best-calibrated credible intervals across the most cells wins.

## Pretrain → fine-tune handoff (halfway checkpoint)

Between the contrastive pretraining stage and the supervised fine-tune, the
pretrained encoder is frozen and validated before any fine-tuning head is attached:

1. Embed a 50k-star held-out sample through the encoder trunk (use the projection-head
   output, D=32).
2. cuML-UMAP the 32-D projection to 2-D with the same hyperparameters used in
   `sanity.py::_continuity_embedding`.
3. Color-code by `teff_apogee`, `logg_apogee`, `mh_apogee`; visually require smooth
   contours across training-regime parameter ranges. Numerical gate: continuity
   statistic ≥ 0.75 (matches the sanity-battery continuity check).
4. Reviewer sign-off in `reports/119_tessml_port/pretrain_checkpoint.md` before any
   fine-tuning head is attached.

If the embedding is not smooth and physically structured, STOP. Fine-tuning heads
inherit the pretrain structure — fixing a messy trunk after heads are attached is
far more expensive than catching it here.

## OOD rejection at inference

Pipeline 1 releases a per-star OOD flag pair with two components:

- **`ood_flag_mahalanobis`** — Mahalanobis distance from the training XP feature
  distribution. At training time, compute the mean vector and full covariance of
  the 108-D `(bp_coef_norm_1..54, rp_coef_norm_1..54)` block on the training pool;
  persist (μ, Σ⁻¹, p99 threshold) in `ood_stats` of the checkpoint. At inference,
  flag stars above the training-set 99th-percentile Mahalanobis distance.
- **`ood_flag_ensemble`** — ensemble-disagreement epistemic proxy. Per star, compute
  epistemic / total σ ratio; flag above a configurable threshold (default 0.5).

Either flag alone is a yellow light; both firing is red (do not trust per-star).
Inference runs unaltered for flagged stars — the flag is metadata only — but
D-Cat-b release documentation notes that flagged-star predictions are valid for
population-level statistics only, not per-star science. Motivated by Ye+2024
blue-flux instability (research_brief §3.1) concentrating on metal-poor / hot /
high-Av stars under-represented in APOGEE, on which the network could produce
confidently-wrong predictions without explicit rejection.

## Block-Cholesky covariance structure

The 21-label output vector is partitioned into 4 physics-motivated blocks for the
Cholesky covariance head:

| Block | Labels | Block size | Off-diag entries |
|---|---|---|---|
| Atmospheric | `teff_apogee`, `logg_apogee`, `mh_apogee` | 3 | 3 |
| α-process | `mg_h_apogee`, `si_h_apogee`, `ca_h_apogee`, `ti_h_apogee` | 4 | 6 |
| Fe-peak | `fe_h_apogee`, `mn_h_apogee`, `ni_h_apogee`, `cr_h_apogee` | 4 | 6 |
| Light / CNO | `c_h_apogee`, `n_h_apogee`, `o_h_apogee`, `na_h_apogee`, `al_h_apogee`, `k_h_apogee` | 6 | 15 |

The 4 blocks cover 17 of the 21 output labels (3+4+4+6 = 17). The remaining 4
labels — `alpha_m_apogee`, `s_h_apogee`, `v_h_apogee`, `ce_h_apogee` — are
**diagonal-only** (no cross-label covariance learned). `alpha_m_apogee` is
redundant with the α-block individual abundances; the other three are Tier-3
audit-only with insufficient APOGEE SNR for robust off-diagonal estimation.

Total covariance-head output dimensionality: 21 means + 21 block-diagonal log-σ +
30 block-internal off-diagonals (3+6+6+15) = 72 outputs (vs 252 for full dense
covariance — a ~3.5× reduction in learned covariance parameters).

### 5-label production head (locked 2026-04-19, sprint #143)

After the 21-label retrain on z-scored features (sprint #142) passed sanity but
produced calibration fits with many sparse Tier-3 cells (see #144 halt-cell
diagnosis), the production head was reduced to **5 labels**:
`{teff_apogee, logg_apogee, mh_apogee, alpha_m_apogee, mg_h_apogee}` as a single
5×5 full-Cholesky block (`model.five_label_block_layout`,
`LabelTiers.five_label()`). Rationale:

- Scientific: the 5-label set covers the D-Cat-b release targets (Tier 1 +
  α-process summary + the strongest individual line signal, Mg b triplet /
  MgH band). Remaining Tier-2 and all Tier-3 elements defer to a follow-up
  sprint with their own §3.3 promotion studies.
- Statistical: with 5 labels, every (Teff × log g × [M/H]) cell has enough
  stars to estimate per-cell Var(z) reliably (n+2 lower bound is trivially
  met). At 21 labels most interior cells still had enough stars but the
  sparse-tail calibration was dominated by Tier-3 elements with large
  per-element NaN fractions.
- Architectural: a single 5×5 block vs a 4-block layout eliminates the
  block-structure choice as a confounder during the initial release.

The 21-label architecture remains in `model.py` and is the forward-compatible
head for the post-D-Cat-b element-promotion sprint.

**Why physics-motivated blocks:** atmospheric parameters covary strongly among
themselves via Teff–logg–metallicity degeneracies; α-elements track each other via
hydrostatic-burning nucleosynthetic origin; Fe-peak elements share Type-Ia SN
yields; light / CNO elements share a looser group (first dredge-up, CNO cycle,
Mg-Al cycle). Cross-block covariances exist but are expected second-order. The
§9.2 information-content audit checks whether any cross-block covariance is
significant; if so, the block structure is widened in a subsequent sprint.

**Positive-definiteness:** softplus on block-diagonal log-σ elements (strictly
positive), free off-diagonals within blocks, zero between blocks. Block-Cholesky
factor L is lower-triangular by construction; Σ = L L^T is guaranteed PD.

## Logging discipline

Every training run writes:

```
models/main/xp_abundances/<YYYYMMDD>_<short-sha>_<config-hash>/
├── config.yaml                      — frozen training config
├── provenance.json                  — git sha, data sha256, seeds, env hash
├── logs/
│   ├── pretrain_epoch_metrics.jsonl  — one record per epoch
│   ├── finetune_epoch_metrics.jsonl
│   └── ensemble_member_<N>.jsonl
├── checkpoints/
│   ├── pretrain_encoder.pt
│   └── finetune_member_<N>.pt
└── reports/
    ├── pretrain_checkpoint.md       — halfway-checkpoint continuity-UMAP sign-off
    ├── reliability_diagrams/        — per (Teff × log g × [M/H]) cell
    └── coverage_tests.json          — 68/95/99% on hold-out
```

JSON-lines logs are the **canonical** record — always on, committed to the
provenance bundle. Wandb is opt-in via `ARQUEOGAL_WANDB=1` env var for dev
ergonomics during the sprint; wandb is never the canonical source of truth for
release artifacts (projects get archived / deleted over 18-month deliverable
timelines; release docs must not depend on third-party SaaS).

## Training

- AdamW + OneCycleLR, pct_start configurable.
- AMP bfloat16 preferred (float16 fallback). `torch.compile(mode="reduce-overhead")`.
- Gradient clipping default norm=1.0.
- Stratified 70/15/15 train/val/test on ([Fe/H]_apogee, Teff_apogee, |b_deg|).
  Quantile-stratified, not random.
- Ensemble trained **sequentially** on RTX 3060 (6 GB VRAM budget); parallel runs
  only on IA HPC (CPU-only there, so throughput-limited).
- Global seed set (torch + numpy + python random + CUDA). Seeds recorded to
  checkpoint metadata.

## Pre-training sanity gate

`sanity.py` runs before the first training epoch; training refuses to start if any
check fails. Gates (frozen 2026-04-18): schema match, no unexpected NaNs in
required cols, label coverage per Teff×[Fe/H] stratification cell above threshold,
coefficient-column statistics within physical bounds, residual-flag fraction in
expected range, **2-D joint-distribution matches** vs published APOGEE DR19 cloud
for `(Teff_apogee, fe_h_apogee)`, `(logg_apogee, alpha_m_apogee)`,
`(Teff_apogee, av_nbhd_median)` — not just 1-D marginals.

## Checkpoint schema (v2)

```python
{
  "version": 2,
  "input_dim": int, "latent_dim": int,
  "n_labels": int, "label_names": list[str],
  "tier_map": dict[str, int],
  "block_cholesky_layout": {
      "blocks": list[list[str]],   # 4 lists of label names (atmospheric/α/Fe-peak/light)
      "diagonal_only_labels": list[str],   # e.g., alpha_m, s_h, v_h, ce_h
  },
  "encoder": state_dict,
  "regressor": state_dict,
  "evol_stage_head": state_dict,
  "label_scaler_mean": np.ndarray, "label_scaler_scale": np.ndarray,
  "calibration": {
      "temperature_per_cell": dict,
      "isotonic_per_label": dict,
      "conformal_scores": np.ndarray,
  },
  "c0_zscore_stats": {"bp": {"mu": float, "sigma": float},
                      "rp": {"mu": float, "sigma": float}},
  "ood_stats": {
      "mahalanobis_mean": np.ndarray,        # 108-D training-pool μ
      "mahalanobis_cov_inv": np.ndarray,     # 108×108 Σ⁻¹
      "mahalanobis_threshold_p99": float,    # training-pool p99 distance
      "ensemble_disagreement_threshold": float,   # default 0.5 (epistemic / total σ)
  },
  "ensemble": {
      "n_members": int,
      "member_seeds": list[int],
      "shares_pretrained_encoder": bool,     # True for sprint #119
  },
  "run_variant": str,                        # "A_108_108", "B_108_41", "C_41_41"
  "config_yaml": str,
  "random_seed": int,
  "git_sha": str,
  "training_metrics": dict,
}
```

Filename: `xp_abundances_main_<YYYYMMDD>_<git-sha7>_seed<N>.pt`. Loaded with
`torch.load(..., weights_only=True)`.

## Calibration — per-cell per-label α shrinkage (production)

`uncertainty.shrunken_per_cell_per_label_scale` is the production calibrator.
For each (Teff × log g × [M/H]) cell `c` and each label `j`, the per-star
predicted Cholesky factor `L` is rescaled along its row axis by a per-(cell,
label) factor `α_{c,j}`, so `L' = diag(α_{c(b),·}) · L_b` and `Σ' = diag(α) Σ
diag(α)`. `α_{c,j}^raw = √Var((y-μ)/σ_diag | cell c, label j)` is shrunk
toward the unconditional `α_j` with empirical-Bayes weight `λ = n_c /
(n_c + τ)`, τ = 50. PD-ness and joint correlation structure are preserved.

**Why per-(cell, label) scalar instead of per-cell scalar temperature**: the
#144 halt-cell diagnosis showed per-cell scalar temperature cannot fix the
global gate because different labels need different scaling within the same
cell (heteroscedasticity is *conditional* on (cell, label), not just cell).

**Regime B galactic-plane exclusion envelope.** A subset of cool-giant cells
in the Galactic plane (low `|b|`, Teff > 4750 K, log g < 2.1 — see
`uncertainty.RegimeBEnvelope`) exhibits a structural Teff mean-bias that
is not correctable by α-shrinkage because it is driven by extinction-
confounding that we cannot resolve within the 5 GB dust-map budget
(research_brief §14 item 11). Stars inside this envelope ship with
`pipeline1_tier1_release = False` and are carried as population-level-only.
D-Cat-b release notes this as expected incompleteness; the excluded
fraction on Stream 3 is reported in the final calibration summary.

### GP α-smoothing — evaluated, rejected, documented as a methodology finding

We evaluated a 3-D Gaussian-process smoothing of per-cell α over
(Teff, log g, [M/H]) parameter space as an alternative to empirical-Bayes
shrinkage (`uncertainty.gp_smoothed_per_cell_per_label_scale`). Rationale:
sparse cells should borrow strength from well-populated neighbours rather
than shrinking to a single global scalar.

**Result (single-member 5-label val set, 41,851 stars, 62 cells):**

| Method                     | global err | halt cells (>30%)          | cov95  |
|----------------------------|-----------:|----------------------------|-------:|
| Shrinkage (τ=50) — production | 0.080   | {4, 15, 28, 34, 49}        | 0.970  |
| GP α-smoothing (cell-center) | 0.136   | {4, 15, 28, 34, 48, 52}    | 0.885  |

**Diagnosis:** the GP under-corrects cells whose true α is structurally
far from neighbours. Example — cell 15 label 0 needs α = 1.109 but the GP
predicts α = 0.840, pulling the cell toward better-calibrated warmer
neighbours. Cells {4, 15, 28} lie at the cool-giant corner of the
(Teff, log g) grid; their calibration factor changes **structurally**
across parameter space (different molecular opacity regimes — TiO bands,
MgH band, different Ye+2024 training-sample density), not smoothly.

**Methodology finding (for the methods paper):**

> We evaluated Gaussian-process smoothing of per-cell calibration factors
> across (Teff, log g, [M/H]) parameter space as an alternative to
> empirical-Bayes shrinkage. The GP assumes smooth variation of
> calibration structure across parameter space, which is violated at
> cool-giant corners (Teff < 4450 K) where the calibration factor
> changes structurally rather than gradually. GP smoothing over-
> regularised the cool-giant cells toward warmer neighbours'
> calibration factors, worsening per-cell reliability (global err 0.136
> vs shrinkage's 0.080). We report this as a methodology finding: the
> assumption of smooth calibration across parameter space is incorrect
> for XP-abundance pipelines trained on APOGEE-dominated samples with
> non-uniform parameter coverage.

The GP machinery (`GpAlphaBundle`, `apply_gp_alpha`) is retained as an
evaluated-and-rejected reference implementation — it stays in
`uncertainty.py` with this caveat, not promoted as a calibrator.

### Cool-giant σ inflation — release documentation

Released per-star σ in the cool-giant regime (Teff < ~4450 K) is inflated
by factors up to ~1.5× relative to warmer giants. This is the honest
statement of how the model handles the cool edge of the training
distribution — uncertainty is genuinely larger there, and shrinkage-
calibrated α captures it. This non-uniform-across-parameter-space
uncertainty is a feature of the calibration, not a bug: users marginalising
over cool-giant samples should take the reported σ at face value.

## Release gates (hard)

- **Reliability diagrams** per (Teff×log g×[Fe/H]×Av×G) cell: predicted σ tracks
  observed residual σ within 10%. Uncalibrated cells fixed by stratified temperature
  scaling.
- **Coverage**: 68/95/99% credible intervals contain truth at those rates within 5
  pp on hold-out.
- **Information-content audit** (§9.2) per label. Labels failing shuffled-spectrum
  null or conditional-MI are NOT released per-star.
- **§3.3 six-test promotion protocol** before any Tier-3 → Tier-2 element enters
  the catalogue.

## Per-star release_tier column

Every Stream-3 prediction parquet carries a `release_tier ∈ {1, 2, 3}` column
assigned by `release.assign_release_tier()`:

| tier | consumer use                           | failure mode |
|------|----------------------------------------|--------------|
| 1    | per-star science                       | —            |
| 2    | statistical / ensemble only            | `regime_b_flag`, `mode_ambiguous_flag`, `ood_disagreement_flag`, or `aux_missing_any` |
| 3    | do not release                         | `ood_joint_flag`, `latent_support_flag`, or NaN in any `*_pred` column |

Hard-kill (Tier 3) trumps caveat (Tier 2). Missing flag columns are treated
as `False` — demotions must be explicit in the input. Contract is frozen:
changes require an ADR + DESIGN.md update in the same commit (invariant #15).
Counts and the flag-column provenance are emitted to
`<parquet>.release_tier.json` alongside each annotated parquet.

## Tests

Under `tests/xp_abundances/main/`. Smoke tests for model forward/backward,
preprocessing round-trip, calibration post-hoc application, ensemble aggregation
correctness, per-cell coverage assertion on a held-out synthetic fixture. Schema
contract tests assert the parquet schema matches `FeatureLayout` exactly — the
sanity battery is the runtime enforcement; the unit tests are the static
enforcement.

## Change log

- **2026-04-18** — Feature contract reconciled to match re-emitted parquet after
  the Hermite-reprojection data-layer sprint (tasks #111, #113, #117). Major
  changes: flat scalar XP columns (not list-typed arrays); c0 normalization done
  at emit time (previously deferred to loader and never executed); frozen c0
  z-score stats persisted in provenance; APOGEE label suffix `_apogee` (not
  `_atm`); extinction features ship multi-column (Edenhofer/SFD/Lallement/
  nbhd-median/GSP-Phot, no `av_los` abstraction); raw Hermite coefficients kept
  as diagnostic-only (not ML inputs); reprojection residuals fed as features
  (not sample weights); `spectrum_pk` is the audit handle for per-source_id
  duplicates; `v_astra` carried as categorical for DR-mix proofing.
- **2026-04-18** (afternoon, sprint #119 kickoff) — Architectural spec locked for
  the TESS_ML port: block-Cholesky 4-block covariance head (atmospheric /
  α-process / Fe-peak / light), physics-motivated not tier-based; halfway
  continuity-UMAP checkpoint between pretraining and fine-tuning; OOD rejection
  via Mahalanobis + ensemble disagreement as first-class per-star metadata
  outputs; three-run experimental design (A 108+108 / B 108+41 / C 41+41) with
  Run A's pretrained encoder reused in Run B; JSON-lines logs canonical, wandb
  opt-in via env var; ensemble members share pretrained encoder and vary only
  head init seed (data splits frozen across members). TESS_ML-ported pieces:
  encoder trunk 256→128→D=32, L2-normalized projection head, contrastive
  pretraining loop, `gnll` loss form generalized to block-Cholesky 21-D.
- **2026-04-19** — Production head reduced from 21 labels to 5 labels
  (`{Teff, log g, [M/H], [α/M], [Mg/H]}`, single 5×5 full block) for the
  D-Cat-b release (#143). 21-label architecture retained for the post-
  D-Cat-b element-promotion sprint. Calibration method locked: per-(cell,
  label) α shrinkage with τ=50 (empirical Bayes,
  `uncertainty.shrunken_per_cell_per_label_scale`). GP α-smoothing
  evaluated and rejected as a methodology finding: spatial smoothing
  assumption fails at cool-giant corners where calibration changes
  structurally rather than gradually. GP machinery retained in
  `uncertainty.py` as reference-only. Regime B galactic-plane warm-
  upper-RGB exclusion envelope (`RegimeBEnvelope`) added as per-star
  `pipeline1_tier1_release = False` for stars in `|b|<5° ∧ Teff>4750 K
  ∧ log g<2.1`. A/B/C 108/41-D experimental comparison (#137) cancelled
  — 5-label single-member already passes the global-err/coverage spec
  on the 108-D baseline; comparative-ablation work deferred to post-
  D-Cat-b sprint.
