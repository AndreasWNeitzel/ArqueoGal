# ArqueoGal — Full Codebase Technical Report
**Working copy date:** 2026-04-19 · **Repo:** `/home/aneitzel/projects/ArqueoGal` · **Branch:** `master` (pre–initial-commit at time of writing)

> **Dated snapshot — read this first.** This report captures the repository
> state on **2026-04-19**. On **2026-04-22** the `src/arqueogal/population_classifier/`
> tree ("Pipeline 2") was spun out to the separate **Starfold** repository
> (<https://github.com/AndreasWNeitzel/Starfold>); all Pipeline-2 sections,
> module counts, config names, and test counts below describe that earlier
> state, not the current one. For current scope see
> [`docs/plan/04_pipeline2_main.md`](../docs/plan/04_pipeline2_main.md).

This report describes, module-by-module, what the ArqueoGal code under `src/arqueogal/` and `scripts/` actually does. It is assembled from (a) a codegraph dump (1 156 functions / 3 513 call edges in `src`; 846 / 3 321 in `scripts`), (b) targeted per-module reads of source files, and (c) the canonical docs (`docs/research_brief.md`, `docs/data_acquisition.md`, `docs/data_overview.md`) plus `README.md`.

> **Note to downstream reviewer.** Section 11 is an explicit alignment check between what the canonical docs / `README.md` claim and what the code actually does. Flag any discrepancy surfaced there first.

> **Post-review fix record (2026-04-19, after the reviewer pass).** Three must-fix items were resolved before this report was committed:
> 1. **Ye+2024 docstring drift** — `src/arqueogal/data/gaia_xp.py` module docstring and `scripts/fetch_gaia_xp.py` (docstring + provenance notes) now describe Ye+2024 as live, driven by `scripts/apply_ye2024_xp.py`. See §11.2 item 3.
> 2. **`halfway_umap.py` tests** — `tests/xp_abundances/main/test_halfway_umap.py` added (10 tests covering `HalfwayEmbedding`, `compute_halfway_embedding`, `save_halfway_plots`; UMAP mocked). Coverage 48/50 main-pipeline modules. See §11.2 item 2 and §12.
> 3. **Scripts count** — `scripts/` contains **31** Python entry points (earlier scratch-pad summary said 36). See §11.2 item 10.

---

## 1. Project identity and deliverables

- **Grant:** FCT 2024.15303.PEX (DOI: 10.54499/2024.15303.PEX). PI: Tiago Campante (CAUP/IA, Porto). Period: 2026-02 – 2027-08 (18 months).
- **Workspace:** personal development area of Andreas Neitzel (PhD student, Co-I). Not the team-wide repo.
- **Deliverables built here** (from `README.md` §Deliverables, corroborated by the research brief):
  - **D-Cat-b** (supporting, Aug 2026 / Month 6) — XP-based chemical abundance catalogue for stars without APOGEE DR19 spectroscopy.
  - **D5.1** (Dec 2026 / Month 10) — open-source ML tool for automated stellar-population classification.
  - **D-Cat-d** (Feb 2027 / Month 12) — stellar-population membership probabilities appended to the ArqueoGal all-sky catalogue.
- **Two pipelines**:
  - **Pipeline 1 — `xp_abundances`** (`src/arqueogal/xp_abundances/main/`) — semi-supervised multi-task regression Gaia DR3 XP → APOGEE-DR19-calibrated abundances, with calibrated covariant uncertainties.
  - **Pipeline 2 — `population_classifier`** (`src/arqueogal/population_classifier/main/`) — unsupervised Parametric UMAP + HDBSCAN on a 10–11D chrono-chemo-kinematic vector, DBCV-optimised, MC-propagated.
- **Three data streams** (from `docs/data_overview.md`):
  - Stream 1 (APOGEE DR19 × Gaia DR3, ~324 k rows) — **training set** for Pipeline 1.
  - Stream 2 (TESS Hon+2021 × Gaia DR3, ~158 k rows) — **pre-staged** for Task 4 asteroseismic ages; not consumed by Pipeline 1 or 2 yet.
  - Stream 3 (Andrae+2023 RGB+RC × Gaia DR3, ~168 k rows sub-selected from ~10.5 M) — **inference set**.

---

## 2. Repository layout

```
ArqueoGal/
├── README.md                    ← elevator pitch, Credentials, References (~119 lines)
├── pyproject.toml               ← pkg name arqueogal 0.1.0, py 3.12, ruff + pytest
├── codegraph.py                 ← present at top level (wrapper)
├── graph.json / graph.md        ← existing codegraph dumps
├── configs/
│   ├── main/                    ← 3 YAML: xp_abundances_baseline, population_classifier_{real,fire2}
│   └── experimental/            ← 1 YAML: xp_abundances_extended_labels
├── docs/                        ← research_brief.md, data_acquisition.md, data_overview.md
├── context/                     ← gitignored PDFs + resumo.txt (project reference)
├── data/                        ← gitignored, 5 GB budget (raw/, interim/, processed/, external/, provenance/)
├── models/                      ← gitignored, main/ + experimental/
├── notebooks/                   ← exploration / eda / viz
├── reports/                     ← committed outputs (figures, codebase_survey, sanity, pipeline1)
├── scripts/                     ← 31 CLI entry points (ingest, fetch, apply, emit, build, run, diagnose, plot)
├── src/arqueogal/
│   ├── data/                    ← 25 modules (ingestion, cross-match, corrections, enrichment)
│   ├── xp_abundances/main/      ← 14 modules (Pipeline 1 production)
│   ├── xp_abundances/experimental/   ← segregated exploration (no tests)
│   ├── population_classifier/main/   ← 8 modules (Pipeline 2 production)
│   ├── population_classifier/experimental/   ← segregated exploration (no tests)
│   └── utils/                   ← 6 modules (config, coordinates, gpu, io, plotting, reproducibility)
├── tests/                       ← 60 files mirroring src/ (tests/{data, xp_abundances/main, population_classifier/main, utils})
└── tools/codegraph/             ← AST-based call-graph tool (~132 L)
```

---

## 3. Data layer — `src/arqueogal/data/` (25 modules)

### 3.1 Core infrastructure

| Module | Responsibility |
|---|---|
| `credentials.py` | YAML loader with file-mode (`0o600`) check. Supports AIP/ESA block + fallback to `GAIA_AIP_TOKEN` env var (Bearer token). Custom path via `ARQUEOGAL_CREDENTIALS_PATH`. |
| `tap.py` | Unified pyvo wrappers for AIP, ESA, GAVO, VizieR. Async submission for > 5 000 rows, batched queries, resumable checkpoint dirs, TAP UPLOAD fallback on 504. Public: `aip_service()`, `gavo_service()`, `batched_upload_fetch_df()`, constants `AIP_TAP_URL`, `XP_TABLE`, etc. |
| `downloads.py` | Atomic HTTPS with streaming + SHA-256 + temp→rename. Never `r.content` for > 100 MB. |
| `provenance.py` | JSON sidecar writer. Source types: `HttpSource`, `TapSource`, `LocalSource`. Auto-embeds git SHA + timestamp + row counts. |

### 3.2 Stream loaders

| Module | Responsibility |
|---|---|
| `apogee_dr19.py` | DR19 FITS + column aliases + `[C/N]` synthesis + **live** Mészáros+2025 Δ[X/M] corrections (12 elements, Teff ∈ [3500, 6000] K, log g < 3.8). |
| `tess_hon2021.py` | Hon+2021 ν_max via VizieR J/ApJ/919/131, Prob > 0.95. ~120 k giants. |
| `tic_v82.py` | TIC v8.2 (IV/39/tic82) lookup; TIC → DR2 Gaia id. |
| `andrae2023.py` | Andrae+2023 RGB FITS/Parquet loader; ~2 M parent. |

### 3.3 Enrichment & corrections

| Module | Responsibility |
|---|---|
| `gaia_corrections.py` | **Live** Lindegren+2021 parallax zpt (5/6-param solutions); **live** Riello+2021 G-band cubic correction (G ≥ 13, BP−RP ∈ [0.25, 3.0]). Emits `parallax_zpt`, `parallax_corr`, `phot_g_mean_mag_corr`. |
| `gaia_xp.py` | 55-coefficient Hermite extraction; **live** Ye+2024 NN flux correction (vendored weights; gaiaxpy offline; CCM89 dereddening; 330-element sampled output); `normalise_xp()`; `zscore_c0()`; Hermite re-projection utilities. Driver: `scripts/apply_ye2024_xp.py`. |
| `dust_maps.py` | 3D composition: Edenhofer+2024 (< 1.25 kpc), Lallement+2022 (1.25–3 kpc), SFD (> 3 kpc); neighborhood-median A_G via cKDTree on 3D Cartesian. |
| `distances.py` | Bailer-Jones+2021 photogeometric (GAVO) + StarHorse2 v2 (AIP); merge + conflict-flag (factor of 2). |
| `kinematics.py` | galpy actions (McMillan17 potential, Staeckel δ = 0.45); central-value bulk + MC subsample for boundary stars. |

### 3.4 Cross-match, selection, dedup

| Module | Responsibility |
|---|---|
| `crossmatch.py` | DR2↔DR3 via `gaiadr3.dr2_neighbourhood`; filters (300 mas, 0.1 mag); tie-break by smallest \|Δmag\|. |
| `stream3_selection.py` | Stratified sub-sample on (Teff × log g × [M/H] × G); 600/cell → ~1.5 M stars; reproducible `rng_seed`. |
| `dedup.py` | Remove APOGEE duplicates on `source_id`; keep highest-SNR per star. |

### 3.5 Orchestrators and enrichment stages

| Module | Responsibility |
|---|---|
| `ingest_stream1.py` | APOGEE × Gaia end-to-end (DR19 download → cuts → corrections → enrichment). |
| `ingest_stream2.py` | Hon+2021 × TIC × DR2→DR3 × Gaia end-to-end. |
| `ingest_stream3.py` | Andrae+2023 → stratified → Gaia end-to-end. |
| `ingest_xp.py` | XP fetch + Ye+2024 NN → sampled-corrected-flux Parquet. |
| `gaia_enrich.py` | Level 2: standard Gaia enrichment (70+ columns); fixed ADQL template. |
| `enrich_geometry.py` | Level 4: distances (BJ21 + optional SH2) + extinction composition + neighborhood-median. |
| `enrich_kinematics.py` | Level 5: galpy actions; left-joins, NaN where unsolved. |
| `build_master_catalogs.py` | Level 6: join Stream 1/3 × XP; schema-validate. |
| `master_schema.py` | Column contracts (`PIPELINE1_TRAINING`, `PIPELINE1_INFERENCE`, `PIPELINE2_FEATURES`). |

### 3.6 Corrections applied by this layer (canonical)

1. **Lindegren+2021** parallax zero-point via official `gaiadr3-zeropoint` package.
2. **Riello+2021** (A&A 649 A3 Appendix A) G-band flux/mag cubic correction.
3. **Mészáros+2025** [X/M]/Teff trend polynomials (12 elements).
4. **Ye+2024** NN flux correction on XP (reduces systematics 3.2–3.7 % → 1.2–2.4 %).

---

## 4. Pipeline 1 — `src/arqueogal/xp_abundances/main/` (14 modules)

Production pipeline: **contrastive pretraining + supervised fine-tune + ensemble + post-hoc calibration** over Gaia DR3 XP → APOGEE DR19 labels. Locked 5-label production head as of 2026-04-19.

### 4.1 Configuration and data contracts

- `config.py` — 39 training hyperparameters across `LossWeights` (SupCon weight, β-NLL weight, Seitzer β = 0.5, label-space bandwidth) and `TrainingConfig` (data paths, architecture, AdamW LR, OneCycleLR pct_start, ensemble seeds, early stopping, AMP dtype, `use_c0_scalars`, `encoder_lr_ratio`).
- `data.py` — Feature layout (`FeatureLayout`: 54 BP + 54 RP + 2 c0-scalars + 3 residuals + 27 aux ≈ 140-D). Three label tiers (T1: Teff/log g/[M/H]; T2: 5 per-element; T3: 13 audit-only). `LabelScaler` fit on train only. `load_arrays()` selective-column Parquet. `stratified_split_ids()` quantile-stratifies on ([Fe/H], Teff, \|b\|) → 70/15/15.

### 4.2 Model and adapter

- `model.py` — `Encoder` (256→128→32-D trunk + L2 projection). `BlockCholeskyHead` (latent → 128 × 2 → means + Cholesky params). `XpAbundanceModel` wrapper. Physics-motivated 4-block covariance (atmospheric / α-process / Fe-peak / light + 4 diagonal-only). Default 21-label layout; **5-label production variant (single 5×5 block)** via `five_label_block_layout()`. `CovarianceBlockLayout` centralises block↔human reorder.
- `adapter.py` — `XpFeatureAdapter` zero-parameter nn.Module: `use_c0_scalars=False` zeros c0_z for SupCon (shape geometry only); `True` passes through for fine-tune. `reorder_labels_human_to_block()`.

### 4.3 Losses

- `losses.py`
  - `supcon_soft_positive()`: SupCon with Gaussian-kernel soft positives in label space (σ = 0.1).
  - `beta_nll_block_cholesky()`: multivariate Gaussian NLL over Cholesky factor; β-weighting via detached geometric mean of diagonals; missing-label mask + triangular solve.

### 4.4 Training, inference, checkpoint schema

- `training.py` — `build_dataloaders()`, `train_model()` (AdamW + OneCycleLR, AMP bfloat16, grad clip, early stop), `train_ensemble()` (sequential M-member, reinit head per seed, reuse pretrained encoder). **Checkpoint v2 schema**: encoder/head state dicts + block-layout dict + calibration artifacts + `c0_zscore_stats` + OOD bundle + ensemble metadata.
- `inference.py` — `load_ensemble()`, `predict_ensemble()`: per-member inference → per-member calibration → Bayesian aggregation (μ_agg = mean(μ_m); Σ_alea = mean(LLᵀ); Σ_epi = cov(μ_m); Σ_total = alea + epi). Returns `EnsemblePrediction`.

### 4.5 Uncertainty — production calibrator

- `uncertainty.py` — **biggest module by outgoing edges (204)**. Largest function is `gp_smoothed_per_cell_per_label_scale` (117 outgoing edges), **retained but rejected** per methodology finding (smoothness fails cool-giant corners). Production calibrator is `shrunken_per_cell_per_label_scale()` (empirical-Bayes per-(cell, label) α-shrinkage, τ = 50, λ = n_c/(n_c + τ)). Fallback `temperature_scaling_per_cell()`. `coverage_at_levels()` per-label Gaussian + joint Mahalanobis empirical coverage. `fit_calibration()` / `apply_calibration()` interface. `RegimeBEnvelope` — Galactic-plane warm-RGB exclusion (\|b\| < 5°, Teff > 4750, log g < 2.1) flagged population-only.

### 4.6 OOD and sanity

- `ood.py` — `MahalanobisOODBundle`: 108-D (bp_coef_norm_1..54, rp_coef_norm_1..54) train distribution (mean, regularised precision, p99 threshold). `fit_mahalanobis_ood()`, `score_mahalanobis_ood()`. Ensemble-disagreement flag (epi/total σ > 0.5 default). Either yellow, both red.
- `sanity.py` — Pre-training gate, 6 checks. Hard-fail: XP-feature NaN invariant, Tier-1 completeness (Teff, log g, [M/H] — explicitly **not** [Fe/H]), parameter bounds (DR19-calibrated). Soft-fail: c0_z z-score validity, dedup idempotency, per-element NaN rate report.

### 4.7 Audit and tier promotion

- `audit.py` — research_brief §9.2 six-test information-content audit: (1) LOOCO attribution, (2) permutation importance, (3) **SHAP (stub)**, (4) shuffled-spectrum null (within-cell), (5) mutual-information KSG (Kraskov+2004 unconditional + Frenzel–Pompe conditional), (6) **decorrelated subsample (stub)**. `audit_report()` aggregates tests 1,2,4,5,6 into JSON report card per label.
- `tier_promotion.py` — §3.3 six-test statistical promotion: (1) physical gate, (2) holdout RMSE + bias stratified, (3) open-cluster precision floor, (4) audit gate (compose 1,2,4,6), (5) conditional-MI bootstrap, (6) **cross-catalogue consistency (stub)**. Decision tree → rejected / internal / T2 / T1.

### 4.8 Halfway UMAP gate

- `halfway_umap.py` — Post-pretrain, pre-fine-tune validation. `compute_halfway_embedding()` embeds X through trunk (h, not z) via UMAP (n_neighbors = 30, min_dist = 0.1). Three-panel plots (Teff / [M/H] / log g). Gate: continuity ≥ 0.75; halt pre-fine-tune if structure shattered. **Test coverage:** `tests/xp_abundances/main/test_halfway_umap.py` (10 tests, UMAP mocked).

### 4.9 Hard release gates (enforced by the code path)

1. Reliability diagrams σ_pred ≈ σ_obs ± 10 % per cell.
2. 68/95/99 % coverage on hold-out ± 5 pp.
3. Information-content audit — labels failing shuffled-spectrum null or conditional-MI do not release per-star.
4. Tier promotion six-test mandatory before T3 → T2.
5. Regime B (Galactic-plane warm RGB) released population-only.

---

## 5. Pipeline 2 — `src/arqueogal/population_classifier/main/` (8 modules)

Parametric UMAP + HDBSCAN with soft memberships, DBCV-optimised hyperparameters, MC ensemble propagation, six-diagnostic real-data validation stack, FIRE-2 hare-and-hounds method-validation only.

### 5.1 Features (10–11D)

- `features.py` — `build_feature_matrix()`, `standardize()`, `apply_c_n_gate()`, `FeatureSpec`, `FeatureMatrix`.
- **Columns (10-D main):** `age, fe_h, mg_fe, al_fe, c_n, J_R, J_z, L_z, ecc, E` (+ optional `r_peri/r_apo`).
- Per-column StandardScaler; [C/N] evolutionary-stage gating (RGB P > 0.5 threshold).
- `include_mask` tracks NaN-dropped rows.
- Baseline preset: Neitzel+2025 (5-D) for reproduction.

### 5.2 Embedding (Parametric UMAP)

- `embedding.py` — NN Parametric UMAP (Sainburg+2021). MLP encoder (LayerNorm + GELU + Dropout), no activation on final layer. Loss = cross-entropy on positive/negative edges from umap-learn fuzzy simplicial set. Public: `ParametricUMAP.fit()/.transform()/.save()/.load()`. Defaults: hidden_dims = (128, 64), lr = 1e-3, n_epochs = 100.

### 5.3 Clustering

- `clustering.py` — `cluster_hdbscan()` → `ClusteringResult(labels, probabilities, soft_memberships[N,K], glosh, boundary_flag)`. Boundary flag: `max(soft) < 0.7`. Config: `min_cluster_size`, `cluster_selection_epsilon`, `method ∈ {eom, leaf}`.

### 5.4 Hyperparameter grid

- `hyperparameter.py` — **3 240 combinations** = 5 (n_neighbors) × 3 (min_dist) × 3 (n_components) × 4 (min_cluster_size) × 3 (min_samples) × 3 (epsilon) × 2 (method). Objective: DBCV (Moulavi+2014); `−1` returned on < 2 clusters or invalid. `grid_search()` caches embeddings per unique UMAP config.

### 5.5 MC ensemble

- `mc_ensemble.py` — N_MC = 50 realisations per star. Noise = Pipeline-1 calibrated covariance (diagonal σ or full (N, D, D)). Sample X_k ~ N(μ, Σ) → fixed reference embedding/clustering → aggregate mean/std soft-membership. Boundary flag: std > 0.15.

### 5.6 Diagnostics (six-tool stack)

- `diagnostics.py` — (1) Bootstrap-ARI (N = 500, Hennig 2007), (2) DBCV, (3) permutation-causal, (4) null-model (MVN/Copula, N = 100), (5) held-out-feature (Kruskal–Wallis), (6) literature cross-reference. Thresholds: bootstrap median > 0.75 stable, < 0.5 artefact; real clusters > null Q95. `DiagnosticStackReport` aggregator. Pluggable generic `cluster_fn` (not tied to Parametric UMAP).

### 5.7 FIRE-2 hare-and-hounds

- `hare_hounds.py` — ARI, AMI, MCC, Youden J (macro-avg TPR + TNR − 1). Hungarian optimal matching on negative contingency. **Method-validation only** — no transfer to real data (§10.5 diagnostics stand alone).

---

## 6. Shared utilities — `src/arqueogal/utils/` (6 modules)

| Module | Public API highlights |
|---|---|
| `config.py` | `load_config(path, schema)`, `to_yaml(obj)`, `ConfigValidationError`. Nested dataclass, type coercion, unknown-key warnings, path resolution vs config dir. Handles `Optional[X]`, `list[X]`, `dict[K,V]`, tuples. |
| `coordinates.py` | `equatorial_to_galactic()` (RA/Dec/parallax/pm/RV → ℓ/b/d/U/V/W); `galactic_velocities_to_cylindrical()` (U/V/W → v_R/v_φ/v_z). astropy.units internal; plain ndarrays in/out. 1/parallax fallback; prefers BJ21. Note: DESIGN lists `compute_orbital_params` but actual function lives in `arqueogal.data.kinematics` (galpy). |
| `gpu.py` | `get_device(prefer)`, `check_vram()`, `get_umap_class()`, `get_hdbscan_class()`. cuML → CPU sklearn fallback (HPC CPU-only path). |
| `io.py` | `load_parquet()`, `save_parquet()`, `streaming_parquet_reader()`, `save_checkpoint()`, `load_checkpoint()`. Atomic writes (temp → rename), version tag (`CHECKPOINT_VERSION = 1`), `_orig_mod.` prefix stripping (torch.compile). `ArqueoGalCheckpointError`. |
| `plotting.py` | `set_aa_style()`, `hexbin_with_colorbar()`, `density_2d()`, `residual_panel()`, `coverage_curve()`, `save_figure()`. A&A single/double column widths, Wong colorblind-safe palette. LaTeX auto-detect + mathtext fallback; lazy matplotlib imports. |
| `reproducibility.py` | `set_global_seed()` seeds random/numpy/torch/CUDA, returns `np.random.Generator`; `set_full_determinism()` opt-in cudnn determinism (slow). |

---

## 7. Scripts — `scripts/` (31 Python entry points)

31 CLI scripts driving the pipeline end-to-end. Categorised below.

### 7.1 Ingestion (7)

| Script | Purpose | Writes |
|---|---|---|
| `fetch_gaia_enrich_stream1.py` | Gaia DR3 enrichment for Stream 1 (APOGEE × Gaia). | `data/interim/stream1_gaia_dr3_raw.parquet` |
| `fetch_gaia_enrich_stream3.py` | Gaia DR3 enrichment for Stream 3 (Andrae+2023 stratified). | `data/interim/stream3_gaia_dr3_raw.parquet` |
| `fetch_gaia_xp.py` | AIP UPLOAD-batched fetch of raw XP 55-coefficient spectra (Stream 1 ∪ Stream 3 has_xp_continuous union). | `data/interim/xp_coeffs_raw.parquet` |
| `fetch_andrae2023_vizier.py` | Andrae+2023 / Ardern-Arentsen+2024 from VizieR TAP. | `data/raw/andrae2023/andrae2023_rgb.parquet` |
| `fetch_bailerjones_stream3.py` | Bailer-Jones+2021 photogeometric distances for Stream 3 source_ids. | `data/raw/bailer_jones_2021/per_stream3.parquet` |
| `ingest_stream2.py` | Hon+2021 × TIC × DR2→DR3 × Gaia end-to-end. | `data/interim/stream2_*.parquet` |
| `select_stream3_stars.py` | Stratified selection over (Teff, log g, [M/H], G). | `data/interim/stream3_selected.parquet` |

### 7.2 Corrections & feature emission (8)

| Script | Purpose |
|---|---|
| `apply_gaia_corrections.py` | Lindegren+2021 zpt + Riello+2021 G-mag. CLI: `raw`, `--out`. |
| `apply_ye2024_xp.py` | Ye+2024 NN flux correction; batched to `data/interim/enrich_batches/ye2024/`, final `xp_sampled_corrected.parquet`. **Live Ye+2024 driver.** |
| `emit_apogee_interim.py` | Pre-correction DR19 interim re-emission with richer columns. |
| `emit_apogee_corrected.py` | Mészáros+2025 [X/M]/Teff correction applied. |
| `build_stream1_apogee_gaia.py` | Join APOGEE-corrected × Gaia-corrected → `stream1_apogee_gaia.parquet`. |
| `build_pipeline1_features_stream1.py` | Full Pipeline-1 feature matrix: Stream 1 + Ye-corrected XP + dust priors → `pipeline1_features_stream1.parquet` (427 L). |
| `emit_stream1_with_hermite.py` | Re-emit with Hermite coefficients (stage B; 529 L). |
| `precompute_stream3_av.py` | Per-star A_V for Stream 3 via Edenhofer + SFD. |

### 7.3 Training / diagnostics (10)

| Script | Role |
|---|---|
| `run_pretraining_sanity.py` | Runs `sanity.py` six-check battery → `reports/sanity_battery/`. |
| `run_contrastive_pretrain.py` | Run A issue #131. AdamW + OneCycle + SupCon. |
| `run_halfway_umap.py` | Issue #132. UMAP gate after pretrain. |
| `run_supervised_finetune.py` | Issue #133. β-NLL fine-tune. |
| `run_ensemble.py` | Issue #134. Sequential M-member ensemble training. |
| `run_calibration.py` | Issue #135. Calibration harness — **largest script (960 L)**. Flags: `--apply-gp-smoothing`, `--apply-regime-b`. |
| `run_ood_eval.py` | Issue #136. Mahalanobis OOD + ensemble-disagreement flag distribution. |
| `diagnose_per_label_calibration.py` | Per-label calibration diagnostic (#140). |
| `diagnose_bias_location.py` | μ-bias parameter-space location (#140/#135). |
| `diagnose_halt_cells.py` | Halt-cell diagnosis for 5-label z-scored calibration. |

### 7.4 Hermite analysis (2)

- `analyze_hermite_pre_emit.py` (532 L) — three pre-re-emit analyses on ~315 k Ye-OK stars.
- `smoke_hermite_reprojection.py` (524 L) — §6.4 step-2 Hermite re-projection smoke test.

### 7.5 Plotting (4)

- `plot_data_overview.py` (434 L) — 8-panel canonical overview (`reports/figures/data_overview/panel_{01..08}`).
- `plot_extraction_diagnostics.py` — six-panel APOGEE interim extraction diagnostics.
- `plot_meszaros_correction_deltas.py` — Mészáros+2025 Table-3 correction visualisation.
- `plot_status_2026_04_18.py` (338 L) — diagnostic suite for 2026-04-18 status report.

### 7.6 Scripts call-graph hotspots

From the `scripts/` codegraph (846 functions / 3 321 edges):

- Largest `main()` bodies: `run_calibration.main` (302 edges out) ≫ `emit_stream1_with_hermite.main` (154) > `run_ood_eval.main` (144) > `diagnose_halt_cells.main` (136) > `build_pipeline1_features_stream1.main` (129) > `ingest_stream2.main` (116).
- Most outgoing calls by module: `run_calibration` (238) ≫ `plot_data_overview` (152) ≈ `smoke_hermite_reprojection` (151) ≈ `plot_status_2026_04_18` (150).

---

## 8. Configs — `configs/`

Four YAML files, one experimental.

### 8.1 `configs/main/xp_abundances_baseline.yaml` (100 L)
- Pipeline 1 baseline. Labels: [Fe/H], [α/M].
- latent_dim = 32, epochs = 1000, batch_size = 2048, lr = 1e-3.
- Contrastive: SupCon, σ = 0.10, learnable temperature.
- Regression: β-NLL, β = 0.5.
- Redundancy: VICReg 0.005.
- Conformal α ∈ {0.10, 0.05, 0.01}.
- Data: `xp_apogee_dr19_crossmatch.parquet`.

### 8.2 `configs/main/population_classifier_real.yaml` (73 L)
- Real-data D-Cat-d application. Features: age, [Fe/H], [α/Fe], V, UV_mag.
- UMAP n_neighbors = 30, HDBSCAN min_cluster_size = 100, MC N = 200.
- Data: `tess_gaia_apogee_5d.parquet`.
- Validation: silhouette, physical consistency, α-bimodality, Gaia Enceladus, AMR.

### 8.3 `configs/main/population_classifier_fire2.yaml` (102 L)
- FIRE-2 hare-and-hounds. Same features as real + hidden ground truth (`population_label`, `circularity`, `exsitu_flag`).
- Error model: age σ_rel = 0.20; [Fe/H] σ = 0.08/0.15; [α/Fe] σ = 0.04; V σ = 2–5 km/s.
- Extended feature sets (orbital, integrals of motion).
- Metrics: ARI, NMI, completeness, purity, F1.

### 8.4 `configs/experimental/xp_abundances_extended_labels.yaml` (92 L)
- Ablation. 5 labels: [Fe/H], [α/M], [Mg/Fe], [C/N], [Si/Fe]. latent_dim = 64. Permutation test n_seeds = 3.

---

## 9. Tests — `tests/` (60 files)

- Tree: `tests/data/` (21), `tests/xp_abundances/main/` (12, incl. new `test_halfway_umap.py`), `tests/population_classifier/main/` (9), `tests/utils/` (6), plus subtree stubs for `experimental/`.
- **Experimental test dirs exist but are empty** (`tests/xp_abundances/experimental/`, `tests/population_classifier/experimental/`).
- Coverage: **48/50 production modules** have a test (96 %). **Remaining gaps: `xp_abundances/main/config.py`, `xp_abundances/main/sanity.py`.**
- Mostly offline unit tests with monkeypatched requests/TAP. ~3–4 integration tests (e.g., `test_integration.py` drives end-to-end D-Cat-d orchestration). `@pytest.mark.slow` marker for real galpy/astropy. `test_halfway_umap.py` mocks `umap` via `sys.modules` monkeypatch.
- No global `conftest.py`; fixtures inline via `monkeypatch`.
- Line counts mostly 20–50 L; largest `test_downloads.py` at 131 L.

---

## 10. Call-graph observations

### 10.1 `src/arqueogal` (1 156 functions, 3 513 edges)

Top modules by outgoing calls (heaviest to other modules/stdlib):

| Rank | Module | Outgoing edges |
|---:|---|---:|
| 1 | `xp_abundances.main.uncertainty` | 204 |
| 2 | `data.gaia_xp` | 181 |
| 3 | `xp_abundances.main.training` | 159 |
| 4 | `xp_abundances.main.audit` | 103 |
| 5 | `data.dust_maps` | 91 |
| 6 | `data.kinematics` | 81 |
| 7 | `population_classifier.main.embedding` | 76 |
| 8 | `population_classifier.main.diagnostics` | 74 |
| 9 | `data.apogee_dr19` | 74 |
| 10 | `data.tap` | 70 |

Top functions by fan-out:

| Rank | Function | Edges out |
|---:|---|---:|
| 1 | `xp_abundances.main.uncertainty.gp_smoothed_per_cell_per_label_scale` | 117 |
| 2 | `data.gaia_xp.apply_ye2024_correction` | 71 |
| 3 | `data.stream3_selection.stratified_subsample` | 61 |
| 4 | `xp_abundances.main.model.__init__` | 56 |
| 5 | `population_classifier.main.hare_hounds.compute_hare_hounds_metrics` | 55 |
| 6 | `data.ingest_stream2.ingest_stream2` | 49 |
| 7 | `xp_abundances.main.training.train_model` | 44 |
| 8 | `data.enrich_geometry.enrich_geometry` | 44 |
| 9 | `data.kinematics._run_galpy` | 44 |
| 10 | `population_classifier.main.diagnostics.literature_cross_reference` | 42 |

**Interpretation** — Uncertainty is by far the densest module (calibration + coverage + GP-smoothing-retained-but-rejected). `apply_ye2024_correction` is the densest single data-layer function. `gp_smoothed_per_cell_per_label_scale` is retained dead code — a methodology finding, not production path.

### 10.2 `scripts/` (846 functions, 3 321 edges)

- `run_calibration.main` (302 edges) is the single largest script entry point — it drives model load → adapter → inference → calibration → coverage → report generation.
- `plot_data_overview` and `smoke_hermite_reprojection` are the densest plotting/smoke modules (≥ 150 edges each).
- No cross-module calls between `scripts/` files — every script is a standalone entry point.

---

## 11. Alignment check — docs / README.md vs code

### 11.1 Claims that are corroborated by code

| Claim | Source | Code evidence |
|---|---|---|
| Gaia zpt + G-mag corrections are mandatory at ingestion. | data_acquisition.md §3.7 | `data/gaia_corrections.py` implements both live; `scripts/apply_gaia_corrections.py` enforces on every raw enrichment Parquet. |
| XP preprocessing order: Ye+2024 → normalise by c_0 → log + z-score c_0. | data_acquisition.md §6.4 | `data/gaia_xp.py` contains `apply_ye2024_correction`, `normalise_xp`, `zscore_c0` as separate functions. `scripts/apply_ye2024_xp.py` drives step 1; `scripts/build_pipeline1_features_stream1.py` drives 2–4. |
| Mészáros+2025 [X/M] corrections. | data_acquisition.md | `data/apogee_dr19.py` applies live; `scripts/emit_apogee_corrected.py`, `plot_meszaros_correction_deltas.py`. |
| Three streams as described. | README.md, data_overview.md | Three separate orchestrators `ingest_stream{1,2,3}.py` + matching enrichment and fetchers. |
| AIP bearer token via `GAIA_AIP_TOKEN`; YAML fallback; YAML wins when both set. | README.md §Credentials | `data/credentials.py` logic matches (YAML has priority, env var fallback, `ARQUEOGAL_CREDENTIALS_PATH` override). |
| Tier 1 per-star / Tier 2 population / Tier 3 not released. | research_brief.md, data_overview.md §6 | `xp_abundances/main/data.py` `FeatureLayout` and tier lists; `tier_promotion.py` six-test protocol. |
| Pipeline 2 = Parametric UMAP + HDBSCAN, DBCV-optimised, MC ensemble, six-diagnostic stack. | research_brief.md | All eight `population_classifier/main/` modules match: `embedding.py` (Parametric UMAP), `clustering.py` (HDBSCAN soft), `hyperparameter.py` (DBCV 3 240-cell grid), `mc_ensemble.py` (N = 50), `diagnostics.py` (six tests). |
| 5 GB storage budget, no FIRE-2 for pipelines, no push beyond G = 17. | working invariants | `dust_maps.py` deliberately omits Bayestar19 per doc; no FIRE-2 imports in Pipeline-1 main; `features.py`/`data.py` do not expose G > 17 feature paths. |
| Segregated main vs experimental. | working invariants | `src/arqueogal/*/main/` and `src/arqueogal/*/experimental/` trees exist; no cross-imports detected in codegraph. |

### 11.2 Discrepancies / cautions the reviewer should note

1. **Test tree gap.** `tests/xp_abundances/experimental/` and `tests/population_classifier/experimental/` directories exist but are **empty**. The project conventions mandate "separate test trees for `main/` and `experimental/`" — this is a gap, not a contradiction, because the experimental trees themselves are also currently sparse.

2. **xp_abundances `config.py`, `sanity.py` lack tests** (2 of 14 main-pipeline modules). `halfway_umap.py` was added to the test tree on 2026-04-19 (`tests/xp_abundances/main/test_halfway_umap.py`, 10 tests, UMAP mocked via `sys.modules`).

3. **Ye+2024 docstring drift — RESOLVED 2026-04-19.** Previously, `scripts/fetch_gaia_xp.py` docstring and provenance notes described Ye+2024 as a "stub / BLOCKED on model-weight availability" while the data-module survey reported `data/gaia_xp.py:apply_ye2024_correction` as a live implementation. Direct read of the function confirmed **live** — vendored NN weights, gaiaxpy offline, CCM89 dereddening, 330-element sampled output. Fixed in two places:
   - `src/arqueogal/data/gaia_xp.py` module docstring — removed "**stub**" descriptor.
   - `scripts/fetch_gaia_xp.py` module docstring + provenance notes — replaced "stub / BLOCKED on model-weight availability" language with pointers to `scripts/apply_ye2024_xp.py` as the live driver and `scripts/build_pipeline1_features_stream1.py` for steps 2–4.

4. **Rejected-but-retained code.** `xp_abundances/main/uncertainty.py:gp_smoothed_per_cell_per_label_scale` (117 outgoing edges, the single largest function in `src/`) is deliberately retained as a methodology-finding artifact but is **not** the production calibrator. Production is `shrunken_per_cell_per_label_scale`. A scanner that assumes "largest function = primary path" will mis-identify it. `run_calibration.py` exposes `--apply-gp-smoothing` so the retained code is user-togglable.

5. **Three stubs/incomplete audit tests** in `xp_abundances/main/audit.py` and `tier_promotion.py`:
   - `audit.py` test 3 (SHAP values) — deferred (requires external lib).
   - `audit.py` test 6 (decorrelated subsample) — incomplete.
   - `tier_promotion.py` test 6 (cross-catalogue consistency) — stub.
   
   Research_brief §3.3 and §9.2 both enumerate six tests each; code implements 5/6 in each. Tier-promotion completeness depends on the cross-catalogue test.

6. **DESIGN.md promise vs code location.** `utils/DESIGN.md` lists `compute_orbital_params` as a utils function, but it actually lives in `arqueogal.data.kinematics` with galpy. Minor; just a doc drift.

7. **README.md claim: "two ML pipelines and three deliverables".** Corroborated. The three deliverables are D-Cat-b (support), D5.1, D-Cat-d.

8. **`fetch_gaia_xp.py` is the only script that explicitly does NOT apply the §6.4 preprocessing** — it only fetches raw coefficients. The preprocessing is applied by `apply_ye2024_xp.py` and then `build_pipeline1_features_stream1.py` / `emit_stream1_with_hermite.py`. Correct separation, but downstream code that bypasses these scripts and reads `xp_coeffs_raw.parquet` directly would silently skip preprocessing. No evidence any production code does this.

9. **Stream 2 consumer status.** data_overview.md §1: "Neither Pipeline 1 nor Pipeline 2 consumes it yet". Pipeline 1 feature builder imports and configs do not reference Stream 2. Corroborated.

10. **Scripts count — confirmed 31.** An earlier-session scratchpad summary had said 36 scripts; direct count of `.py` files under `scripts/` confirms 31. No code-vs-docs issue.

### 11.3 Invariant "Don't" rules vs code

| Rule | Check |
|---|---|
| Don't commit data to git | `data/` gitignored; all fetchers write there. |
| Don't hardcode credentials | `credentials.py` reads YAML + env var; no hardcoded tokens found. |
| Don't use astroquery.gaia | Data layer uses pyvo everywhere; astroquery.vizier only for Andrae+2023. |
| Don't query TAP sync > 5 000 rows | `tap.py` uses async `submit_job` and batched upload. |
| Don't skip Gaia corrections | `apply_gaia_corrections.py` is a prerequisite of every training feature matrix. |
| Don't reuse DR17 weights on DR19 data | No DR17 code path; Mészáros+2025 DR19-specific corrections applied. |
| Don't release Tier 3 per-star | `tier_promotion.py` decision tree classifies rejected/internal/T2/T1 only. |
| Don't select Pipeline 2 hyperparameters by visual inspection | `hyperparameter.py` grid search maximises DBCV, not persistence. |
| Don't report clusters without bootstrap stability | `diagnostics.py` implements bootstrap-ARI as test 1 of six. |
| Don't conflate FIRE-2 and real-data metrics | `hare_hounds.py` docstring explicitly gates this. |

No rule violations detected in the main-pipeline code path.

---

## 12. Known stubs, blockers, incomplete code (consolidated, as of 2026-04-19)

| Location | Status |
|---|---|
| `xp_abundances/main/audit.py` test 3 (SHAP values) | deferred; external lib needed |
| `xp_abundances/main/audit.py` test 6 (decorrelated subsample) | incomplete |
| `xp_abundances/main/tier_promotion.py` test 6 (cross-catalogue consistency) | stub |
| `xp_abundances/main/uncertainty.py:gp_smoothed_per_cell_per_label_scale` | **retained but rejected** — methodology finding; production uses `shrunken_per_cell_per_label_scale` |
| `tests/xp_abundances/experimental/` and `tests/population_classifier/experimental/` | empty |
| `xp_abundances/main/{config,sanity}.py` | no tests |

No NotImplementedError or explicit `raise` discovered outside these cases.

---

## 13. File / line inventory

| Group | File count | Line count range (approx.) |
|---|---:|---|
| `src/arqueogal/data/` | 25 | 20–800 L per module (heaviest: orchestrators and `gaia_xp.py`) |
| `src/arqueogal/xp_abundances/main/` | 14 | small (`adapter.py`) to large (`uncertainty.py`, `training.py`) |
| `src/arqueogal/population_classifier/main/` | 8 | 100–400 L per module |
| `src/arqueogal/utils/` | 6 | 80–250 L per module |
| `scripts/` | 31 | 87 L (`plot_meszaros…`) – 960 L (`run_calibration.py`) |
| `tests/` | 60 | 20–250 L per file |
| `configs/` | 4 | 73–102 L per YAML |
| `docs/` | 3 | `data_overview.md` 120 L, others longer |

Total tracked Python files: ≈ 84 in `src/` + 31 in `scripts/` + 60 in `tests/` = **175 files**. Add `tools/codegraph/codegraph.py` (132 L) and top-level `codegraph.py`.

---

## 14. How the pieces compose end-to-end (canonical paths)

### 14.1 Stream 1 training-data build

```
fetch_gaia_enrich_stream1.py        → stream1_gaia_dr3_raw.parquet
apply_gaia_corrections.py (raw→corr) → stream1_gaia_dr3_corrected.parquet
emit_apogee_interim.py               → apogee_dr19_precorrected.parquet
emit_apogee_corrected.py             → apogee_dr19_corrected.parquet         (Mészáros+2025)
build_stream1_apogee_gaia.py         → stream1_apogee_gaia.parquet
fetch_gaia_xp.py                     → xp_coeffs_raw.parquet
apply_ye2024_xp.py                   → xp_sampled_corrected.parquet         (Ye+2024 NN, LIVE)
build_pipeline1_features_stream1.py  → pipeline1_features_stream1.parquet    (+ dust priors + §6.4 steps 2–4)
emit_stream1_with_hermite.py         → pipeline1_features_stream1.parquet    (stage B, Hermite)
```

Every step writes a `.provenance.json` sidecar via `data/provenance.py`.

### 14.2 Pipeline 1 training-to-inference

```
run_pretraining_sanity.py      → sanity battery report
run_contrastive_pretrain.py    → pretrained encoder checkpoint
run_halfway_umap.py            → halfway gate (continuity ≥ 0.75)
run_supervised_finetune.py     → single-seed β-NLL model
run_ensemble.py                → M-member ensemble (sequential)
run_calibration.py             → per-(cell,label) calibration artifacts
run_ood_eval.py                → Mahalanobis OOD bundle + ensemble-disagreement flag
diagnose_{per_label_calibration,bias_location,halt_cells}.py  → escalation diagnostics
```

### 14.3 Stream 3 inference-data build (for downstream Pipeline 2 feeding)

```
select_stream3_stars.py          → stream3_selected.parquet (stratified on Teff, log g, [M/H], G)
fetch_gaia_enrich_stream3.py     → stream3_gaia_dr3_raw.parquet
apply_gaia_corrections.py        → stream3_gaia_dr3_corrected.parquet
fetch_bailerjones_stream3.py     → per_stream3.parquet
precompute_stream3_av.py         → stream3_av.parquet
```

Stream 3's Pipeline 1 inference feature matrix and Pipeline 2 feature matrix are **not yet built** in this workspace — this is flagged as the next deliverable-sequence step.

---

## 15. Pyproject and environment

- `pyproject.toml`: `arqueogal` 0.1.0, Python ≥ 3.12, MIT-ish (TBD per README). Minimal runtime deps; primary deps come from the shared `~/.venvs/rapids25.10_python3.12_cuda13/` venv.
- Ruff: line length 100. Ignores `N803`, `N806` (astronomical naming), `PLR2004` (physics constants).
- Pytest: `testpaths = tests`, `pythonpath = src`, `addopts = "-v --tb=short"`, markers include `slow`.
- Dev deps: ruff, pytest, pytest-cov.

Environment rules (project conventions, §Environment, §Hardware):
- Host: WSL2 Ubuntu, RTX 3060 6 GB VRAM.
- Python venv `~/.venvs/rapids25.10_python3.12_cuda13/`, activated via `rapidsenv` shell alias.
- No new venvs. No `pip install` that bumps cudf/cuml/numpy/pandas/pyarrow.
- Storage budget: 5 GB.
- GPU-only code paths must degrade gracefully via `torch.device("cuda" if cuda else "cpu")`.
- Portable to IA HPC `quasar → pc127` (CPU-only GT 1030).

---

## 16. Summary for the scanning reviewer

This codebase implements, end-to-end, the data acquisition + Pipeline-1 training machinery described in the project docs. Pipeline 2 code is present and structurally complete but its input feature matrix (`pipeline2_features.parquet`) is not yet built; Stream 3 Pipeline-1 inference is likewise not yet run. Pipeline 1 is at a frozen 5-label production-head configuration (locked 2026-04-19 per the in-file notes), with a full ensemble training + calibration + OOD + audit + tier-promotion path wired through the 10 `run_*`/`diagnose_*` scripts.

The most important things for a discrepancy scanner to verify against source:

1. **`data/gaia_xp.py:apply_ye2024_correction` is LIVE** (not stub). Docstring drift in this module and in `scripts/fetch_gaia_xp.py` was resolved 2026-04-19.
2. **GP-smoothing uncertainty calibrator is retained dead code**, not production. Production is `shrunken_per_cell_per_label_scale`.
3. **Three audit/tier-promotion subtests are stubs/deferred** (SHAP, decorrelated subsample, cross-catalogue consistency) — these are the only known completeness gaps in the release-gate code.
4. **Experimental test trees are empty** — a convention aspiration rather than an implementation gap, but flag it.
5. **No TODO/FIXME/NotImplementedError scattered across the main pipelines** beyond the four items above.
