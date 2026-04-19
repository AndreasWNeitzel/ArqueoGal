# `arqueogal.population_classifier.main` — Design

## Status

Production pipeline for **D5.1** (Dec 2026) and **D-Cat-d** (Feb 2027). The main tree
holds the frozen deliverable implementation; feature work is segregated under
`../experimental/` and does not import from `main/`.

See `../DESIGN.md` for pipeline-level scope/rules and `docs/research_brief.md` §10 for
science details.

## Module layout

```
arqueogal.population_classifier.main
├── features.py         — 10–11D feature vector construction from
│                          data/processed/pipeline2_features.parquet. Standardisation.
│                          k-NN imputation from Pipeline-1 where confident. Evolutionary-
│                          stage gating on [C/N].
├── embedding.py        — Parametric UMAP (main) + cuML UMAP baseline. Train on
│                          representative subsample, apply to full catalogue.
├── clustering.py       — HDBSCAN (cuML GPU / hdbscan CPU fallback). Soft memberships
│                          via `all_points_membership_vectors`. GLOSH outlier scores.
├── hyperparameter.py   — DBCV grid search (research_brief §10.4). Grid:
│                          n_neighbors ∈ {15,30,50,100,200}; min_dist ∈ {0.0,0.05,0.1};
│                          n_components ∈ {2,3,5}; min_cluster_size ∈ {50,100,200,500};
│                          min_samples ∈ {5,10,20}; cluster_selection_epsilon ∈ {0,0.1,0.2};
│                          cluster_selection_method ∈ {eom, leaf}.
│                          Log full search as part of D5.1 release.
├── diagnostics.py      — Six-tool stack (research_brief §10.5): bootstrap-ARI stability
│                          (N=500), DBCV per cluster, permutation-feature causal attribution,
│                          null-model (Gaussian + Gaussian copula, N=100), held-out-feature
│                          consistency, literature cross-reference
│                          (Dodd+2023 / Myeong+2019 / Horta+2021 / Ceccarelli+2024).
├── mc_ensemble.py      — N=50 MC over per-star feature posteriors (including calibrated σ
│                          from Pipeline 1). Mean + std of soft-membership distribution per
│                          star per cluster. Boundary-star flag at max<0.7 or std>0.15.
└── hare_hounds.py      — Subtask 5.1 FIRE-2 Ananke validation. Metrics: ARI, AMI,
                           Youden J / informedness, MCC against simulation ground truth.
                           Prefer Remus/Romulus; fall back m12f for Neitzel+2025 parity.
                           FIRE-2 ingestion via arqueogal.data.fire2_ananke.
```

## Feature vector (10–11D)

```
age, age_err                 (TESS seismology, Task 4 output; null for now)
fe_h, fe_h_err               (APOGEE if overlap, else Pipeline 1 Tier 1)
mg_fe, mg_fe_err             (APOGEE if overlap, else Pipeline 1 Tier 2)
al_fe, al_fe_err             (APOGEE if Tier 1/2 supports)
c_n, c_n_err                 (RGB only, evolutionary-stage-gated)
J_R, J_z, L_z                (galpy McMillan17 Staeckel)
ecc, r_peri, r_apo, z_max, E
```

Backward-compatibility baseline (Neitzel+2025 reproduction): `(V_φ, √(U²+W²))` retained as
auxiliary feature set; run both side-by-side in the release note.

## Algorithm choice (fixed)

1. **StandardScaler** fit on representative training subset.
2. **Parametric UMAP** (Sainburg, McInnes & Gentner 2021, arXiv:2009.12981). Trained on
   sub-sample; applied to full catalogue. Euclidean metric on scaled features.
3. **HDBSCAN** with `prediction_data=True` for soft memberships.
4. DBCV-maximised hyperparameters (no visual selection).
5. MC ensemble N=50 over Pipeline-1 feature uncertainty.
6. Bootstrap stability N=500 on central run.

## Release gates (hard)

- DBCV > 0 per released cluster.
- Bootstrap-ARI median > 0.5 per cluster (hard threshold); clusters with ARI < 0.5 are
  dropped. Stable clusters have ARI > 0.75.
- Soft memberships + GLOSH outlier scores released for every star.
- Boundary-star flag released.
- For D-Cat-d: MC mean + std of membership probability per (star, cluster). Uncertainty
  propagated end-to-end from Pipeline-1 calibrated σ. Stars failing Pipeline-1 calibration
  gates do not receive D-Cat-d memberships.
- Per-cluster "stability report card" in D5.1 documentation: DBCV, bootstrap ARI,
  permutation-causal attribution results, null-model rate, literature cross-matches.

## Tests

Under `tests/population_classifier/main/`. Smoke tests for DBCV on a synthetic Gaussian
mixture, bootstrap-ARI stability on same, Parametric UMAP save/load round-trip, MC ensemble
aggregation correctness, soft-membership normalisation (sum to 1 within tolerance).

## Non-goals (explicit)

- Persistence-score-based hyperparameter selection (research_brief §10.7).
- Visual hyperparameter selection.
- 2D-only clustering (clustering uses the higher-D UMAP output; 2D is for viz only).
- Any dependency on FIRE-2 for D-Cat-d science.
