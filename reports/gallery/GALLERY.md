# ArqueoGal Pipeline-1 Visual Gallery (hybrid release v5)

**Purpose.** Every stage of the deployment graph has a visualisation here.
Walk top-to-bottom and you see exactly what the pipeline does at every step,
from the TAP fetch of raw Gaia DR3 records through to the final hybrid
catalog. The gallery is the auditable evidence that the deployment is
producing what we claim — when a stage's plots look wrong, the deployment is
wrong, and you know which script to fix.

**Canonical model:** strong-contrastive-v2 ensemble (SupCon=1.0 + β-NLL=1.0
+ Barlow=0.5, single seed) plus the latent-kNN rescue, composed by
`release_pipeline.attach_hybrid_columns` into the v5 hybrid surface.

**Schema:** v5. Reproduces from `bash scripts/run_full_pipeline.sh` plus
`bash scripts/gallery/build_all.sh`. Build time: ~15 min on RTX 3060.

---

## Stages, in deployment order

| # | Stage | What the deploy step is | What we plot | Source script | Plot script |
|---:|---|---|---|---|---|
| 00 | Source coverage | Stream-1 (APOGEE × Gaia, ~324 k) and Stream-3 (Andrae+23 RGB × Gaia DR3, ~614 k) row counts and sky maps | sky Mollweide + cumulative-G histogram per stream | `data/ingest_stream1.py`, `ingest_stream3.py` | `plot_00_source_coverage.py` |
| 01 | Gaia DR3 corrections | Lindegren+21 parallax zpt + Riello+21 G-mag cubic | Δparallax vs G; Δ(G_corr − G_raw) vs colour | `data/gaia_corrections.py`, `scripts/apply_gaia_corrections.py` | `plot_01_gaia_corrections.py` |
| 02 | Raw Gaia XP | 55 BP + 55 RP Hermite coefficients per star | per-coef distribution; example-star SED reconstructions across HRD cells | `data/ingest_xp.py`, `scripts/fetch_gaia_xp.py` | `plot_02_gaia_xp_raw.py` |
| 03 | Ye+2024 NN flux correction | per-star multiplicative correction against synthetic photometry | before/after SED stacks; Δcoef distribution; NO_SYNTH_PHOT flag map | `data/gaia_xp.py`, `scripts/apply_ye2024_xp.py` | `plot_03_ye_correction.py` |
| 04 | Hermite normalisation + frozen z-score | coefs 1-54 / coef 0; log+z-score on coef 0; per-coef z-score from frozen v1 stats (`0d34b565...`) | per-coef Hermite z-score histograms (Stream 1 vs Stream 3, frozen-stats verified) | `data/frozen_stats.py` | `plot_04_hermite_zscore.py` |
| 05 | Distance fusion | Bailer-Jones+2021 photogeo + Edenhofer+24 / Lallement+22 / SFD A_V cascade | per-source distance histograms; A_V map stacked by source; budget compliance | `data/distances.py`, `data/dust_maps.py` | `plot_05_distance_extinction.py` |
| 06 | IR photometry join | 2MASS JHK + AllWISE W1/W2 per Stream-1 / Stream-3 source | coverage sky map; missingness vs G; J-K vs G-K colour-colour | `data/ir_photometry.py`, `scripts/fetch_ir_photometry.py` | `plot_06_ir_photometry.py` |
| 07 | APOGEE DR19 labels + Mészáros+2025 [X/M] correction | RGB-only validity guard; Table 3 polynomial corrections for 14 elements | per-element NaN rate; Mészáros Δ vs Teff per element; pairwise hexbin | `data/apogee_dr19.py` | `plot_07_apogee_labels.py` |
| 08 | Stream-1 APOGEE × Gaia join | many-to-one tie-break on Δmag within 300 mas / 0.1 mag (ADR-0001) | merge cardinality histogram; multi-spectrum dedup statistics | `data/ingest_stream1.py`, `data/dedup.py`, `scripts/build_stream1_apogee_gaia.py` | `plot_08_stream1_join.py` |
| 09 | Selection function ω(s) | Ye × IR × ϖ × A_V (per-factor) | per-factor decomposition; ω(s) total sky map; ω vs G | `data/selection_function.py`, `scripts/build_selection_function_v1.py` | `plot_09_selection_function.py` |
| 10 | Kinematic enrichment | Galpy actions (J_R, L_z, J_z), eccentricity from astrometry + RV | E vs L_z plane; action diagram; orbit-family fractions | `data/enrich_kinematics.py`, `data/kinematics.py` | `plot_10_kinematics.py` |
| 11 | Geometry enrichment | Galactocentric (X, Y, Z, R_gal, v_φ) | R_gal vs Z scatter; (X, Y) Galactic plane projection | `data/enrich_geometry.py` | `plot_11_geometry.py` |
| 12 | Pipeline-1 feature matrix | 110 XP + 43 aux features per star, post-frozen-stats | feature distribution panels (XP block + aux block); correlation heatmap; layout schema | `scripts/build_pipeline1_features_stream1.py`, `build_pipeline1_features_stream3.py` | `plot_12_feature_matrix.py` |
| 13 | Strong-contrastive-v2 training | single-stage SupCon=1.0 + β-NLL=1.0 + Barlow=0.5, full 290k Stream-1, 12 epochs, 1 seed | training curves (4 loss components); grad-norm + τ stability | `scripts/run_ensemble.py` (now defaults to strong-contrastive-v2 recipe) | `plot_13_training.py` |
| 14 | Stream-3 inference (regressor) | strong-contrastive-v2 forward pass on 614k Stream-3 | per-element σ histograms; pred-vs-G; σ-inflation rate per element | `scripts/run_pipeline1_inference.py`, `xp_abundances/main/inference.py` | `plot_14_regressor_inference.py` |
| 15 | Latent-kNN rescue | encoder z + GPU cosine kNN (K=50) against Stream-1 training pool | distance-to-top-1 distribution; kNN-IQR vs σ_regressor per element | `scripts/run_knn_rescue.py`, `xp_abundances/main/knn_rescue.py` | `plot_15_knn_rescue.py` |
| 16 | OOD gates | Mahalanobis on 108-D XP + dual-Mahalanobis on aux + latent support | per-gate ROC/score distributions; aux-vs-XP scatter coloured by joint flag | `xp_abundances/main/ood.py`, `scripts/build_latent_support_gate.py` | `plot_16_ood_gates.py` |
| 17 | Hybrid composer | regressor when σ ≤ threshold; kNN-median when σ > threshold AND kNN finite; regressor_caveat otherwise | per-element source split (stacked bar); per-element hybrid_tier counts | `data/release_pipeline.py:attach_hybrid_columns` | `plot_17_hybrid_composer.py` |
| 18 | Hybrid Kiel + chemistry on Stream 3 | the catalog's user-facing planes | hybrid Kiel (Tier-1) split by source; hybrid chemistry plane faceted by tier | (same parquet as 17) | `plot_18_hybrid_inference_planes.py` |
| 19 | Release-tier composition | composite + per-element; v3 + v4 caveat flags | regime diagnostics: tier composition vs G / Av / distance / sky / HR / chem; σ-by-tier; flag-contribution | `xp_abundances/main/release.py`, `data/release_pipeline.py:run_release_pipeline` | `plot_19_release_tier_regime.py` |
| 20 | Pred-vs-truth on splits | 70/15/15 train/val/test, LOO kNN on train, frozen kNN on val/test | 3 splits × 5 labels hexbin scatter (n / RMSE / bias / std) + Kiel & chem truth-vs-pred | (recomputed in-script) | `plot_20_pred_vs_truth_splits.py` |
| 21 | 3-component GMM cluster tracking | structure-preservation criterion 2 of methods.md §3.6 | truth-derived GMM colours followed into pred plane; centroid drift, ARI, purity | (recomputed in-script) | `plot_21_gmm_cluster_tracking.py` |
| 22 | Per-class contamination | criterion 3 | confusion matrix + precision/recall/F1 + flow + Hellinger/TV per cluster | (recomputed in-script) | `plot_22_contamination.py` |
| 23 | Hybrid stress battery | 7-test integration suite | summary panel: 5-fold CV, σ-coverage, K-sensitivity, leakage Δ, permutation importance, multi-spectrum consistency | `tests/integration/test_hybrid_stress_battery.py` (run with `--run-stress`) | `plot_23_stress_battery.py` |

Stages 15-16 (population classification) moved to the **Starfold** repository
on 2026-04-22; the historical artefacts live under
`reports/gallery/_archive_pre_hybrid_2026-04-25/`.

---

## How to rebuild from scratch

1. Run the full deploy: `bash scripts/run_full_pipeline.sh`
2. Run the gallery: `bash scripts/gallery/build_all.sh`

Each plot script reads only the canonical artefacts that the deploy produced,
so a wrong plot means a wrong artefact, which means a wrong deploy step. The
gallery is the audit surface.
