# 14 — Pipeline-1 inference (Stream 3)

**What this shows.** Predictions on the Andrae+2023 RGB × Gaia DR3 XP
deployment sample. Two arms: volume-limited (249k, OOD-joint 10.1%)
and uniform (365k, OOD-joint 48.6%); union = 613,939 stars. Current
release: joint-loss rebuild (2026-04-22), superseding v1.1.

## OOD / support gates

Two complementary OOD gates are applied, both as additive parquet columns:

1. **`ood_joint_flag`** — 108-D XP-block Mahalanobis **OR** ensemble
   disagreement threshold (the original v1 gate, ellipsoidal support).
2. **`latent_support_flag`** *(2026-04-22)* — convex-hull surrogate: kNN-mean
   distance in the SupCon trunk's 32-D latent to the Stream-1 training
   reference exceeds the 99th-percentile threshold calibrated on the
   Stream-1 val split. Captures **non-convex concavities** in the training
   manifold that the Mahalanobis ellipse misses. Rationale: Sun et al. 2022,
   *Out-of-Distribution Detection with Deep Nearest Neighbors* (NeurIPS).

## Artefacts
- `data/processed/pipeline1_predictions_stream3_joint.parquet` — union
- `data/processed/pipeline1_predictions_stream3_joint_volume.parquet` — volume arm
- `data/processed/pipeline1_predictions_stream3_joint_uniform.parquet` — uniform arm
- `data/processed/pipeline1_latent_support_stream3.parquet` — per-star
  `latent_knn_dist` + `latent_support_flag` (standalone gate output)

## Figures

| # | file | what to look at | status |
|--:|---|---|---|
| 01 | [stream3_predictions.png](stream3_predictions.png) | 8-panel: HRD + chemistry planes ([M/H]–[α/M], [M/H]–[Mg/H]) side-by-side volume vs uniform arm, OOD sky-rate Mollweide, σ_α distribution. | existing |
| 02 | [latent_support_diagnostics.png](latent_support_diagnostics.png) | 6-panel: latent kNN-dist distributions (val threshold + Stream-3), latent × OOD-joint overlap confusion, sky-rate Mollweide, chemistry colour-coded by flag. | existing |
| 03 | stream3_pred_sky.png | Sky-map of each predicted label; exposes systematic sky gradients. | planned |
| 04 | stream3_regime_b_sky.png | Regime-B flag rate per sky pixel (|b|<5° + warm-upper-RGB). | planned |
| 05 | stream3_aux_missingness.png | `ir_missing_flag` + `extinction_missing_flag` per pixel. | planned |

## Failure modes
- A chemistry plane with a dense ribbon at fixed [α/M] for |M/H|≈-1 is
  the prior-collapse signature that motivated the v1.2 σ-gate.
- `ood_joint_flag` rate > 20% in a region means the model is outside its
  training manifold **in the input XP distribution**; do not release
  per-star predictions there.
- `latent_support_flag` = True means the star's trunk embedding is far
  from any training cluster in the learned representation — often catches
  shape-space drift that the 108-D Mahalanobis misses. Expected to overlap
  meaningfully with `ood_joint_flag` but not fully coincide; stars flagged
  by both are the strongest OOD signal.
