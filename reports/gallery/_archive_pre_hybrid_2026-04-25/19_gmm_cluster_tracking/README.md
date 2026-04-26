# Stage 19: 3-component GMM cluster tracking on chemistry

Structure-preservation diagnostic for the kNN+strong-contrastive-v2 hybrid.
Fits a 3-component GMM on the *truth* chemistry plane ([M/H], [α/M]) and
tracks the resulting hard-assigned labels into the predicted chemistry plane.

## Figures

- **`gmm_cluster_tracking.pdf`** — 3 splits × 3 columns:
  - col 0: truth chemistry coloured by truth-derived GMM labels;
  - col 1: pred chemistry coloured by the **same truth-derived label** so
    the reader can see how each star moves under prediction;
  - col 2: pred chemistry coloured by an independent GMM re-fit on the
    pred plane, Hungarian-aligned to the truth components by centroid
    distance.
- **`gmm_cluster_tracking_metrics.json`** — per-split quantitative metrics:
  - `centroid_drift_per_comp` and `_total_rms` (in dex);
  - `adjusted_rand_index` (truth-clustering vs pred-refit);
  - `purity_truth_vs_pred_assignment` (membership preservation);
  - `within_cluster_pred_rms_chem` (per-cluster scatter in pred plane);
  - `n_per_cluster`.

## How it's built

`scripts/gallery/plot_19_gmm_cluster_tracking.py`. Same encoder + kNN
infrastructure as stage 18; uses scikit-learn `GaussianMixture(n_components=3,
covariance_type="full", random_state=20260425, n_init=5)` on both planes,
then `scipy.optimize.linear_sum_assignment` for Hungarian alignment.

## Why it lives here

Answers the second of the three release-criteria: structure preservation. A
model that minimises per-element RMSE alone (stage 18) but smashes the disc
chemical bimodality fails the science use case. The ARI ≈ 0.566 and
membership purity ≈ 0.870 reported in the JSON sidecar are the empirical
quantification.
