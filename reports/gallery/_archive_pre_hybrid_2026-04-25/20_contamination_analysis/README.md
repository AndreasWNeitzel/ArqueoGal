# Stage 20: Per-class contamination analysis

Contamination diagnostic for the kNN+strong-contrastive-v2 hybrid surface.
Builds on stage 19's truth-vs-pred GMM clusterings and quantifies the
truth-Gi ↔ pred-Gj preservation in two complementary ways: discrete (confusion
matrix + per-class precision/recall/F1) and continuous (Hellinger / Total
Variation distance between truth-Gi and pred-Gi 2-D densities).

## Figures

- **`contamination_analysis.pdf`** — 3 splits × 4 panels:
  - **col 0** Confusion matrix (row-normalised), 3 × 3, with cell counts
    + percentages annotated. Reads "of stars truth-classified as Gi, what
    fraction were pred-classified as Gj?"
  - **col 1** Per-class precision (purity), recall (completeness), F1 bars,
    with macro-F1 annotated.
  - **col 2** Stacked-bar contamination flow: for each truth class, what
    fraction lands in pred-G1 vs pred-G2 vs pred-G3.
  - **col 3** Per-cluster Hellinger and TV distances between truth-Gi and
    pred-Gi 2-D densities on a 64 × 64 chemistry grid (catches centroid +
    shape preservation jointly).
- **`contamination_metrics.json`** — per-split quantitative metrics:
  - `confusion_matrix`, `n_per_truth_class`, `n_per_pred_class`;
  - `completeness_recall`, `purity_precision`, `f1_per_class`, `macro_f1`;
  - `flow_pct_truth_to_pred` (3 × 3 row-normalised);
  - `hellinger_per_class`, `tv_per_class`;
  - `ari` (cross-reference with stage 19).

## How it's built

`scripts/gallery/plot_20_contamination_analysis.py`. Same encoder + kNN +
GMM infrastructure as stages 18-19. Hellinger between two 2-D histograms
on a 64 × 64 chemistry grid: `H(p, q) = sqrt(0.5 × Σ (sqrt(p) - sqrt(q))²)`.

## Why it lives here

Answers the third of the three release-criteria: contamination. A model that
preserves global structure (stage 19, ARI = 0.566) can still mis-route stars
between adjacent populations. The dominant contamination flow on this model
is G2 (mid disc) → G3 (thin disc) at ≈ 33.7 % per split — consistent with
the regression head defaulting to the high-density [α/M]-poor disc when in
prior collapse. macro-F1 = 0.819 across all splits.
