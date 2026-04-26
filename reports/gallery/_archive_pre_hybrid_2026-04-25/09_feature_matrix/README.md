# 09 — Feature matrix (Pipeline-1 NN input)

**What this shows.** Final NN input tensor: 110 Hermite XP coefs + 43 aux
features (A_V, parallax, IR magnitudes, missingness flags). Schema contract
lives in `src/arqueogal/xp_abundances/main/DESIGN.md`.

## Figures

| # | file | what to look at | status |
|--:|---|---|---|
| 01 | feature_layout_schema.png | Schematic of tensor layout: the FeatureLayout contract, visualized. | batch 3 |
| 02 | feature_distributions_xp.png | 10×11 panel of all 110 XP coefs post-z-score; should be ~N(0,1). | batch 3 |
| 03 | feature_distributions_aux.png | 7×7 panel of aux features; missingness flags and physical-unit features. | batch 3 |
| 04 | feature_correlation_heatmap.png | 153×153 correlation matrix, blocked by XP / aux; diagonal band + aux block. | batch 3 |

## Failure modes
- A coef with a wide or skewed distribution post-z-score means z-scoring
  failed for that coef (frozen-stats mismatch, see stage 03).
- Non-zero correlations between XP and aux features are fine; suspicious
  correlations inside aux (e.g. A_V strongly correlated with parallax) are
  not, and deserve inspection.
