# 13 — Ensemble + uncertainty

**What this shows.** 5-seed ensemble moment-match per star:
Σ̄ = mean(Σ_k) + between(μ_k). Aleatoric = diag(mean Σ); epistemic =
sqrt(diag between). OOD = Mahalanobis on 108-D XP block ∪ ensemble
disagreement threshold.

## Figures

| # | file | what to look at | status |
|--:|---|---|---|
| 01 | [pipeline1_joint_diagnostics.png row 4 cols 3 & 5](pipeline1_joint_diagnostics.png) | σ_α val-by-truth-[M/H] vs Stream-3 halo (col 3) and ensemble-disagreement per label (col 5) — the collapse-diagnosis panels. | existing |
| 02 | aleatoric_vs_epistemic_per_label.png | Scatter of aleatoric σ vs epistemic σ per star, 5 panels. Tells you whether uncertainty is model-limited or data-limited per label. | planned |
| 03 | ood_mahalanobis_distribution.png | Mahalanobis distance distribution with threshold line; val + Stream-3 overlaid. | planned |
| 04 | ood_disagreement_distribution.png | Per-star ensemble disagreement distribution with threshold. | planned |
| 05 | ood_joint_decision_plot.png | 2-D scatter: Mahalanobis vs disagreement, with both thresholds + joint flag region shaded. | planned |
| 06 | regime_b_envelope_footprint.png | Kiel diagram showing the regime-B exclusion region. | planned |

## Failure modes
- If epistemic σ is always tiny (< 1% of aleatoric), the ensemble collapsed
  to the same solution — retrain with more seed diversity. Happened to
  Stream-3 halo stars in v1.1 and drove the σ-gate.
- If Mahalanobis and disagreement are uncorrelated, the OOD gates are
  catching different things — both needed. If strongly correlated, one
  is redundant.
