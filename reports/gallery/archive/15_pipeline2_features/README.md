# 15 — Pipeline-2 features

**What this shows.** 8-D feature matrix for the population classifier:
3 chemistry ([M/H], [α/M], [Mg/M]) + 5 kinematics (J_R, J_z, L_z, ecc, E).
v1 used volume-limited raw predictions; v1.1 used v1.1 ensemble
predictions; v1.2 adds the σ-gate that removes Pipeline-1 prior-collapsed
stars.

## Figures

| # | file | what to look at | status |
|--:|---|---|---|
| 01 | feature_pairplot_8d.png | 8×8 triangular pair plot, contours + stratified by label. | batch 6 |
| 02 | feature_distributions_1d.png | Per-feature histogram; v1 vs v1.1 vs v1.2 overlay. | batch 6 |
| 03 | sigma_gate_drop_rates.png | Fraction dropped by the σ-gate vs reported σ_α; shows threshold choice rationale. | batch 6 |
| 04 | sigma_gate_before_after_chemistry.png | [M/H]-[α/M] before and after σ-gate; where the prior-collapse pocket was. | batch 6 |

## Failure modes
- A strong correlation inside the 8 features (e.g. [M/H] ↔ E) makes UMAP
  find trivial structure. Expected, but interpret downstream clusters
  carefully.
- A σ-gate drop rate above 20% in one [M/H] bin means Pipeline-1 is
  reliably uncertain there — flag in the methods-paper as a
  domain-shift residual.
