# 12 — Pipeline-1 validation

**What this shows.** Performance on held-out Stream 1 val split
(n=41,851, split seed locked to ensemble config). Tells you how well
the model does **on-distribution**. Out-of-distribution (Stream-3
deployment) performance is in stage 14.

## Figures

| # | file | what to look at | status |
|--:|---|---|---|
| 01 | [val_truth_vs_pred.png](val_truth_vs_pred.png) | **Side-by-side truth vs pred:** Kiel diagram, [M/H]–[α/M] chemistry, [M/H]–[Mg/H] chemistry, per-label density histograms. | existing |
| 02 | [pipeline1_joint_diagnostics.png](pipeline1_joint_diagnostics.png) | 20-panel consolidated figure (rows 1-3 live here; row 4 overlaps stages 11 and 13). | existing |
| 03 | [reliability_precal_joint.png](reliability_precal_joint.png) | Pre-calibration reliability per label. | existing |
| 04 | [reliability_postcal_joint.png](reliability_postcal_joint.png) | Post-calibration reliability (shrinkage τ=50). | existing |
| 05 | [reliability_gp_joint.png](reliability_gp_joint.png) | GP-smoothed α-calibration — methodology only, NOT production (retained-but-rejected per ADR-0003). | existing |
| 06 | residual_by_g_mag.png | Residual mean + std vs G-mag bin, per label. | planned |
| 07 | residual_by_sky.png | Residual sky-map per label (Mollweide). | planned |
| 08 | residual_by_teff_logg.png | 2-D Kiel-cell bias heatmap per label. | planned |

## Failure modes
- A residual sky-map that mirrors the training-set footprint gradient
  means the model learned sky position; domain-shift red flag.
- A systematic logg bias at the warm upper-RGB edge is regime-B (see
  ADR-0004); gated out via `RegimeBEnvelope`.
