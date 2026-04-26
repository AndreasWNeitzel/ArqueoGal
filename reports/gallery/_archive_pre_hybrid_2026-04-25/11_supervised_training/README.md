# 11 — Supervised training

**What this shows.** Single-stage joint-loss training (SupCon + β-NLL +
Barlow Twins) of the full XP→5-label model. 5-seed ensemble, 200 epochs
per seed, momentum queue (size=8192). Replaces the v1/v1.1 two-stage
contrastive-pretrain → fine-tune pipeline.

## Figures

| # | file | what to look at | status |
|--:|---|---|---|
| 01 | [pipeline1_joint_diagnostics.png](pipeline1_joint_diagnostics.png) | 20-panel figure; row 4 col 1 is per-seed training / val loss curves. | existing |
| 02 | [val_truth_vs_pred.png](val_truth_vs_pred.png) | Side-by-side truth vs pred: Kiel + chemistry planes + per-label histograms. | existing |
| 03 | joint_loss_decomposition.png | Per-seed per-epoch trajectories of supcon / β-NLL / Barlow sub-losses. | planned |
| 04 | tau_schedule.png | Learned SupCon temperature τ per seed vs epoch. | planned |
| 05 | grad_norm_trace.png | Pre-clip gradient max + mean per seed vs epoch. | planned |

## Failure modes
- Pre-clip grad-norm > 1000 sustained across multiple epochs means the
  β-NLL block-Cholesky head is unstable (check for NaN labels leaking
  past the `mask=`). Mid-warmup spikes to ~500 are normal under the
  joint recipe.
- Per-seed best val-loss spread > 10% of the mean means the ensemble is
  not coherent; retrain with seed-split audit. Current run: spread 0.017
  / mean 3.14 = 0.5% — tight.
- SupCon τ collapsing below 10⁻³ means the contrastive loss saturated
  (learned temperature bounds are 10⁻³, 0.5); rethink the label-space
  Gaussian-kernel σ.
