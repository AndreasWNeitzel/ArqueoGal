# 16 — Pipeline-2 classification

**What this shows.** cuML UMAP (8-D → 2-D) + HDBSCAN-DBCV grid. Grid is
tight-by-default (12 cells, `mcs ∈ {200,500,1000} × ms ∈ {10,20} × ε ∈ {0,0.1}
× eom`). Winner maximises DBCV (Moulavi+14); visual tuning was a hard
invariant of the Pipeline 2 methodology (now lives in Starfold).

## Figures

| # | file | what to look at | status |
|--:|---|---|---|
| 01 | [pipeline2_v1_diagnostics.png](../../pipeline2/figures/pipeline2_v1_diagnostics.png) | v1 (pipeline1-v1 features) — DBCV=+0.265. | existing |
| 02 | [pipeline2_v11_diagnostics.png](../../pipeline2/figures/pipeline2_v11_diagnostics.png) | v1.1 (inverse-freq Pipeline-1) — DBCV=+0.180. | existing |
| 03 | [pipeline2_v12_diagnostics.png](../../pipeline2/figures/pipeline2_v12_diagnostics.png) | v1.2 (σ-gated) — DBCV=-0.004; honest classification. | existing |
| 04 | umap_embedding_by_feature.png | UMAP 2-D colored by each of the 8 features (8 panels). Dissociates kinematics-driven vs chemistry-driven structure. | batch 6 |
| 05 | hdbscan_grid_dbcv_heatmap.png | DBCV heatmap over (mcs × ms × ε); winner cell highlighted. | batch 6 |
| 06 | chemical_plane_evolution.png | Side-by-side: v1 / v1.1 / v1.2 chem plane. Tells the σ-gate story visually. | batch 6 |
| 07 | cluster_medians_table.png | Per-cluster median of each feature across v1/v1.1/v1.2; tabular view. | batch 6 |

## Failure modes
- A "halo cluster" that has [α/M] dispersion below ~0.05 dex is
  prior-collapse masquerading as a population — v1/v1.1 both shipped with
  this artifact. v1.2's σ-gate exposes it honestly at the cost of DBCV.
- Noise fraction jumping > 10% from one version to the next means the
  input feature distribution shifted — check σ-gate drop accounting.
