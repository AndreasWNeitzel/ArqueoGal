# 00 — Data sources

**What goes in.** Three independent streams, ingested separately and never
cross-contaminated. Stream 1 trains the model; Stream 2 is held out for
information-content audit; Stream 3 is the deployment sample.

## Inputs
| stream | catalogue | N raw | N after cuts | role |
|---|---|---:|---:|---|
| 1 | APOGEE DR19 × Gaia DR3 XP | 733,901 | 354,003 | training + validation |
| 2 | Hon+2021 TESS RG × Gaia DR3 | 158,505 | 32,108 | held-out (audit only) |
| 3 | Andrae+2023 RGB × Gaia DR3 | 10,483,688 | 613,939 (vol 249 k + unif 365 k) | deployment |

## Figures

| # | file | what to look at | status |
|--:|---|---|---|
| 01 | [panel_01_data_flow.png](../../figures/data_overview/panel_01_data_flow.png) | Top-to-bottom flow from raw → feature matrix → model. | existing |
| 02 | [panel_02_sky_mollweide.png](../../figures/data_overview/panel_02_sky_mollweide.png) | Sky footprint of the training set. | existing |
| 03 | [panel_05_magnitude_hist.png](../../figures/data_overview/panel_05_magnitude_hist.png) | G-mag distribution; informs the G<17 XP-native cut. | existing |
| 04 | [panel_08_rowcount_waterfall.png](../../figures/data_overview/panel_08_rowcount_waterfall.png) | Stream-1 row-count waterfall: where stars are lost in the pipeline. | existing |
| 05 | [stream_sky_maps.png](stream_sky_maps.png) | Four Mollweide panels: Streams 1, 2, 3-volume, 3-uniform sky footprints. | existing |
| 06 | [stream_row_counts.png](stream_row_counts.png) | Row-count waterfall across the full pipeline (log y). | existing |

## Failure modes to watch for
- A stream with a sky footprint concentrated in one hemisphere means
  sample bias downstream. Stream 1 footprint skews toward the APOGEE
  pointings; Stream 3 is all-sky.
- If the row-count waterfall shows a cut dropping > 50%, that cut
  deserves its own sensitivity analysis — see `reports/stream1_waterfall.md`.
