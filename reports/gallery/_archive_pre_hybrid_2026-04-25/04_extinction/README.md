# 04 — Extinction

**What this shows.** A_V is composed from four sources, per distance regime
(per `docs/data_acquisition.md §8`):
- **Edenhofer+2024** (d < 1.25 kpc, highest precision)
- **Lallement+2022** (1.25–3 kpc, cross-check only per ADR #91)
- **SFD** (asymptotic beyond 3 kpc; low-|b| caveats per the no-Bayestar19 invariant)
- **GSP-Phot neighborhood-median** (fallback for stars missing all three)

Provenance per star in `data/interim/stream3_av.parquet`; `av_los_source`
column names which one was used.

## Figures

| # | file | what to look at | status |
|--:|---|---|---|
| 01 | [panel_07_extinction_priors.png](../../figures/data_overview/panel_07_extinction_priors.png) | Extinction prior distributions on the training corpus. | existing |
| 02 | [av_map_stack_stream3.png](av_map_stack_stream3.png) | Four A_V sources side-by-side on Mollweide (Edenhofer / Lallement / SFD / nbhd-median) with valid-fraction and median A_V per panel. | existing |
| 03 | [av_source_breakdown_stream3.png](av_source_breakdown_stream3.png) | Bar chart of primary `av_los_source` + sky map coloured by source. | existing |
| 04 | [av_histograms_by_source.png](av_histograms_by_source.png) | A_V distribution histograms overlaid across all four sources (log y). | existing |
| 05 | [av_scatter_edenhofer_vs_lallement.png](av_scatter_edenhofer_vs_lallement.png) | Stream-1 Edenhofer-vs-Lallement hexbin + Δ(Lallement−Edenhofer) vs distance with 1.25 / 3 kpc horizons marked. | existing |

## Failure modes
- If two sources disagree by > 0.3 mag on the same star, check
  `av_los_source` priority: Edenhofer wins by construction.
- SFD at |b|<5° overpredicts A_V (no background subtraction in the
  Galactic plane) — the low-|b| dust-stack invariant hard-flags this regime.
- A sky-map hot spot where `av_los_source == "fallback"` dominates means
  the 3-D dust cubes don't cover that region — expected at high-b
  faint-star volumes.
