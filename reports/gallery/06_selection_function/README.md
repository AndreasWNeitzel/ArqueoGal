# 06 — Selection function

**What this shows.** ω_i = ω_Ye · ω_IR · ω_ϖ · ω_Av per star. Composed in
`src/arqueogal/data/selection_function.py`. Used for downstream density-
aware diagnostics and as a per-star weight in population-level summaries.
**NOT** used as a training sample weight (research_brief §11.2).

## Figures

| # | file | what to look at | status |
|--:|---|---|---|
| 01 | [omega_total_sky.png](omega_total_sky.png) | omega_total on Mollweide for Stream 3. Magma colorbar [0, 1]. | existing |
| 02 | [omega_components.png](omega_components.png) | Four-panel sky decomposition: P_ye / P_ir / P_parallax / P_extinction with per-panel mean. | existing |
| 03 | [omega_vs_g.png](omega_vs_g.png) | Mean omega per G-mag bin with overall baseline. | existing |
| 04 | [omega_histograms.png](omega_histograms.png) | Left: per-component histograms overlaid (log y). Right: omega_total distribution with mean marker. | existing |

## Failure modes
- If ω_Ye dominates everywhere, the Ye selection is swamping the other
  factors — check the NO_SYNTH_PHOT clip and reconsider the ω composition
  (multiplicative vs additive).
- A low-ω region that's also a high-population region (e.g. Galactic
  plane) means the deployment sample is biased against it — flag in any
  population-level result.
