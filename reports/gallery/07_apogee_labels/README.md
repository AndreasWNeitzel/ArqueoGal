# 07 — APOGEE labels (training targets)

**What this shows.** APOGEE DR19 ASPCAP labels after Mészáros+2025 [X/M]/
Teff corrections (mandatory per project invariant before use as training
targets). Also shows element-level NaN rates which drive the `mask=`
path in `beta_nll_block_cholesky`.

## Figures

| # | file | what to look at | status |
|--:|---|---|---|
| 01 | [panel_06_label_matrix.png](../../figures/data_overview/panel_06_label_matrix.png) | Per-element label coverage matrix. | existing |
| 02 | [apogee_dr19_diagnostics.png](../../figures/apogee_dr19_diagnostics.png) | APOGEE-specific diagnostics panel. | existing |
| 03 | [apogee_dr19_meszaros_deltas.png](../../figures/apogee_dr19_meszaros_deltas.png) | Mészáros+25 Δ[X/M] vs Teff per element — the correction applied. | existing |
| 04 | label_pairwise_hexbin.png | [M/H] vs [α/M], [M/H] vs [Fe/H], [α/M] vs [Mg/Fe] — chemical consistency. | batch 3 |
| 05 | label_nan_rates.png | Per-element NaN rate bar chart; α/M 0%, V ~5%, C/N ~10%. | batch 3 |

## Failure modes
- If `m_h_atm` and `fe_h_atm` disagree by > 0.05 dex on average, the DR19
  column renaming aliasing is biting — see `docs/data_acquisition.md §7.4`
  for the history (documented footgun).
- Meszaros Δ values larger than ~0.3 dex for some element means the star
  is outside the correction's Teff envelope — `corr_envelope_flag` handles
  this.
