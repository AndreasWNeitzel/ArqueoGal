# 05 — IR photometry

**What this shows.** 2MASS JHK + AllWISE W1 W2 magnitudes, cross-matched
via VizieR TAP. Used as auxiliary feature and for Teff priors.

## Figures

| # | file | what to look at | status |
|--:|---|---|---|
| 01 | [ir_coverage_sky.png](ir_coverage_sky.png) | Two-panel Mollweide: IR-complete (all of JHK+W1W2) vs IR-missing (any band). | existing |
| 02 | [ir_missing_vs_g.png](ir_missing_vs_g.png) | Left: total vs IR-missing per G-bin. Right: missingness rate vs G with overall-mean baseline. | existing |
| 03 | [ir_color_color.png](ir_color_color.png) | J−K vs W1−W2 hexbin for Stream 3. | existing |
| 04 | [ir_magnitude_distributions.png](ir_magnitude_distributions.png) | J / H / K / W1 / W2 mag-count histograms (log y). | existing |

## Failure modes
- AllWISE bright-star saturation (W1 < 8) produces biased W1 mags — the
  `allwise_xm_quality_flag` handles this.
- Crowded-field 2MASS contamination shows up as anomalously large
  `tmass_angular_distance` — filter by the 300 mas cross-match radius.
