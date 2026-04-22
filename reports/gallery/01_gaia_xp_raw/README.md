# 01 — Gaia XP (raw)

**What this shows.** Each DR3 XP source supplies 55 BP + 55 RP Hermite
coefficients. Coef 0 is the absolute flux scale; coefs 1–54 are the shape.
This stage visualizes raw coefs before any corrections.

## Figures

| # | file | what to look at | status |
|--:|---|---|---|
| 01 | [c0_vs_g.png](../../figures/hermite_smoke/c0_vs_g.png) | c0 (flux scale) vs G magnitude: tight power-law = sanity check. | existing |
| 02 | [xp_sed_atlas_by_hrd.png](xp_sed_atlas_by_hrd.png) | 4×4 grid of Stream-1 post-Ye SEDs stratified by (Teff, logg) cell, coloured by [M/H]. Shows the SED signal that Pipeline-1 is decoding. | existing |
| 03 | [xp_coef_distributions.png](xp_coef_distributions.png) | Median + 16/84 + 2/98 envelopes for 55 BP and 55 RP coefficients across 30 k Stream-1 stars. | existing |
| 04 | [xp_example_stars.png](xp_example_stars.png) | Six representative SEDs spanning the (Teff, [M/H]) box, each labelled with APOGEE Teff / logg / [M/H]. | existing |
| 05 | [noise_floor.png](../../figures/hermite_smoke/noise_floor.png) | Coef error magnitude vs G — sets the SNR floor. | existing |

## Failure modes
- If a coefficient shows a strongly skewed distribution at bright G, it
  probably holds a saturation artifact — check against `bp_n_measurements` /
  `rp_n_measurements` stratification.
- Coefs 50+ at G > 17 are noise-dominated; the G < 17 XP-native cut exists
  to avoid training on them.
