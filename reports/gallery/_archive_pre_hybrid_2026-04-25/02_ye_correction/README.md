# 02 — Ye+2024 flux correction

**What this shows.** Ye+2024 (Zenodo 14028588) publishes a NN that
corrects XP *flux samples* against synthetic photometry. Applied to all
stars ingested; `NO_SYNTH_PHOT` flag stars are excluded.

**Order of operations** (per `docs/data_acquisition.md §6.4`):
1. Ye+2024 NN flux correction
2. Normalise coefs 1–54 by coef 0
3. log + z-score coef 0
4. per-coefficient z-scoring at emit

## Figures

| # | file | what to look at | status |
|--:|---|---|---|
| 01 | [ye_before_after_sed.png](ye_before_after_sed.png) | Six random Stream-1 stars — raw gaiaxpy calibrate flux vs post-Ye dereddened flux. λ<420 nm shaded (correction is largest there). | existing |
| 02 | [ye_delta_distribution.png](ye_delta_distribution.png) | Δflux / peak vs wavelength, 2/16/50/84/98 envelopes across 60 stars. Quantifies the wavelength-dependent correction magnitude. | existing |
| 03 | [ye_flag_sky_map.png](ye_flag_sky_map.png) | Two-panel sky split: Ye=OK (left) vs flag≠OK (right) across Stream 1. Counts in titles. | existing |
| 04 | [ye_flag_vs_g.png](ye_flag_vs_g.png) | Left: flag distribution vs G-mag. Right: NO_SYNTH_PHOT rate vs \|b\|, with the low-|b|<5° regime marked. | existing |

## Failure modes
- If the Δ histogram has bimodal structure concentrated at extreme G, the
  Ye NN is extrapolating — flag those stars and check against
  `reports/pipeline1/ir_fetch_stream3_existing.md` for the NO_SYNTH_PHOT
  selection-function treatment.
- A sky-map hot spot at |b|<5° is expected (high extinction + missing
  synthetic photometry); do NOT correct it by patching the NN.
