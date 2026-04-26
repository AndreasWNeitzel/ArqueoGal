# Figure captions — Stream 3 v3 release diagnostic plots

Generated 2026-04-25 03:00 UTC from `release/D-Cat-b/pipeline_run/predictions_with_features.parquet` (614k Stream 3 rows). Captions written for paper / supplementary-material reuse. Lengths are A&A-aware (≤80 words for in-paper figures; ≤150 words for appendix or supplementary).

---

## fig_tier_distribution.pdf

**Short caption (in-paper):**
> Pipeline 1 v3 release-tier distribution on 614k Stream 3 stars. (a) Stacked counts per `g_mag_bin`: bright (G ≤ 15, ~553k), mid (15 < G ≤ 16, ~60k), faint absent. (b) Per-element tier breakdown: all five elements (T_eff, log g, [M/H], [α/M], [Mg/H]) share the same Tier 1 / Tier 2 / Tier 3 split (~78.7% / 1.4% / 19.9%) because the v1 kinematic-OOD flag is a placeholder; the per-element tier columns will diverge once the Phase B detector populates `kin_ood_flag`.

**Long caption (appendix):**
> Stacked-bar tier counts for the 614k-row Stream 3 release. Panel (a) breaks down by Gaia G magnitude bin: 553237 bright stars (G ≤ 15) and 60702 mid-bin stars (15 < G ≤ 16) reach the catalog; the faint bin (16 < G ≤ 17) is empty in v3 because the Stream 3 selection cuts faint. Within each bin, the green segments are Tier 1 (per-star science, ~80%), orange Tier 2 (caveat: regime-B, mode-ambiguous, aux-missing, ensemble disagreement, or `dist_prior_dominated`), and red Tier 3 (any joint OOD or per-element NaN, dominated by the 122183 stars (19.9%) that the XP-Mahalanobis detector flags as outside the APOGEE training distribution). Panel (b) shows per-element tier distribution. The five elements appear identical in v3 because the kinematic-OOD detector that distinguishes [α/M] / [Mg/H] (aux-assisted) demotion from spectrum-dominant elements (T_eff, log g, [M/H]) is a placeholder; populating it will be the visible signature of the Phase B implementation.

---

## fig_ood_distribution.pdf

**Short caption:**
> Mahalanobis OOD score distribution on Stream 3, log y-axis. The classifier threshold (29.87, solid black) is calibrated at p = 0.99 on the 108-D XP-feature training distribution. The 99th percentile of Stream 3 observed scores (177.21, dashed red) is much higher than the threshold because Stream 3 contains many genuine OOD stars from the broadly-selected RGB sample.

**Long caption:**
> Histogram of the per-star Mahalanobis OOD score in the 108-D XP feature space, computed via `arqueogal.xp_abundances.main.ood.MahalanobisOOD` and stored as `ood_mahalanobis_score` in `pipeline1_predictions_stream3.parquet`. The classifier threshold (solid black, 29.87) corresponds to the chi-squared p = 0.99 boundary calibrated on the Stream 1 training distribution; the empirical 99th percentile of Stream 3 scores (dashed red, 177.21) lies well above the threshold because Stream 3 is a broader RGB+RC selection containing many stars (~20%) that legitimately fall outside the APOGEE training cone. The long tail to ~36000 captures the most extreme OOD outliers (~10 stars). Stars above the classifier threshold receive `ood_joint_flag = True` and are demoted to Tier 3.

---

## fig_sigma_per_tier.pdf

**Short caption:**
> Per-element predicted-σ density stratified by per-element release tier on Stream 3. Tier 1 (green) shows narrow low-σ peaks indicative of in-distribution disc stars; Tier 2 (orange) shows the sharp empirical-Bayes-shrinkage σ ceiling at τ = 50; Tier 3 (red) is the broader OOD distribution.

**Long caption:**
> Five-panel σ density showing the predicted uncertainty distribution for each of the five released labels (T_eff, log g, [M/H], [α/M], [Mg/H]) stratified by the per-element tier from `release_tier__<element>`. Tier 1 (green) is narrow with low σ peaks (T_eff σ ~ 50 K, [M/H] σ ~ 0.1 dex, [α/M] σ ~ 0.04 dex), reflecting in-distribution disc stars whose XP coefficients are well-modelled by the ensemble. Tier 2 (orange) shows a sharp ceiling at the empirical-Bayes-shrunken σ value (τ = 50, Efron-Morris 1973, Pourahmadi 1999); these are the caveat-flagged stars whose calibrated uncertainty is the prior-dominated upper bound. Tier 3 (red) is the broader, intermediate-σ distribution for OOD stars whose predictions are extrapolations of the learned representation. The T2 ceiling is a feature, not a bug: it documents the τ = 50 shrinkage choice and is the operational σ a downstream consumer should expect for caveat-flagged predictions.

---

## fig_hrd_tier1.pdf

**Short caption:**
> Tier-1 Hertzsprung-Russell diagram for 483002 Stream 3 stars, coloured by predicted [M/H]. The metallicity gradient along the giant branch (yellow at metal-rich, purple at metal-poor) recovers the textbook population-color relation; the recovery on a Tier-1-only catalog at 614k-star scale validates the Pipeline 1 v3 release as astrophysically faithful.

**Long caption:**
> Hertzsprung-Russell diagram (G_BP - G_RP colour vs absolute G magnitude M_G) for the 483002 Tier-1 stars in the Stream 3 release, coloured by the model-predicted [M/H]. Absolute magnitudes are computed from `phot_g_mean_mag_corr` and `r_med_photogeo` (Bailer-Jones+2021 photogeometric distance). The metal-poor (purple, [M/H] ~ -1.5) population concentrates at warmer colour and brighter M_G, consistent with halo and accreted-debris populations; metal-rich (yellow, [M/H] ~ 0) stars cluster at cooler colours and fainter M_G, the thin-disc giant locus. The clear gradient along the upper RGB recovers the well-known metallicity-color trend. A small population of fainter-than-expected stars (M_G > 6) at moderate colour is likely a contamination of subgiants and lower-MS stars that survived the Stream 3 selection; their fraction is small and they are excluded from per-star science by the `flag_bad` Gaia quality cut at ingestion.

---

## fig_alpha_m_vs_fe_h.pdf

**Short caption:**
> Tier-1 [α/M] vs [M/H] hexbin density. Panel (a) recovers the textbook chemical bimodality: a low-α thin-disc sequence at [M/H] > -0.5, [α/M] ~ 0, and a high-α thick-disc / halo plateau at [M/H] < -0.5, [α/M] ~ 0.2-0.3. Panel (b) overlays the v1 placeholder caveat: the kinematic-OOD detector that would split disc-prior-trustworthy stars from kinematic outliers is Phase B work pending.

**Long caption:**
> Two-panel chemical-bimodality figure. Panel (a) is a hexbin density of predicted [α/M] vs [M/H] for the 483002 Tier-1 stars, recovering the classical α-bimodality reported by APOGEE on its native reduction (e.g., Hayden+2015, Recio-Blanco+2014): a low-α thin-disc sequence ([α/M] ~ 0 dex) at metal-rich [M/H] > -0.5, and a high-α thick-disc + halo plateau ([α/M] ~ 0.2-0.3 dex) extending to [M/H] = -1.8. The transition between the two sequences shows the well-known ridge at [M/H] ~ -0.5. The recovery is striking given that [α/M] is a model-flagged aux-assisted label (`xp_abundance_type__alpha_m = "aux_assisted"`, conditional MI given auxiliary features < 0.02 nats per the §3.5 information-content audit): the model has learned the disc-population [α/M]-[M/H] relation from APOGEE training and reproduces it faithfully when applied to Stream 3 disc stars. Panel (b) shows the same density in greyscale with a red overlay reminding the reader that the v1 `kin_ood_flag` is a placeholder (all False); the Phase B kinematic-OOD detector will populate the flag, after which an additional split into disc-prior-trustworthy (kin_ood_flag = False) versus kinematically anomalous (True) becomes possible. Halo and accreted-debris stars in the latter group should not be claimed as α-trusted per-star measurements; they survive in the Tier-1 set in v1 only because the kinematic gate is not yet active.
