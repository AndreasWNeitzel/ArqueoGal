# 6 Limitations and caveats

This section documents, with explicit honesty, the boundaries of the
Pipeline 1 v1 release. Most items are surfaced as per-star flags in the
catalog (see §3.5.4 and `docs/CATALOG_SCHEMA.md`); this section
quantifies their scientific implications and points users to the
appropriate filter for their use case.

## 6.1 Tier-promotion protocol coverage at 5/6

The six-test promotion protocol of research_brief §3.3 gates per-element
release. Pipeline 1 v1 ships with five tests fully implemented (physical
feasibility, hold-out RMSE, shuffle-null permutation, conditional MI,
within-cell calibration) and two as documented stubs:

- **Test 3 (SHAP feature attribution):** deferred. The implementation
  cost of full-catalogue SHAP on the 108-D XP block plus aux features
  exceeds the v1 timeline. Permutation importance with the XP-vs-aux
  grouping (research_brief §3.3.1 question 2) provides a partial
  substitute and is reported.

- **Test 6 (cross-catalogue consistency):** deferred to v1.1 / D-Cat-d
  (February 2027) when Stream 3 inference completes and the overlap
  with GALAH DR4 (Buder et al. 2024, arXiv:2410.12272), AspGap (Li et
  al. 2024), Guiglion et al. 2024 (A&A 682, A9), Anders et al. 2024
  SHBoost (A&A 691, A127), and Fallows and Sanders 2024 (MNRAS 531,
  2126) becomes available at scale.

The release language and the catalog `release_tier` columns reflect 5/6
coverage explicitly; no claim of 6/6 coverage is made anywhere in this
work or in the catalog provenance metadata.

## 6.2 [α/M] and [Mg/H] as aux-assisted predictions

As documented in §3.5, the conditional mutual information of XP with
[α/M] and [Mg/H] given full aux conditioning is below the 0.02 nats
threshold. The model uses these labels primarily through the disc-
population prior implicit in the APOGEE-Gaia training set. The released
catalog flags this via per-element `xp_abundance_type__alpha_m` and
`xp_abundance_type__mg_h` strings ("aux_assisted"); users are
encouraged to filter on these flags when per-star spectroscopic fidelity
is required.

For halo stars, accreted-debris streams, and counter-rotating-disc
populations where the disc prior breaks down, aux-assisted predictions
are systematically unreliable. The `kin_ood_flag` column is the
operational lever: when True, the per-element tier for [α/M] and [Mg/H]
demotes to Tier 2. The kinematic-OOD detector is a single-Gaussian fit
on the disc-only training subset in 3D Galactocentric (v_R, v_φ, v_z)
space; threshold at the disc 99th percentile.

In Pipeline 1 v1.0 the `kin_ood_flag` column was a False placeholder
(Phase A2). v1.0.1 (Phase A2-followup) adds the actual detector and
thereby promotes the [α/M] reframing from documentation to a per-star
release decision. Consumers using Pipeline 1 should retrieve the v1.0.1
or later catalog.

## 6.3 NLTE departures in cool giants

ASPCAP DR19 (Mészáros et al. 2025) is LTE-only; NLTE departures in cool
giants (Teff < 4200 K, [Fe/H] < −1) reach 0.05–0.15 dex on [Mg/Fe] per
Bergemann and Cescutti 2010 (A&A 522, A9; arXiv:1006.4814) and Lind et
al. 2012. Pipeline 1 inherits this systematic. The methods paper does
not attempt NLTE correction; consumers comparing thin/thick disc
[α/Fe] values across a wide Teff range should account for this. A
sensitivity ablation (Limitations supplementary material) is left as
future work.

## 6.4 Microturbulence degeneracy

ASPCAP fits stars at fixed empirical microturbulence as a function of
logg/Teff. At APOGEE H-band resolution, a ±0.5 km/s shift in v_micro
shifts individual abundances by 0.2–0.3 dex (Gray et al. 2011, A&A 530,
A129; Bruntt et al. 2011). At Gaia XP resolution (R ~ 30–100), v_micro
is unmeasurable and degenerate with the abundance scale. Pipeline 1 v1
inherits the v_micro choice baked into the ASPCAP DR19 labels; this is
an irreducible systematic on the per-element abundance scale.

## 6.5 Vanadium not released

V abundances are not released in Pipeline 1 v1 despite being present in
APOGEE DR19 ASPCAP. The information-theoretic justification: V H-band
features are weak and blended (Smith et al. 2021, ApJS 254, 32; APOGEE
DR17 documentation), the reported APOGEE V scatter exceeds the
empirical 0.10 dex threshold, and Mészáros et al. 2025 do not publish
correction coefficients for V (arXiv:2506.07845). At Gaia XP
resolution, V has no measurable spectroscopic signature. The catalog's
`xp_abundance_type__v_h` would be `aux_assisted` were V released, but
the aux-only baseline RMSE for V is large enough that the prediction
would carry no scientific value above a population-level prior. We
therefore decline to release V at all.

## 6.6 Regime B Teff over-prediction (excluded from per-star release)

Stars at |b| < 5°, Teff in 4750–5200 K, logg in 1.5–2.1 (warm upper-RGB
in the Galactic plane) show a systematic ~+1σ Teff over-prediction
(50–70 K). The exclusion is implemented via `RegimeBEnvelope` in
`uncertainty.py` and surfaces as `regime_b_flag` in the release.

Three hypotheses for the bias have been considered (META_META §4):

1. **Av-prior under-correction at d > 3 kpc.** The 3D dust map stack
   ends Edenhofer et al. 2024 at 1.25 kpc; beyond, Lallement et al.
   2022 thins (1.25–3 kpc) and SFD over-corrects at low latitudes
   beyond 3 kpc (Sale et al. 2009; Schlafly et al. 2014). If true Av is
   lower than estimated, the model trained with implicit Av in DR19
   over-corrects in inference and biases predicted Teff upward.
2. **Plane-parallel atmosphere geometry inadequate for warm upper-RGB
   stars** with photospheric scale heights approaching a non-trivial
   fraction of stellar radius.
3. **Training-set spatial bias.** APOGEE selection at low |b| favours
   bright, less-reddened stars; the calibration learns a Teff-Av coupling
   that inverts for fainter Stream 3 stars at the same |b|.

The hypotheses are not mutually exclusive. The leading candidate (Av
under-correction) is amenable to falsification via an IRFM-Teff
diagnostic on Regime B holdout stars with d > 3 kpc, |b| < 5°,
stratified by dust-map source (Casagrande et al. 2021, MNRAS 507, 2684;
arXiv:2107.12792). This validation is scoped in
`docs/protocols/fisher_crlb.md` and the Regime B subsection is updated
post-D-Cat-d (February 2027) once the diagnostic completes. Until then,
the Regime B exclusion is documented as an applicability boundary, not
a model failure.

## 6.7 Magnitude-dependent degradation

Pipeline 1 caps releases at G < 17 (XP-native regime). At fainter
magnitudes, SNR per XP coefficient drops by ~6× from G = 14 to G = 17,
and per-element CMI is expected to drop accordingly. The catalog's
`g_mag_bin` column (`bright`, `mid`, `faint`) is the consumer's
operational handle for magnitude-dependent reliability stratification.

A per-magnitude-bin CMI and reliability audit is recommended (METRICS
gap §1.3 in metrics_diagnostics.md, P1 work plan in META_META §14.4)
and will be reported in a v1.1 update; the v1 release tier assignments
are not yet stratified by magnitude.

## 6.8 Distance prior dominance

For sources with parallax SNR < 5 (σ_π/π > 0.2), Bailer-Jones et al.
2021 photogeometric distances are dominated by the Galactic prior
rather than parallax. The catalog flags such sources via the
`dist_prior_dominated` boolean column; this flag adds a global Tier 2
caveat. Bulge-direction RGB stars at d > 6 kpc are almost always
prior-dominated. Consumers using the catalog for kinematic studies
should filter on this flag and quote the photogeometric prior
(Bailer-Jones et al. 2021 §4) as the assumed Galactic structure.

## 6.9 Cross-survey scale offsets

Pipeline 1 predictions are on the DR19 (post-Mészáros+2025) abundance
scale. Direct numerical comparison to DR17-based catalogues, Gaia GSP-
Spec (Recio-Blanco et al. 2023), or other surveys requires accounting
for documented offsets. The methods paper does not attempt to harmonise
across survey scales; consumers requiring such harmonisation should
consult the relevant survey's calibration paper.

## 6.10 Gaia DR4 forward compatibility

Pipeline 1 v1 trains on Gaia DR3. Gaia DR4 is scheduled for ~2026
(post-D-Cat-d in our timeline; ESA COSMOS, arXiv:2503.01533). The
Lindegren parallax zero-point and Riello G-mag correction will require
re-derivation for DR4; the current corrections are DR3-specific. A v2
retrain on DR4 is post-D-Cat-d work.
