# ArqueoGal — Research Brief

**Author:** Andreas Neitzel (Co-I, PhD student, CAUP/IA)
**Project:** 2024.15303.PEX (DOI: 10.54499/2024.15303.PEX) — PI: Tiago Campante
**Workspace scope:** Personal development area for D5.1 (ML tool for automated stellar population classification) and contributions to D-Cat-b (chemical abundances) and D-Cat-d (population membership probabilities)
**Companion document:** `data_acquisition.md` (data pipeline specification)
**Last revision:** April 2026 (v2)

---

## 0. Executive summary

This document defines the scientific rationale, the state of the art, and the methodological programme for the machine-learning pipeline that Andreas Neitzel leads within the ArqueoGal project in this repository:

1. **Pipeline 1 — `xp_abundances`**: a semi-supervised regression from Gaia DR3 XP (BP/RP) Hermite coefficients to APOGEE DR19 chemical abundances, covering red giants within the XP-native magnitude regime (G ≲ 17.65). The focus is *depth of treatment within this regime* — rigorous uncertainty calibration, extension to additional elements with APOGEE-DR19-backed statistical validation, and honest separation of spectrum-driven from prior-driven predictions — rather than pushing to fainter magnitudes.

Population classification — the UMAP+HDBSCAN tool originally scoped as "Pipeline 2" — has been spun out as the separate **Starfold** repository (2026-04-22). D5.1 and D-Cat-d are delivered by Starfold, which consumes this repository's Pipeline 1 predictions via the parquet contract described in `docs/plan/04_pipeline2_main.md`. Sections of this document that still discuss population-classification methodology are retained as historical rationale and to document the D-Cat-b/D-Cat-d interface requirements; the active methodological reference for Starfold lives in that repo.

The project operates on **real observational data**. FIRE-2 Ananke synthetic surveys are used *only* for the method-validation hare-and-hounds exercise in Subtask 5.1; all downstream science is conducted on real Gaia DR3 × APOGEE DR19 × TESS data. The concrete data acquisition and preprocessing plan — three streams (APOGEE DR19 × Gaia, TESS Hon+2021 × Gaia, Gaia RGB+RC application sample), 5 GB disk budget, pyvo-over-astroquery ingestion, Edenhofer+2024 + GSP-Phot neighborhood-median dust handling, StarHorse2 distances, galpy orbits — is specified in the companion `data_acquisition.md`. This document distinguishes sharply between what can be validated with ground truth (training-set hold-out, FIRE-2) and what must be validated without it (the real catalogue).

The differentiator this project stakes out, relative to the saturated field of XP-abundance pipelines (Andrae+2023, Zhang+2023, Li+2024/AspGap, Guiglion+2024, Hattori+2024, Fallows & Sanders 2024, Anders+2024/SHBoost, Ye+2024, Buck & Schwarz 2024), is not catalogue size or magnitude reach. It is **honesty under uncertainty**: every abundance label carries a *calibrated* covariant uncertainty (not just a quoted σ), an explicit separation of evolutionary-stage-dependent from galactic-archaeology quantities, and a pipeline-level information-content audit that quantifies how much of each prediction is driven by spectrum versus by training-set priors. No publicly released XP-abundance catalogue does this rigorously. That is the niche we fill.

---

## 1. Scientific framing

### 1.1 ArqueoGal in one paragraph

ArqueoGal will produce the first all-sky, kiloparsec-scale chrono-chemo-kinematic map of the Milky Way disc by combining TESS asteroseismic ages, Gaia DR3 astrometry and kinematics, and APOGEE spectroscopy on red giants. Three Research Objectives motivate everything: RO1 (origin of the chemically bimodal disc — two-infall vs shock-heating vs merger-driven), RO2 (dynamical agents of disc flaring), RO3 (efficiency and timescales of radial migration). The catalogue will contain ~10⁵ stars. The bottleneck that limits all three ROs is not data volume but *per-star parameter dimensionality*: for most red giants in the target sample we will have ages (seismic, 10–20% precision), kinematics (Gaia, excellent), but chemistry only for the APOGEE footprint (~30–40% of the sample). Pipeline 1 (this repo) closes the chemistry gap. Starfold (separate repo) then turns the completed chrono-chemo-kinematic vector into membership probabilities for the disc sub-populations (α-rich, α-poor) and for accreted/in-situ halo components that the sample will inevitably contain at low-latitude and metal-poor tails.

### 1.2 Andreas Neitzel's role

| Task | Role | Person-months | Period |
|---|---|---|---|
| Task 3 — Astrometry, Kinematics, Spectroscopy | Participant (lead: Bossini) | 1 | ~Aug 2026 |
| Task 5 — Stellar Population Classification | Lead contributor (leads: Campante, Miglio) | 4.4 | Jun–Nov 2026 |
| Task 6 — Galactic Modeling | Participant (leads: Campante, Miglio) | 6 | Dec 2026–May 2027 |

The workspace targets:
- **Supporting contribution to D-Cat-b** (Month 6 / Aug 2026): chemical abundances from Gaia XP for stars without APOGEE spectroscopy — Pipeline 1 in this repo delivers here.
- **D5.1** (Month 10 / Dec 2026): open-source GitHub release of the population-classifier tool — delivered by Starfold (separate repo), which builds on Neitzel et al. 2025 (A&A 695, A243; arXiv:2501.16294) and consumes Pipeline 1 predictions produced here.
- **D-Cat-d** (Month 12 / Feb 2027): stellar population membership probabilities appended to the all-sky catalogue — produced by Starfold from this repo's Pipeline 1 output.

---

## 2. The Gaia XP → chemical abundances landscape

### 2.1 What exists and what each paper actually delivers

Since Gaia DR3 (June 2022) made 220 M BP/RP spectra available, a dozen independent pipelines have mined them for stellar parameters and abundances. Accurate positioning requires knowing what each actually does.

| Pipeline | Method | Training | Output labels | Sample | Primary limitation |
|---|---|---|---|---|---|
| **Andrae, Rix & Chandra 2023** (MNRAS 521, 3527) | XGBoost on XP + parallax + WISE | APOGEE DR17 | Teff, log g, [M/H] | 175 M | Only [M/H]; no α; no covariances |
| **Zhang, Green & Rix 2023** (MNRAS 524, 1855) | Forward-model NN, joint Av + distance | LAMOST LRS + APOGEE | Teff, log g, [Fe/H], Av, d | 220 M | No α; Av as latent only |
| **AspGap (Li+2024)** (ApJ 974, 42) | Teacher–student NN, masked prediction | APOGEE DR17 RGB | Teff, log g, [M/H], [α/M], [C/Fe], [N/Fe], [Mg/Fe], [Al/Fe] | 23 M RGB | Narrow Teff–log g; low-[Fe/H] [α/M] unreliable |
| **Guiglion et al. 2024** (A&A 682, A9) | Hybrid CNN on RVS + XP + photom + parallax | APOGEE DR17 (45 k) | Teff, log g, [M/H], [Fe/H], [α/M] | 886 080 | RVS-limited (G ≲ 14); [M/H] < −2.3 biased; no individual elements |
| **SHBoost (Anders+2024)** (A&A 691, A127) | XGBoost per-label | Spectroscopic compilation | Av, Teff, log g, [M/H], mass | 217 M | Per-label independent → lost covariances |
| **Fallows & Sanders 2024** (MNRAS 531, 2126) | Uncertain NN on XP + photometry | APOGEE | Teff, log g, [Fe/H], [C/Fe], [N/Fe], [α/M] | ~170 M | Authors explicitly flag training-prior vs spectrum ambiguity |
| **Hattori 2024** (ApJ 969, 81) | XGBoost | APOGEE | [Fe/H], [α/Fe] | 48 M giants in low-extinction regions | No uncertainty covariance; low-Av selection |
| **Ye, Allende Prieto et al. 2024** (A&A, arXiv:2411.19105) | FERRE + synthetic grid + NN flux-correction | Synthetic + PASTEL | Teff, log g, [M/H] | 68 M | Teff 4000–7000 K only; no α. **Valuable:** reduces relative flux systematics from 3.2–3.7% to 1.2–2.4%. |
| **Buck & Schwarz 2024** (arXiv:2410.16081, NeurIPS ML4PS) | Contrastive multimodal learning (CLIP-style), RVS + XP shared 32-D embedding; k-NN regression on embedding | Self-supervised on 841 300 RVS+XP pairs; 44 780 APOGEE labels for validation only | Teff, log g, [M/H], [α/M] via k-NN (k=13) | Workshop proof-of-concept | R²(Teff)=0.987, R²([α/M])=0.849. **Not a production catalogue**; no released all-sky abundances. Clean demonstration that CL works for stellar spectra. |
| **Ardern-Arentsen et al. 2025** (MNRAS 537, 1984) | NN | Pristine + APOGEE | [Fe/H], [C/Fe], CEMP flag | ~10 M metal-poor | CEMP-targeted |
| **Leung & Bovy 2024** (StellarPerceptron; arXiv:2411.04750) | Transformer, missing-input tolerant | APOGEE | Multi-label | Proof of concept | No all-sky catalogue yet |

### 2.2 The epistemological problem these papers share

Fallows & Sanders 2024 state it clearly: *"An important question when using discriminative models is 'where is our information coming from'? … reliance on non-physical parameters can stop the model from adapting to new data."* When an NN predicts an [α/M] bimodality from XP that resembles APOGEE's, three mechanisms can produce it:

1. **Genuine spectral information** from CN, CH (G-band 4300 Å), MgH (~5100 Å), Mgb triplet (5167/5172/5183 Å), Ca H+K, TiO bands surviving R ≈ 30–100.
2. **Training-set correlations** — the network learns that [α/Fe] covaries with Teff, log g, Av, position, kinematics, and uses those as the real predictors.
3. **Survey-selection leakage** — APOGEE preferentially samples certain sightlines and colour–magnitude regions; the ML learns survey geometry.

Without a deliberate audit, the three are indistinguishable. **Our pipeline will be the first XP-abundance catalogue to require an information-content audit for every released label** (§9). Not Buck & Schwarz 2024, not Guiglion+2024, not AspGap, not Fallows & Sanders — none currently provide this.

### 2.3 Specific transcendence targets

| Reference | What it does well | What our Pipeline 1 adds |
|---|---|---|
| **Buck & Schwarz 2024** | Elegant CLIP-style contrastive representation of RVS+XP; excellent k-NN downstream performance (R²=0.987 Teff, 0.849 [α/M]); demonstrates shared-embedding-space utility | Production catalogue vs their proof-of-concept; **calibrated heteroscedastic uncertainties with reliability diagrams** (they report R² only, no σ calibration); information-content audit per label; extension to individual abundances beyond [α/M]; APOGEE DR19 vs DR17; their multimodal CL becomes a candidate method in our experimental arm (§8.5). |
| **Guiglion+2024** | Strong hybrid CNN, 886 k stars with [α/M], CoNN architecture proven | XP-only avoids RVS G<14 ceiling (we cover the full native XP regime G ≲ 17.65); DR19 labels; neighborhood-median Av prior (§5); Tier 1/2/3 tagging (§3); calibrated uncertainties |
| **AspGap (Li+2024)** | First large-scale individual-element XP catalogue for RGB | Joint distance–Av fit; broader Teff–log g window with evolutionary-stage classification head; heteroscedastic covariance output; per-label information-content audit |
| **Fallows & Sanders 2024** | Uncertainty via deep ensembles + uncertain NN | Same ensemble philosophy + §9.2 audit transforms their caveat ("training priors may drive bimodality") into a measured, per-label, released quantity |
| **Ye+2024** | NN flux-correction preprocessing that reduces XP flux systematics 2–3× | We adopt their flux-correction as mandatory preprocessing; extend to α and [C/N] predictions Ye+2024 does not attempt |

---

## 3. Physics of Gaia XP spectra — what is actually learnable

### 3.1 Instrumental facts

BP covers 330–680 nm; RP covers 640–1050 nm, overlapping near 650–680 nm (De Angeli et al. 2023, A&A 674, A2; Carrasco et al. 2021). Resolution is *non-uniform*: R(λ) ≈ 30 at 400 nm rising to R ≈ 100 at 800 nm. Spectra are distributed as 55 Hermite-function coefficients per band; the zeroth coefficient sets flux normalisation, and **useful abundance signal in BP typically extends to coefficient ~45 and in RP to ~30**, with higher orders noise-dominated at G ≳ 13 (Guiglion+2024 gradient-map analysis; Weiler+2023 for systematic wiggles).

The `GaiaXPy` package (current version 2.1) converts coefficients to sampled spectra and back. The "wiggles" — systematic residuals at the 1–3% level correlating with colour, magnitude, and position — are characterised and partially corrected by Huang et al. 2024 and Ye et al. 2024. **We adopt Ye et al. 2024's NN flux-correction as mandatory preprocessing** in the main pipeline.

**Empirical information content after Ye+2024 flux correction and Hermite re-projection onto the 55+55 orthonormal basis (this work, 2026-04):** on a 1 490-star stratified sample from Stream 1 (Ye flag=0, catastrophic-residual rows excluded) the per-mode σ_MAD noise floor plateaus at n ≈ 20 (BP) and n ≈ 23 (RP), indicating ~43 effective modes carry abundance-relevant information. A PCA comparison on 50 000 normal-population stars confirms noise-dilution rather than intrinsic spreading: PC1 rises from 40.8% on the full 110-D basis to **73.7% on the 43-D noise-floor truncation** (BP coeffs 0–19 ⊕ RP coeffs 0–22), and PC1+PC2 rises from 54.9% to **89.8%** — consistent with a Teff/log g-dominated stellar-physics manifold with mild [Fe/H] variation. These numbers are reproducible from the frozen canonical basis (fingerprint `0d34b5659e97e5891b57005215a59b0b70fc56f23d8ffb22f442c4ad5101eab7`) and `scripts/smoke_hermite_reprojection.py` + `scripts/analyze_hermite_pre_emit.py`. This contrasts with the per-coefficient noise thresholds of n ≈ 45 (BP) and n ≈ 30 (RP) reported by Guiglion et al. 2024 for raw Gaia Hermite coefficients at G ≳ 13; the difference reflects the denoising effect of the Ye+2024 NN, which trades raw per-coefficient precision for a lower-dimensional systematic-corrected representation.

**Implication for Pipeline 1 architecture (open):** the main-pipeline supervised regression head will default to the 43-D noise-floor truncation as input. Whether the self-supervised contrastive pretraining stage should instead see the full 110-D vector — so the network learns signal-vs-noise discrimination on its own for downstream OOD detection — is deferred to the Pipeline 1 design sprint; the parquet therefore materialises all 110 coefficients plus residuals so either choice remains available at model-training time.

### 3.2 Which elements have actual spectral features in the XP window

| Element | Dominant XP features | Expected information content at R ≈ 50 |
|---|---|---|
| Fe | Ca H+K blanketing, Mg b + Fe blends, metal-line forest 500–600 nm | **High** — [Fe/H] to 0.1–0.2 dex down to [Fe/H] ≈ −3 (established) |
| α (Mg, Si, Ca, Ti combined) | Mg b triplet 5170, MgH band 5100, Ca I 4227, Ca triplet 850–866 | **Moderate** — [α/M] to ~0.05–0.1 dex at [M/H] > −1.5 (AspGap, Guiglion+2024, Buck & Schwarz 2024) |
| Mg alone | Mg b triplet 5170, MgH band 5100 | **Low–moderate** — partially separable from other α |
| C | CH G-band 4300 Å, Swan bands 4737/5165 (cool giants) | **Moderate for giants** — [C/Fe] to ~0.15 dex |
| N | CN 3883, 4216 Å | **Moderate for giants** — [N/Fe] to ~0.15 dex; evolutionary-stage dependent |
| C/N ratio | Joint CH + CN band use | **Moderate for giants** — RGB [C/N]-mass-age clock partially accessible |
| Na | Na D 5890/5896 | Very weak at R=50; degenerate with ISM |
| Al | Al I 3944/3961 (often drowned), 6696 | Weak; partial recovery in RGB possible |
| Mn | Mn triplet 4030 | Weak; feasible for metal-rich giants only |
| Ni, Ti, O | many blends; TiO in cool giants | Inaccessible as individual labels; prior-driven if predicted |
| Ca | Ca H+K (dominant), Ca I 4227, Ca II 850–866 | **Moderate** — but confused with [Fe/H] and gravity |
| Li | Li 6708 | **Low** at R=50; only strong-Li giants detectable |
| n-capture (Ba, Eu) | Ba II 4554/4934; Eu II 4129 | **Very low** — essentially prior-driven |

**Tiering adopted throughout this project:**

- **Tier 1** (per-star reliable, individual uncertainties meaningful): Teff, log g, **[M/H]** (ASPCAP global metallicity fit over Fe-peak + α lines jointly), Av, distance. **[M/H] ≠ [Fe/H] in ASPCAP DR19**: the per-element [Fe/H] fit runs after the global parameters are fixed and can legitimately fail (saturated Fe lines in metal-rich regimes, insufficient SNR per individual line in the blue, unresolved Fe blends in cool giants — consistent with Andrae+2023 and Guiglion+2024). For most RGB stars [M/H] and [Fe/H] agree to 0.02–0.05 dex; when they disagree or when [Fe/H] is NaN, [M/H] is the robust quantity. Pipeline 1's Tier-1 training target is [M/H]; [Fe/H] sits with the Tier-2 per-element abundances.
- **Tier 2** (per-star release with explicit uncertainty inflation + population-level recommended): **[Fe/H]** (per-element), [α/M], [C/Fe], [N/Fe], [C/N] for RGB in validated domain, [Mg/Fe] if separable.
- **Tier 3** (do not release per-star; diagnostic only if released): individual Mg distinct from α, Al, Mn, Ca as separate labels, Na, Ni, Ti, O, n-capture.

No XP-abundance catalogue currently publishes this tiering honestly.

### 3.3 Statistical method to validate extension to new elements against APOGEE DR19

A *statistical method* is required for promoting new elements from Tier 3 to Tier 2 using APOGEE DR19 as the reference. The following six-test protocol is applied to each candidate element:

1. **Physical gate**: does the element have absorption features in the XP wavelength window (§3.2 table)? If no, reject immediately — no amount of ML can invent information that isn't in the photons.

2. **Hold-out RMSE vs APOGEE DR19**: train on 70% of APOGEE × XP overlap, validate on 15%, test on 15%. Report RMSE, bias, and reduced χ² stratified by Teff (4 bins), log g (2 bins), [Fe/H] (4 bins), Av (3 bins), G (3 bins). **Rejection criterion**: worst-cell RMSE > 2× median RMSE, OR worst-cell bias > 0.05 dex.

3. **Precision floor via open clusters**: within a single open cluster, all RGB members share [X/Fe] to within a few 0.01 dex (Bovy 2016; Spina+2021; Casamiquela+2020). Predicted intra-cluster dispersion σ_intra is computed; the element is promotable only if σ_intra < σ_APOGEE × 1.5.

4. **Information-content audit passing** (§9.2 tests 1–4):
   - Permutation importance must show at least 3 XP coefficients with ΔR² > 0.02 per label.
   - LOOCO must show non-trivial label shift under removal of physically relevant coefficients.
   - Shuffled-spectrum null: predictive skill on shuffled data must be < 20% of the real-data skill (otherwise the label is prior-driven).
   - Decorrelated sub-sample test: predictive skill on decorrelated sample (§9.2.6) must retain ≥ 50% of the original R².

5. **Cross-catalogue consistency**: compare against AspGap, Guiglion+2024, SHBoost, GALAH DR4, Fallows & Sanders 2024 on common stars. Mean pairwise bias vs the consensus must be within 0.05 dex; scatter within 2× the APOGEE precision. Flag systematic disagreements.

6. **Conditional mutual information test**: estimate MI(XP coefficients; [X/Fe] | Teff, log g, [Fe/H]) via the `npeet` or `sklearn.feature_selection.mutual_info_regression` implementations with bootstrap confidence intervals (N = 1000 resamples). The 95% CI must exclude zero. This is the formal test of "is there any information about [X/Fe] in the XP coefficients beyond what's already carried by the atmospheric parameters?"

**Promotion decision tree:**
- Passes 1–3 only → **Tier 3 stays Tier 3**. Element is not released per-star.
- Passes 1–4 but fails 5 or 6 → **Tier 3, internal use only**. Catalogue documents that the element was attempted and why it was excluded.
- Passes 1–5, passes 6 at conservative threshold → **Tier 2**. Release per-star with explicit uncertainty-inflation factor and population-level-only recommendation.
- Passes all six *and* the reliability diagram (§9.1) shows calibrated σ across parameter space → **Tier 1**.

This is a rigorous, pre-registered protocol. It is applied before any element appears in the D-Cat-b release.

#### 3.3.1 Three-question diagnostic — failure-mode protocol for the shuffled-spectrum null

Codified 2026-04-19 after the Pipeline 1 v1 audit (see `reports/pipeline1/audit/SUMMARY.md`). When the §9.2 Test 4 shuffled-spectrum null test fails for a label — operationalised as null skill_ratio > 0.20, i.e. the model retains more than 20% of its real-data skill on the shuffled XP block — the response is **not** automatic Tier demotion. The literal null-skill-ratio gate is sensitive to aux-only labels carrying an extinction/distance-correlated signal that the model can partially reconstruct under shuffled XP; this does not imply XP is uninformative. The response is the three-question diagnostic:

1. **PCA-CMI with richer summary.** Recompute the §9.2 Test 5 conditional MI estimate with a PCA summary of the XP block that retains ≥ 95% variance (typically 5–10 components for Gaia XP), holding the conditioning set, KSG k, and subsample cap fixed. The 2-D summary used by the production audit can inflate or collapse the CMI estimate — see §9.2.1 below. The PCA estimate is primary; the 2-D estimate is supplementary.
2. **Per-feature permutation importance with XP-vs-aux grouping.** Emit the full per-feature permutation ΔRMSE ranking (not just family aggregates). Tag each feature by group (XP shape + c0 vs auxiliary) and report the top-10 composition plus group Σ(ΔRMSE). This disentangles "the aux features are individually strong" from "XP is globally uninformative".
3. **Auxiliary-only baseline MLP.** Retrain a head-matched aux-only MLP (no XP shape, no c0 scalars) on the same train/val split and report the per-label RMSE ratio aux_only / full_model. Thresholds: ratio ≈ 1.00 (within ~5%) → XP contribution is noise; ratio > 1.10 → XP contributes meaningfully; intermediate → judgment call.

**Tier assignment follows the combined evidence of these diagnostics.** A label that fails the literal Test 4 gate but passes at least two of (aux-ratio > 1.10, PCA-CMI > 0.02 nats, ≥ 3 XP features in top-10 permutation) is Tier 1 with an explicit prior-augmented release caveat documenting the XP-vs-aux relative contribution. A label that fails Test 4 and fails all three diagnostics is demoted: Tier 2 with population-level-only recommendation. A label that passes Test 4 but has a suspiciously low 2-D-summary CMI should still have its PCA-summary CMI reported for methodology consistency; the load-bearing evidence for such labels is the shuffled-spectrum null and the XP joint shuffle ΔRMSE/σ, not the 2-D CMI.

The driver scripts for this protocol are `scripts/run_three_question_diagnostic.py` (Q1–Q3 for a target label subset) and `scripts/run_pca_cmi_all_labels.py` (Q1 cross-label consistency pass). The Pipeline 1 v1 application directed the ratified Option 2 tier decisions (Teff → Tier 1 clean; log g → Tier 1 with explicit prior-augmented caveat).

---

## 4. APOGEE DR19 as label source

### 4.1 What DR19 gives us

SDSS-V Data Release 19 (Mészáros et al. 2025, arXiv:2506.07845, AJ in press) contains **964 989 ASPCAP stars** — 628 378 previously released APOGEE-1/2 stars plus 336 511 new APO stars observed through July 2023. The pipeline has been entirely rewritten inside Astra. Typical precisions for red giants: σ(Teff) ≈ 50–70 K, σ(log g) ≈ 0.07–0.09, σ([M/H]), σ([α/M]), σ([Mg/Fe]), σ([Si/Fe]) ≈ 0.02–0.04 dex; ≥ 10 elements at < 0.1 dex. All abundances LTE; the NLTE Na/Mg/K/Ca from DR17 are *not* in DR19.

### 4.2 What changed from DR17

- ASPCAP code rewritten; raw and calibrated outputs delivered separately (`raw_*` vs calibrated columns).
- Calibration zero-points and Teff-trend corrections are new.
- Flag schema reorganised: `result_flags`, `flag_bad`, element-specific line-threshold flags.
- Residual Teff-dependent [X/M] trends along the giant branch remain; apply Mészáros+2025 correction polynomials before use as labels.

### 4.3 Consequences for ArqueoGal

1. **Retrain from scratch on DR19**. Do not reuse DR17-trained weights.
2. **Apply Mészáros+2025 [X/M]/Teff corrections** before using [X/M] as labels.
3. **Quality cuts**: `flag_bad == 0`, per-element line-threshold flags, SNR_APOGEE > 70, reasonable `ASPCAP_CHI2` for the Teff–log g bin.
4. **Gaia XP × APOGEE DR19 overlap**: ≈ 700 k stars in the nominal overlap (pre-cut, dwarfs+giants). After the RGB-focused quality cuts (`flag_bad==0`, SNR > 70, Teff ∈ [4000, 5500] K, log g ∈ [1.0, 3.5], [M/H] ∈ [−2, 0.5], Gaia source_id present, `has_xp_continuous`) the 2026-04 Stream 1 emission delivers **324 054 rows spanning 292 948 unique Gaia source_ids** — i.e. about 293 k unique RGB training stars. The ~31 k row-inflation factor is DR19 Astra task-level re-runs (see §4.4); dedup on `source_id` keeping the highest-SNR task is mandatory before train/val/test splitting.
5. **LTE-only** restricts credible NLTE-sensitive labels (Na, K especially). Avoid or apply external NLTE corrections (Bergemann-group grids).
6. **DR19 Astra emits multiple ASPCAP rows per Gaia source_id when the same spectrum is processed by more than one task** (median 2, max 15, 27 920 affected source_ids in the post-cut 2026-04 Stream 1). Within-star label scatter is well below APOGEE precision (Teff σ ≈ 7.6 K vs 50–70 K; [Fe/H] σ ≈ 0.010 dex vs 0.02–0.04 dex), so these are not independent label realisations. `v_astra` is constant at `0.6.0` across the pool — no DR17-vs-DR19 pipeline-variant mix — so a single-level sort on SNR is the correct dedup key. Use `arqueogal.data.dedup.dedup_by_source_id` before any train/val/test split; random splits on the raw matrix leak the same physical star across folds and would quietly corrupt the uncertainty-calibration deliverable. If a future DR mixes `v_astra` values, upgrade the dedup key to `(v_astra desc, snr desc)`.

Concrete ingestion plan: `data_acquisition.md` §3.

---

## 5. Extinction, distance, and the "neighborhood-median" strategy

### 5.1 Why extinction is the dominant systematic

At R ≈ 30–50, reddening alters BP spectral shape in ways that overlap with Teff and [Fe/H] degeneracies. An unaccounted-for Av = 1 mag shifts the BP Hermite coefficients at the same order as a ~150–250 K Teff change for a G-type star; larger at cooler Teff. None of the current published XP-abundance pipelines uses the latest Edenhofer+2024 parsec-scale map.

### 5.2 State of 3D dust mapping, April 2026

| Map | Resolution / range | Input | Best for |
|---|---|---|---|
| **Edenhofer et al. 2024** (A&A 685, A82; arXiv:2308.01295) | 14′ angular (Nside 256); 516 distance bins, 69 pc – 1.25 kpc; extended product to 2 kpc | ZGR23 XP-derived (Av, d) for ~54 M stars, Gaussian-process prior | **SOTA within 1.25 kpc** |
| **Bayestar19** (Green+2019) | 3.4′–13.7′; ≲5 kpc | 799 M PS1+2MASS+Gaia stars | Wide area to ~5 kpc, northern hemisphere dominant. ~3 GB download → **excluded from our 5 GB budget** |
| **Lallement+2022** (A&A 661, A147) | 25 pc voxels; 3 kpc | Gaia+2MASS | Mid-distance smooth disc; southern hemisphere better than Bayestar |
| **Vergely, Lallement & Cox 2022** | 10–50 pc; 3 kpc | Gaia+2MASS | Disc-plane structure |
| SFD/Schlegel+1998 | 2D only | COBE/IRAS | All-sky integrated column; baseline prior for high latitudes |

### 5.3 The neighborhood-median strategy — implementation

The neighborhood-median approach doubles as a budget-driven substitute when Bayestar19 is too large to cache locally (our case). Formalisation:

1. **Primary 3D map**: Edenhofer+2024 for d < 1.25 kpc. Small enough to fit on disk (~600 MB in the Gaussian-process reconstruction product).
2. **For d > 1.25 kpc**: use **Gaia GSP-Phot `ag_gspphot` and `ebpminrp_gspphot`** per-star, with a *3D spatial median*. For each target star at (l, b, d), compute the Gaia-star-weighted median of `ag_gspphot` over a ball of radius 50–100 pc in (l, b, d) space. This "neighborhood-median" prescription is scientifically defensible: individual GSP-Phot A_G values are noisy but their *ensemble* at a given 3D position is a direct tracer of the local ISM.
3. **Per star, compute two Av features:**
   - *Line-of-sight Av* — from the primary map at the star's most-probable distance.
   - *Neighborhood-median Av* — as above.
4. **Inject both as features** into Pipeline 1, plus their difference and the local Av dispersion as auxiliaries. The network learns when to deviate from the prior.
5. **Propagate Av uncertainty** at inference via MC over the prior posterior.

No published XP-abundance pipeline does this. Zhang+2023 treats Av as a latent; SHBoost uses SFD-like priors; Guiglion+2024 and AspGap largely ignore Av or treat it as nuisance. The neighborhood-median approach is a genuine methodological differentiator and fits the 5 GB budget.

Concrete implementation: `data_acquisition.md` §8.

### 5.4 Distance — when to use what

For parallax S/N > 10, inverse parallax with Lutz–Kelker correction is defensible. For G > 14–15, parallax S/N typically drops below 5 and **photogeometric distances** become mandatory:

- **Bailer-Jones et al. 2021** (AJ 161, 147) photogeometric distances — the baseline. Uses Gaia parallax + colour + magnitude + Galactic-model prior; ~10–20% precision at G ≤ 16. Served from GAVO TAP (`gedr3dist.main`).
- **StarHorse2** (Queiroz et al. 2023, A&A 673, A155) — adds spectroscopic parameters; ~10–15% precision at G = 16 with spectroscopic input. Served from Gaia@AIP (`gaiadr3_contrib.aqueiroz2023_*`). Use **v2 tables only**; v1 has a known age-prior bug.

**Our strategy:** Bailer-Jones+2021 baseline across the catalogue; StarHorse2 for the APOGEE × XP training sample (spectroscopically-informed, internally consistent with DR19). Inside Pipeline 1, distance is a *joint output label* with Av and stellar parameters — not a frozen input. Bailer-Jones value serves as prior and cross-check.

Concrete query recipes: `data_acquisition.md` §7.

---

## 6. Stellar evolution and surface abundances — galactic vs stellar clocks

### 6.1 What evolution does to surface abundances

Surface C, N, Li, 12C/13C, and subtler trends are not preserved from birth:

- **First dredge-up (FDU)** (Iben 1965) — near the base of the RGB, convective envelope deepens into CNO-processed layers. Post-FDU: [C/Fe] decreases 0.1–0.3 dex; [N/Fe] increases 0.1–0.4 dex; 12C/13C falls from ~90 to 20–30. **Amplitude scales with birth mass** — FDU deeper for lower-mass stars → [C/N] stellar clock (Salaris+2015).
- **Thermohaline / extra mixing** post-RGB-bump (Charbonnel & Zahn 2007; Eggleton+2006; Charbonnel & Lagarde 2010, A&A 522, A10; Lagarde+2012, A&A 543, A108) — further C depletion; at [Fe/H] < −0.5 can erase the [C/N] mass signal.
- **Red clump** — additional mixing; RGB [C/N]–mass calibration does not apply. Hawkins+2018, Bovy 2016: [C/N] itself can spectroscopically separate RC from RGB.

### 6.2 Galactic clocks (unchanged by stellar evolution in low-mass stars)

- **[Fe/H]** — ISM at birth. Canonical galactic clock.
- **[α/Fe]** — Type II vs Ia SN timescale; separates α-rich and α-poor discs.
- **Individual [Mg/Fe], [Ca/Fe], [Si/Fe], [Ti/Fe]** — different nucleosynthetic channels.
- **[Mn/Fe], [Ni/Fe]** — iron-peak, Ia-sensitive.
- **[Al/Fe]** — separates in-situ from accreted halo cleanly (Hawkins+2015; Das+2020; Belokurov & Kravtsov 2022). Key discriminator for downstream population classification (Starfold).
- **[Eu/Fe], [Ba/Fe]** — r/s-process indicators.

### 6.3 Stellar clocks (evolutionary-stage-dependent)

- **[C/N]** — RGB-only, [Fe/H] > −0.8, Teff 4200–5100 K (Masseron & Gilmore 2015; Martig+2016; Hawkins+2018; Spoo+2022).
- **[C/Fe], [N/Fe]** individually — evolutionary-stage-dependent.
- **12C/13C** — inaccessible from XP.
- **[Li/Fe]** — post-MS Li depletion tracker; mainly for Li-rich giants.

### 6.4 Implications for Pipeline 1

1. **Evolutionary-stage classification head** — RGB / RC / subgiant / MS-turnoff / AGB classifier trained jointly with the regression heads.
2. **Cool-giant cut** — train on Teff > 4000 K, flag extrapolation below 4200 K (APOGEE DR19 [N/Fe] systematics).
3. **[C/N] age calibration** applied only to RGB, [Fe/H] > −0.8, Teff 4200–5100 K. Flag elsewhere.
4. **Tier 2 labels carry evolutionary-stage tags** — downstream users (Starfold included) must respect them.

---

## 7. Pipeline 1 — `xp_abundances` (main)

### 7.1 Architecture

A **semi-supervised multi-task regression with heteroscedastic covariant uncertainty**, trained on APOGEE DR19 labels, with Gaia XP coefficients + auxiliaries as input. The existing contrastive-pretraining → multi-task regression architecture (Neitzel+2025 prototype) is retained as the backbone; we add calibrated uncertainties, ensembling, and the preprocessing / audit scaffolding.

**Inputs per star:**
- Gaia DR3 XP coefficients: 55 BP + 55 RP, **after Ye+2024 NN flux-correction preprocessing, then normalised per `data_acquisition.md` §6.4** (divide by first coefficient, log+z-score the first coefficient).
- Gaia DR3 auxiliaries: G, BP−G, G−RP, parallax + σ, RUWE.
- Photometric auxiliaries: 2MASS J, H, K; WISE W1, W2 (where available).
- Extinction features: line-of-sight Av, neighborhood-median Av, local Av dispersion (§5.3).
- Optional (ablation only): Galactic position (l, b, d). **Must be flagged in audits** — common route for training-prior leakage.

**Outputs per star (label vector + full covariance):**
- Tier 1: Teff, log g, **[M/H]** (ASPCAP global metallicity, not per-element [Fe/H] — see §3.2), Av_model, d_model.
- Tier 2: **[Fe/H]** (per-element), [α/M], [C/Fe], [N/Fe] (RGB only, gated by evolutionary-stage head), [Mg/Fe] if separable, evolutionary-stage probability vector.
- Heteroscedastic covariance matrix (network predicts mean + lower-triangular Cholesky factor).

**Architecture:**
- **Contrastive pre-training** on the full XP corpus (unsupervised). Positive pairs from same-star multi-epoch or within-cell (same Teff–log g–[Fe/H]) sampling. Architecturally aligned with Buck & Schwarz 2024's CLIP-style framework but targeted at production regression rather than proof-of-concept embedding.
- **Supervised fine-tuning** on APOGEE DR19 × XP overlap.
- **Multi-task heteroscedastic head** (Kendall & Gal 2017) — predicts label means and covariance.
- **Ensemble**: 5–10 independent training runs with different random seeds. Output ensemble mean and total uncertainty decomposed into aleatoric (heteroscedastic) and epistemic (ensemble spread) components.

**Uncertainty calibration is a first-class deliverable requirement.** The catalogue must carry *calibrated* σ, not merely quoted σ. Specifically:
- **Reliability diagrams** (Guo+2017) must show predicted σ tracking observed residual σ within 10% across the full parameter space, stratified by Teff, log g, [Fe/H], Av, G.
- **Coverage tests**: do 68%, 95%, 99% credible intervals contain truth at those rates on APOGEE hold-out? Miscalibration > 5 percentage points rejects the label from release.
- **Post-hoc calibration** via temperature scaling or isotonic regression if the raw heteroscedastic head is miscalibrated.
- **Conformal prediction intervals** (Angelopoulos & Bates 2023) released as alternative uncertainty product for downstream users needing distribution-free guarantees.

**Training set selection:**
- APOGEE DR19 giants: `flag_bad == 0`, SNR_APOGEE > 70, Teff 4000–5500 K, log g < 3.5, Mészáros+2025 corrections applied.
- Gaia XP available with SNR thresholds per De Angeli+2023.
- Bailer-Jones+2021 or StarHorse2 distances.
- 70 / 15 / 15 train/val/test, stratified on [Fe/H], Teff, Galactic latitude.

Data preparation, cross-matching, and feature-matrix construction: `data_acquisition.md` §3, §6, §10.

### 7.2 Flux representation — coefficients, not sampled flux

**Decision (2026-04-18): Pipeline 1 operates on the 55 BP + 55 RP Hermite coefficients, not on Ye+2024's 330-element sampled-flux output.**

This is an architectural commitment, not a preference, and it is binding on `src/arqueogal/xp_abundances/main/` before any model code is written against the Stream-1 feature matrix.

**Why coefficients:**

1. **Benchmark alignment.** Every peer Pipeline-1-class model we must match or exceed operates on coefficients: Andrae+2023 (RGB vetted catalogue, 7-param stellar parameters from XP), Li+2024 (AspGap, xgboost on coefficients + colours), Buck & Schwarz 2024 (NeurIPS ML4PS, CLIP-style contrastive on coefficients), Guiglion+2024 (deep ensemble on coefficients + parallax + photometry). Zhang+2023 is the lone sampled-flux peer and is a physics-informed forward model (their loss requires a predicted flux to compare against observed flux) — a class of method we do not use.
2. **Native XP representation.** Coefficients are the actual Gaia DR3 release product. The sampled-flux axis is a derived representation chosen by Ye+2024 for their correction NN; it is not what Gaia publishes and not what downstream users work with.
3. **Dimensionality.** 110 coefficients vs 330 flux points. Fewer parameters → tighter priors → less capacity for representation bloat → smaller ensembles fit within the 6 GB VRAM budget.
4. **Normalisation mathematics.** The §6.4 recipe (divide c₁..c₅₄ by c₀, log+z-score c₀ per band) is defined on coefficients. It extracts a physically meaningful quantity — the total flux proxy in c₀ and the shape information in the normalised higher orders — and is the same scheme Buck & Schwarz 2024 and Guiglion+2024 use.
5. **Ye+2024 is a preprocessing step, not an input layer.** The correction reduces 3.2–3.7 % relative flux systematics to 1.2–2.4 %. We want its *correction* applied to the coefficients, not its *representation* propagated to our model.

**The adapter pipeline:**

```
Ye+2024 sampled flux on geomspace(360, 990, 330) nm  (per-star, 97.4 % yield)
    │
    ▼
linear least-squares re-projection onto the 55+55 Hermite basis
    │   ├── emit coefficients c_bp[55], c_rp[55]
    │   └── emit re-projection residual RMS as per-star QC feature
    ▼
normalise: c_bp[1:]/c_bp[0], c_rp[1:]/c_rp[0]   (§6.4 step 2)
    │
    ▼
log10(c_bp[0]), log10(c_rp[0]); z-score across Stream 1 training set;
    freeze (μ, σ) for Stream 3 inference   (§6.4 step 3)
    │
    ▼
Pipeline-1 feature matrix: {c_bp_norm[54], c_rp_norm[54], log_c_bp_0_z,
    log_c_rp_0_z, reprojection_residual_rms, ye2024_flag, ...}
```

**Consequences:**

- `data_acquisition.md` §6.4 is amended: steps 2–5 apply *after* Ye+2024 sampled-flux output has been re-projected onto the Hermite basis, not directly on the stored `corrected_flux` column. The §6.4 step 1 output (`xp_sampled_corrected.parquet`) is an *intermediate* now, not the Pipeline-1 input.
- `pipeline1_features_stream1.parquet` as currently materialised (74 cols, with `corrected_flux` as a 330-element array) is the **intermediate representation**, not the training matrix. It will be re-emitted once the Hermite re-projection adapter ships in `src/arqueogal/data/gaia_xp.py`. Code in `src/arqueogal/xp_abundances/main/` must not consume `corrected_flux` directly.
- The re-projection residual is a feature, not a rejection criterion. Stars with high residual are retained; the model (and downstream user) can read the residual as a trust signal.
- Ye+2024's OOD diagnostic stays on the sampled-flux side: the Mahalanobis distance from the Ye training-sample mean in the 14-feature input space is independent of our coefficient choice and travels alongside the coefficients as an uncertainty feature.

**This decision closes Task #98. It does not override future promotion protocols:** if an experimental-arm model demonstrably benefits from sampled flux (e.g. a physics-informed forward model per Zhang+2023), that remains a valid `experimental/` track. The main pipeline is committed.

### 7.3 Scope and magnitude regime

Project scope (April 2026): **we do not push beyond the XP-native release (G ≲ 17.65), and we do not chase the G > 17 regime**. Practical usable abundance regime is G ≲ 15–16 for red giants with good XP SNR. Within this regime we focus on:

1. **Perfection, not reach** — match or exceed AspGap/Guiglion+2024 RMSE/bias levels while delivering calibrated uncertainties they lack.
2. **Extension to additional elements** where §3.3 statistical promotion criteria pass.
3. **Information-content audit** of every released label.
4. **Tier 1/2/3 release honesty**.

This focus aligns with the Pipeline 1 deliverable (D-Cat-b contribution, Month 6) which does not require G > 17.

### 7.4 Training-prior leakage guards

Three leakage routes and their guards:

1. **Teff – [α/M] correlation in APOGEE**. APOGEE giants span narrow Teff, and [α/M]–Teff correlate weakly because the α-rich disc is kinematically hot, older, slightly different on the RGB. Guard: stratified sampling + explicit permutation test on Teff (§9.2).
2. **Position-chemistry correlation** (Galactic chemical gradient). APOGEE samples specific sightlines; network can learn "star at (l, b) → [Fe/H]" without spectral evidence. Guard: ablation trains with/without position features; release two variants.
3. **Extinction-spectrum-position triple correlation**. Guard: permutation test on Av + position (§9.2).

---

## 8. Pipeline 1 — experimental arm

**Strictly segregated** from the main pipeline, living in `src/arqueogal/xp_abundances/experimental/` with its own DESIGN.md, diagnostics, tests, configs. Does not contribute to D-Cat-b unless promoted after passing the same validation bar as main (§8.6).

### 8.1 Normalising-flow label posteriors
Replace the heteroscedastic Gaussian head with a conditional normalising flow (`zuko`, `nflows`) over the label vector given XP + auxiliaries. Allows non-Gaussian, multi-modal posteriors — critical for boundary stars and [α/M] in transition regions. Precedent: Green et al. 2023 for stellar parameters.

### 8.2 Transformer-on-spectra with masked prediction
Leung & Bovy's StellarPerceptron (arXiv:2411.04750) as template. Missing-input tolerance at inference; natural multi-task structure. Rózański, Ting & Jabłońska 2023 attention approach is a related reference.

### 8.3 Physics-informed synthetic-spectrum regularisation
Inject a prior regularisation term penalising label vectors that produce forward-modelled XP spectra (PHOENIX or MARCS grids convolved to XP response) inconsistent with observation. Computationally heavy but the only route to honest out-of-distribution detection. Enables simulation-based inference (SBI) as a further extension.

### 8.4 Diffusion models for conditional density estimation
Recent demonstrations in adjacent astronomical contexts (Rouhiainen+2024). Potentially more expressive than normalising flows at higher inference cost. Note: Buck & Schwarz 2024 explicitly flag diffusion models as the natural next step from their CL framework.

### 8.5 Multimodal contrastive co-learning (Buck & Schwarz 2024 style)
This is **the experimental route most closely aligned with existing SOTA**. Buck & Schwarz 2024 (arXiv:2410.16081) showed CLIP-style contrastive learning on Gaia DR3 RVS + XP produces highly structured 32-D shared embeddings (τ = 0.01, LAMB optimiser, batch size 16384), with k-NN (k = 13) on the embedding achieving R² = 0.987 on Teff and R² = 0.849 on [α/M] on 44 780 APOGEE-labelled validation stars. They use a CNN for RVS (adapted from Guiglion+2024's architecture) and a 1-layer MLP for XP. Their 841 300-pair training corpus is entirely self-supervised.

**Key adaptations for our experimental arm:**

- **Use RVS as a training-time modality only.** At inference, the main pipeline remains XP-only, preserving applicability to the full G ≲ 17.65 XP footprint. This is the "co-learning" branch of multimodal taxonomy (helper modality during training only) per Baltrušaitis+2019.
- **Train CL on ~800 k RVS+XP pairs** (B&S protocol), fine-tune for regression on DR19 labels. Their 44 780 APOGEE-labelled stars are in DR17; our ~700 k DR19 overlap is ~15× larger and label-richer.
- **Test candidate gains over main:** does the CL-pretrained XP encoder outperform the contrastive-pretrained XP-only baseline on (i) uncertainty calibration, (ii) out-of-distribution robustness to missing 2MASS/WISE, (iii) individual-element recovery under the §3.3 promotion protocol? If it wins on ≥2 of 3 criteria, it becomes a promotion candidate.
- **Architecture improvements over B&S**: replace their 1-layer MLP XP encoder with attention over coefficients (per B&S's own forward-looking §4 recommendation); replace k-NN zero-shot with a learned regression head carrying heteroscedastic loss; add an ensemble for epistemic uncertainty. Test τ = 0.01 (B&S) vs a trainable temperature (CLIP standard) — B&S note that results depend on τ.
- **Honest comparison baseline**: on the same DR19 × XP training split, the main XP-only contrastive pipeline should match or exceed B&S's R²(Teff) ≈ 0.987 and R²([α/M]) ≈ 0.849 *at the same k-NN baseline* before declaring the CL-pretraining approach "better". If the XP-only baseline already hits these numbers, the multimodal variant is not needed for regression; only for the cross-modal generation capability B&S demonstrate.

### 8.6 Promotion rule
An experimental method is promoted to main only when it:
- Passes every validation test main passes (§9).
- Beats main by ≥ 0.02 dex on at least one Tier 1 label or ≥ 0.03 dex on one Tier 2 label on hold-out test set.
- Passes the interpretability audit (§9.2) at least as cleanly.
- Matches or exceeds main's calibration quality (reliability diagram error ≤ main's).

---

## 9. Validation methodology for Pipeline 1

**All validation is on real data** — APOGEE DR19 hold-out, open clusters with literature abundances, Gaia FGK benchmark stars. No FIRE-2 for Pipeline 1.

### 9.1 Standard statistical performance and uncertainty calibration

Uncertainty calibration is elevated here to an equal standing with point-estimate accuracy.

- **Per-label RMSE, bias, reduced χ² vs APOGEE hold-out**, stratified by Teff, log g, [Fe/H], Av, G (4 × 2 × 4 × 3 × 3 cells). Report worst cell, not just mean.
- **Precision floor via open-cluster differential abundances** (Bovy 2016, Spina+2021, Casamiquela+2020). Within-cluster predicted dispersion sets empirical precision floor.
- **Gaia FGK benchmark stars** (Jofré et al. 2014, 2018) external cross-validation.
- **Reliability diagrams** (Guo+2017) — predicted σ vs observed residual σ, target agreement within 10%. Apply temperature scaling or isotonic regression if miscalibrated.
- **Coverage tests** — 68%, 95%, 99% credible intervals must contain truth at those rates within 5 percentage points. Labels failing coverage are not released per-star.
- **Per-cell calibration audit** — reliability diagrams computed *per (Teff, log g, [Fe/H]) cell* separately, not just globally. Uncalibrated cells identified and fixed via stratified temperature scaling.
- **Conformal prediction intervals** — released as alternative uncertainty product for downstream users needing distribution-free guarantees.

### 9.2 Information-content audit (the transcendence contribution)

Six tests per released label:

1. **Leave-one-coefficient-out (LOOCO).** Drop one of 110 XP coefficients at inference, measure shift in predicted label. Aggregate over coefficients → "XP dependence" vector per label.
2. **Permutation feature importance.** Permute one feature across the test set, measure R² drop per label. Heatmap: label × feature.
3. **SHAP values** on N=10k subsample. Per-star feature attributions for outlier inspection.
4. **Shuffled-spectrum null.** Within each Teff–log g cell, randomly permute XP coefficients across stars. Retrain or evaluate. Labels retaining skill are 100% prior-driven — flag in catalogue.
5. **Mutual information** MI(XP; [X/Fe] | Teff, log g, [Fe/H]). Residual MI bootstrap confidence interval above zero is required for Tier 2 promotion. **Primary estimator (v1.1): PCA-summary KSG CMI.** The XP block (BP + RP normalised Hermite coefficients; 108 features on the Pipeline 1 v1 layout) is projected onto a PCA summary retaining ≥ 95% of its variance (default 7 components for Gaia XP; 15 components ≈ 99% when aux-absorption diagnosis is needed — see §9.2.1 and the [α/M] triage finding below). The 2-D summary used by v1 is deprecated (§9.2.1); reproducibility of historical report cards is preserved via a `legacy_2d` flag in `src/arqueogal/xp_abundances/main/audit.py`.
6. **Decorrelated sub-sample test.** Stratified sample in which [X/Fe] is decorrelated from [Fe/H] (match α-rich and α-poor at similar [Fe/H], Teff, log g, Av, position). Retrain. If prediction survives decorrelation, spectrum-driven; if collapses, prior-driven.

#### 9.2.1 KSG on low-dimensional XP summaries — 2-D deprecated, PCA primary (audit protocol v1.1)

Added 2026-04-19 after the Pipeline 1 v1 audit; audit protocol bumped **v1 → v1.1** at the same time. KSG-based CMI estimates on 2-D summaries of high-dimensional spectral signals (e.g. ``(|BP|-sum, |RP|-sum)``) are prone to upward bias when the spectral signal is carried by low-order Hermite structure that correlates with the summary, and to downward collapse when the signal is carried by higher-order structure that the 2-D summary discards. **In the Pipeline 1 v1 audit, three of five labels showed substantial 2-D bias** — Teff 4.6× upward inflation (0.1352 nats 2-D vs 0.0296 PCA), [M/H] 4× underestimation (0.0088 vs 0.0357; 2-D collapsed below the release-gate floor), [Mg/H] pathological collapse (0 vs 0.0357). Only log g showed mild bias (0.0401 vs 0.0311, 1.3× inflation). The 2-D estimator is therefore no longer fit-for-purpose as the primary Test-5 methodology.

**Protocol (v1.1): audits must use PCA summaries with ≥ 95% variance explained as the primary CMI estimate. The default is 7 components for Gaia XP, which retained 95.8% of the BP + RP normalised-shape variance on the Pipeline 1 v1 val split.** The 2-D estimator is retained behind a `legacy_2d` flag in `src/arqueogal/xp_abundances/main/audit.py` for reproducibility of historical report cards only; calling it emits a `DeprecationWarning`. The KSG k, conditioning set, and subsample cap should match across PCA (and any legacy 2-D) estimates so only the summary dimensionality changes.

**Escape hatch for high-order or aux-absorbing signals.** When the 7-component default returns CMI near zero for a label that is otherwise information-rich (passing shuffle-null and joint-XP-shuffle), the `scripts/triage_alpha_m_cmi.py` three-test sequential protocol disambiguates three candidate causes: H1 high-order Hermite structure (push the PCA to 15 components, ~99% variance), H3 KSG clamp artefact (rerun with the clip off), H2 aux absorption (reduce conditioning to parallax alone). The Pipeline 1 v1 [α/M] case — where 7-PC PCA-CMI was 0.0000 — was adjudicated by this triage: H2 confirmed, with parallax-only 15-PC CMI = 0.1125 nats (~56× the 0.02 floor) vs full-aux 15-PC CMI = 0.0000. See `reports/pipeline1/audit/alpha_m_triage.md` for the decision record. When triage is needed, the tier decision remains adjudicated by the shuffle-null and the three-question diagnostic (§3.3.1); CMI is a methodological cross-check, not a release-blocking quantity for labels flagged H2.

### 9.3 Interpretability audit — the per-label report card

Each released label carries a "report card" summarising §9.2 results in the release documentation. Catalogue consumers get a direct answer to *how much of this prediction is spectrum vs prior?*. Labels self-identify as spectrum-dominant (per-star use) or prior-dominant (population-level use only).

### 9.4 Cross-catalogue validation

Compare against AspGap, Guiglion+2024, SHBoost, Andrae+2023, Ye+2024, Buck & Schwarz 2024 embedding k-NN, GALAH×RVS on common stars. Disagreement is informative — two pipelines agreeing on [Fe/H] but disagreeing on [α/M] isolates the spectral-vs-prior contribution.

---

## 10. Population classification — moved to Starfold (2026-04-22)

The UMAP+HDBSCAN population classifier originally scoped as "Pipeline 2" has been spun out into the separate **Starfold** repository. The methodological detail previously in §§10–11 (architecture, feature-space redesign, soft memberships, DBCV hyperparameter selection, §10.5 unsupervised diagnostic stack, FIRE-2 hare-and-hounds, experimental-arm candidates, collaborator HPC-sweep assessment) now lives there; Starfold is the authoritative reference for that work.

What stays in this document is the **interface contract** between the two repositories:

- This repo ships Pipeline 1 predictions (5-label μ + full 5×5 Σ, OOD flags, Regime-B flag, `selection_prob`, `aux_missingness_*`, `release_tier`) on the Stream 3 inference parquets.
- Starfold consumes Tier 1 (and optionally Tier 2) rows from those parquets, applies its own feature assembly (age, kinematics, [C/N], etc.), and produces the D-Cat-d soft memberships.
- The kinematics module (`src/arqueogal/data/kinematics.py`) may be exposed as an importable utility or duplicated inside Starfold — choice deferred to Starfold's own architecture review.
- FIRE-2 method validation (Subtask 5.1) follows Starfold, not this repo; the real-data-only policy in `data_acquisition.md` §0 still applies to everything that feeds Pipeline 1.

See `docs/plan/04_pipeline2_main.md` for the integration contract in full.

---

## 12. Real data vs FIRE-2 — a sharp methodological boundary

For Pipeline 1 in this repo:

| | Pipeline 1 (`xp_abundances`) |
|---|---|
| Training data | Real: APOGEE DR19 × Gaia XP |
| Validation data | Real: APOGEE hold-out, open clusters, Gaia FGK benchmarks |
| Ground truth available | Yes (APOGEE labels) |
| Quantitative metrics | RMSE, bias, reliability diagrams, coverage, §9.2 audit |
| Role of FIRE-2 | None |
| Data acquisition scope | `data_acquisition.md` streams 1, 2, 3 |

The FIRE-2 hare-and-hounds (Subtask 5.1) and the real-data D-Cat-d validation (Subtask 5.2) are Starfold concerns. The corresponding methodological boundary — FIRE-2 as method-validation only, real-data diagnostics for D-Cat-d — is enforced there.

---

## 13. What this programme transcends

| Reference | How Pipeline 1 transcends |
|---|---|
| **Buck & Schwarz 2024** | Production catalogue vs their proof-of-concept; calibrated heteroscedastic covariant uncertainties with reliability diagrams and coverage tests (they report R² only); information-content audit; APOGEE DR19 vs DR17; extension to individual elements beyond [α/M]; their multimodal CL becomes our experimental benchmark (§8.5). |
| **Guiglion+2024** | XP-only drops RVS G<14 ceiling; DR19 labels; neighborhood-median Av prior; Tier 1/2/3 tagging; calibrated uncertainties. |
| **AspGap (Li+2024)** | Joint distance-Av fit; broader Teff-log g window; evolutionary-stage head; heteroscedastic covariance; per-label audit. |
| **Fallows & Sanders 2024** | Ensemble/uncertainty philosophy + §9.2 audit converts their caveat ("training priors may drive bimodality") into measured, per-label released quantity. |
| **Ye+2024** | Adopt their flux-correction; extend to α and [C/N] they do not attempt. |
| **All current XP-abundance catalogues** | The information-content audit and Tier 1/2/3 honesty. No one else does this. |

(Transcendence claims over population-classification prior art — Neitzel+2025 and the Gaia-population catalogues — are documented in Starfold.)

The transcendence is not "more stars" or "more elements" or "fainter magnitudes" — all three races are saturated. It is **honesty under uncertainty**: every prediction carrying its information-content report card, every uncertainty calibrated and coverage-tested, and a clean separation between method-validation and science-validation (real data + §9). Defensible as PhD contribution.

---

## 14. Risks, gaps, and honest limitations

1. **DR19 systematics**: Mészáros+2025 Teff-trend corrections are new, not fully stress-tested.
2. **Ye+2024 flux correction**: not yet peer-reviewed in final form. Freeze the version we use; log version.
3. **Extinction beyond 1.25 kpc**: Edenhofer+2024 limit. Beyond, rely on GSP-Phot neighborhood-median (5 GB budget forces this choice over Bayestar19; acceptable substitute with documented caveats in `data_acquisition.md` §8). Far-disc Av uncertainty is a known limitation.
4. **Red-clump contamination** in the "RGB" evolutionary-stage class propagates into [C/N]-age errors if > 5%. Seismic evolutionary-stage classification from Task 2 ground-truths a subsample.
5. **FIRE-2 non-representativity**: even bimodal Ananke galaxies may not match MW α-bimodality precisely. Handled in Starfold's Subtask 5.1.
6. **[Al/Fe] availability from XP**: Tier 3 at R = 50. Rely on APOGEE [Al/Fe] for APOGEE-observed subsample; ~60% of catalogue will lack [Al/Fe], limiting halo discrimination downstream. Honest gap in D-Cat-d (Starfold).
7. **Downstream compute budget** for Starfold's MC × DBCV × bootstrap grid is Starfold's concern; this repo's Pipeline 1 inference cost is separately tracked in `docs/plan/03_stream3_inference.md`.
8. **Age uncertainty**: 15–30% seismic precision is a fundamental chronological-resolution floor for any downstream population analysis. If Task 4 delivers > 25% median, expect fewer populations resolvable in age dimension.
9. **Buck & Schwarz 2024 is a workshop paper** (NeurIPS ML4PS 2024): contribution established but not peer-reviewed to journal standard. Cite appropriately in our publications.
10. **Data-acquisition fragility**: `data_acquisition.md` relies on AIP TAP availability, SDSS DR19 server uptime, VizieR cross-match stability. Script-level retries, sidecar provenance, and batch checkpointing mitigate but do not eliminate. Budget ~1 week for an unattended full-sample acquisition.
11. **Low-latitude, high-extinction incompleteness of the D-Cat-b release.** Pipeline 1 Tier 1/2 outputs exclude stars with `xp_fit_flag_residual_high` set. On Stream 1 (2026-04 materialisation) these Hermite-catastrophic rows concentrate on A_V_SFD ≈ 23 and low galactic latitude (|b| < 10°), with median G ≈ 14.6 and median Teff ≈ 4900 K — Galactic-plane disc territory where Ye+2024's NN extrapolates against saturated SFD columns and near-zero blue flux. Flag rate in Stream 1: 2 521 / 315 616 ≈ 0.80%. This is a structural selection effect, not a bug: the true line-of-sight extinction is unknowable from SFD alone at those columns, so neither we nor any downstream user can recover those stars without a better 3D dust prior. The D-Cat-b release therefore carries documented low-latitude high-extinction incompleteness. The scientifically sensitive halo-structural population is less affected than expected — the catastrophic group sits firmly in the disc, not the halo — so the cost of the exclusion is modest.
12. **Ye+2024 out-of-distribution behaviour at the Teff edges.** In the ≥ 6000 K Stream 1 bin, 4.1% of Ye-corrected spectra fail the Hermite catastrophic-residual test (p99 in-bin = 2.9e-12, ~440× the global normal p99); in the < 4000 K bin the rate reaches 30% on a small-N tail. Both are structural: Ye's APOGEE training is thin at the hot end (Paschen lines at the RP edge), and cool giants have strong TiO/VO molecular bands that the NN was not trained hard on. The Pipeline 1 release policy should almost certainly exclude Teff > 6000 K and Teff < 4000 K from Tier 2 per-star output with an explicit note — a Pipeline 1 decision, not a data-layer decision.

---

## 15. Key references

**Gaia XP data**: De Angeli et al. 2023 (A&A 674, A2); Carrasco et al. 2021; Montegriffo et al. 2023; Weiler et al. 2023; Huang et al. 2024.

**XP abundance pipelines**: Andrae, Rix & Chandra 2023 (MNRAS 521, 3527; arXiv:2302.02611); Zhang, Green & Rix 2023 (MNRAS 524, 1855; arXiv:2303.03420); Li, Zhang, Rix et al. 2024 — AspGap (ApJ 974, 42; arXiv:2309.14294); Guiglion et al. 2024 (A&A 682, A9; arXiv:2306.05086); Anders et al. 2024 — SHBoost (A&A 691, A127; arXiv:2407.06963); Fallows & Sanders 2024 (MNRAS 531, 2126; arXiv:2405.10699); Hattori 2024 (ApJ 969, 81; arXiv:2404.01269); Ye, Allende Prieto et al. 2024 (A&A in press; arXiv:2411.19105); **Buck & Schwarz 2024 — Deep Multimodal Representation Learning for Stellar Spectra (arXiv:2410.16081; NeurIPS 2024 ML4PS workshop)**; Leung & Bovy 2024 — StellarPerceptron (arXiv:2411.04750); Ardern-Arentsen et al. 2025 (MNRAS 537, 1984; arXiv:2410.11077); Rix et al. 2022 (ApJ 941, 45).

**APOGEE DR19**: SDSS Collaboration 2025 (arXiv:2507.07093); Mészáros et al. 2025 (arXiv:2506.07845); https://www.sdss.org/dr19/mwm/data/.

**3D dust**: Edenhofer et al. 2024 (A&A 685, A82; arXiv:2308.01295); Green et al. 2019 — Bayestar19; Lallement et al. 2022 (A&A 661, A147); Vergely, Lallement & Cox 2022.

**Distances**: Bailer-Jones et al. 2021 (AJ 161, 147; arXiv:2012.05220); Queiroz et al. 2023 — StarHorse2 (A&A 673, A155); Khalatyan et al. 2024 — SHBoost (A&A 691, A127).

**Stellar evolution & clocks**: Iben 1965; Charbonnel & Lagarde 2010 (A&A 522, A10); Lagarde et al. 2012 (A&A 543, A108); Charbonnel & Zahn 2007; Masseron & Gilmore 2015 (MNRAS 453, 1855); Martig et al. 2016 (MNRAS 456, 3655); Hawkins et al. 2018 (MNRAS 481, 5592); Spoo et al. 2022; Salaris et al. 2015; Bovy 2016.

**TESS asteroseismology**: Hon et al. 2021 (ApJ 919, 131; arXiv:2108.01241); TASOC; Silva Aguirre et al. 2020; Serenelli et al. 2017.

**Uncertainty & calibration**: Kendall & Gal 2017 — heteroscedastic deep learning; Guo et al. 2017 — temperature scaling; Angelopoulos & Bates 2023 — conformal prediction; Venna & Kaski 2001 — trustworthiness.

**Population classification, actions, halo substructure**: McInnes, Healy & Melville 2018 — UMAP (arXiv:1802.03426); Sainburg, McInnes & Gentner 2021 — Parametric UMAP (arXiv:2009.12981); Campello, Moulavi & Sander 2013 — HDBSCAN; Moulavi et al. 2014 — DBCV; Hennig 2007 — cluster stability; Myeong et al. 2019; Koppelman et al. 2019; Belokurov et al. 2018; Dodd et al. 2023; Horta et al. 2021; Belokurov & Kravtsov 2022 — [Al/Fe] accreted/in-situ; Hawkins et al. 2015; Das et al. 2020; Ceccarelli et al. 2024.

**FIRE-2 and synthetic surveys**: Hopkins et al. 2014, 2018; Wetzel et al. 2016, 2023 (FIRE-2 DR; ApJS 265, 44); Sanderson et al. 2020 — Ananke; Nguyen et al. 2024 (ApJ 966, 108); Barry et al. 2026 (arXiv:2601.02520); Parul et al. 2025 (MNRAS 537, 1571); Bellardini et al. 2022.

**Kinematics**: Recio-Blanco et al. 2023 (A&A 674 A29; arXiv:2206.05534); Bovy 2015, ApJS 216, 29; McMillan 2017, MNRAS 465, 76; Reid & Brunthaler 2020; GRAVITY Collaboration 2018, 2021; Schönrich, Binney & Dehnen 2010 (solar motion); Binney 2012 (Staeckel fudge).

**Multimodal ML taxonomy**: Baltrušaitis, Ahuja & Morency 2019 (IEEE TPAMI 41, 423).

**ArqueoGal foundational**: Campante et al. 2016 (ApJ 830, 138); Miglio et al. 2021 (A&A 645, A85); Neitzel et al. 2025 (A&A 695, A243; arXiv:2501.16294); Campante FCT proposal 2024.15303.PEX.

---

*End of research_brief.md (v2). Living reference; update when significant new literature emerges or pipeline decisions are revised. See companion `data_acquisition.md` for all data pipeline specifics.*
