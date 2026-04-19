# ArqueoGal — Data Acquisition & Preprocessing Plan

**Author:** Andreas Neitzel (Co-I, PhD student, CAUP/IA)
**Scope:** This document defines the complete data acquisition, cross-matching, and preprocessing pipeline for Andreas Neitzel's workspace contributions to ArqueoGal. It is the primary reference for every data-related task and should be consulted before writing any data-handling code.
**Storage budget:** 5 GB total (local laptop). This is a hard constraint and shapes every decision below.
**Last revision:** April 2026 (v1)

---

## 0. Overview and constraints

### The three data streams

| Stream | Purpose | Approx. size (stars) | Target disk footprint |
|---|---|---|---|
| **1. APOGEE DR19 × Gaia DR3** | Training labels for Pipeline 1 (`xp_abundances`) | ~700 k | ~1.0 GB |
| **2. TESS Hon+2021 + TASOC × Gaia DR3** | Pre-staging for future asteroseismic ages (Task 4) | ~160 k | ~250 MB |
| **3. Gaia RGB+RC application sample** | Pipeline 1 inference set + Pipeline 2 feature basis | 1–2 M | ~2.0 GB |
| **Shared**: dust maps, distances, orbits, master catalog | | | ~1.0 GB |
| **Headroom** | | | ~750 MB |

**Total target: 4.25 GB within a 5 GB budget.** The budget forces three non-trivial decisions up front:
- **No Bayestar19** (~3 GB download) on local disk. Use the smaller Edenhofer+2024 product for d < 1.25 kpc and Gaia GSP-Phot neighborhood-median A_G for the rest. Full 3D dust-map substitution is analysed in §8.
- **No full Gaia DR3 dump.** All Gaia queries are selective (column subsets, source_id-IN batches, quality cuts). Never `SELECT *`.
- **No XP covariance matrices.** The full 55×55 correlation matrix per band is ~24 KB per star; for 2 M stars that's 48 GB. Keep only coefficient means + per-coefficient errors. Revisit only if demonstrably needed (it isn't for Pipeline 1 main).

### Architectural principles

1. **Pyvo over astroquery.** `astroquery.gaia` has shown recurring instability in recent months. Use `pyvo` for all TAP-based queries (Gaia@AIP, GAVO, VizieR). Use `requests` for direct HTTP downloads of catalogue files. Keep `astroquery` as a last-resort fallback only when no TAP endpoint exists.
2. **AIP TAP+auth wherever possible.** The project team holds an AIP account. AIP mirrors Gaia DR3 and hosts the Queiroz+2023 StarHorse2 tables under `gaiadr3_contrib`. Authenticated access gives better quotas and async query support than public ESA TAP.
3. **Immutable raw, derived outputs in Parquet.** Raw downloads go to `data/raw/` and are never modified. Intermediate cleaned products go to `data/interim/`, analysis-ready feature matrices go to `data/processed/`, all as Parquet (columnar, compressed, cudf-compatible).
4. **Provenance is logged.** Every Parquet file ships with a companion `*.provenance.json` containing source URL, query timestamp, ADQL query string, row count, quality cuts applied, and git SHA of the ingestion script.
5. **Batched, resumable queries.** Large queries split into chunks of 5 000–10 000 `source_id`s, checkpointed to disk. A crash mid-query does not restart from zero.
6. **All Gaia corrections applied at ingestion**: Lindegren+2021 parallax zero-point, Riello+2021 (A&A 649, A3 Appendix A) G-band flux/magnitude correction. Never work with raw Gaia astrometry downstream.

---

## 1. Prerequisites

### Credentials and accounts

| Service | URL | Auth | Purpose |
|---|---|---|---|
| **Gaia@AIP TAP** | https://gaia.aip.de/tap | User/password (AIP account) | Gaia DR3 + StarHorse2 + SHBoost |
| **ESA Gaia TAP** (backup) | https://gea.esac.esa.int/tap-server/tap | Optional Gaia Archive account | Fallback if AIP down |
| **GAVO TAP** (backup) | https://dc.g-vo.org/tap | None | Bailer-Jones+2021 distances |
| **SDSS DR19** | https://dr19.sdss.org/ | Public | APOGEE DR19 summary files |
| **MAST** | https://mast.stsci.edu/ | None for public | TIC v8.2 |
| **VizieR** | https://vizier.cds.unistra.fr/ | None | Hon+2021 (J/ApJ/919/131) |

**Store credentials in** `~/.arqueogal/credentials.yaml` with `0600` permissions, parsed by the ingestion code. Never commit or hardcode.

### Python packages (already in the `rapidsenv` venv)

```
pyvo                 # TAP queries — primary interface for Gaia/AIP/GAVO
requests             # direct HTTPS downloads
astropy              # FITS I/O, coordinates, units
pandas, polars       # tabular
pyarrow              # Parquet
cudf, cuml           # GPU tabular (RAPIDS 25.10)
GaiaXPy              # XP coefficient → sampled spectrum, calibration
dustmaps             # 3D dust map interface (but see §8 for selective use)
galpy                # orbits
healpy               # healpix for dust maps
tqdm                 # progress bars
```

Verify with `python -c "import pyvo, astropy, GaiaXPy, galpy, dustmaps; print('ok')"`. If any are missing, install with `pip install --no-deps` to avoid bumping RAPIDS pins.

---

## 2. Directory layout

```
data/
├── raw/                                    # immutable downloads
│   ├── apogee_dr19/
│   │   └── astraAllStarASPCAP-0.6.0.fits.gz
│   ├── tess_hon2021/
│   │   └── J_ApJ_919_131.fits.gz
│   ├── tic_v8_2/
│   │   └── tic_v8_2_crossmatch.parquet     # only stars of interest, not full TIC
│   ├── bailer_jones_2021/
│   │   └── gedr3dist_subset.parquet        # only our source_ids
│   ├── starhorse2/
│   │   └── aqueiroz2023_apogee_dr17.fits.gz
│   │   └── aqueiroz2023_gaia_rvs.fits.gz   # only if needed — large
│   └── dust/
│       └── edenhofer2024/                  # managed by dustmaps package
├── interim/                                # cross-matched, cleaned, single-stream
│   ├── stream1_apogee_gaia.parquet
│   ├── stream2_tess_gaia.parquet
│   └── stream3_gaia_rgbrc.parquet
├── processed/                              # analysis-ready feature matrices
│   ├── pipeline1_training.parquet          # APOGEE × Gaia with XP + labels
│   ├── pipeline1_inference.parquet         # Gaia RGB+RC with XP, no labels
│   └── pipeline2_features.parquet          # chrono-chemo-kinematic
├── external/                               # third-party shared catalogues
│   └── hot_stuff/                          # anything else if needed
└── provenance/                             # JSON provenance files (one per Parquet)
    └── stream1_apogee_gaia.provenance.json
```

---

## 3. Stream 1 — APOGEE DR19 × Gaia DR3

### 3.1 Primary source

**APOGEE DR19 summary file** (Astra pipeline, Mészáros+2025):
- URL: `https://dr19.sdss.org/sas/dr19/spectro/astra/0.6.0/summary/astraAllStarASPCAP-0.6.0.fits.gz`
- Approximate size compressed: ~500 MB
- Row count: ~1.05 M ASPCAP results (Mészáros+2025 quotes 964,989 ASPCAP stars; file contains all plus duplicates across visits)
- Reference: Mészáros et al. 2025, AJ in press, arXiv:2506.07845; https://www.sdss.org/dr19/mwm/data/

**Download strategy:** streaming with `requests.get(url, stream=True)` to `data/raw/apogee_dr19/`. Do not un-gzip on disk; astropy.io.fits reads gzipped FITS natively.

### 3.2 Column selection

Read HDU 2 (the aggregated catalogue), select only the columns actually used. Approximate size of selected-column subset: **~150 MB uncompressed, ~50 MB Parquet**.

**Identifiers and cross-match keys:**
- `sdss_id`, `apogee_id`, `source_id` (Gaia DR3 — DR19 publishes this directly; no manual cross-match needed!)

**Atmospheric parameters (calibrated):**
- `teff`, `e_teff`, `logg`, `e_logg`, `m_h_atm`, `e_m_h_atm` (overall [M/H]), `alpha_m_atm`, `e_alpha_m_atm`, `vsini`, `vmicro`, `vmacro`

**Individual abundances (calibrated, [X/H] and per-element where published):**
- DR19 releases ~24 measurements of 21 elements. Keep: `c_h_atm`, `n_h_atm`, `o_h_atm`, `na_h_atm`, `mg_h_atm`, `al_h_atm`, `si_h_atm`, `s_h_atm`, `k_h_atm`, `ca_h_atm`, `ti_h_atm`, `v_h_atm`, `cr_h_atm`, `mn_h_atm`, `fe_h_atm`, `ni_h_atm`, `ce_h_atm` — each with `e_*` uncertainty.
- Also `c_fe`, `n_fe`, and derived C/N if DR19 provides it; if not, compute `c_n = c_fe - n_fe` at ingestion.
- **Note on column naming**: DR19 may use suffix `_atm` or `_1`/`_raw` variants. Use calibrated values (no suffix or `_atm`); the `raw_*` columns are diagnostic only. Cross-check against the Astra documentation at https://www.sdss.org/dr19/mwm/astra/accessing-astra-files/ at ingestion time.

**Flags and quality:**
- `flag_bad` (DR19's primary reject flag), `snr`, `result_flags`, per-element line-threshold flags (e.g., `flag_*_h`)
- `vhelio_avg`, `vhelio_err` (radial velocity)
- `teff_flag`, `logg_flag`, `m_h_flag`, `alpha_m_flag` (element-level flags)

**Spectroscopic metadata:**
- `v_astra` (Astra version tag), `task_id`, `spectrum_pk`

### 3.3 Quality cuts (training set)

Apply at ingestion, store both pre-cut and post-cut row counts in provenance.

```
flag_bad == 0
snr > 70
teff between 4000 and 5500            # red giants only
logg between 1.0 and 3.5              # RGB + RC
m_h_atm between -2.0 and +0.5         # well-calibrated range
```

Expected surviving rows: ~600 k–700 k.

### 3.4 Mészáros+2025 Teff-trend corrections

DR19 abundances exhibit residual Teff-dependent trends along the giant branch. Mészáros+2025 (arXiv:2506.07845) publishes correction polynomials `[X/M]_corrected = [X/M]_raw - poly(Teff, logg)`. **These must be applied before using [X/M] as ML training labels.** Check the paper's supplementary materials or the DR19 release notes for coefficient tables; implement as a single function `apply_dr19_corrections(df)` at `src/arqueogal/data/apogee_dr19.py`. Log pre/post statistics to provenance.

### 3.5 Cross-match with Gaia DR3

**DR19 publishes `source_id` directly in the summary file** (via SDSS-V's internal cross-match). Verify at ingestion: `df['source_id'].isna().sum()` should be small (<1% of rows). For unmatched stars, do not attempt manual positional cross-match; drop them — the sample is already large enough.

### 3.6 Gaia DR3 enrichment query (at AIP)

Batch the post-cut `source_id`s in chunks of 10 000 and query AIP TAP with `pyvo`:

```sql
-- Illustrative ADQL. Batch of source_ids substituted via pyvo parameter binding.
SELECT
    g.source_id, g.ra, g.dec,
    g.parallax, g.parallax_error,
    g.pmra, g.pmra_error, g.pmdec, g.pmdec_error,
    g.ra_dec_corr, g.ra_parallax_corr, g.ra_pmra_corr, g.ra_pmdec_corr,
    g.dec_parallax_corr, g.dec_pmra_corr, g.dec_pmdec_corr,
    g.parallax_pmra_corr, g.parallax_pmdec_corr, g.pmra_pmdec_corr,
    g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag,
    g.phot_g_mean_flux_over_error, g.bp_rp, g.bp_g, g.g_rp,
    g.ruwe, g.visibility_periods_used, g.astrometric_excess_noise,
    g.astrometric_params_solved, g.ipd_gof_harmonic_amplitude,
    g.ipd_frac_multi_peak,
    g.has_xp_continuous, g.has_rvs,
    g.radial_velocity, g.radial_velocity_error,
    g.nu_eff_used_in_astrometry, g.pseudocolour,
    g.ecl_lat,                                 -- for Lindegren+2021 zpt
    ap.teff_gspphot, ap.teff_gspphot_lower, ap.teff_gspphot_upper,
    ap.logg_gspphot, ap.logg_gspphot_lower, ap.logg_gspphot_upper,
    ap.mh_gspphot,   ap.mh_gspphot_lower,   ap.mh_gspphot_upper,
    ap.ag_gspphot,   ap.ag_gspphot_lower,   ap.ag_gspphot_upper,
    ap.ebpminrp_gspphot,
    ap.distance_gspphot, ap.distance_gspphot_lower, ap.distance_gspphot_upper,
    ap.teff_gspspec, ap.logg_gspspec, ap.mh_gspspec, ap.alphafe_gspspec,
    ap.flags_gspspec
FROM gaiadr3.gaia_source AS g
LEFT JOIN gaiadr3.astrophysical_parameters AS ap USING (source_id)
WHERE g.source_id IN (__batch__)
```

Keep `has_xp_continuous` and `has_rvs` — the former is the prerequisite for fetching XP coefficients in §6.

### 3.7 Apply Gaia corrections

At ingestion, on the concatenated result:
- **Lindegren+2021 parallax zero-point**: use the official `zero_point.py` (https://www.cosmos.esa.int/web/gaia/edr3-code). Compute per-star `zpt`, store as `parallax_zpt`, apply as `parallax_corrected = parallax - zpt` stored as `parallax_corr`.
- **Riello+2021 G-band flux/magnitude correction** (A&A 649, A3 Appendix A): apply the cubic-polynomial factor f(BP−RP) to sources with 2-parameter (`astrometric_params_solved == 3`) or 6-parameter (`== 95`) solutions at G ≥ 13, with separate bright (13 ≤ G ≤ 16) and faint (G > 16) coefficients and BP−RP clipped to [0.25, 3.0]. Reference code: agabrown/gaiaedr3-6p-gband-correction. Store as `phot_g_mean_mag_corr` (and `phot_g_mean_flux_corr` when a flux column is available).

### 3.8 Output schema — `data/interim/stream1_apogee_gaia.parquet`

One row per APOGEE star with matched Gaia DR3. ~700 k rows × ~80 columns × ~8 bytes/cell ≈ **~450 MB uncompressed, ~150 MB Parquet with Snappy compression**.

---

## 4. Stream 2 — TESS Hon+2021 + TASOC × Gaia DR3

Pre-staged for Task 4 (asteroseismic ages, led by others). **Not used by Pipelines 1 or 2 yet.** Kept minimal to preserve budget.

### 4.1 Hon et al. 2021 catalogue

- VizieR: **J/ApJ/919/131**
- Reference: Hon, Huber, Kuszlewicz et al. 2021, ApJ 919, 131; arXiv:2108.01241. 158 505 oscillating red giants from TESS primary-mission FFIs via the MIT QLP.
- Columns needed: `TIC` (integer), `RAJ2000`, `DEJ2000`, `numax` (μHz), `e_numax`, `Teff`, `R` (radius in R_sun), `Prob` (detection probability), `Sector` flags.
- Download via VizieR TAP (https://tapvizier.cds.unistra.fr/TAPVizieR/tap) with `pyvo`:

```sql
SELECT TIC, RAJ2000, DEJ2000, numax, e_numax, Teff, R, Prob
FROM "J/ApJ/919/131/table1"
WHERE Prob > 0.95
```

Prob > 0.95 is the authors' recommended high-confidence threshold.
Expected rows: ~120 k post-cut. Size: ~15 MB Parquet.

### 4.2 TASOC / additional seismic parameters

"TASOC" is the TESS Asteroseismic Science Operations Center. For TESS red giants, published supplementary parameters come from:
- **Mackereth et al. 2021** (MNRAS 502, 1947) — 5 574 giants with Δν and mass estimates combining TESS + APOGEE.
- **Stokholm et al. 2023** (MNRAS 524, 1080) — ~12 k stars with seismic masses.
- **Hon et al. 2024** (in prep / arXiv recently) — extension of the 2021 catalogue with Δν for a subset.
- **Silva Aguirre et al. 2020** (TASC-I) — for methodology reference.

**Not all of these are publicly released as tables yet.** Fetch what is available from VizieR or from the papers' Zenodo repositories. For the 2026 cut of the project, accept that Hon+2021 ν_max is the primary deliverable and the rest is opportunistic.

### 4.3 TIC → Gaia DR3 cross-match

The TESS Input Catalog (TIC v8.2; Paegert+2021; Stassun+2019) publishes a `GAIA` column — **but this is Gaia DR2 source_id**, not DR3. Two-step cross-match required:

**Step 1**: fetch TIC v8.2 rows for our Hon+2021 TICs via MAST CasJobs or the CDS VizieR mirror `IV/39/tic82`. Columns: `TIC`, `GAIA` (DR2 ID), `RAJ2000`, `DEJ2000`, `Tmag`, `plx`.

**Step 2**: map DR2 → DR3 via the Gaia cross-match table `gaiadr3.dr2_neighbourhood` at AIP:

```sql
SELECT
    nbh.dr2_source_id,
    nbh.dr3_source_id,
    nbh.angular_distance,
    nbh.magnitude_difference
FROM gaiadr3.dr2_neighbourhood AS nbh
WHERE nbh.dr2_source_id IN (__batch_dr2_ids__)
```

Accept only matches with `angular_distance < 300 mas` AND `abs(magnitude_difference) < 0.1`. Where a DR2 ID resolves to multiple DR3 IDs (binaries split by DR3 improved resolution), select the brightest. Log ambiguities.

### 4.4 Enrichment and output

Join the cross-matched `source_id`s to Gaia DR3 using the same query template as §3.6. No XP coefficients needed yet — deferred to Task 4.

Output: `data/interim/stream2_tess_gaia.parquet`. ~120 k rows × ~40 columns ≈ **~80 MB uncompressed, ~30 MB Parquet**.

---

## 5. Stream 3 — 1–2 M Gaia RGB+RC application sample

This is the inference set for Pipeline 1 (`xp_abundances`) and the feature basis for Pipeline 2. **Selection criterion matters most here** — an honest, defensible, reproducible cut is essential for the D-Cat-b/D-Cat-d releases.

### 5.1 Three candidate selection strategies

| Option | Source | Strengths | Weaknesses |
|---|---|---|---|
| **A. Andrae+2023 vetted RGB** | Zenodo 7945154; 17.5 M giants G<16 pre-selected | Clean, publicly released, ML-validated, metallicity-rich | Uses XGBoost labels (circular for Pipeline 1 training); selection function not documented in detail |
| **B. StarHorse2 Kiel cut** | Queiroz+2023; AIP `gaiadr3_contrib.aqueiroz2023_*` | Bayesian-inferred Teff/logg, spectroscopically-informed where spectra exist, propagates parallax uncertainty | Only covers stars with spectroscopic inputs in the StarHorse2 sample; excludes XP-only stars |
| **C. GSP-Phot Kiel cut** | Gaia DR3 `astrophysical_parameters` | All-sky, all 219 M XP stars, fully Gaia-internal | GSP-Phot logg is noisy; many contaminants |

**Recommendation: hybrid A + C.**

- **Primary**: Andrae+2023 vetted RGB subsample for stars with SNR(XP) that passes their cuts. This is ~17.5 M stars, out of which our 1–2 M cut is a random draw stratified on `(Teff, logg, [M/H], G)` to maximise parameter-space coverage.
- **Extension at low latitudes / high extinction**: where Andrae+2023 has holes (they cut Av > 1.5 and |b| < 5°), supplement with GSP-Phot Kiel cuts (Teff 4000–5200 K, logg 1.0–3.5) to avoid a latitude-biased catalogue.
- **Cross-check** (not selection): verify each kept star against StarHorse2 (Queiroz+2023) where available; flag disagreements (|logg_A23 − logg_SH2| > 0.5 dex) for downstream scrutiny.

**Why not StarHorse2 as primary?** StarHorse2 is scientifically excellent but its coverage is defined by the inputs — and critically, its largest table (Gaia RVS StarHorse, ~4.2 M stars, Queiroz+2023) requires G<sub>RVS</sub> < 14, which excludes the faint-end XP population our Pipeline 1 targets. Use StarHorse2 where it overlaps (as a cross-check and to import precision distances/extinctions), not as the primary selector.

### 5.2 Andrae+2023 vetted RGB download

- Source: Zenodo record **7945154** (DOI 10.5281/zenodo.7945154); file `giants_vetted.fits.gz` (name may vary; check Zenodo record on download).
- Reference: Andrae, Rix & Chandra 2023, ApJS 267:8; arXiv:2302.02611.
- Size: full vetted catalogue ~800 MB; we only need columns `source_id`, `teff_xgboost`, `logg_xgboost`, `mh_xgboost` and their uncertainties. ~150 MB Parquet after column trimming.

### 5.3 Stratified sub-sampling to 1–2 M

Target: **1.5 M** stars evenly distributed in `(Teff, logg, [M/H], G)`.

```python
# Illustrative — the real implementation lives in
# src/arqueogal/data/stream3_selection.py
bins_teff  = np.linspace(4000, 5500, 7)
bins_logg  = np.linspace(1.0, 3.5, 6)
bins_mh    = np.linspace(-2.0, 0.5, 6)
bins_g     = np.linspace(7, 16, 10)
# ≈ 7 × 6 × 6 × 10 = 2520 cells. Sample 600 stars per cell → 1.5 M.
# Cells with < 600 stars: take all; cells with more: random.
```

This stratification is essential for Pipeline 2 diagnostics — an unstratified sample dominated by disc stars at solar metallicity would under-sample the halo and metal-poor tails where interesting population structure lives.

Log the exact stratification parameters and random seed to provenance.

### 5.4 Gaia DR3 enrichment

Same query template as §3.6. **With XP coefficients this time** — the full purpose of Stream 3 is to feed XP into Pipeline 1 inference.

### 5.5 Output

- `data/interim/stream3_gaia_rgbrc.parquet`: 1.5 M rows × ~80 Gaia columns ≈ **~950 MB uncompressed, ~320 MB Parquet** (without XP).
- With XP coefficients attached (§6): **~1.8 GB Parquet**. This is the single largest disk consumer in the workspace.

---

## 6. XP coefficient extraction and preprocessing

### 6.1 Source and schema

Gaia DR3 XP continuous mean spectra live in **`gaiadr3.xp_continuous_mean_spectrum`** at AIP (also at ESA). Schema:

| Column | Type | Description |
|---|---|---|
| `source_id` | BIGINT | join key |
| `bp_coefficients` | ARRAY(DOUBLE, 55) | BP Hermite coefficients |
| `bp_coefficient_errors` | ARRAY(DOUBLE, 55) | per-coefficient errors |
| `bp_coefficient_correlations` | ARRAY(DOUBLE, 1485) | lower-triangle correlations, 55*(55-1)/2 = 1485 |
| `rp_coefficients` | ARRAY(DOUBLE, 55) | RP Hermite coefficients |
| `rp_coefficient_errors` | ARRAY(DOUBLE, 55) | per-coefficient errors |
| `rp_coefficient_correlations` | ARRAY(DOUBLE, 1485) | |
| `bp_basis_function_id`, `rp_basis_function_id` | INT | basis metadata |
| `bp_standard_deviation`, `rp_standard_deviation` | DOUBLE | global |
| `bp_n_measurements`, `rp_n_measurements` | INT | epoch count |
| `bp_n_relevant_bases`, `rp_n_relevant_bases` | INT | effective dimensionality |

**Keep**: coefficients, coefficient_errors, standard_deviation, n_measurements, n_relevant_bases. **Drop**: coefficient_correlations (too heavy for the budget). Flag this decision in provenance.

Per-star footprint after dropping correlations:
- Means: 55 × 2 bands × 8 bytes = 880 B (double)
- Errors: 55 × 2 × 8 = 880 B
- Metadata: ~100 B
- Total: ~1.9 KB per star × 1.5 M ≈ **~2.8 GB uncompressed**, **~1.4 GB Parquet** with float32 conversion and compression.

### 6.2 Float32 downcast is safe here

Gaia XP coefficients are published as float64 but the relative precision needed for abundance ML is ~1e-4, well within float32's ~1.2e-7 floor. Downcast at ingestion, halve the footprint, note in provenance.

### 6.3 Batched pyvo query

XP arrays are large; TAP queries with large arrays are slow. Batch in groups of **5 000 source_ids** (not 10 000) for XP queries specifically:

```sql
SELECT source_id,
       bp_coefficients, bp_coefficient_errors,
       rp_coefficients, rp_coefficient_errors,
       bp_standard_deviation, rp_standard_deviation,
       bp_n_measurements, rp_n_measurements,
       bp_n_relevant_bases, rp_n_relevant_bases
FROM gaiadr3.xp_continuous_mean_spectrum
WHERE source_id IN (__batch_5000__)
```

Checkpoint each batch to `data/interim/xp_batches/batch_NNNN.parquet` so a crash resumes gracefully.

### 6.4 Preprocessing sequence

Applied in `src/arqueogal/data/gaia_xp.py`, in order:

1. **Ye+2024 NN flux-correction.** Reference: Ye et al. 2025, *A&A* 695 A75 (peer-reviewed version of arXiv:2411.19105). Public data and code release: **concept DOI 10.5281/zenodo.14028588** (resolves to all versions); **version DOI 10.5281/zenodo.14712749** (v2, published 2025-01-21, `GaiaXP-correction_V0.zip`, ~1.5 GB, MD5 `c7136ede1fcada9b1e0a0373c59741b1`). CDS mirror: `J/A+A/695/A75`. **Implemented** via path (a): trained weights (`nn_model_pattern.pth`) + per-feature StandardScaler (`scaler_mean.txt`, `scaler_scale.txt`) vendored under `data/external/ye2024/GaiaXP-correction_V0/model/` (42 MB). Path (b) — cross-matching the Zenodo catalog — is not viable: `catalog_all.vot` only contains atmospheric parameters for the 68 M cross-sample, not corrected coefficients/spectra for arbitrary source_ids. The NN operates in *sampled* flux space on `np.geomspace(360, 990, 330)` nm; its output is stored as `xp_sampled_corrected.parquet` (intermediate).
   **GaiaXPy-required columns we didn't fetch** (covariance drop in §6.1): `bp_n_parameters` / `rp_n_parameters` are injected as `55` (matching the fixed-length Hermite arrays); `bp_coefficient_correlations` / `rp_coefficient_correlations` are injected as zero vectors of length `55·54/2 = 1485`. GaiaXPy only consumes correlations to propagate `flux_error` and `flux_error_*`, and the Ye NN reads only mean flux + mean magnitudes, so the zero substitution leaves the NN inputs exact. Propagated `flux_error` is consequently uncalibrated and **must not be used** — use `YE2024_FLAG_OK` / `YE2024_FLAG_NO_SYNTH_PHOT` / `YE2024_FLAG_CALIBRATE_FAIL` as the only per-star reliability signal out of this step. Recorded in the `xp_sampled_corrected.provenance.json` sidecar.
2. **Hermite re-projection.** Linear least-squares projection of the 330-element Ye-corrected flux onto the 55+55 Hermite basis used by Gaia DR3 (BP and RP separately, wavelength→pseudo-wavelength per De Angeli+2023 §3). Outputs: `bp_coeffs_corrected[55]`, `rp_coeffs_corrected[55]`. **Per-star QC feature:** `reprojection_residual_rms` (RMS of Ye sampled flux − Hermite-basis reconstruction). Retained, not used for rejection. This step re-enters the coefficient representation used by Andrae+2023, AspGap, Buck & Schwarz 2024, and Guiglion+2024, and closes the §7.2 research_brief.md decision on flux representation.
3. **Normalisation by first coefficient.** For each star:
   ```
   bp_coeff[1:] /= bp_coeff[0]
   rp_coeff[1:] /= rp_coeff[0]
   ```
   This is the Guiglion+2024 and Buck & Schwarz 2024 convention: the zeroth coefficient carries the overall flux scale; dividing the rest makes the representation magnitude-independent.
4. **First-coefficient transformation.** `bp_coeff[0] → log10(bp_coeff[0])`; then z-score `bp_coeff[0]` and `rp_coeff[0]` across the Stream-1 training set. **Freeze (μ, σ) in the pipeline-1 training provenance** and re-apply identically at Stream-3 inference — never re-z-score against the inference sample. Buck & Schwarz 2024's recipe.
5. **Error propagation.** Under division by `bp_coeff[0]`, the normalised error is:
   ```
   σ_norm_i = sqrt( (σ_i / bp_coeff[0])^2 + (bp_coeff_i * σ_0 / bp_coeff[0]^2)^2 )
   ```
   Implement this exactly, do not approximate. Ye's re-projected coefficients inherit their σ from the propagation of `flux_error`; since our `flux_error` is uncalibrated per step 1, the coefficient σ is a lower bound only — release notes must say so.
6. **Output columns.** `bp_coeffs_norm[55]`, `rp_coeffs_norm[55]`, `bp_coeff_errs_norm[55]`, `rp_coeff_errs_norm[55]`, the z-scored zeroth-coefficient features `bp_c0_z`, `rp_c0_z`, the Hermite-reprojection QC feature `reprojection_residual_rms`, and `ye2024_flag`. Float32. Drop raw coefficients from the analysis-ready file (keep them in `data/interim/` for diagnostic reanalysis).

### 6.5 Sanity checks (run at ingestion)

- `bp_coeffs` and `rp_coeffs` arrays have length exactly 55 each. Reject stars with NaN or missing arrays.
- `bp_coeff[0] > 0` and `rp_coeff[0] > 0`. (Flux normalisation should be positive for real stars.)
- No coefficient should be > 10× the typical magnitude-stratified median of its column — outliers flagged with `xp_outlier_flag`.
- `has_xp_continuous` from the `gaia_source` join should be True for every star. If not, the XP query silently failed — halt and investigate.

---

## 7. Distances

### 7.1 Bailer-Jones et al. 2021 photogeometric distances

- Reference: Bailer-Jones, Rybizki, Fouesneau, Demleitner, Andrae 2021, AJ 161, 147; arXiv:2012.05220.
- VizieR catalogue: **I/352**.
- GAVO mirror: `gedr3dist.main` at https://dc.g-vo.org/tableinfo/gedr3dist.main (primary, recommended).
- Full catalogue: 1,467,744,818 geometric + 1,346,621,631 photogeometric distances. Full dump is ~60 GB — **do not download all**. Query only our source_ids.
- Columns: `source_id`, `r_med_geo`, `r_lo_geo`, `r_hi_geo`, `r_med_photogeo`, `r_lo_photogeo`, `r_hi_photogeo`, `flag`.

GAVO TAP query via pyvo:

```sql
SELECT source_id,
       r_med_geo, r_lo_geo, r_hi_geo,
       r_med_photogeo, r_lo_photogeo, r_hi_photogeo,
       flag
FROM gedr3dist.main
WHERE source_id IN (__batch__)
```

**Use `r_med_photogeo` as the primary distance** — Bailer-Jones+2021 show photogeometric outperforms geometric for stars with parallax S/N < 10 by incorporating colour and magnitude as an absolute-magnitude prior. Asymmetric errors: use `(r_hi_photogeo - r_lo_photogeo) / 2` as the symmetric σ approximation only where required; otherwise carry the full asymmetric pair.

Fallback: if GAVO is down or slow, AIP hosts the same data under `gaiaedr3.distances_bailerjones` (confirm exact table name via AIP schema browser at first ingestion).

### 7.2 StarHorse2 (Queiroz+2023)

- Reference: Queiroz, Anders, Chiappini et al. 2023, A&A 673, A155; arXiv:2303.09926. Version 2 files **only** (v1 had a piecewise age prior that biased old stars — release notes at https://data.aip.de/projects/aqueiroz2023.html).
- Landing page: https://data.aip.de/projects/aqueiroz2023.html. S3 bucket: `https://s3.data.aip.de:9000/shaqueiroz2023/`.
- Eight files, one per parent spectroscopic survey. For our training-set enrichment the relevant ones are:
  - `aqueiroz2023_apogee_dr17_v2.fits` — APOGEE DR17 (562 k stars). **Our Stream 1 overlap.** Size ~250 MB.
  - `aqueiroz2023_gaia_rvs_v2.fits` — Gaia RVS (4.2 M). Useful for Stream 3 cross-check. Size ~1.5 GB — **exceeds budget if downloaded in full**. Query via AIP TAP instead (see below).
- Key columns: `source_id`, `dist16`, `dist50`, `dist84` (pc), `av16`, `av50`, `av84` (mag), `teff16/50/84`, `logg16/50/84`, `met16/50/84`, `mass16/50/84`, `age16/50/84` (Gyr), `starhorse_outputflag`, `starhorse_ageflag`.
- **Age cautions**: SH2 ages are reliable only on the SGB (σ_age/age ~15%) and MSTO (σ_age/age ~30%). For RGB and RC stars, ages are uninformative (the SGB prior dominates). Respect `starhorse_ageflag`. Do not use SH2 ages for red giants; use them only where the flag says valid.

**Preferred ingestion**: query the TAP-exposed table at AIP rather than downloading the FITS file, to avoid disk blow-out:

```sql
SELECT source_id,
       dist16, dist50, dist84,
       av16, av50, av84,
       mass16, mass50, mass84,
       age16, age50, age84,
       starhorse_outputflag, starhorse_ageflag
FROM gaiadr3_contrib.aqueiroz2023_apogee_dr17_v2
WHERE source_id IN (__batch__)
```

(Confirm exact table name at first use via `SELECT TOP 1 * FROM tap_schema.tables WHERE schema_name = 'gaiadr3_contrib'` — AIP occasionally reorganises its `_contrib` namespace.)

### 7.3 Distance selection logic

For Pipeline 1 and Pipeline 2:
- Primary: `r_med_photogeo` from Bailer-Jones+2021.
- Cross-check and uncertainty inflation: where StarHorse2 `dist50` is available, compare. If |log10(r_BJ/d_SH2)| > 0.3 (factor of 2 disagreement), flag the star as `dist_conflict` and inflate the distance uncertainty to encompass both.
- Inside Pipeline 1 itself, distance is also a model output — the final feature vector carries a `distance_prior` (Bailer-Jones) and its σ, and the ML jointly re-fits.

---

## 8. Extinction — strategy within the 5 GB budget

### 8.1 The budget problem

| Dust product | Disk footprint | Coverage | Resolution |
|---|---|---|---|
| **Bayestar19** (Green+2019) | ~3 GB (h5 file) | Pan-STARRS footprint, δ > −30° | 3.4′–13.7′, d ≲ 5 kpc |
| **Edenhofer+2024** (A&A 685 A82) | ~500 MB–1 GB | All-sky within 1.25 kpc (2 kpc extension available) | 14′ (Nside 256), 516 distance bins |
| **Lallement+2022** (A&A 661 A147) | ~300 MB FITS | All-sky, 3 kpc | 25 pc voxels |
| **SFD1998** (2D) | ~100 MB | All-sky integrated column | 6.1′ |
| **Gaia GSP-Phot per-star A_G/E(BP-RP)** | ~0 (already in Stream 3 query) | Wherever GSP-Phot ran (219 M stars) | per-star |

**The 5 GB budget forbids Bayestar19** and makes the combination of Bayestar19 + Edenhofer+2024 impossible. Practical options:

### 8.2 Primary strategy (aligned with research_brief §5.3)

**Two Av features per star, computed independently:**

1. **Line-of-sight Av from Edenhofer+2024** for stars with d < 1.25 kpc (all-sky, Nside 256, 516 distance bins). Beyond 1.25 kpc the line-of-sight feature is left as NaN; the ML sees the missingness explicitly and falls back to the neighborhood-median. Install Edenhofer via the `dustmaps` package:

   ```python
   from dustmaps.config import config
   config['data_dir'] = 'data/external/dustmaps'
   from dustmaps import edenhofer, sfd
   edenhofer.fetch()   # ~600 MB
   sfd.fetch()         # ~100 MB, 2D high-latitude prior
   ```

   ```python
   from dustmaps.edenhofer2023 import Edenhofer2023Query
   q = Edenhofer2023Query()
   av = q(coord)  # SkyCoord with distance attached
   ```

2. **GSP-Phot 3D neighborhood-median Av** computed from the per-star `ag_gspphot` values already carried in Stream 1 and Stream 3 Gaia queries. See §8.3 for the recipe — pure Gaia data, zero additional disk, all distances covered (no 1.25 kpc regime cut-off).

Both features are injected into Pipeline 1 independently; the ML learns when to trust each and when to deviate from the prior. Per research_brief §5.3 this is the published-XP-pipeline methodological differentiator — no prior XP-abundance work uses the neighborhood-median.

**SFD+SFD_to_Av** stays on disk as a 2D high-latitude prior for sanity checks and as the third tier for stars with NaN Edenhofer and insufficient GSP-Phot neighbours — not as a primary feature.

**Lallement+2022** is a *cross-check* feature, not the 1.25–3 kpc primary. See §8.5.

### 8.3 Neighborhood-median implementation (primary Av feature #2)

Implemented in `src/arqueogal/data/dust_maps.py::neighborhood_av_features`. Recipe:

1. Use `ag_gspphot` from the Stream 1 / Stream 3 Gaia query (zero additional disk).
2. For each star, compute a neighborhood-median: all stars within a 3D ball of radius 50–100 pc (default 75 pc) in heliocentric Cartesian coordinates, median of `ag_gspphot` over that ball, excluding the star itself. `scipy.spatial.cKDTree`.
3. Columns written per star: `av_nbhd_median`, `av_nbhd_std`, `n_neighbors`. Stars with fewer than `MIN_NEIGHBORS_FOR_MEDIAN = 5` neighbours get NaN so the ML sees explicit missingness instead of a shot-noise garbage value.
4. Propagate A_G uncertainties via the GSP-Phot quantile bounds `ag_gspphot_lower/upper` when feeding downstream MC.

**Why this is scientifically defensible** (research_brief §5.3): individual GSP-Phot A_G values are noisy, but the *ensemble* at a given 3D position is a direct ISM tracer. This is essentially what Edenhofer+2024 does internally (GP smoothing of ZGR23 per-star Av's). The neighborhood-median extends that logic beyond 1.25 kpc where Edenhofer does not reach, at zero disk cost.

**Pros**: zero disk footprint; uses Gaia-internal consistency; scales to 1.5 M stars trivially; covers all distances.

**Cons**: GSP-Phot per-star A_G has systematic biases at high extinction (Av > 3) and at low metallicity. The neighborhood median smooths these but does not eliminate them — hence the cross-check against Lallement+2022 (§8.5) at mid-distance.

### 8.4 Final recommendation — primary Av features

**Per star, two independent Av features go into Pipeline 1:**

| Feature | Source | Coverage | Disk |
|---|---|---|---|
| `av_los_edenhofer` | Edenhofer+2024 LOS integral to d | d < 1.25 kpc | ~600 MB |
| `av_nbhd_median`, `av_nbhd_std`, `n_neighbors` | §8.3 GSP-Phot 3D median | all d (subject to n_neighbors ≥ 5) | 0 (uses Stream data) |

Plus two auxiliaries:

| Feature | Source | Role |
|---|---|---|
| `av_sfd` | SFD1998 2D | High-latitude prior, sanity check | ~100 MB |
| `av_los_lallement` (§8.5) | Lallement+2022 cube | Cross-check for 1.25–3 kpc stars | ~117 MB |

**Total dust-map disk**: ~820 MB — well inside the 5 GB budget.

The ML network receives all four as features and learns to weight them. No prior-based distance-regime hand-off. If Lallement is unavailable in the local `dustmaps` build (see §8.5), the primary pipeline still runs end-to-end on `av_los_edenhofer` + `av_nbhd_*` + `av_sfd`.

### 8.5 Lallement+2022 as cross-check (not 1.25–3 kpc primary)

Lallement+2022 (A&A 661 A147) is a 6×6×0.8 kpc³ Cartesian extinction-density cube at 25 pc voxels. It is fed as a **cross-check feature** alongside Edenhofer and the neighborhood-median — never as the sole 1.25–3 kpc primary.

**The `dustmaps.lallement2022` submodule is not in every `dustmaps` wheel.** Upgrading `dustmaps` to pull the submodule risks pinned RAPIDS / numpy / pandas dependency churn and is **forbidden**. Use a direct CDS fetch instead:

- **Source:** CDS J/A+A/661/A147, anonymous FTP at `ftp://cdsarc.u-strasbg.fr/cats/J/A+A/661/A147/`.
- **File:** `cube_ext.fits.gz` (~100 MB gzipped, ~117 MB uncompressed; single 3D image cube of extinction density in nanomag/pc, Sun-centred Cartesian axes: X → Galactic centre, Y → rotation direction, Z → NGP).
- **Checksum-pinned download:** fetched once into `data/external/lallement2022/cube_ext.fits.gz`, SHA-256 recorded in provenance.
- **Reader:** `astropy.io.fits` for the cube, `scipy.ndimage.map_coordinates` for trilinear interpolation. Minimal helper `lallement2022_query(ra_deg, dec_deg, distance_pc)` lives in `src/arqueogal/data/dust_maps.py` — no package dependency, no env risk.

The voxel-integrated Av along the LOS is computed by interpolating extinction density along the (X(s), Y(s), Z(s)) sightline in voxel space and multiplying by voxel step in parsecs.

Stars outside the cube bounds (|X|, |Y| > 3 kpc or |Z| > 0.4 kpc) return NaN and fall back to Edenhofer + neighborhood-median + SFD.

---

## 9. Orbital parameters via galpy

Implemented in `src/arqueogal/data/kinematics.py`. Reference: Recio-Blanco et al. 2023, A&A 674 A29 (arXiv:2206.05534) for the Gaia DR3 kinematics methodology.

### 9.1 Inputs per star

From Streams 1 and 3:
- `ra`, `dec` (deg)
- `r_med_photogeo` (pc) with `r_lo_photogeo`, `r_hi_photogeo`
- `pmra`, `pmdec`, `pmra_error`, `pmdec_error` (mas/yr)
- `radial_velocity`, `radial_velocity_error` (km/s) from Gaia RVS where available, else from APOGEE `vhelio_avg`/`vhelio_err`.
- Full astrometric correlation coefficients (`ra_dec_corr`, etc.) for proper MC propagation.

### 9.2 Solar and galactic constants

- `R_0 = 8.122 kpc` (GRAVITY Collaboration 2018, 2021).
- `z_0 = 20.8 pc` (Bennett & Bovy 2019).
- `V_0 = 233.1 km/s` (Reid & Brunthaler 2020, A_sag).
- Solar peculiar motion: `(U, V, W)_sun = (11.1, 12.24, 7.25) km/s` (Schönrich, Binney & Dehnen 2010).

### 9.3 Potential

**Primary**: `McMillan17` via galpy (McMillan 2017, MNRAS 465, 76). Well-constrained rotation curve, disc+bulge+halo components.
**Sensitivity check**: run a subset with `MWPotential2014` (Bovy 2015) and record the per-star action differences — if systematic actions shift by > 20% between potentials, Pipeline 2 conclusions must carry a "potential-dependent" caveat in the D5.1 release.

### 9.4 Outputs

Per star:
- Actions: `J_R`, `J_z`, `L_z` (kpc·km/s). Computed via `actionAngleStaeckel(pot, delta=0.45)` (the Staeckel-fudge method; see Binney 2012 and Mackereth et al. 2019 for the standard-choice delta).
- Orbit: `ecc`, `r_peri`, `r_apo` (kpc), `z_max` (kpc), `E` (energy, km²/s²).
- Velocities: `v_R`, `v_T` (tangential/azimuthal), `v_Z`, plus their LSR-referenced counterparts for backward compatibility with Neitzel+2025 feature set.

### 9.5 MC uncertainty propagation

Naive MC for actions is expensive (2 M stars × N_MC × ~1 ms/sample = impractical at N_MC > 10). Two-tier strategy:
- **Stream 3 bulk (1.5 M stars)**: compute central-value actions only. Reports uncertainties from analytic error propagation (Jacobian) as a first-order approximation.
- **Pipeline 2 D-Cat-d subsample**: for stars that Pipeline 2 soft-assigns to boundary clusters (see research_brief §10.3), compute N_MC = 100 full MC draws from the astrometric covariance using `numpy.random.multivariate_normal` with the 5×5 Gaia covariance matrix. This is ~10⁴ stars × 100 × 1 ms = ~15 min. Affordable only because we restrict to boundary cases.

### 9.6 Output schema

Append to `data/processed/pipeline2_features.parquet`: one row per star with `J_R`, `J_z`, `L_z`, `ecc`, `r_peri`, `r_apo`, `z_max`, `E`, plus LSR-referenced velocities and their uncertainties. ~1.5 M × 20 float32 columns ≈ **~120 MB Parquet**.

---

## 10. Master catalog schema

After all streams are ingested, cross-matched, and enriched, the analysis-ready products live in `data/processed/`:

### 10.1 `pipeline1_training.parquet`

~700 k rows (APOGEE × Gaia × StarHorse2 × dust × XP). Feeds Pipeline 1 supervised training.

| Category | Columns | Provenance |
|---|---|---|
| Identifiers | `source_id`, `apogee_id`, `sdss_id` | Streams 1+3 |
| Astrometry | `ra`, `dec`, `parallax_corr`, `parallax_error`, `pmra`, `pmdec`, 10 covariances | Gaia DR3 |
| Photometry | `phot_g_mean_mag_corr`, `bp_rp`, `bp_g`, `g_rp` | Gaia DR3 |
| XP | `bp_coeffs_norm[55]`, `rp_coeffs_norm[55]`, errors, `bp_c0_z`, `rp_c0_z` | Stream 3 XP |
| Distance | `r_med_photogeo`, asym bounds | Bailer-Jones+2021 |
| Extinction | `av_edenhofer`, `av_lallement`, `av_sfd`, `av_nbhd_median` + σ | §8 |
| Labels (training targets) | `teff_apogee`, `logg_apogee`, `mh_apogee`, `alpha_m_apogee`, per-element `*_h_apogee`, all with σ | Stream 1 (Mészáros+2025-corrected) |
| Flags | `flag_bad`, `ruwe`, `xp_outlier_flag`, `dist_conflict` | |

### 10.2 `pipeline1_inference.parquet`

~1.5 M rows (Gaia RGB+RC with XP, no APOGEE labels). Feeds Pipeline 1 inference.

Same columns as 10.1 minus the APOGEE-sourced labels, plus Andrae+2023 labels as cross-reference diagnostics.

### 10.3 `pipeline2_features.parquet`

~1.5 M rows, chrono-chemo-kinematic feature vector for Pipeline 2 clustering.

| Column | Source |
|---|---|
| `source_id` | join key |
| `age`, `age_err` | Task 4 output (future); null for now |
| `fe_h`, `fe_h_err` | APOGEE if overlap, else Pipeline 1 Tier 1 |
| `mg_fe`, `mg_fe_err` | APOGEE if overlap, else Pipeline 1 Tier 2 |
| `al_fe`, `al_fe_err` | APOGEE where Tier 1/2 supports |
| `c_n`, `c_n_err` | RGB only, evolutionary-stage-gated |
| `J_R`, `J_z`, `L_z`, `ecc`, `r_peri`, `r_apo`, `z_max`, `E` | §9 |
| `v_phi`, `sqrt_u2_plus_w2` | backward-compatibility with Neitzel+2025 |
| `evolutionary_stage_prob` | from Pipeline 1 head |

---

## 11. Order of operations

Dependencies between steps. Parallelisable within each level.

```
Level 0 (downloads, no dependencies):
  - Gaia DR3 schema browser (AIP)       → verify table names
  - APOGEE DR19 summary FITS            → data/raw/apogee_dr19/
  - Hon+2021 catalogue                  → data/raw/tess_hon2021/
  - Andrae+2023 vetted RGB              → data/raw/andrae2023/
  - Install dustmaps + fetch Edenhofer+2024, Lallement+2022, SFD

Level 1 (single-stream processing):
  - Stream 1: filter APOGEE, apply Mészáros+2025 corrections → data/interim/apogee_dr19_cut.parquet
  - Stream 2a: Hon+2021 cuts → data/interim/hon2021_cut.parquet
  - Stream 2b: TIC v8.2 lookup → data/interim/tic_v8_2.parquet
  - Stream 2c: DR2→DR3 via dr2_neighbourhood → data/interim/tess_dr3_xmatch.parquet
  - Stream 3: Andrae+2023 stratified sub-sample → data/interim/andrae_rgb_sample.parquet

Level 2 (Gaia enrichment):
  - Source_id lists from Level 1 → AIP TAP queries → Gaia columns + GSP-Phot/Spec
  - Apply Lindegren+2021 zpt, CG+Brandt G correction
  - Join to Level 1 → data/interim/stream{1,2,3}_gaia.parquet

Level 3 (XP extraction):
  - Source_id lists from Level 2 with has_xp_continuous=True
  - Batched queries to gaiadr3.xp_continuous_mean_spectrum
  - Ye+2024 flux correction
  - Normalise by first coeff, z-score, float32
  → data/interim/xp_coeffs.parquet

Level 4 (distance and extinction):
  - Bailer-Jones distances via GAVO TAP
  - StarHorse2 for APOGEE overlap via AIP TAP
  - dustmaps queries for Edenhofer/Lallement/SFD
  - GSP-Phot neighborhood-median Av (scipy.spatial.cKDTree)

Level 5 (kinematics):
  - galpy orbits per star (McMillan17 potential, Staeckel fudge)
  - Central-value actions for bulk 1.5M; MC for boundary stars

Level 6 (master catalogs):
  - Join all intermediates → data/processed/pipeline1_training.parquet,
                               pipeline1_inference.parquet,
                               pipeline2_features.parquet
  - Provenance JSON for each
```

Expected total wall time from cold start, assuming reasonable AIP/GAVO responsiveness: **~12–16 hours** over 2–3 sessions. Most time is TAP queries (XP is the slowest) and galpy integration.

---

## 12. Disk footprint accounting

Final budget audit at the end of ingestion. If the total exceeds 4.5 GB, re-evaluate (drop Lallement+2022, drop StarHorse2 for Gaia-RVS overlap, or prune Stream 3 to 1 M stars instead of 1.5 M).

| Location | Contents | Size |
|---|---|---|
| `data/raw/apogee_dr19/` | astraAllStarASPCAP-0.6.0.fits.gz | 500 MB |
| `data/raw/tess_hon2021/` | Hon+2021 FITS | 20 MB |
| `data/raw/andrae2023/` | Andrae+2023 vetted giants (trimmed) | 150 MB |
| `data/external/dustmaps/edenhofer2024/` | | 800 MB |
| `data/external/dustmaps/lallement2022/` | | 300 MB |
| `data/external/dustmaps/sfd/` | | 100 MB |
| `data/interim/xp_coeffs.parquet` | XP float32, 1.5 M stars | 1.4 GB |
| `data/interim/stream1_apogee_gaia.parquet` | | 150 MB |
| `data/interim/stream2_tess_gaia.parquet` | | 30 MB |
| `data/interim/stream3_gaia_rgbrc.parquet` (without XP — joined at use time) | | 320 MB |
| `data/processed/pipeline1_training.parquet` | | 400 MB |
| `data/processed/pipeline1_inference.parquet` | | 700 MB |
| `data/processed/pipeline2_features.parquet` | | 250 MB |
| `data/provenance/` | JSON sidecars | <10 MB |
| **Total** | | **~5.1 GB** |

**This is at the edge of the 5 GB budget.** To create headroom, the first cuts to make (in order of preference):
1. Drop Lallement+2022; rely on Edenhofer+2024 + GSP-Phot neighborhood-median beyond 1.25 kpc (saves 300 MB).
2. Drop the `pipeline1_inference.parquet` materialisation; compute inference in-memory streaming from `stream3 + xp_coeffs` (saves 700 MB).
3. Reduce Stream 3 to 1 M stars (saves ~500 MB).
4. Drop `bp_coefficient_errors`/`rp_coefficient_errors` from XP, keep only means (saves ~700 MB but costs uncertainty propagation — **don't do this if calibration is a first-class requirement per research_brief §7.1**).

Apply cuts in priority order until budget is met.

---

## 13. Known pitfalls

1. **DR19 column naming is unstable.** Between 0.5.x and 0.6.0 Astra releases, some column names shifted (`fe_h` → `m_h_atm` → `fe_h_atm`). Re-verify at ingestion against https://www.sdss.org/dr19/mwm/data/abundances/ and the FITS HDU header.
2. **APOGEE DR17 abundances are NOT interchangeable with DR19.** The Astra rewrite changed zero-points. Do not mix DR17-trained models with DR19 labels.
3. **Gaia DR3 radial_velocity is G_RVS < 14 only for most stars.** At fainter magnitudes, rely on APOGEE `vhelio_avg` (Stream 1) or leave RV null and skip kinematics for that star.
4. **RUWE cuts**: the community standard is RUWE < 1.4 for astrometrically well-behaved single stars (Lindegren+2018). Apply at Stream 3 ingestion; retain RUWE ≥ 1.4 stars in Stream 1 (they may be binaries but their APOGEE abundances are still valuable) with a `binary_suspect` flag.
5. **Gaia parallax zero-point requires `ecl_lat`**, `nu_eff_used_in_astrometry` (for 5p solutions) or `pseudocolour` (for 6p), and `astrometric_params_solved`. Fetch all at Stream 1/3 query time. The `zero_point.py` function fails silently on missing inputs.
6. **dr2_neighbourhood is many-to-many.** A single DR2 source may resolve to 2+ DR3 sources in close binaries; a single DR3 source may match 2+ DR2 sources in mergers. Log multiplicity and apply a tie-breaker (brightest by magnitude), never pick the first arbitrarily.
7. **StarHorse2 v1 vs v2.** Release notes at https://data.aip.de/projects/aqueiroz2023.html explicitly mark v1 as deprecated due to an age-prior bug. Confirm you are pulling v2 tables. If the TAP schema still serves v1, query with explicit `_v2` suffix.
8. **GaiaXPy calibration to sampled spectra is not needed for our main pipeline.** Pipeline 1 ingests raw Hermite coefficients, not sampled fluxes. Use `GaiaXPy.calibrate` only in experimental or diagnostic contexts. Calling it on 1.5 M stars would blow the budget (sampled spectra are ~8× larger than coefficients).
9. **Ye+2024 flux-correction may not have public code.** Check arXiv:2411.19105 for code/model links. If not released, either contact Ye et al. or implement locally from the paper description; document the choice in provenance. Do not silently skip.
10. **Mészáros+2025 correction polynomials must be applied BEFORE `flag_bad` cut**, because the flag definitions reference calibrated values. Order: load raw → apply corrections → apply flag cut.
11. **TAP async vs sync.** For queries returning > 5 000 rows, use `pyvo.dal.TAPService(url).submit_job(query)` (async) not `.search(query)` (sync). Sync queries time out at ~90 s on AIP.
12. **XP arrays in Parquet.** Arrow/Parquet support fixed-length lists natively. Ensure the Parquet schema declares `bp_coefficients: list<float32>[55]` (not variable-length list) for fast cudf reading.
13. **Float32 vs float64 for kinematics.** Keep actions and orbits in float64 during galpy integration (numerical stability). Downcast to float32 only at Parquet-write time.

---

## 14. Tooling: pyvo, requests, astroquery alternatives

### 14.1 pyvo — primary TAP client

```python
import pyvo
from pyvo.auth import authsession

# AIP authenticated
session = authsession.AuthSession()
session.credentials.set_password("user", "password")
aip = pyvo.dal.TAPService("https://gaia.aip.de/tap", session=session)

# Simple query
result = aip.search("SELECT TOP 10 source_id FROM gaiadr3.gaia_source")
df = result.to_table().to_pandas()

# Async for large queries
job = aip.submit_job("SELECT ... FROM gaiadr3.xp_continuous_mean_spectrum WHERE source_id IN (...)")
job.run()
job.wait(phases=["COMPLETED", "ERROR"])
result = job.fetch_result().to_table()
```

Documentation: https://pyvo.readthedocs.io/

### 14.2 requests — direct HTTPS downloads

```python
import requests
from pathlib import Path
from tqdm import tqdm

def download(url, dest: Path, chunk_size=1024*1024):
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with dest.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
        for chunk in r.iter_content(chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))
```

Always stream (not `r.content`) for large files; write to temp path, rename on success, for atomicity.

### 14.3 astroquery — last-resort fallback only

Use `astroquery.vizier` for VizieR catalogue fetches when TAP is not available. Never use `astroquery.gaia` unless pyvo explicitly fails — it has been unstable in recent months.

### 14.4 Provenance logging

Every ingestion step writes a sidecar JSON:

```json
{
  "output_file": "data/interim/stream1_apogee_gaia.parquet",
  "script": "src/arqueogal/data/ingest_stream1.py",
  "git_sha": "a1b2c3d",
  "timestamp_utc": "2026-05-14T10:23:17Z",
  "sources": [
    {
      "name": "APOGEE DR19 ASPCAP summary",
      "url": "https://dr19.sdss.org/sas/dr19/...",
      "size_bytes": 524288000,
      "sha256": "abc..."
    },
    {
      "name": "AIP Gaia DR3 TAP",
      "endpoint": "https://gaia.aip.de/tap",
      "query": "SELECT g.source_id, g.ra, ... FROM gaiadr3.gaia_source ...",
      "n_batches": 70,
      "batch_size": 10000
    }
  ],
  "cuts_applied": ["flag_bad == 0", "snr > 70", "teff in [4000,5500]"],
  "row_count_before": 964989,
  "row_count_after": 682344,
  "corrections": [
    "Meszaros+2025 Teff-trend polynomial",
    "Lindegren+2021 parallax zero-point",
    "Riello+2021 G-mag correction"
  ]
}
```

This is what makes the pipeline reproducible and auditable.

---

## 15. Reproducibility and provenance

**Every run of every ingestion script must:**
1. Log its git SHA (from `src/arqueogal/`) into the provenance JSON.
2. Record the exact query strings and endpoint URLs.
3. Record row counts at every cut.
4. Record the SHA-256 of every downloaded file.
5. Use a fixed random seed (hardcoded in config) for any stratified sub-sampling.

**The test suite** (`tests/data/`) must include:
- A smoke test for each TAP endpoint (TOP 10 query).
- A round-trip test for the XP preprocessing (Hermite coefficients → normalised → denormalised back to original within float32 tolerance).
- A cross-match validation test: re-run DR2→DR3 mapping on a 1 k-star test sample and verify against a known ground-truth subset.
- A Mészáros+2025 correction sanity test: before/after [X/M]–Teff trend plots on a test set.

---

## 16. Key references (for implementation)

**Gaia DR3**: Gaia Collaboration et al. 2023, A&A 674, A1; De Angeli et al. 2023, A&A 674, A2 (XP); Recio-Blanco et al. 2023, A&A 674, A29 (GSP-Spec); Creevey et al. 2023 / Andrae et al. 2023 (Apsis).

**Gaia corrections**: Lindegren et al. 2021 (parallax zero-point), A&A 649, A4; Riello et al. 2021 (G-band flux/magnitude correction), A&A 649, A3 Appendix A; https://www.cosmos.esa.int/web/gaia/edr3-code; reference code https://github.com/agabrown/gaiaedr3-6p-gband-correction.

**APOGEE DR19**: Mészáros et al. 2025, arXiv:2506.07845; SDSS Collaboration 2025, arXiv:2507.07093. Release page https://www.sdss.org/dr19/mwm/data/; Astra documentation https://www.sdss.org/dr19/mwm/astra/accessing-astra-files/.

**Hon+2021**: ApJ 919, 131; arXiv:2108.01241. VizieR J/ApJ/919/131.

**Bailer-Jones+2021**: AJ 161, 147; arXiv:2012.05220. Primary: GAVO `gedr3dist.main` (dc.g-vo.org/tableinfo/gedr3dist.main). Catalogue: VizieR I/352.

**StarHorse2 / Queiroz+2023**: A&A 673, A155; arXiv:2303.09926. Landing: https://data.aip.de/projects/aqueiroz2023.html. **Use v2 files only.**

**Andrae+2023**: ApJS 267, 8; arXiv:2302.02611. Zenodo 7945154.

**Ye+2024** (XP flux correction): A&A in press; arXiv:2411.19105.

**Dust maps**: Edenhofer+2024, A&A 685, A82; arXiv:2308.01295. Lallement+2022, A&A 661, A147. Green+2019 (Bayestar19), ApJ 887, 93. SFD: Schlegel+1998. Python: https://dustmaps.readthedocs.io/.

**Galpy & kinematics**: Bovy 2015, ApJS 216, 29; galpy.org. McMillan 2017, MNRAS 465, 76. Reid & Brunthaler 2020, ApJ 892, 39. GRAVITY Collaboration 2018, 2021 (A_sgrA*). Schönrich, Binney & Dehnen 2010, MNRAS 403, 1829 (solar motion). Binney 2012 (Staeckel fudge), MNRAS 426, 1324.

**TIC v8.2**: Paegert+2021, arXiv:2108.04778. MAST.

**dr2_neighbourhood**: https://gaia.aip.de/metadata/gaiadr3/dr2_neighbourhood/ and https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/.

---

*End of data_acquisition.md (v1). Update whenever a new data release or a schema change invalidates the recipes above.*
