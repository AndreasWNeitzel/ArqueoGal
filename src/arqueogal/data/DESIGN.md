# `arqueogal.data` — Design

## Purpose

Data acquisition, cross-matching, and feature engineering for the three data streams
(Stream 1 APOGEE DR19×Gaia, Stream 2 TESS Hon+2021×Gaia, Stream 3 Gaia RGB+RC application
sample). Produces analysis-ready Parquet feature matrices in `data/processed/` with JSON
provenance sidecars in `data/provenance/`.

**Primary reference**: `docs/data_acquisition.md`. Read that before writing or modifying
any code in this module. All ADQL queries, TAP endpoints, quality cuts, correction
polynomials, and disk-footprint accounting live there.

## Module layout

```
arqueogal.data
├── credentials.py        — reads ~/.arqueogal/credentials.yaml (AIP user/password, etc.)
├── tap.py                — pyvo TAP wrappers (AIP, GAVO, VizieR). Async submission >5k rows.
├── downloads.py          — requests-based HTTPS downloads with streaming, tqdm, SHA-256,
│                           atomic temp→rename.
├── provenance.py         — JSON sidecar writer (§14.4 of data_acquisition.md).
├── gaia_corrections.py   — Lindegren+2021 parallax zpt + Riello+2021 G-mag correction.
│                           Mandatory at Stream 1/3 ingestion.
├── apogee_dr19.py        — Stream 1 loader; DR19 column-naming handling; Mészáros+2025
│                           [X/M]/Teff correction polynomials.
├── tess_hon2021.py       — Stream 2 loader; Hon+2021 ν_max catalogue; TASOC pre-staging.
├── stream3_selection.py  — Andrae+2023 vetted-RGB stratified sampling to 1.5M stars.
├── gaia_xp.py            — XP coefficient extraction; Ye+2024 NN flux-correction;
│                           normalisation (§6.4 — FIXED ORDER: Ye+2024 → divide by c0 →
│                           log10+z-score c0 → error propagation).
├── dust_maps.py          — Edenhofer+2024 (d<1.25 kpc) / Lallement+2022 (1.25–3 kpc) /
│                           SFD composition + GSP-Phot neighborhood-median Av via cKDTree.
├── distances.py          — Bailer-Jones+2021 (GAVO TAP) + StarHorse2 v2 (AIP TAP).
├── kinematics.py         — galpy orbits (McMillan17, Staeckel fudge, δ=0.45); central-value
│                           for bulk, full MC driven by downstream consumers (Starfold,
│                           separate repo) for boundary-star subsamples only.
├── crossmatch.py         — DR2↔DR3 via gaiadr3.dr2_neighbourhood; TIC v8.2↔Gaia DR3.
│                           Many-to-many handling; brightest-tie-break.
└── fire2_ananke.py       — Subtask 5.1 hare-and-hounds. Follows Starfold (separate repo);
                            retained here only if still imported for method-validation
                            harness and strictly segregated from real-data science.
```

## Hard rules

- **pyvo over astroquery.gaia** (unstable in recent months). Astroquery.vizier is a last
  resort only.
- **Async TAP (`submit_job`)** for queries returning >5 000 rows. Sync `search()` times out
  on AIP at ~90 s.
- **Batched, resumable**: 10 000 source_ids per Gaia main-table batch; 5 000 per XP batch.
  Per-batch checkpoint files in `data/interim/xp_batches/`.
- **Provenance sidecar is mandatory**. No Parquet output ships without its
  `*.provenance.json` (source URL, query string, row counts, cuts, corrections, git SHA,
  timestamp, input SHA-256). An artefact without provenance is not reproducible.
- **Gaia corrections at ingestion, never later**: Lindegren+2021 parallax zpt
  (official `zero_point.py`), Riello+2021 (A&A 649, A3 Appendix A) G-mag correction.
  Downstream code assumes corrected values.
- **5 GB disk budget** (data_acquisition.md §12). Audit before adding artefacts.
- **Float32 downcast** for XP coefficients and kinematics outputs at Parquet write time;
  float64 inside galpy integration for numerical stability.
- **No XP coefficient correlation matrices** — 48 GB for 2 M stars; excluded from budget.
- **FIRE-2 in `data/fire2/`, strictly separated from `data/processed/`** (real-data science).

## Tests

Under `tests/data/`. Smoke test per TAP endpoint (`TOP 10`), XP preprocessing round-trip
(within float32 tolerance), DR2→DR3 cross-match on a 1 k ground-truth subset, Mészáros+2025
before/after [X/M]–Teff trend check. See data_acquisition.md §15.
