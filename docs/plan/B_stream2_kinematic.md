# Phase B — Stream-2 kinematic catalogue (D-Cat-b MVP)

**Status (2026-04-29): code shipped, production run pending.**
**Module:** `src/arqueogal/data/build_stream2_kinematic_catalogue.py`.
**CLI driver:** `scripts/build_stream2_kinematic_catalogue.py`.
**Tests:** `tests/data/test_build_stream2_kinematic_catalogue.py` (11 tests).

## Goal

Produce the **astrometric / kinematic** component of D-Cat-b on the
Stream-2 (Hon+2021 TESS asteroseismic-giant) cohort, before chemistry
inference (Phase C) or Stream-3 generalisation (Phase E). This is the
MVP that the brief identifies as the highest priority for Aug 2026.

## Pipeline

```
Stream-2 base parquet (ingest_stream2 output)
  → BJ21 photogeometric distances (GAVO TAP, fetch_bailerjones)
  → percentile-spread distance trust flags (assign_distance_trust_flags)
  → dust-map fusion av_los (Edenhofer / Lallement / SFD + neighbourhood,
        compose_av or external pre-fetch)
  → Yuan+2013 + CCM89 R_V=3.1 broadband dereddening (apply_extinction_corrections)
        — no-op when Stream 2 lacks the IR cross-match
  → Av trust flags (av_is_neighborhood_fallback, av_distance_prior_dominated,
        av_neighbourhood_high_dispersion)
  → galpy actions under McMillan+2017 (compute_actions)
        with the GALACTOCENTRIC_FRAME constants pinned
  → write parquet + provenance.json
```

## Output schema (per-row)

| Group | Columns |
|---|---|
| Identifier | `source_id` (Gaia DR3) |
| Astrometry (raw, Lindegren+2021 corrected at ingest_stream2) | `ra`, `dec`, `parallax`, `parallax_error`, `parallax_over_error`, `pmra`, `pmdec`, `radial_velocity`, ten correlations |
| Distance | `r_med_photogeo`, `r_lo_photogeo`, `r_hi_photogeo` (BJ21) |
| Distance trust flags | `dist_has_bj21`, `dist_relative_spread_high`, `dist_negative_parallax`, `dist_trustworthy` |
| Photometry (Riello+2021 G-mag corrected at ingest_stream2) | `phot_g_mean_mag_corr`, `bp_rp`, `bp_g`, `g_rp` |
| IR (optional) | `j_mag`, `h_mag`, `k_mag`, `w1_mag`, `w2_mag` (raw, retained for diagnostics) plus `*_dered` if extinction applied |
| Extinction | `av_los`, `av_los_source` (int8 categorical, see `data.extinction.AV_SOURCE_CODES`) |
| Av trust flags | `av_is_neighborhood_fallback`, `av_distance_prior_dominated`, `av_neighbourhood_high_dispersion` |
| Galpy | `R_galcen_kpc`, `z_galcen_kpc`, `phi_galcen_rad`, `v_R_kms`, `v_T_kms`, `v_z_kms`, `J_R_kpc_kms`, `L_z_kpc_kms`, `J_z_kpc_kms`, `ecc`, `r_peri_kpc`, `r_apo_kpc`, `z_max_kpc`, `E_kms2` |

The provenance sidecar (`*.provenance.json`) carries the extinction-law
fingerprint (Yuan+2013 ratios, R_V), kinematics config (McMillan17,
Staeckel δ=0.45), distance-trust threshold, per-flag firing counts.

## Acceptance criteria

When the production run completes:

1. `dist_trustworthy` rate ≥ 70 % on the Hon+2021 cohort (TESS giants
   are bright; most have parallax SNR ≫ 5).
2. `av_los_source` populated for ≥ 95 % of rows; `missing` rate ≤ 5 %.
3. Galpy actions solved on ≥ 95 % of `dist_trustworthy` stars.
4. Provenance sidecar carries the extinction-law fingerprint + the four
   trust-flag firing rates.
5. The Stream-2 inference path (Phase C) reads the parquet and emits
   chemistry predictions on `dist_trustworthy=True` rows only.

## Halt conditions

- BJ21 trust-flag rate < 50 % on the Hon+2021 cohort (would indicate a
  TAP / cross-match regression).
- Dust-map fusion fails on > 20 % of rows (would indicate a missing
  coverage layer).
- galpy fails to converge on > 10 % of trustworthy stars (would
  indicate a kinematics-config drift).

## Running it

```bash
# 1. Ensure the Stream-2 base parquet exists.
python -m arqueogal.data.ingest_stream2 ...

# 2. Pre-fetch BJ21 distances (GAVO TAP):
PYTHONPATH=src python scripts/fetch_bj21_for_stream.py \
    --source-id-parquet data/interim/stream2_tess_gaia.parquet \
    --out                data/interim/stream2_bj21.parquet

# 3. Pre-fetch dust-map fusion (uses dustmaps + neighbourhood-median):
PYTHONPATH=src python scripts/fetch_dust_layer_for_stream.py \
    --source-id-parquet data/interim/stream2_tess_gaia.parquet \
    --out                data/interim/stream2_av_layer.parquet

# 4. Compose the kinematic catalogue:
PYTHONPATH=src python scripts/build_stream2_kinematic_catalogue.py \
    --stream2-parquet data/interim/stream2_tess_gaia.parquet \
    --bj21-parquet    data/interim/stream2_bj21.parquet \
    --av-parquet      data/interim/stream2_av_layer.parquet \
    --out             release/D-Cat-b/stream2_kinematic_catalogue.parquet
```

Steps 2 and 3 are user-side pre-fetches; the underlying functions
(`fetch_bailerjones`, `compose_av`) exist in `arqueogal.data.distances`
and `arqueogal.data.dust_maps`. Per-stream wrappers can be added once
the production run is scheduled.

## What this enables

Once Phase B ships its production run:

- **Phase C** (Stream-2 chemistry inference) can be triggered: the
  Pipeline-1 v2 ensemble reads the kinematic catalogue + the dereddened
  broadband features and emits {Teff, log g, [M/H], [α/M], [Mg/H]} per
  star. The RGB+HeCB filter from
  `arqueogal.data.evolutionary_stage.filter_to_rgb_or_hecb` should be
  applied at this step.
- **Phase D** (Stream-2 cross-catalogue Test-6) can be triggered: the
  framework already lives in `arqueogal.xp_abundances.main.cross_catalogue`;
  per-catalogue cross-matches against AspGap / SHBoost / Guiglion+2024 /
  GALAH DR4 produce the methods-paper Figure 7.
- **Methods paper** can be drafted with the per-element Stream-2
  validation residual diagrams and the cross-catalogue rank-summary
  heatmap (gallery stage 27).

## Open items for the user

- Run Phase B in production (gated on AIP / GAVO credentials and the
  v2 ensemble). Pipeline 1 v2 inference can technically run on the
  Stream-2 base parquet without Phase B's BJ21 + Av layers, but the
  kinematic columns and extinction-aware features will be missing.
- Extend Stream 2 to fetch IR (2MASS + AllWISE) at ingest, so the
  Yuan+2013 dereddening fires natively. Currently only Stream 1 / 3 do
  the IR cross-match.
