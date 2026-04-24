"""Stream 2 end-to-end ingestion — TESS Hon+2021 × TIC v8.2 × Gaia DR3.

Mirrors the Stream 1/3 fetch pattern but chains three TAP stages:

    Hon+2021 (already in data/raw/tess_hon2021/hon2021.parquet)
        → VizieR TIC v8.2 lookup (TIC → DR2 GAIA), inline IN batching
        → drop TICs without a DR2 counterpart
        → AIP DR2→DR3 via gaiadr3.dr2_neighbourhood, TAP UPLOAD
        → apply §4.3 cuts (angular < 300 mas, |Δmag| < 0.1) + tie-break
        → AIP Gaia DR3 enrichment via ENRICHMENT_ADQL_UPLOAD
        → Lindegren+2021 parallax zpt + Riello+2021 G-mag correction
        → write data/interim/stream2_tess_gaia.parquet

UPLOAD is mandatory for the AIP stages: inline IN on dr2_neighbourhood or
gaia_source above ~5k IDs produces 504 Gateway Timeout from AIP's proxy
(see memory/reference_aip_tap_upload.md).

Expected wall-time: ~32 VizieR batches + ~14 AIP xmatch batches + ~13 AIP
enrichment batches ≈ 10–15 min total.

Output
------
- data/interim/stream2_xmatch.parquet — Hon+2021 + TIC + DR2→DR3 mapping
- data/interim/stream2_gaia_dr3_raw.parquet — Gaia enrichment for DR3 IDs
- data/interim/stream2_gaia_dr3_corrected.parquet — with corrections applied
- data/interim/stream2_tess_gaia.parquet — final joined training-ready frame

Per ``docs/research_brief.md``: Stream 2 is pre-staging for Task 4
asteroseismic ages (led by others). Pipelines 1 and 2 do not consume it yet.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from arqueogal.data.crossmatch import (
    DEFAULT_ANGULAR_DISTANCE_MAS,
    DEFAULT_MAG_DIFF_LIMIT,
    resolve_dr2_to_dr3,
)
from arqueogal.data.gaia_corrections import apply_g_mag_correction, apply_parallax_zpt
from arqueogal.data.gaia_enrich import ENRICHMENT_ADQL_UPLOAD
from arqueogal.data.provenance import (
    LocalSource,
    Provenance,
    TapSource,
    write_sidecar,
)
from arqueogal.data.tap import (
    AIP_TAP_URL,
    VIZIER_TAP_URL,
    aip_service,
    batched_upload_fetch_df,
    vizier_service,
)
from arqueogal.data.tess_hon2021 import HON2021_DEFAULT_PROB_THRESHOLD
from arqueogal.data.tic_v82 import TIC_V82_ADQL, fetch_tic_v82

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest_stream2")

TIC_BATCH_SIZE = 5_000
XMATCH_BATCH_SIZE = 10_000
ENRICH_BATCH_SIZE = 10_000

DR2_NEIGHBOURHOOD_UPLOAD_ADQL = """\
SELECT
    nbh.dr2_source_id, nbh.dr3_source_id,
    nbh.angular_distance, nbh.magnitude_difference
FROM gaiadr3.dr2_neighbourhood AS nbh
JOIN tap_upload.ids AS u ON nbh.dr2_source_id = u.source_id
"""


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _cast_gaia_float32(df: pd.DataFrame) -> pd.DataFrame:
    # Same cast list as Stream 1/3 enrichment scripts.
    float32_cols = [
        "ra",
        "dec",
        "parallax",
        "parallax_error",
        "pmra",
        "pmra_error",
        "pmdec",
        "pmdec_error",
        "ra_dec_corr",
        "ra_parallax_corr",
        "ra_pmra_corr",
        "ra_pmdec_corr",
        "dec_parallax_corr",
        "dec_pmra_corr",
        "dec_pmdec_corr",
        "parallax_pmra_corr",
        "parallax_pmdec_corr",
        "pmra_pmdec_corr",
        "phot_g_mean_mag",
        "phot_bp_mean_mag",
        "phot_rp_mean_mag",
        "phot_g_mean_flux_over_error",
        "bp_rp",
        "bp_g",
        "g_rp",
        "ruwe",
        "astrometric_excess_noise",
        "ipd_gof_harmonic_amplitude",
        "radial_velocity",
        "radial_velocity_error",
        "nu_eff_used_in_astrometry",
        "pseudocolour",
        "ecl_lat",
        "teff_gspphot",
        "teff_gspphot_lower",
        "teff_gspphot_upper",
        "logg_gspphot",
        "logg_gspphot_lower",
        "logg_gspphot_upper",
        "mh_gspphot",
        "mh_gspphot_lower",
        "mh_gspphot_upper",
        "ag_gspphot",
        "ag_gspphot_lower",
        "ag_gspphot_upper",
        "ebpminrp_gspphot",
        "distance_gspphot",
        "distance_gspphot_lower",
        "distance_gspphot_upper",
        "teff_gspspec",
        "logg_gspspec",
        "mh_gspspec",
        "alphafe_gspspec",
    ]
    for col in float32_cols:
        if col in df.columns and df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
    return df


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    hon_path = repo / "data" / "raw" / "tess_hon2021" / "hon2021.parquet"
    tic_ckpt = repo / "data" / "interim" / "xmatch_batches" / "stream2_tic"
    nbh_ckpt = repo / "data" / "interim" / "xmatch_batches" / "stream2_dr2_nbh"
    enrich_ckpt = repo / "data" / "interim" / "enrich_batches" / "stream2_gaia"

    xmatch_out = repo / "data" / "interim" / "stream2_xmatch.parquet"
    gaia_raw_out = repo / "data" / "interim" / "stream2_gaia_dr3_raw.parquet"
    gaia_corr_out = repo / "data" / "interim" / "stream2_gaia_dr3_corrected.parquet"
    final_out = repo / "data" / "interim" / "stream2_tess_gaia.parquet"

    if not hon_path.exists():
        raise SystemExit(f"missing {hon_path}; Hon+2021 extraction must be done first")

    # ------------------------------------------------------------------
    # Stage 1: Hon+2021 + TIC v8.2 lookup + DR2→DR3 crossmatch
    # ------------------------------------------------------------------
    logger.info("loading %s", hon_path)
    hon = pd.read_parquet(hon_path)
    # hon2021.parquet has column 'tic' (int64); normalise casing for the join.
    assert "tic" in hon.columns, f"expected 'tic' column in {hon_path}"
    n_hon = len(hon)
    logger.info(
        "Hon+2021: %d rows (pre-cut by Prob > %s upstream)", n_hon, HON2021_DEFAULT_PROB_THRESHOLD
    )

    logger.info("TIC v8.2 lookup via VizieR (batch=%d)", TIC_BATCH_SIZE)
    vz = vizier_service()
    tic = fetch_tic_v82(
        vz,
        hon["tic"].astype("int64").to_list(),
        batch_size=TIC_BATCH_SIZE,
        checkpoint_dir=tic_ckpt,
    )
    # VizieR column is 'TIC' / 'GAIA' (uppercase).
    tic_cols_lower = {c: c.lower() for c in tic.columns}
    tic = tic.rename(columns=tic_cols_lower)
    n_tic_total = len(tic)
    tic_valid = tic.dropna(subset=["gaia"]).copy()
    tic_valid["gaia"] = tic_valid["gaia"].astype("int64")
    n_tic_dr2 = len(tic_valid)
    logger.info(
        "TIC rows: %d total, %d with DR2 GAIA (%d dropped)",
        n_tic_total,
        n_tic_dr2,
        n_tic_total - n_tic_dr2,
    )

    logger.info("DR2→DR3 crossmatch via AIP UPLOAD (batch=%d)", XMATCH_BATCH_SIZE)
    aip = aip_service()
    dr2_ids = tic_valid["gaia"].astype("int64").to_list()
    n_nbh_batches = (len(dr2_ids) + XMATCH_BATCH_SIZE - 1) // XMATCH_BATCH_SIZE
    raw_nbh = batched_upload_fetch_df(
        aip,
        dr2_ids,
        DR2_NEIGHBOURHOOD_UPLOAD_ADQL,
        upload_name="ids",
        batch_size=XMATCH_BATCH_SIZE,
        checkpoint_dir=nbh_ckpt,
        checkpoint_prefix="batch",
        queue="2h",
        runid="arqueogal-stream2-dr2-dr3",
    )
    logger.info("dr2_neighbourhood raw rows: %d", len(raw_nbh))

    resolved = resolve_dr2_to_dr3(
        raw_nbh,
        max_angular_distance_mas=DEFAULT_ANGULAR_DISTANCE_MAS,
        max_mag_diff=DEFAULT_MAG_DIFF_LIMIT,
    )
    n_dr3 = len(resolved)
    logger.info("DR2→DR3 resolved: %d / %d DR2 sources pass §4.3 cuts", n_dr3, n_tic_dr2)

    # Stage 1 intermediate: Hon × TIC × DR2→DR3 mapping only (no Gaia enrichment).
    xmatch = hon.merge(tic_valid, on="tic", how="inner")
    xmatch = xmatch.merge(
        resolved,
        left_on="gaia",
        right_on="dr2_source_id",
        how="inner",
    )
    _write_parquet_atomic(xmatch, xmatch_out)
    logger.info(
        "wrote xmatch intermediate: %s (%d rows, %d cols, %.1f MB)",
        xmatch_out,
        len(xmatch),
        len(xmatch.columns),
        xmatch_out.stat().st_size / 1024**2,
    )

    # ------------------------------------------------------------------
    # Stage 2: Gaia DR3 enrichment via UPLOAD
    # ------------------------------------------------------------------
    dr3_ids = xmatch["dr3_source_id"].astype("int64").drop_duplicates().to_list()
    n_enrich = len(dr3_ids)
    n_enrich_batches = (n_enrich + ENRICH_BATCH_SIZE - 1) // ENRICH_BATCH_SIZE
    logger.info(
        "Gaia DR3 enrichment via AIP UPLOAD: %d unique DR3 source_ids (%d batches)",
        n_enrich,
        n_enrich_batches,
    )
    gaia_raw = batched_upload_fetch_df(
        aip,
        dr3_ids,
        ENRICHMENT_ADQL_UPLOAD,
        upload_name="ids",
        batch_size=ENRICH_BATCH_SIZE,
        checkpoint_dir=enrich_ckpt,
        checkpoint_prefix="batch",
        queue="2h",
        runid="arqueogal-stream2-enrich",
    )
    logger.info("Gaia enrichment: fetched %d rows", len(gaia_raw))
    gaia_raw = _cast_gaia_float32(gaia_raw)
    _write_parquet_atomic(gaia_raw, gaia_raw_out)
    logger.info(
        "wrote raw enrichment: %s (%.1f MB)", gaia_raw_out, gaia_raw_out.stat().st_size / 1024**2
    )

    # Provenance for raw enrichment (mirrors Stream 1/3 convention).
    prov_raw = Provenance(
        output_file=str(gaia_raw_out.relative_to(repo)),
        script="scripts/ingest_stream2.py",
        sources=[
            LocalSource(
                name="Stream 2 xmatch (Hon+2021 × TIC × DR2→DR3)",
                path=str(xmatch_out.relative_to(repo)),
                sha256=_sha256_of(xmatch_out),
            ),
            TapSource(
                name="AIP gaiadr3.gaia_source ⨝ gaiadr3.astrophysical_parameters",
                endpoint=AIP_TAP_URL,
                query=ENRICHMENT_ADQL_UPLOAD,
                n_batches=n_enrich_batches,
                batch_size=ENRICH_BATCH_SIZE,
            ),
        ],
        cuts_applied=[],
        corrections=["float64 → float32 on astrometric/photometric/APs columns"],
        row_count_before=n_enrich,
        row_count_after=int(len(gaia_raw)),
        notes=(
            "Raw AIP Gaia DR3 enrichment for Stream 2 DR3 source_ids. "
            "Lindegren+2021 zpt + Riello+2021 G-mag correction applied in "
            "stream2_gaia_dr3_corrected.parquet (below)."
        ),
        extra={"batch_size": ENRICH_BATCH_SIZE},
    )
    write_sidecar(prov_raw)

    # ------------------------------------------------------------------
    # Stage 3: Gaia corrections (Lindegren zpt + Riello G-mag)
    # ------------------------------------------------------------------
    logger.info("applying Lindegren+2021 parallax zpt")
    gaia_zpt = apply_parallax_zpt(gaia_raw)
    logger.info("applying Riello+2021 G-mag correction")
    gaia_corr = apply_g_mag_correction(gaia_zpt)
    for col in ("parallax_zpt", "parallax_corr", "phot_g_mean_mag_corr"):
        if col in gaia_corr.columns and gaia_corr[col].dtype == np.float64:
            gaia_corr[col] = gaia_corr[col].astype(np.float32)
    _write_parquet_atomic(gaia_corr, gaia_corr_out)
    logger.info(
        "wrote corrected enrichment: %s (%.1f MB, %d cols)",
        gaia_corr_out,
        gaia_corr_out.stat().st_size / 1024**2,
        len(gaia_corr.columns),
    )

    n_zpt_applied = int(gaia_corr["parallax_zpt"].notna().sum())
    prov_corr = Provenance(
        output_file=str(gaia_corr_out.relative_to(repo)),
        script="scripts/ingest_stream2.py",
        sources=[
            LocalSource(
                name="Stream 2 Gaia DR3 raw enrichment",
                path=str(gaia_raw_out.relative_to(repo)),
                sha256=_sha256_of(gaia_raw_out),
            ),
        ],
        cuts_applied=[],
        corrections=[
            f"Lindegren+2021 parallax zero-point — {n_zpt_applied}/{len(gaia_corr)} rows",
            "Riello+2021 G-band mag correction (2-/6-param at G ≥ 13)",
        ],
        row_count_before=int(len(gaia_raw)),
        row_count_after=int(len(gaia_corr)),
        notes=(
            "Adds parallax_zpt, parallax_corr, phot_g_mean_mag_corr columns. "
            "Downstream joins go through stream2_tess_gaia.parquet below."
        ),
        extra={"zpt_rows_applied": n_zpt_applied},
    )
    write_sidecar(prov_corr)

    # ------------------------------------------------------------------
    # Stage 4: Assemble final frame (Hon × TIC × xmatch × Gaia_corrected)
    # ------------------------------------------------------------------
    overlap_hon_gaia = set(xmatch.columns) & set(gaia_corr.columns) - {"dr3_source_id"}
    if overlap_hon_gaia:
        logger.warning(
            "dropping overlapping non-key cols from xmatch: %s", sorted(overlap_hon_gaia)
        )
        xmatch = xmatch.drop(columns=list(overlap_hon_gaia))

    final = xmatch.merge(
        gaia_corr,
        left_on="dr3_source_id",
        right_on="source_id",
        how="inner",
    )
    n_final = len(final)
    logger.info("final joined: %d rows × %d cols", n_final, len(final.columns))
    _write_parquet_atomic(final, final_out)
    logger.info("wrote final: %s (%.1f MB)", final_out, final_out.stat().st_size / 1024**2)

    prov_final = Provenance(
        output_file=str(final_out.relative_to(repo)),
        script="scripts/ingest_stream2.py",
        sources=[
            LocalSource(
                name="Hon+2021 TESS νmax catalogue",
                path=str(hon_path.relative_to(repo)),
                sha256=_sha256_of(hon_path),
            ),
            TapSource(
                name="VizieR TIC v8.2 (IV/39/tic82)",
                endpoint=VIZIER_TAP_URL,
                query=TIC_V82_ADQL,
                n_batches=(n_hon + TIC_BATCH_SIZE - 1) // TIC_BATCH_SIZE,
                batch_size=TIC_BATCH_SIZE,
            ),
            TapSource(
                name="AIP gaiadr3.dr2_neighbourhood (UPLOAD)",
                endpoint=AIP_TAP_URL,
                query=DR2_NEIGHBOURHOOD_UPLOAD_ADQL,
                n_batches=n_nbh_batches,
                batch_size=XMATCH_BATCH_SIZE,
            ),
            TapSource(
                name="AIP gaiadr3.gaia_source ⨝ astrophysical_parameters (UPLOAD)",
                endpoint=AIP_TAP_URL,
                query=ENRICHMENT_ADQL_UPLOAD,
                n_batches=n_enrich_batches,
                batch_size=ENRICH_BATCH_SIZE,
            ),
            LocalSource(
                name="Stream 2 corrected Gaia enrichment",
                path=str(gaia_corr_out.relative_to(repo)),
                sha256=_sha256_of(gaia_corr_out),
            ),
        ],
        cuts_applied=[
            f"Hon+2021 Prob > {HON2021_DEFAULT_PROB_THRESHOLD} (upstream VizieR cut)",
            "drop TICs without DR2 GAIA counterpart",
            f"dr2_neighbourhood angular_distance < {DEFAULT_ANGULAR_DISTANCE_MAS} mas",
            f"dr2_neighbourhood |Δmag| < {DEFAULT_MAG_DIFF_LIMIT}",
            "tie-break on smallest |Δmag| per DR2 source",
        ],
        corrections=[
            "Lindegren+2021 parallax zero-point",
            "Riello+2021 G-band mag correction",
        ],
        row_count_before=n_hon,
        row_count_after=n_final,
        notes=(
            "Stream 2 pre-staged for Task 4 asteroseismic ages (led by others). "
            "Not consumed by Pipelines 1 or 2 yet. Carries Hon+2021 νmax and "
            "TESS-derived Teff/R/L alongside the full Gaia DR3 bundle."
        ),
        extra={
            "n_hon_in": n_hon,
            "n_tic_with_dr2": n_tic_dr2,
            "n_dr3_resolved": n_dr3,
            "n_final_joined": n_final,
        },
    )
    write_sidecar(prov_final)
    logger.info("Stream 2 ingestion complete")


if __name__ == "__main__":
    main()
