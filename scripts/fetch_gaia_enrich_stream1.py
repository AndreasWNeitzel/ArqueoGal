"""Gaia DR3 enrichment for Stream 1 — APOGEE DR19 × Gaia DR3.

Reads 320 k unique source_ids from ``interim/apogee_dr19_source_ids.parquet``
and fetches the full §3.6 Gaia DR3 bundle (astrometry + 21 correlation
coefficients + photometry + GSP-Phot + GSP-Spec) via AIP TAP. Output is the
raw enrichment Parquet — further processing (Lindegren zpt, Riello+2021 G,
merge with APOGEE) happens in a downstream script.

Uses TAP UPLOAD rather than an inline ``IN (…)`` clause. AIP's gateway 504s
async submissions whose ADQL body exceeds ~100 KB (10 k int64 IDs ≈ 190 KB),
and attaching the IDs as a VOTable sidesteps that ceiling entirely. 10 k-ID
upload batches complete in ~6 s each — faster than inline-IN on GAVO.

Expected wall-time: ~32 batches × ~6 s/batch ≈ 3–4 min on AIP async.

Output
------
``data/interim/stream1_gaia_dr3_raw.parquet`` with all ENRICHMENT_COLS
plus the provenance sidecar. The 'raw' suffix is because Lindegren+2021 zpt
and CG&Brandt G-mag corrections haven't been applied yet.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from arqueogal.data.gaia_enrich import ENRICHMENT_ADQL_UPLOAD
from arqueogal.data.provenance import LocalSource, Provenance, TapSource, write_sidecar
from arqueogal.data.tap import AIP_TAP_URL, aip_service, batched_upload_fetch_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fetch_gaia_enrich_stream1")

BATCH_SIZE = 10_000


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


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ids_path = repo / "data" / "interim" / "apogee_dr19_source_ids.parquet"
    out = repo / "data" / "interim" / "stream1_gaia_dr3_raw.parquet"
    ckpt = repo / "data" / "interim" / "enrich_batches" / "stream1_gaia"

    logger.info("loading %s", ids_path)
    src_ids = pd.read_parquet(ids_path)["source_id"].astype("int64").to_list()
    n_src = len(src_ids)
    n_batches = (n_src + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(
        "%d unique Stream 1 source_ids → AIP Gaia DR3 (%d batches of %d)",
        n_src,
        n_batches,
        BATCH_SIZE,
    )

    svc = aip_service()
    df = batched_upload_fetch_df(
        svc,
        src_ids,
        ENRICHMENT_ADQL_UPLOAD,
        upload_name="ids",
        batch_size=BATCH_SIZE,
        checkpoint_dir=ckpt,
        checkpoint_prefix="batch",
        queue="2h",
        runid="arqueogal-stream1-enrich",
    )
    logger.info("fetched %d rows from AIP", len(df))

    # Float32 cast on magnitudes/astrometry where float64 is wasteful.
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

    _write_parquet_atomic(df, out)
    size_mb = out.stat().st_size / 1024**2
    logger.info("wrote %s (%.1f MB, %d cols)", out, size_mb, len(df.columns))

    ids_sha = _sha256_of(ids_path)
    prov = Provenance(
        output_file=str(out.relative_to(repo)),
        script="scripts/fetch_gaia_enrich_stream1.py",
        sources=[
            LocalSource(
                name="APOGEE DR19 unique source_ids (Stream 1 keys)",
                path=str(ids_path.relative_to(repo)),
                sha256=ids_sha,
            ),
            TapSource(
                name="AIP gaiadr3.gaia_source ⨝ gaiadr3.astrophysical_parameters",
                endpoint=AIP_TAP_URL,
                query=ENRICHMENT_ADQL_UPLOAD,
                n_batches=n_batches,
                batch_size=BATCH_SIZE,
            ),
        ],
        cuts_applied=[],
        corrections=["float64 → float32 on astrometric/photometric/APs columns"],
        row_count_before=n_src,
        row_count_after=int(len(df)),
        notes=(
            "Raw AIP Gaia DR3 enrichment for Stream 1 APOGEE DR19 source_ids. "
            "Async TAP with 10k-batch checkpoints. Downstream: apply Lindegren+2021 "
            "zero-point (needs astrometric_params_solved, nu_eff_used_in_astrometry, "
            "pseudocolour, ecl_lat, phot_g_mean_mag — all included here), then "
            "Riello+2021 (A&A 649, A3 Appendix A) G-mag correction, then merge "
            "into interim/stream1_apogee_gaia.parquet per task #76."
        ),
        extra={"batch_size": BATCH_SIZE},
    )
    write_sidecar(prov)


if __name__ == "__main__":
    main()
