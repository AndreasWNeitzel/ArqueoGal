"""Gaia DR3 enrichment for Stream 3 — delta 449k Ye-OK (Phase 3a expansion).

Delta variant of ``fetch_gaia_enrich_stream3.py``. Reads Ye-OK delta
source_ids (stream3_delta_ye_ok_source_ids.parquet) rather than the
168k existing selection, writes to stream3_delta_gaia_dr3_raw.parquet.

Wall-time: 45 batches × ~20 s/batch ≈ 15 min via AIP TAP UPLOAD.

Output
------
``data/interim/stream3_delta_gaia_dr3_raw.parquet`` with the §3.6
enrichment columns + provenance sidecar.
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
logger = logging.getLogger("fetch_gaia_enrich_stream3_delta")

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
    ids_path = repo / "data" / "interim" / "stream3_delta_ye_ok_source_ids.parquet"
    out = repo / "data" / "interim" / "stream3_delta_gaia_dr3_raw.parquet"
    ckpt = repo / "data" / "interim" / "enrich_batches" / "stream3_gaia_delta"

    logger.info("loading %s", ids_path)
    src_ids = pd.read_parquet(ids_path)["source_id"].astype("int64").to_list()
    n_src = len(src_ids)
    n_batches = (n_src + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(
        "%d Stream 3 delta Ye-OK source_ids → AIP Gaia DR3 (%d batches of %d)",
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
        runid="arqueogal-stream3-delta-enrich",
    )
    logger.info("fetched %d rows from AIP", len(df))

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
        script="scripts/fetch_gaia_enrich_stream3_delta.py",
        sources=[
            LocalSource(
                name="Stream 3 delta Ye-OK source_ids (Phase 3a)",
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
            "Raw AIP Gaia DR3 enrichment for Stream 3 delta (Phase 3a Ye-OK "
            "subset of 454k expansion). TAP UPLOAD batches of 10 k. "
            "Downstream: Lindegren+2021 zpt + Riello+2021 G correction, then "
            "union with stream3_gaia_dr3_corrected.parquet for full 622k "
            "Stream 3 feature matrix."
        ),
        extra={"batch_size": BATCH_SIZE, "phase": "Phase 3b delta enrichment"},
    )
    write_sidecar(prov)


if __name__ == "__main__":
    main()
