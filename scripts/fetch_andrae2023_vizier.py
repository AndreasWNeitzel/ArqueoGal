"""Fetch the Andrae+2023 / Ardern-Arentsen+2024 reissue from VizieR TAP.

The Zenodo table_1_catwise.fits.gz is 3.59 GB verified (see
reports/extraction_budget_20260418.md). We only use ~15 columns from it, so
we pull the equivalent reissue table ``J/MNRAS/537/1984/a23`` at VizieR's
TAPVizieR service directly. Expected wire-time: ~10 min for the async job
plus ~10 min for the VOTable download (10.48 M rows × 26 cols).

The result is written to:
    data/raw/andrae2023/andrae2023_rgb.parquet
    data/raw/andrae2023/andrae2023_rgb.provenance.json

Columns are renamed to canonical snake_case (``source_id``, ``ra_deg``,
``dec_deg``, ``teff``, ``logg``, ``fe_h``, ``c_fe``, ``ebv``, ``energy``,
``l_z``, etc) and floats are cast to float32 to keep the Parquet around
~800 MB per the revised budget.

Response-format escape hatch
----------------------------
Default submission uses the TAPVizieR default (VOTable/TABLEDATA — XML).
TABLEDATA parsing is CPU-bound at ~10–20 k rows/s, so a 10.5 M-row pull
takes 20–30 min just to parse client-side and peaks around 3–4 GB RSS.
If that fails, set one of:

    ANDRAE_FORMAT=binary2   # VOTable/BINARY2 — ~5–10× faster parse, same path
    ANDRAE_FORMAT=fits      # FITS stream — fastest, manual download path

FITS mode bypasses ``pyvo.DALResults.to_table`` (which expects VOTable) and
reads the job result URL with ``astropy.io.fits`` directly.
"""

from __future__ import annotations

import gc
import io
import logging
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyvo
import requests

from arqueogal.data.provenance import Provenance, TapSource, write_sidecar

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fetch_andrae2023_vizier")

VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"
TABLE = '"J/MNRAS/537/1984/a23"'

ADQL = f"""
SELECT
    "GaiaDR3" AS source_id,
    "RA_ICRS" AS ra_deg,
    "DE_ICRS" AS dec_deg,
    "Gmag"    AS g_mag,
    "plx"     AS parallax_mas,
    "e_plx"   AS parallax_err_mas,
    "pvar"    AS pvar,
    "E(B-V)"  AS ebv,
    "(BP-RP)0" AS bp_rp_0,
    "Gmag0"   AS g_mag_0,
    "Teff"    AS teff,
    "e_Teff"  AS e_teff,
    "s_Teff"  AS s_teff,
    "logg"    AS logg,
    "e_logg"  AS e_logg,
    "s_logg"  AS s_logg,
    "[Fe/H]"  AS fe_h,
    "e_[Fe/H]" AS e_fe_h,
    "s_[Fe/H]" AS s_fe_h,
    "[C/Fe]"  AS c_fe,
    "e_[C/Fe]" AS e_c_fe,
    "s_[C/Fe]" AS s_c_fe,
    "Ccor"    AS c_cor,
    "Energy"  AS energy,
    "Lz"      AS l_z
FROM {TABLE}
"""

FLOAT32_COLS = (
    "ra_deg",
    "dec_deg",
    "g_mag",
    "parallax_mas",
    "parallax_err_mas",
    "ebv",
    "bp_rp_0",
    "g_mag_0",
    "teff",
    "e_teff",
    "s_teff",
    "logg",
    "e_logg",
    "s_logg",
    "fe_h",
    "e_fe_h",
    "s_fe_h",
    "c_fe",
    "e_c_fe",
    "s_c_fe",
    "c_cor",
    "energy",
    "l_z",
)


def _fits_fetch(job: "pyvo.dal.tap.AsyncTAPJob"):
    """FITS fallback path — bypasses pyvo's VOTable parser.

    pyvo stores result URIs on the job object; for FITS we GET the first
    result URL into memory (astropy requires a seekable stream) and let
    astropy parse it directly. Peak RSS stays roughly 2× the file size.
    """
    from astropy.io import fits
    from astropy.table import Table

    uris = job.result_uris
    if not uris:
        raise RuntimeError(f"job {job.url} reports no result URIs")
    url = uris[0]
    logger.info("FITS mode: downloading %s", url)
    resp = requests.get(url, timeout=3600)
    resp.raise_for_status()
    logger.info("FITS mode: received %.1f MB, parsing", len(resp.content) / 1024**2)
    with fits.open(io.BytesIO(resp.content), memmap=False) as hdul:
        return Table.read(hdul[1])


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / "data" / "raw" / "andrae2023"
    out = out_dir / "andrae2023_rgb.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt_alias = os.environ.get("ANDRAE_FORMAT", "default").lower()
    fmt_map = {
        "default": None,
        "binary2": "votable/binary2",
        "fits": "fits",
    }
    if fmt_alias not in fmt_map:
        raise SystemExit(f"ANDRAE_FORMAT must be one of {sorted(fmt_map)}; got {fmt_alias!r}")
    responseformat = fmt_map[fmt_alias]

    logger.info("async TAP submit on %s (format=%s)", VIZIER_TAP, responseformat or "default")
    svc = pyvo.dal.TAPService(VIZIER_TAP)
    submit_kwargs: dict[str, str] = {"language": "ADQL"}
    if responseformat is not None:
        submit_kwargs["responseformat"] = responseformat
    job = svc.submit_job(ADQL, maxrec=20_000_000, **submit_kwargs)
    job.run()
    logger.info("async TAP job submitted: %s", job.url)

    t0 = time.time()
    last_phase = ""
    while True:
        phase = job.phase
        if phase != last_phase:
            logger.info("phase=%s elapsed=%.0fs", phase, time.time() - t0)
            last_phase = phase
        if phase in ("COMPLETED", "ERROR", "ABORTED"):
            break
        time.sleep(15)

    if job.phase != "COMPLETED":
        raise RuntimeError(f"Andrae TAP job ended in {job.phase}: {job.url}")

    if fmt_alias == "fits":
        logger.info("fetching results (FITS direct-download path)")
        table = _fits_fetch(job)
    else:
        logger.info("fetching results (%s via pyvo VOTable path)", fmt_alias)
        r = job.fetch_result()
        table = r.to_table()
    n_rows = len(table)
    col_names = list(table.colnames)
    logger.info("fetched %d rows × %d cols", n_rows, len(col_names))

    # Stream column-by-column into pyarrow to avoid holding astropy Table +
    # pandas DataFrame in RAM simultaneously (~6-7 GB peak → WSL OOM).
    arrays: list[pa.Array] = []
    for name in col_names:
        data = np.asarray(table[name])
        if name == "source_id":
            data = np.asarray(data, dtype=np.int64)
        elif name in FLOAT32_COLS and data.dtype == np.float64:
            data = data.astype(np.float32)
        arrays.append(pa.array(data))
        table.remove_column(name)
        del data
    del table
    gc.collect()

    arrow_table = pa.Table.from_arrays(arrays, names=col_names)
    del arrays
    gc.collect()
    logger.info(
        "arrow table built — schema=%s", arrow_table.schema.to_string(show_field_metadata=False)
    )

    tmp = out.with_suffix(out.suffix + ".part")
    pq.write_table(arrow_table, tmp, compression="snappy")
    os.replace(tmp, out)
    size_mb = out.stat().st_size / 1024**2
    logger.info("wrote %s (%.1f MB)", out, size_mb)

    prov = Provenance(
        output_file=str(out.relative_to(repo)),
        script="scripts/fetch_andrae2023_vizier.py",
        sources=[
            TapSource(
                name="Andrae+2023 reissue (Ardern-Arentsen+2024 MNRAS 537:1984) at VizieR",
                endpoint=VIZIER_TAP,
                query=ADQL.strip(),
                n_batches=1,
            )
        ],
        cuts_applied=[],
        corrections=[
            "float64 → float32 cast on all astrophysical columns (save 50% disk)",
        ],
        row_count_before=None,
        row_count_after=int(arrow_table.num_rows),
        notes=(
            "Direct column-projected pull of the Andrae+2023 vetted-RGB sample via "
            "VizieR TAP, substituting for the 3.59 GB Zenodo FITS. Keeps disk footprint "
            "at ~800 MB per reports/extraction_budget_20260418.md. "
            "XGBoost labels are selection-only, never Pipeline 1 training targets."
        ),
        extra={"columns": list(arrow_table.column_names), "tap_job_url": job.url},
    )
    write_sidecar(prov)


if __name__ == "__main__":
    main()
