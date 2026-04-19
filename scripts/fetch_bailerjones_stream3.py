"""Fetch Bailer-Jones+2021 photogeometric distances for Stream 3.

Mirrors the Stream 1 fetch at ``logs/bj_stream1.log`` / ``per_stream1.parquet``
but against the Andrae+2023 stratified Stream 3 subsample (~168 k source_ids).

GAVO TAP is unauthenticated, so this runs without the AIP token. Expected
wall-time: ~17 batches × ~25 s/batch ≈ 7–8 min.

Output
------
``data/raw/bailer_jones_2021/per_stream3.parquet`` (~8 MB) with columns
    ``source_id``, ``r_med_geo``, ``r_lo_geo``, ``r_hi_geo``,
    ``r_med_photogeo``, ``r_lo_photogeo``, ``r_hi_photogeo``, ``flag``
plus the provenance sidecar.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from arqueogal.data.distances import BAILERJONES_ADQL, fetch_bailerjones
from arqueogal.data.provenance import LocalSource, Provenance, TapSource, write_sidecar
from arqueogal.data.tap import GAVO_TAP_URL, gavo_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fetch_bailerjones_stream3")

BATCH_SIZE = 10_000
FLOAT32_COLS = (
    "r_med_geo", "r_lo_geo", "r_hi_geo",
    "r_med_photogeo", "r_lo_photogeo", "r_hi_photogeo",
)


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
    ids_path = repo / "data" / "interim" / "stream3_selected_source_ids.parquet"
    out = repo / "data" / "raw" / "bailer_jones_2021" / "per_stream3.parquet"
    ckpt = repo / "data" / "raw" / "bailer_jones_2021" / "_ckpt_stream3"

    logger.info("loading %s", ids_path)
    src_ids = pd.read_parquet(ids_path)["source_id"].astype("int64").to_list()
    n_src = len(src_ids)
    n_batches = (n_src + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info("%d Stream 3 source_ids → GAVO Bailer-Jones (%d batches of %d)",
                n_src, n_batches, BATCH_SIZE)

    svc = gavo_service()
    df = fetch_bailerjones(
        svc,
        src_ids,
        batch_size=BATCH_SIZE,
        mode="async",
        checkpoint_dir=ckpt,
    )
    logger.info("fetched %d rows", len(df))

    # float32 cast on all distance columns (50% disk).
    for col in FLOAT32_COLS:
        if col in df.columns and df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)

    _write_parquet_atomic(df, out)
    size_mb = out.stat().st_size / 1024**2
    logger.info("wrote %s (%.1f MB)", out, size_mb)

    ids_sha = _sha256_of(ids_path)
    prov = Provenance(
        output_file=str(out.relative_to(repo)),
        script="scripts/fetch_bailerjones_stream3.py",
        sources=[
            LocalSource(
                name="Stream 3 selected source_ids (Andrae+2023 RGB stratified)",
                path=str(ids_path.relative_to(repo)),
                sha256=ids_sha,
            ),
            TapSource(
                name="GAVO gedr3dist.main (Bailer-Jones+2021 photogeometric)",
                endpoint=GAVO_TAP_URL,
                query=BAILERJONES_ADQL,
                n_batches=n_batches,
                batch_size=BATCH_SIZE,
            ),
        ],
        cuts_applied=[],
        corrections=["float64 → float32 on all distance columns"],
        row_count_before=n_src,
        row_count_after=int(len(df)),
        notes=(
            "Bailer-Jones+2021 photogeometric distances (r_med_photogeo primary) "
            "for the 168 k Stream 3 application sample. Async TAP with 10k-batch "
            "checkpoints — ~7 min wire time on GAVO. Dropouts (no distance "
            "solution) expected to match Stream 1 rate of ~0.1%."
        ),
        extra={"batch_size": BATCH_SIZE},
    )
    write_sidecar(prov)


if __name__ == "__main__":
    main()
