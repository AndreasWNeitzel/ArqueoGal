"""Assemble ``data/interim/stream1_apogee_gaia.parquet`` — the Pipeline-1
training pool.

Inputs
------
1. ``data/interim/apogee_dr19_corrected.parquet`` — APOGEE DR19 post-cuts,
   Mészáros+2025 Teff-trend corrections applied, BJ21 photogeometric distances
   already embedded.
2. ``data/interim/stream1_gaia_dr3_corrected.parquet`` — Gaia DR3 enrichment
   for post-cut APOGEE source_ids, Lindegren+2021 zpt and Riello+2021 G-mag
   corrections applied.

Join
----
INNER JOIN on ``source_id``. Drops the (tiny) set of APOGEE rows whose Gaia
source_id failed enrichment on AIP. Left-join variant is not useful for a
training set — no-Gaia rows have no XP features.

Output
------
``data/interim/stream1_apogee_gaia.parquet`` + provenance sidecar. This is
the training-pool candidate for Pipeline-1 (xp_abundances). Per-star XP
coefficients still need to be pulled (task #80) before the file becomes a
feature matrix — that happens downstream.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import pandas as pd

from arqueogal.data.provenance import LocalSource, Provenance, write_sidecar

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_stream1_apogee_gaia")


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
    apogee_path = repo / "data" / "interim" / "apogee_dr19_corrected.parquet"
    gaia_path = repo / "data" / "interim" / "stream1_gaia_dr3_corrected.parquet"
    out = repo / "data" / "interim" / "stream1_apogee_gaia.parquet"

    for p in (apogee_path, gaia_path):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    logger.info("loading APOGEE: %s", apogee_path)
    apogee = pd.read_parquet(apogee_path)
    logger.info("  %d rows × %d cols", len(apogee), len(apogee.columns))

    logger.info("loading Gaia: %s", gaia_path)
    gaia = pd.read_parquet(gaia_path)
    logger.info("  %d rows × %d cols", len(gaia), len(gaia.columns))

    assert apogee["source_id"].dtype == gaia["source_id"].dtype == "int64", (
        "source_id dtype mismatch — both must be int64"
    )

    # Drop columns that APOGEE and Gaia both carry that would collide. APOGEE
    # already has BJ21 distances (embedded upstream, identical to per_stream1)
    # and ra/dec — Gaia enrichment is authoritative for ra/dec, but APOGEE's
    # ra/dec are the pointing coords and fine to keep with no suffix if Gaia
    # doesn't clash. Confirmed: only 'source_id' overlaps between the files.
    overlap = set(apogee.columns) & set(gaia.columns) - {"source_id"}
    if overlap:
        logger.warning("unexpected column overlap: %s — using Gaia values",
                       sorted(overlap))
        apogee = apogee.drop(columns=list(overlap))

    merged = apogee.merge(gaia, on="source_id", how="inner")
    n_in = len(apogee)
    n_out = len(merged)
    logger.info(
        "inner join: %d × %d → %d rows (%d APOGEE rows dropped, no Gaia)",
        n_in, len(gaia), n_out, n_in - n_out,
    )

    _write_parquet_atomic(merged, out)
    size_mb = out.stat().st_size / 1024**2
    logger.info("wrote %s (%.1f MB, %d cols)", out, size_mb, len(merged.columns))

    prov = Provenance(
        output_file=str(out.relative_to(repo)),
        script="scripts/build_stream1_apogee_gaia.py",
        sources=[
            LocalSource(
                name="APOGEE DR19 corrected (Mészáros+2025 Teff-trend, BJ21 embedded)",
                path=str(apogee_path.relative_to(repo)),
                sha256=_sha256_of(apogee_path),
            ),
            LocalSource(
                name="Stream 1 Gaia DR3 corrected (Lindegren+2021 zpt, Riello+2021 G)",
                path=str(gaia_path.relative_to(repo)),
                sha256=_sha256_of(gaia_path),
            ),
        ],
        cuts_applied=[
            f"INNER JOIN on source_id — drop {n_in - n_out} APOGEE rows without Gaia match",
        ],
        corrections=[
            "inherited: Mészáros+2025 Teff-trend Δ[X/M]",
            "inherited: Lindegren+2021 parallax zero-point",
            "inherited: Riello+2021 G-band mag correction",
        ],
        row_count_before=n_in,
        row_count_after=n_out,
        notes=(
            "Stream 1 Pipeline-1 training pool: APOGEE DR19 labels × Gaia DR3 "
            "astrometry/photometry/GSP-Phot/GSP-Spec. BJ21 photogeometric "
            "distances already embedded in APOGEE inputs. XP coefficients "
            "still pending (task #80) — this file is the feature-matrix "
            "precursor, not the feature matrix itself. Closes task #76."
        ),
        extra={
            "apogee_rows_in": n_in,
            "gaia_rows_in": len(gaia),
            "merged_rows_out": n_out,
            "apogee_dropped_no_gaia": n_in - n_out,
        },
    )
    write_sidecar(prov)
    logger.info("wrote provenance sidecar")


if __name__ == "__main__":
    main()
