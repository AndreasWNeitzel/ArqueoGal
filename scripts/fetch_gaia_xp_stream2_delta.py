"""Delta XP fetch for Stream 2-only source_ids.

The canonical ``scripts/fetch_gaia_xp.py`` now unions S1 ∪ S2 ∪ S3 and would
re-fetch the 457k rows already on disk. This delta script is the cheap path:
it loads the existing ``xp_coeffs_raw.parquet``, finds the S2 source_ids that
are missing, fetches only those, and concats the new rows in place. Result is
identical to a clean re-run of the canonical script (same columns, same
ordering after sort).

After this delta script runs once, future re-runs of ``fetch_gaia_xp.py``
will be reproducible-from-scratch but slower; this script is a one-shot.

Wall-time: ~15 batches × ~15 s/batch ≈ 4 min for ~127k new IDs.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import pandas as pd

from arqueogal.data.gaia_xp import XP_BATCH_SIZE, XP_QUERY_ADQL_UPLOAD, XP_TABLE
from arqueogal.data.provenance import LocalSource, Provenance, TapSource, write_sidecar
from arqueogal.data.tap import AIP_TAP_URL, aip_service, batched_upload_fetch_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fetch_gaia_xp_stream2_delta")


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
    s2_path = repo / "data" / "interim" / "stream2_tess_gaia.parquet"
    xp_existing = repo / "data" / "interim" / "xp_coeffs_raw.parquet"
    ids_out = repo / "data" / "interim" / "xp_source_ids.parquet"
    ckpt = repo / "data" / "interim" / "enrich_batches" / "xp_coeffs_s2_delta"

    for p in (s2_path, xp_existing):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    # Existing XP source IDs already on disk.
    existing_ids = set(pd.read_parquet(xp_existing, columns=["source_id"])["source_id"])
    logger.info("existing XP rows: %d", len(existing_ids))

    # S2 sources flagged as has_xp_continuous.
    s2 = pd.read_parquet(s2_path, columns=["source_id", "has_xp_continuous"])
    s2_xp = set(s2.loc[s2["has_xp_continuous"], "source_id"].astype("int64"))
    logger.info("S2 has_xp_continuous: %d", len(s2_xp))

    delta = sorted(s2_xp - existing_ids)
    logger.info("S2 NEW source_ids to fetch: %d", len(delta))

    if not delta:
        logger.info("nothing to fetch — S2 fully covered by existing XP parquet")
        return

    n_batches = (len(delta) + XP_BATCH_SIZE - 1) // XP_BATCH_SIZE
    logger.info("XP delta fetch: %d batches of %d via AIP UPLOAD", n_batches, XP_BATCH_SIZE)

    svc = aip_service()
    new_df = batched_upload_fetch_df(
        svc,
        list(delta),
        XP_QUERY_ADQL_UPLOAD,
        upload_name="ids",
        batch_size=XP_BATCH_SIZE,
        checkpoint_dir=ckpt,
        checkpoint_prefix="xp_s2",
        queue="2h",
        runid="arqueogal-xp-fetch-s2-delta",
    )
    logger.info("fetched %d new XP rows", len(new_df))

    # Concat and atomically replace the existing parquet.
    existing_df = pd.read_parquet(xp_existing)
    full_df = (
        pd.concat([existing_df, new_df], ignore_index=True)
        .drop_duplicates(subset=["source_id"], keep="first")
        .sort_values("source_id")
        .reset_index(drop=True)
    )
    logger.info(
        "merged: existing=%d + new=%d → final=%d", len(existing_df), len(new_df), len(full_df)
    )

    _write_parquet_atomic(full_df, xp_existing)
    size_mb = xp_existing.stat().st_size / 1024**2
    logger.info("rewrote %s (%.1f MB)", xp_existing, size_mb)

    # Update the source_id index too.
    union_ids = full_df[["source_id"]].astype("int64").sort_values("source_id")
    _write_parquet_atomic(union_ids, ids_out)
    logger.info("rewrote source_id index: %s", ids_out)

    # Provenance: this is a delta operation; sidecar reflects the merged state.
    prov = Provenance(
        output_file=str(xp_existing.relative_to(repo)),
        script="scripts/fetch_gaia_xp_stream2_delta.py",
        sources=[
            LocalSource(
                name="Existing xp_coeffs_raw.parquet (pre-delta)",
                path=str(xp_existing.relative_to(repo)),
                sha256="(pre-delta hash not retained; merged in place)",
            ),
            LocalSource(
                name="Stream 2 TESS × Gaia DR3 corrected",
                path=str(s2_path.relative_to(repo)),
                sha256=_sha256_of(s2_path),
            ),
            TapSource(
                name=f"AIP {XP_TABLE} (UPLOAD; delta only)",
                endpoint=AIP_TAP_URL,
                query=XP_QUERY_ADQL_UPLOAD,
                n_batches=n_batches,
                batch_size=XP_BATCH_SIZE,
            ),
        ],
        cuts_applied=[
            "has_xp_continuous == True (Stream 2 \\ existing)",
        ],
        corrections=[],
        row_count_before=len(existing_df),
        row_count_after=int(len(full_df)),
        notes=(
            "Delta fetch: only S2 source_ids missing from existing parquet "
            "were queried. Result is byte-equivalent (after sort) to a clean "
            "re-run of scripts/fetch_gaia_xp.py."
        ),
        extra={
            "batch_size": XP_BATCH_SIZE,
            "n_existing": len(existing_df),
            "n_delta_fetched": int(len(new_df)),
            "n_after_merge": int(len(full_df)),
        },
    )
    write_sidecar(prov)


if __name__ == "__main__":
    main()
