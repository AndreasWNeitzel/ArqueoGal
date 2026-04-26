"""Fetch raw Gaia DR3 XP continuous-mean-spectrum coefficients for the
Stream 1 + Stream 2 + Stream 3 union of has_xp_continuous sources.

Uses AIP's ``gaiadr3.xp_continuous_mean_spectrum`` TAP table via UPLOAD —
astroquery DataLink (XP_CONTINUOUS retrieval_type) works but is slower and
does not chunk gracefully. UPLOAD TAP is faster, resumable, and consistent
with the rest of the Stream 1/2/3 ingestion pattern.

Output columns (per data_acquisition.md §6.3):
- ``source_id``
- ``bp_coefficients``, ``bp_coefficient_errors``  (55-element arrays)
- ``rp_coefficients``, ``rp_coefficient_errors``  (55-element arrays)
- ``bp_standard_deviation``, ``rp_standard_deviation``
- ``bp_n_measurements``, ``rp_n_measurements``
- ``bp_n_relevant_bases``, ``rp_n_relevant_bases``

``coefficient_correlations`` are **not** fetched — per §6.3 that full
55×55 covariance per band is too heavy for the 5 GB budget. Errors are
retained as the diagonal; provenance records the drop.

IMPORTANT: this script only fetches RAW XP. The §6.4 preprocessing
sequence (Ye+2024 NN flux-correction → normalise by c_0 → log+zscore c_0)
is applied by downstream scripts — primarily ``scripts/apply_ye2024_xp.py``
for step 1, then ``scripts/build_pipeline1_features_stream1.py`` for
steps 2–4. Pipeline-1 feature-matrix code must apply the full sequence
before training.

Expected wall-time: ~92 batches × ~15 s/batch ≈ 25 min.

Output
------
``data/interim/xp_coeffs_raw.parquet`` + provenance sidecar.
Est. size: 400–600 MB.
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
logger = logging.getLogger("fetch_gaia_xp")


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
    s1_path = repo / "data" / "interim" / "stream1_gaia_dr3_corrected.parquet"
    s2_path = repo / "data" / "interim" / "stream2_tess_gaia.parquet"
    s3_path = repo / "data" / "interim" / "stream3_gaia_dr3_corrected.parquet"
    ids_out = repo / "data" / "interim" / "xp_source_ids.parquet"
    xp_out = repo / "data" / "interim" / "xp_coeffs_raw.parquet"
    ckpt = repo / "data" / "interim" / "enrich_batches" / "xp_coeffs"

    for p in (s1_path, s3_path):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    logger.info("loading has_xp_continuous source_ids from Stream 1 + Stream 2 + Stream 3")
    s1 = pd.read_parquet(s1_path, columns=["source_id", "has_xp_continuous"])
    s3 = pd.read_parquet(s3_path, columns=["source_id", "has_xp_continuous"])
    s1_xp = s1.loc[s1["has_xp_continuous"], "source_id"]
    s3_xp = s3.loc[s3["has_xp_continuous"], "source_id"]
    if s2_path.exists():
        s2 = pd.read_parquet(s2_path, columns=["source_id", "has_xp_continuous"])
        s2_xp = s2.loc[s2["has_xp_continuous"], "source_id"]
    else:
        logger.warning("Stream 2 interim parquet missing at %s — skipping", s2_path)
        s2_xp = pd.Series([], dtype="int64", name="source_id")

    # Union, dedupe, preserve int64.
    union = (
        pd.concat([s1_xp, s2_xp, s3_xp], ignore_index=True)
        .drop_duplicates()
        .astype("int64")
        .sort_values()
        .reset_index(drop=True)
    )
    n_src = len(union)
    n_batches = (n_src + XP_BATCH_SIZE - 1) // XP_BATCH_SIZE
    logger.info(
        "Stream 1 has_xp: %d, Stream 2 has_xp: %d, Stream 3 has_xp: %d, "
        "union: %d (dedup by %d)",
        len(s1_xp),
        len(s2_xp),
        len(s3_xp),
        n_src,
        len(s1_xp) + len(s2_xp) + len(s3_xp) - n_src,
    )
    logger.info("XP fetch: %d batches of %d via AIP UPLOAD", n_batches, XP_BATCH_SIZE)

    # Persist the source_id list so provenance + rerunnable downstream prep can
    # verify which stars were targeted.
    _write_parquet_atomic(union.to_frame("source_id"), ids_out)
    logger.info("wrote source_id index: %s", ids_out)

    svc = aip_service()
    df = batched_upload_fetch_df(
        svc,
        union.to_list(),
        XP_QUERY_ADQL_UPLOAD,
        upload_name="ids",
        batch_size=XP_BATCH_SIZE,
        checkpoint_dir=ckpt,
        checkpoint_prefix="xp",
        queue="2h",
        runid="arqueogal-xp-fetch",
    )
    logger.info("fetched %d XP rows", len(df))

    _write_parquet_atomic(df, xp_out)
    size_mb = xp_out.stat().st_size / 1024**2
    logger.info("wrote %s (%.1f MB, %d cols)", xp_out, size_mb, len(df.columns))

    sources = [
        LocalSource(
            name="Stream 1 Gaia DR3 corrected",
            path=str(s1_path.relative_to(repo)),
            sha256=_sha256_of(s1_path),
        ),
        LocalSource(
            name="Stream 3 Gaia DR3 corrected",
            path=str(s3_path.relative_to(repo)),
            sha256=_sha256_of(s3_path),
        ),
        LocalSource(
            name="XP target source_id union",
            path=str(ids_out.relative_to(repo)),
            sha256=_sha256_of(ids_out),
        ),
        TapSource(
            name=f"AIP {XP_TABLE} (UPLOAD)",
            endpoint=AIP_TAP_URL,
            query=XP_QUERY_ADQL_UPLOAD,
            n_batches=n_batches,
            batch_size=XP_BATCH_SIZE,
        ),
    ]
    if s2_path.exists():
        sources.insert(
            1,
            LocalSource(
                name="Stream 2 TESS × Gaia DR3 corrected",
                path=str(s2_path.relative_to(repo)),
                sha256=_sha256_of(s2_path),
            ),
        )

    prov = Provenance(
        output_file=str(xp_out.relative_to(repo)),
        script="scripts/fetch_gaia_xp.py",
        sources=sources,
        cuts_applied=[
            "has_xp_continuous == True (Stream 1 ∪ Stream 2 ∪ Stream 3)",
        ],
        corrections=[
            "coefficient_correlations dropped (data_acquisition.md §6.3 — "
            "full covariances too heavy for 5 GB budget)",
        ],
        row_count_before=n_src,
        row_count_after=int(len(df)),
        notes=(
            "RAW XP coefficients. Downstream §6.4 preprocessing is applied "
            "by separate scripts: (1) Ye+2024 NN flux-correction via "
            "scripts/apply_ye2024_xp.py (live, uses vendored NN weights); "
            "(2) normalise c[1:] by c[0]; (3) log10 + z-score c[0]; "
            "(4) downcast to float32 — all three performed by "
            "scripts/build_pipeline1_features_stream{1,2,3}.py."
        ),
        extra={
            "batch_size": XP_BATCH_SIZE,
            "n_stream1_xp": int(len(s1_xp)),
            "n_stream2_xp": int(len(s2_xp)),
            "n_stream3_xp": int(len(s3_xp)),
            "n_union": n_src,
        },
    )
    write_sidecar(prov)


if __name__ == "__main__":
    main()
