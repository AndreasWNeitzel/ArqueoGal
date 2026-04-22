"""Fetch 2MASS + AllWISE infrared photometry for a source_id list.

Runs the ArqueoGal IR cross-match via TAP UPLOAD against the Gaia-hosted
best-neighbour tables (``gaiadr3.tmass_psc_xsc_best_neighbour``,
``gaiadr3.allwise_best_neighbour``). Default service is the ESA Gaia
Archive (anonymous access; TAP UPLOAD works fine for ≤ 200 k ids at
10 k/batch). AIP is available as a fallback.

Typical use — fetch IR for the existing Stream 3 Ye-OK subset::

    python scripts/fetch_ir_photometry.py \\
        --source-id-parquet data/interim/stream3_ye_ok_source_ids.parquet \\
        --output-parquet data/raw/ir_photometry/stream3_existing_ir.parquet

Outputs
-------
- ``<output-parquet>``: one row per input source_id with joined 2MASS PSC
  and AllWISE columns plus ``ir_missing_flag`` (True iff any of J/H/K/W1/W2
  is missing). Float32 magnitudes; nullable Int8 quality flags.
- ``<output-parquet>.part`` written atomically, then renamed.
- ``<output>.provenance.json`` sidecar recording TAP endpoint, joined
  tables, query text, row counts, per-catalogue missing-counterpart counts,
  SHA-256, git SHA, timestamp.
- Per-chunk checkpoints at
  ``data/interim/enrich_batches/ir/{tmass,allwise}/batch_NNNN.parquet``
  so reruns after a crash resume without refetching completed chunks.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from pathlib import Path
from typing import Final

import pandas as pd
import polars as pl

from arqueogal.data.ir_photometry import (
    ALLWISE_ADQL_UPLOAD_AIP,
    ALLWISE_ADQL_UPLOAD_ESA,
    TMASS_ADQL_UPLOAD_AIP,
    TMASS_ADQL_UPLOAD_ESA,
    assemble_ir_photometry,
)
from arqueogal.data.provenance import LocalSource, Provenance, TapSource, write_sidecar
from arqueogal.data.tap import AIP_TAP_URL, ESA_TAP_URL, aip_service, esa_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fetch_ir_photometry")

DEFAULT_BATCH_SIZE: Final[int] = 10_000


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_parquet_atomic(pdf: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    pdf.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-id-parquet",
        required=True,
        type=Path,
        help="Parquet with a 'source_id' column (Gaia DR3 int64).",
    )
    p.add_argument(
        "--output-parquet",
        required=True,
        type=Path,
        help="Destination Parquet file. Sidecar provenance written alongside.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="TAP UPLOAD chunk size per query (default 10 000).",
    )
    p.add_argument(
        "--service",
        choices=("gaia_esa", "aip"),
        default="gaia_esa",
        help=(
            "TAP service. 'gaia_esa' = ESA Gaia archive (anonymous ok for "
            "small jobs, default). 'aip' = Gaia@AIP mirror (needs "
            "GAIA_AIP_TOKEN or credentials.yaml)."
        ),
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Checkpoint root. Defaults to data/interim/enrich_batches/ir/ "
            "under the repo root. 2MASS batches go in <root>/tmass/, "
            "AllWISE in <root>/allwise/."
        ),
    )
    p.add_argument(
        "--queue",
        default=None,
        help="Optional TAP queue name. Only used by AIP.",
    )
    p.add_argument(
        "--runid",
        default="arqueogal-ir-fetch",
        help="UWS runid label (echoed in AIP's job listing).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    repo = Path(__file__).resolve().parents[1]

    if args.checkpoint_dir is None:
        args.checkpoint_dir = repo / "data" / "interim" / "enrich_batches" / "ir"

    if not args.source_id_parquet.is_file():
        raise SystemExit(f"source-id parquet not found: {args.source_id_parquet}")

    logger.info("loading source_ids from %s", args.source_id_parquet)
    ids_df = pd.read_parquet(args.source_id_parquet, columns=["source_id"])
    source_ids = ids_df["source_id"].astype("int64").drop_duplicates().to_list()
    n_src = len(source_ids)
    n_batches = (n_src + args.chunk_size - 1) // args.chunk_size
    logger.info(
        "%d unique source_ids → IR cross-match (chunks of %d → %d batches per catalogue)",
        n_src,
        args.chunk_size,
        n_batches,
    )

    if args.service == "gaia_esa":
        svc = esa_service()
        endpoint = ESA_TAP_URL
        service_flavor = "esa"
        tmass_adql = TMASS_ADQL_UPLOAD_ESA
        allwise_adql = ALLWISE_ADQL_UPLOAD_ESA
    else:
        svc = aip_service()
        endpoint = AIP_TAP_URL
        service_flavor = "aip"
        tmass_adql = TMASS_ADQL_UPLOAD_AIP
        allwise_adql = ALLWISE_ADQL_UPLOAD_AIP
    logger.info("TAP service: %s (flavour=%s)", endpoint, service_flavor)

    combined = assemble_ir_photometry(
        svc,
        source_ids,
        batch_size=args.chunk_size,
        checkpoint_dir=args.checkpoint_dir,
        queue=args.queue,
        runid=args.runid,
        service_flavor=service_flavor,
    )

    n_out = combined.height
    logger.info("received %d rows from IR cross-match (expected %d)", n_out, n_src)
    if n_out != n_src:
        logger.warning(
            "row count mismatch: input=%d output=%d (should never happen — "
            "assemble_ir_photometry reindexes to the input id list)",
            n_src,
            n_out,
        )

    # Counterpart stats for the report.
    n_tmass_match = int(combined.select(pl.col("j_mag").is_not_null().sum()).item())
    n_allwise_match = int(combined.select(pl.col("w1_mag").is_not_null().sum()).item())
    n_ir_complete = int(
        combined.select((~pl.col("ir_missing_flag")).sum()).item()
    )
    n_ir_missing = int(combined.select(pl.col("ir_missing_flag").sum()).item())
    logger.info(
        "counterpart rates: 2MASS=%.3f%% AllWISE=%.3f%% IR-complete=%.3f%%",
        100.0 * n_tmass_match / max(n_src, 1),
        100.0 * n_allwise_match / max(n_src, 1),
        100.0 * n_ir_complete / max(n_src, 1),
    )

    # Atomic write as Parquet (pandas path — pyarrow handles polars natively,
    # but going through pandas matches the other fetch scripts' idiom and
    # keeps the Parquet schema identical to what downstream pandas readers
    # expect).
    out_pdf = combined.to_pandas()
    _write_parquet_atomic(out_pdf, args.output_parquet)
    size_mb = args.output_parquet.stat().st_size / 1024**2
    logger.info(
        "wrote %s (%.2f MB, %d rows × %d cols)",
        args.output_parquet,
        size_mb,
        len(out_pdf),
        len(out_pdf.columns),
    )

    ids_sha = _sha256_of(args.source_id_parquet)
    out_sha = _sha256_of(args.output_parquet)

    tmass_join = (
        "gaiadr1.tmass_original_valid"
        if service_flavor == "esa"
        else "catalogs.tmass"
    )
    allwise_join = (
        "gaiadr1.allwise_original_valid"
        if service_flavor == "esa"
        else "catalogs.allwise"
    )
    tap_sources = [
        TapSource(
            name=(
                "Gaia best-neighbour 2MASS PSC ⨝ "
                "gaiadr3.tmass_psc_xsc_best_neighbour ⨝ "
                f"{tmass_join} "
                "(joined on designation = bn.original_ext_source_id, "
                "bypassing the broken-on-ESA gaiadr3.tmass_psc_xsc_join "
                "middleman)"
            ),
            endpoint=endpoint,
            query=tmass_adql,
            n_batches=n_batches,
            batch_size=args.chunk_size,
        ),
        TapSource(
            name=(
                "Gaia best-neighbour AllWISE ⨝ "
                f"gaiadr3.allwise_best_neighbour ⨝ {allwise_join} "
                "(joined on designation = bn.original_ext_source_id; "
                "allwise_oid join deterministically 500s on ESA TAP)"
            ),
            endpoint=endpoint,
            query=allwise_adql,
            n_batches=n_batches,
            batch_size=args.chunk_size,
        ),
    ]

    # Path.relative_to requires an absolute path on both sides — argparse
    # gives us whatever the user typed, which may be a repo-relative string.
    # Resolve both before diffing, then fall back to the absolute path if
    # the output somehow lives outside the repo.
    def _rel_or_abs(p: Path) -> str:
        abs_p = p.resolve()
        try:
            return str(abs_p.relative_to(repo))
        except ValueError:
            return str(abs_p)

    prov = Provenance(
        output_file=_rel_or_abs(args.output_parquet),
        script="scripts/fetch_ir_photometry.py",
        sources=[
            LocalSource(
                name="input source_id list (Gaia DR3 int64)",
                path=_rel_or_abs(args.source_id_parquet),
                sha256=ids_sha,
            ),
            *tap_sources,
        ],
        cuts_applied=[],
        corrections=[
            "float64 → float32 on magnitude and angular-distance columns",
            "pd.Int8 nullable dtype for xm_quality_flag columns",
            (
                "LEFT JOIN from uploaded source_ids: stars without 2MASS or "
                "AllWISE neighbour retained with NaN magnitudes and "
                "ir_missing_flag=True"
            ),
        ],
        row_count_before=n_src,
        row_count_after=int(n_out),
        notes=(
            "2MASS + AllWISE infrared photometry cross-match via the Gaia-"
            "hosted best-neighbour tables. Downstream Pipeline 1 inference "
            "uses j/h/k/w1/w2 as auxiliary features alongside the XP "
            "coefficients; zero-imputation diagnostics showed all five "
            "labels degrade 28-130% RMSE without IR, so per-star IR is "
            "non-negotiable. Stars with ir_missing_flag=True should be "
            "dropped (or handled via the missingness branch) before "
            "inference; NaN in the IR columns crashes the adapter."
        ),
        extra={
            "batch_size": args.chunk_size,
            "n_batches_per_catalogue": n_batches,
            "tap_service": args.service,
            "tap_endpoint": endpoint,
            "n_tmass_matches": n_tmass_match,
            "n_allwise_matches": n_allwise_match,
            "n_ir_complete": n_ir_complete,
            "n_ir_missing_any": n_ir_missing,
            "tmass_counterpart_rate": float(n_tmass_match) / max(n_src, 1),
            "allwise_counterpart_rate": float(n_allwise_match) / max(n_src, 1),
            "ir_complete_rate": float(n_ir_complete) / max(n_src, 1),
            "output_sha256": out_sha,
        },
    )
    write_sidecar(prov)
    logger.info("wrote provenance sidecar")


if __name__ == "__main__":
    main()
