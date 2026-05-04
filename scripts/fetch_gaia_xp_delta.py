"""Phase 3a — fetch raw Gaia DR3 XP coefficients for a delta source_id list.

Delta-aware variant of ``scripts/fetch_gaia_xp.py``. Reads an arbitrary
``source_id`` Parquet (default: ``data/interim/stream3_expansion_delta_source_ids.parquet``)
and fetches the raw XP continuous-mean-spectrum coefficients from AIP via
``gaiadr3.xp_continuous_mean_spectrum`` using batched TAP UPLOAD
(5 k per batch — §6.3 mandates 5 k for XP).

Differences from the stream1∪stream3 driver
-------------------------------------------
- Accepts ``--source-id-parquet`` and ``--output-parquet`` so the same
  machinery can be reused for any future delta fetch.
- Does **not** require ``has_xp_continuous`` pre-filtering: AIP returns
  the XP rows for every source_id that has one, silently drops the rest.
  We measure and report the drop rate; stars with no XP row are later
  excluded from Ye+2024 application and from IR fetch.
- Writes to a fresh parquet (not ``xp_coeffs_raw.parquet``) so the existing
  Stream 1 ∪ Stream 3 XP artefact is untouched.
- File-based monitor: writes ``<output>.status.json`` every N chunks so
  progress can be tailed from another process without opening a PyVO session.

Halt conditions
---------------
- Wall-clock > 2.5 h.
- 3 consecutive unrecoverable chunk failures → abort (no IN-list fallback;
  AIP does not suffer the BJ21-style ceiling for XP 5k batches).
- ``data/`` footprint > 9.5 GB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.table import Table as AstropyTable
from pyvo.dal.exceptions import DALQueryError
from pyvo.dal.tap import AsyncTAPJob, TAPService
from tqdm import tqdm

from arqueogal.data.gaia_xp import XP_BATCH_SIZE, XP_QUERY_ADQL_UPLOAD, XP_TABLE
from arqueogal.data.provenance import LocalSource, Provenance, TapSource, write_sidecar
from arqueogal.data.tap import AIP_TAP_URL, aip_service

REPO = Path(__file__).resolve().parents[1]

MAX_WALLCLOCK_SEC = 2.5 * 3600
MAX_RETRIES_PER_BATCH = 3
MAX_CONSECUTIVE_FAILURES = 3
TIMEOUT_PER_JOB_SEC = 1200.0
DATA_FOOTPRINT_CEILING_GB = 16.0


def _setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("fetch_gaia_xp_delta")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)
    log.propagate = False
    return log


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


def _data_footprint_gb() -> float:
    out = subprocess.run(
        ["du", "-sb", str(REPO / "data")],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.split()[0]) / 1024**3


def _write_status(status_path: Path, payload: dict[str, Any]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = status_path.with_suffix(status_path.suffix + ".part")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, status_path)


def _fetch_chunk_upload(
    svc: TAPService,
    source_ids: list[int],
    *,
    log: logging.Logger,
    chunk_idx: int,
) -> tuple[pd.DataFrame, int]:
    """TAP-UPLOAD fetch for one XP batch, with retries."""
    retries = 0
    last_exc: Exception | None = None
    while retries <= MAX_RETRIES_PER_BATCH:
        try:
            upload = AstropyTable({"source_id": source_ids})
            job: AsyncTAPJob = svc.submit_job(
                XP_QUERY_ADQL_UPLOAD,
                uploads={"ids": upload},
                language="ADQL",
            )
            try:
                job.run()
                job.wait(
                    phases=["COMPLETED", "ERROR", "ABORTED"],
                    timeout=TIMEOUT_PER_JOB_SEC,
                )
                if job.phase != "COMPLETED":
                    try:
                        job.raise_if_error()
                    except Exception as exc:  # noqa: BLE001
                        raise RuntimeError(f"async job ended in {job.phase!r}: {exc!r}") from exc
                    raise RuntimeError(f"async job ended in {job.phase!r}")
                table = job.fetch_result().to_table()
            finally:
                try:
                    job.delete()
                except Exception as exc:  # noqa: BLE001
                    log.warning("chunk %d: failed to delete async job: %r", chunk_idx, exc)
            df = table.to_pandas()
            if "source_id" in df.columns:
                df["source_id"] = df["source_id"].astype("int64")
            return df, retries
        except (DALQueryError, RuntimeError, TimeoutError, OSError) as exc:
            last_exc = exc
            retries += 1
            wait = min(30 * retries, 120)
            log.warning(
                "chunk %d UPLOAD attempt %d failed: %r — sleeping %ds",
                chunk_idx,
                retries,
                exc,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"chunk {chunk_idx}: UPLOAD exceeded {MAX_RETRIES_PER_BATCH} retries; "
        f"last error: {last_exc!r}"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-id-parquet",
        type=Path,
        default=REPO / "data" / "interim" / "stream3_expansion_delta_source_ids.parquet",
    )
    p.add_argument(
        "--output-parquet",
        type=Path,
        default=REPO / "data" / "interim" / "xp_coeffs_raw_delta.parquet",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO / "data" / "interim" / "enrich_batches" / "xp_coeffs_delta",
    )
    p.add_argument(
        "--status-path",
        type=Path,
        default=None,
        help="File-based progress monitor. Defaults to <output>.status.json",
    )
    p.add_argument(
        "--log-path",
        type=Path,
        default=REPO / "logs" / "stream3_delta_xp_fetch_20260419.log",
    )
    return p.parse_args()


def main() -> None:  # noqa: PLR0912, PLR0915 — linear driver
    args = _parse_args()
    log = _setup_logging(args.log_path)
    if args.status_path is None:
        args.status_path = args.output_parquet.with_suffix(".status.json")
    t0 = time.time()

    log.info("=== Phase 3a XP delta fetch — start ===")
    log.info("input: %s", args.source_id_parquet)
    log.info("output: %s", args.output_parquet)
    log.info("checkpoints: %s", args.checkpoint_dir)

    ids_df = pd.read_parquet(args.source_id_parquet, columns=["source_id"])
    source_ids = ids_df["source_id"].astype("int64").drop_duplicates().sort_values().to_list()
    n_src = len(source_ids)
    batch_size = XP_BATCH_SIZE
    n_batches = (n_src + batch_size - 1) // batch_size
    log.info(
        "%d unique source_ids → %d batches of %d (AIP TAP UPLOAD)",
        n_src,
        n_batches,
        batch_size,
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    svc = aip_service()

    chunk_wallclocks: list[float] = []
    chunk_retries_total = 0
    chunk_failure_docs: list[str] = []
    fetched_frames: list[pd.DataFrame] = []
    consecutive_failures = 0
    halt_reason: str | None = None
    n_chunks_incomplete = 0

    pbar = tqdm(total=n_batches, desc="XP delta", unit="chunk")

    _write_status(
        args.status_path,
        {
            "state": "running",
            "n_src": n_src,
            "n_batches": n_batches,
            "batch_size": batch_size,
            "completed_batches": 0,
            "started_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    )

    for idx in range(n_batches):
        wall_elapsed = time.time() - t0
        if wall_elapsed > MAX_WALLCLOCK_SEC:
            halt_reason = (
                f"wall-clock {wall_elapsed / 3600:.2f} h exceeds {MAX_WALLCLOCK_SEC / 3600:.1f} h"
            )
            log.error("HALT: %s", halt_reason)
            break

        footprint = _data_footprint_gb()
        if footprint > DATA_FOOTPRINT_CEILING_GB:
            halt_reason = (
                f"data/ footprint {footprint:.2f} GB exceeds {DATA_FOOTPRINT_CEILING_GB} GB ceiling"
            )
            log.error("HALT: %s", halt_reason)
            break

        chunk = source_ids[idx * batch_size : (idx + 1) * batch_size]
        chunk_file = args.checkpoint_dir / f"xp_{idx:06d}.parquet"
        if chunk_file.is_file():
            try:
                frame = pd.read_parquet(chunk_file)
                fetched_frames.append(frame)
                pbar.update(1)
                consecutive_failures = 0
                continue
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "chunk %d: corrupt ckpt (%r) — refetching",
                    idx,
                    exc,
                )
                chunk_file.unlink(missing_ok=True)

        t_batch = time.time()
        try:
            df_batch, retries = _fetch_chunk_upload(svc, chunk, log=log, chunk_idx=idx)
        except Exception as exc:  # noqa: BLE001
            consecutive_failures += 1
            msg = f"chunk {idx}: unrecoverable after retries — {exc!r}"
            chunk_failure_docs.append(msg)
            log.error(msg)
            n_chunks_incomplete += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                halt_reason = f"{consecutive_failures} consecutive chunk failures — aborting"
                log.error("HALT: %s", halt_reason)
                break
            pbar.update(1)
            continue

        elapsed = time.time() - t_batch
        chunk_wallclocks.append(elapsed)
        chunk_retries_total += retries
        consecutive_failures = 0

        n_in = len(chunk)
        n_out = len(df_batch)
        miss_frac = 1.0 - (n_out / n_in if n_in else 1.0)
        if miss_frac > 0.10:
            chunk_failure_docs.append(
                f"SOFT: chunk {idx}: {n_out}/{n_in} rows ({miss_frac:.2%} missing XP)"
            )

        _write_parquet_atomic(df_batch, chunk_file)
        fetched_frames.append(df_batch)

        pbar.set_postfix(
            rows=n_out,
            s=f"{elapsed:.1f}",
            retries=retries,
        )
        pbar.update(1)
        log.info(
            "chunk %d/%d: %d rows in %.1f s (retries=%d)",
            idx + 1,
            n_batches,
            n_out,
            elapsed,
            retries,
        )

        # Status snapshot every 10 chunks (or last).
        if (idx + 1) % 10 == 0 or idx == n_batches - 1:
            _write_status(
                args.status_path,
                {
                    "state": "running",
                    "completed_batches": idx + 1,
                    "n_batches": n_batches,
                    "total_rows_so_far": int(sum(len(f) for f in fetched_frames)),
                    "elapsed_s": time.time() - t0,
                    "consecutive_failures": consecutive_failures,
                    "halt_reason": halt_reason,
                },
            )

    pbar.close()

    # Concat + atomic write.
    log.info("concatenating %d fetched frames", len(fetched_frames))
    if fetched_frames:
        df_full = pd.concat(fetched_frames, ignore_index=True)
    else:
        df_full = pd.DataFrame()
    log.info("final: %d XP rows (input source_ids: %d)", len(df_full), n_src)

    _write_parquet_atomic(df_full, args.output_parquet)
    size_mb = args.output_parquet.stat().st_size / 1024**2
    out_sha = _sha256_of(args.output_parquet)
    log.info(
        "wrote %s (%.1f MB, %d cols, sha256=%s…)",
        args.output_parquet,
        size_mb,
        len(df_full.columns),
        out_sha[:12],
    )

    # Stars returned vs requested.
    returned_ids = set(df_full["source_id"].tolist()) if "source_id" in df_full.columns else set()
    n_missing_xp = n_src - len(returned_ids)
    log.info("stars with no XP row (dropped by AIP): %d / %d", n_missing_xp, n_src)

    total_wall = time.time() - t0

    prov = Provenance(
        output_file=str(args.output_parquet.relative_to(REPO)),
        script="scripts/fetch_gaia_xp_delta.py",
        sources=[
            LocalSource(
                name="Phase 3a delta source_ids (Stream 3 expansion)",
                path=str(args.source_id_parquet.relative_to(REPO)),
                sha256=_sha256_of(args.source_id_parquet),
            ),
            TapSource(
                name=f"AIP {XP_TABLE} (UPLOAD)",
                endpoint=AIP_TAP_URL,
                query=XP_QUERY_ADQL_UPLOAD,
                n_batches=n_batches,
                batch_size=batch_size,
            ),
        ],
        cuts_applied=[
            "source_ids = Phase 3a union \\ existing Stream 3 (168,099)",
        ],
        corrections=[
            "coefficient_correlations dropped (data_acquisition.md §6.3 — "
            "5 GB → 10 GB budget still excludes full covariances)",
        ],
        row_count_before=n_src,
        row_count_after=int(len(df_full)),
        notes=(
            "RAW XP coefficients for the Phase 3a delta. Downstream §6.4 "
            "step 1 (Ye+2024 NN flux-correction) is applied by "
            "scripts/apply_ye2024_xp_delta.py; later steps (Hermite re-proj, "
            "c0 normalisation, z-score, float32) by the Pipeline-1 feature "
            "builder."
        ),
        extra={
            "batch_size": batch_size,
            "n_batches_planned": n_batches,
            "n_batches_completed": len(chunk_wallclocks),
            "n_chunks_incomplete": n_chunks_incomplete,
            "chunk_retries_total": chunk_retries_total,
            "wallclock_seconds": total_wall,
            "chunk_wallclock_mean_s": (
                float(np.mean(chunk_wallclocks)) if chunk_wallclocks else None
            ),
            "chunk_wallclock_median_s": (
                float(np.median(chunk_wallclocks)) if chunk_wallclocks else None
            ),
            "n_source_ids_requested": n_src,
            "n_source_ids_with_xp_row": len(returned_ids),
            "n_source_ids_missing_xp": n_missing_xp,
            "missing_xp_fraction": n_missing_xp / max(n_src, 1),
            "chunk_failure_docs": chunk_failure_docs,
            "output_sha256": out_sha,
            "halt_reason": halt_reason,
            "halt_conditions_tripped": {
                "wallclock_exceeded": total_wall > MAX_WALLCLOCK_SEC,
                "footprint_exceeded": _data_footprint_gb() > DATA_FOOTPRINT_CEILING_GB,
                "consecutive_failures": consecutive_failures,
            },
        },
    )
    write_sidecar(prov)

    _write_status(
        args.status_path,
        {
            "state": "done" if halt_reason is None else f"halted: {halt_reason}",
            "completed_batches": len(chunk_wallclocks),
            "n_batches": n_batches,
            "total_rows": int(len(df_full)),
            "wallclock_s": total_wall,
            "output": str(args.output_parquet.relative_to(REPO)),
            "output_sha256": out_sha,
            "missing_xp": n_missing_xp,
            "halt_reason": halt_reason,
            "finished_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    )
    log.info("=== done in %.1f min (halt_reason=%s) ===", total_wall / 60, halt_reason)


if __name__ == "__main__":
    main()
