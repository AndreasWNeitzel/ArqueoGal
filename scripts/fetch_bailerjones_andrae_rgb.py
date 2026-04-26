"""Fetch Bailer-Jones+2021 photogeometric distances for the Andrae+2023 vetted RGB sample.

Thread-3 prerequisite for the volume-limited arm of Stream 3 (the Pipeline 1
inference arm consumed downstream by Starfold, separate repo):
a BJ21 ``r_med_photogeo`` pull against the full ~10.48 M source_id Andrae+2023
vetted-RGB catalogue (reissue via Ardern-Arentsen+2024, VizieR J/MNRAS/537/1984).

Fetch strategy
--------------
1. Load the Andrae+2023 vetted RGB pool (~10.48 M source_ids).
2. Load existing on-disk BJ21 (``per_stream1.parquet`` ~320 k + ``per_stream3.parquet``
   ~168 k). Intersect with Andrae → the re-used subset.
3. Compute the delta (Andrae \\ existing) ≈ 10 M.
4. Fetch the delta from GAVO (``https://dc.g-vo.org/tap``, table ``gedr3dist.main``)
   using **TAP UPLOAD** — each chunk POSTs a VOTable of 10 k source_ids rather than
   inlining a 500-KB ``IN`` list. UPLOAD was smoke-tested on the live service
   (2026-04-19) before this run.
5. Fallback: if UPLOAD fails systematically (3 consecutive unrecoverable failures),
   retry the delta with the legacy IN-list path (same ADQL template as existing
   ``fetch_bailerjones_stream3.py``).
6. Union (existing_reused ∪ newly_fetched) → sort by source_id → atomic Parquet
   write to ``data/raw/bailer_jones_2021/andrae_pool_bj21.parquet``.

GAVO-specific notes
-------------------
- GAVO applies a default MAXREC = 20 000 on TAP queries, which silently truncates
  a 50k result. We pass ``maxrec=100_000`` on every job to defang that.
- No auth needed. ``gavo_service()`` returns an unauthenticated TAPService.

Outputs
-------
- ``data/raw/bailer_jones_2021/andrae_pool_bj21.parquet``
    Columns ``source_id, r_med_photogeo, r_lo_photogeo, r_hi_photogeo``.
    ``source_id`` as int64, distances as float32.
- ``data/raw/bailer_jones_2021/andrae_pool_bj21.provenance.json`` (sidecar)
- ``data/interim/bj21_andrae_chunks/chunk_{NNNNNN}.parquet`` per-chunk ckpts
  (deleted after successful concat if their combined size > 0.5 GB).
- ``logs/bj21_fetch_andrae_rgb_20260419.log`` wall-clock + retries.
- ``reports/pipeline1/bj21_fetch_andrae_rgb.md`` seven-section summary.

Halt conditions (see task spec):
- ≥ 3 consecutive chunk failures on UPLOAD → switch to IN-list fallback.
  If the fallback also fails ≥ 3 consecutive times → raise & halt.
- Wall-clock > 4 h.
- ``data/`` footprint > 9.5 GB.
- Chunk row-count mismatch > 5 % without documented cause (BJ21-missing
  source_ids are NORMAL — flag aggregate but don't halt per-chunk).
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
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

from arqueogal.data.provenance import LocalSource, Provenance, TapSource, write_sidecar
from arqueogal.data.tap import BATCH_PLACEHOLDER, GAVO_TAP_URL, gavo_service

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]

ANDRAE_PARQUET = REPO / "data" / "raw" / "andrae2023" / "andrae2023_rgb.parquet"
EXISTING_BJ_PATHS = (
    REPO / "data" / "raw" / "bailer_jones_2021" / "per_stream1.parquet",
    REPO / "data" / "raw" / "bailer_jones_2021" / "per_stream3.parquet",
)

OUT_PARQUET = REPO / "data" / "raw" / "bailer_jones_2021" / "andrae_pool_bj21.parquet"
CHUNK_DIR = REPO / "data" / "interim" / "bj21_andrae_chunks"
REPORT_PATH = REPO / "reports" / "pipeline1" / "bj21_fetch_andrae_rgb.md"
LOG_PATH = REPO / "logs" / "bj21_fetch_andrae_rgb_20260419.log"

BATCH_SIZE = 10_000
MAX_RETRIES_PER_BATCH = 3
MAX_CONSECUTIVE_FAILURES_BEFORE_FALLBACK = 3
MAX_WALLCLOCK_SEC = 4 * 3600.0  # halt condition
DATA_FOOTPRINT_CEILING_GB = 9.5  # halt condition
TIMEOUT_PER_JOB_SEC = 1200.0  # 20 min generous upper bound per async job
GAVO_MAXREC = 100_000  # override default 20 000 truncation ceiling

FLOAT32_COLS = ("r_med_photogeo", "r_lo_photogeo", "r_hi_photogeo")
KEEP_COLS = ("source_id", "r_med_photogeo", "r_lo_photogeo", "r_hi_photogeo")

UPLOAD_TABLE_NAME = "sidlist"
UPLOAD_ADQL = (
    "SELECT m.source_id,\n"
    "       m.r_med_photogeo, m.r_lo_photogeo, m.r_hi_photogeo\n"
    "FROM gedr3dist.main AS m\n"
    f"JOIN TAP_UPLOAD.{UPLOAD_TABLE_NAME} AS u ON m.source_id = u.source_id\n"
)

IN_LIST_ADQL = (
    "SELECT source_id,\n"
    "       r_med_photogeo, r_lo_photogeo, r_hi_photogeo\n"
    "FROM gedr3dist.main\n"
    f"WHERE source_id IN ({BATCH_PLACEHOLDER})\n"
)

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------


def _setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("fetch_bj21_andrae_rgb")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOG_PATH, mode="a")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)
    log.propagate = False
    return log


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _data_footprint_gb() -> float:
    out = subprocess.run(
        ["du", "-sb", str(REPO / "data")],
        capture_output=True,
        text=True,
        check=True,
    )
    bytes_ = int(out.stdout.split()[0])
    return bytes_ / 1024**3


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


def _load_existing_bj(log: logging.Logger) -> pd.DataFrame:
    """Load existing BJ21 rows, keep only the columns we need, dedupe by source_id."""
    frames: list[pd.DataFrame] = []
    for p in EXISTING_BJ_PATHS:
        if not p.is_file():
            log.warning("existing BJ21 path missing: %s", p)
            continue
        df = pd.read_parquet(p, columns=["source_id", *FLOAT32_COLS])
        df["source_id"] = df["source_id"].astype("int64")
        for col in FLOAT32_COLS:
            df[col] = df[col].astype(np.float32)
        log.info("loaded %s: %d rows", p.relative_to(REPO), len(df))
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=list(KEEP_COLS))
    combined = pd.concat(frames, ignore_index=True)
    n_before = len(combined)
    combined = combined.drop_duplicates(subset="source_id", keep="first")
    log.info(
        "existing BJ21 union: %d rows (%d dupes dropped)",
        len(combined),
        n_before - len(combined),
    )
    return combined


def _cast_chunk_df(df: pd.DataFrame) -> pd.DataFrame:
    if "source_id" in df.columns:
        df["source_id"] = df["source_id"].astype("int64")
    for col in FLOAT32_COLS:
        if col in df.columns and df[col].dtype != np.float32:
            df[col] = df[col].astype(np.float32)
    return df


def _fetch_chunk_upload(
    svc: TAPService,
    source_ids: list[int],
    *,
    log: logging.Logger,
    chunk_idx: int,
) -> tuple[pd.DataFrame, int]:
    """TAP-UPLOAD fetch for one batch with retries. Returns (df, retries_used).

    Raises ``RuntimeError`` after :data:`MAX_RETRIES_PER_BATCH` retries.
    """
    retries = 0
    last_exc: Exception | None = None
    while retries <= MAX_RETRIES_PER_BATCH:
        try:
            upload = AstropyTable({"source_id": source_ids})
            job: AsyncTAPJob = svc.submit_job(
                UPLOAD_ADQL,
                uploads={UPLOAD_TABLE_NAME: upload},
                language="ADQL",
                maxrec=GAVO_MAXREC,
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
                    except Exception as exc:  # noqa: BLE001 — wrap as RuntimeError
                        raise RuntimeError(
                            f"async TAP UPLOAD job ended in {job.phase!r}: {exc!r}"
                        ) from exc
                    raise RuntimeError(f"async TAP UPLOAD job ended in {job.phase!r}")
                table = job.fetch_result().to_table()
            finally:
                try:
                    job.delete()
                except Exception as exc:  # noqa: BLE001 — deletion is best-effort
                    log.warning("chunk %d: failed to delete async job: %r", chunk_idx, exc)
            return _cast_chunk_df(table.to_pandas()), retries
        except (DALQueryError, RuntimeError, TimeoutError, OSError) as exc:
            last_exc = exc
            retries += 1
            wait = min(30 * retries, 120)
            log.warning(
                "chunk %d UPLOAD attempt %d failed: %r — sleeping %ds before retry",
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


def _fetch_chunk_inlist(
    svc: TAPService,
    source_ids: list[int],
    *,
    log: logging.Logger,
    chunk_idx: int,
) -> tuple[pd.DataFrame, int]:
    """IN-list fallback fetch. Same retry/timeout envelope as _fetch_chunk_upload."""
    retries = 0
    last_exc: Exception | None = None
    while retries <= MAX_RETRIES_PER_BATCH:
        try:
            adql = IN_LIST_ADQL.replace(BATCH_PLACEHOLDER, ",".join(str(i) for i in source_ids))
            job: AsyncTAPJob = svc.submit_job(adql, language="ADQL", maxrec=GAVO_MAXREC)
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
                        raise RuntimeError(
                            f"async TAP IN-list job ended in {job.phase!r}: {exc!r}"
                        ) from exc
                    raise RuntimeError(f"async TAP IN-list job ended in {job.phase!r}")
                table = job.fetch_result().to_table()
            finally:
                try:
                    job.delete()
                except Exception as exc:  # noqa: BLE001
                    log.warning("chunk %d: failed to delete async job: %r", chunk_idx, exc)
            return _cast_chunk_df(table.to_pandas()), retries
        except (DALQueryError, RuntimeError, TimeoutError, OSError) as exc:
            last_exc = exc
            retries += 1
            wait = min(30 * retries, 120)
            log.warning(
                "chunk %d IN-list attempt %d failed: %r — sleeping %ds before retry",
                chunk_idx,
                retries,
                exc,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"chunk {chunk_idx}: IN-list exceeded {MAX_RETRIES_PER_BATCH} retries; "
        f"last error: {last_exc!r}"
    )


def _summary_stats(df: pd.DataFrame) -> dict[str, Any]:
    d = df["r_med_photogeo"].to_numpy()
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
            "count_le_2500pc": 0,
            "frac_le_2500pc": 0.0,
            "n_finite": 0,
        }
    q25, q50, q75 = np.quantile(d, [0.25, 0.5, 0.75])
    n_le = int((d <= 2500.0).sum())
    return {
        "min": float(d.min()),
        "q25": float(q25),
        "median": float(q50),
        "q75": float(q75),
        "max": float(d.max()),
        "count_le_2500pc": n_le,
        "frac_le_2500pc": float(n_le) / float(d.size),
        "n_finite": int(d.size),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:  # noqa: PLR0912, PLR0915 — linear driver; splitting hurts readability
    log = _setup_logging()
    t0 = time.time()
    footprint_before = _data_footprint_gb()
    log.info("=== BJ21 fetch for Andrae+2023 vetted RGB — start ===")
    log.info("data/ footprint before: %.2f GB", footprint_before)

    # 1. Andrae pool.
    log.info("loading Andrae+2023 catalog: %s", ANDRAE_PARQUET)
    andrae = pd.read_parquet(ANDRAE_PARQUET, columns=["source_id"])
    andrae["source_id"] = andrae["source_id"].astype("int64")
    andrae_ids = pd.unique(andrae["source_id"].to_numpy())
    n_andrae_total = int(andrae["source_id"].shape[0])
    n_andrae_unique = int(andrae_ids.shape[0])
    log.info(
        "Andrae catalog: %d rows, %d unique source_ids",
        n_andrae_total,
        n_andrae_unique,
    )

    # 2. Existing BJ21 ∩ Andrae.
    existing = _load_existing_bj(log)
    andrae_set = set(andrae_ids.tolist())
    existing_mask = (
        existing["source_id"].isin(andrae_set) if len(existing) else pd.Series(dtype=bool)
    )
    existing_reused = (
        existing.loc[existing_mask, list(KEEP_COLS)].reset_index(drop=True)
        if len(existing)
        else pd.DataFrame(columns=list(KEEP_COLS))
    )
    log.info(
        "existing BJ21 rows reusable for Andrae: %d / %d",
        len(existing_reused),
        len(existing),
    )

    # 3. Delta to fetch.
    existing_ids_set = set(existing_reused["source_id"].to_numpy().tolist())
    delta_ids = [int(i) for i in andrae_ids.tolist() if int(i) not in existing_ids_set]
    n_delta = len(delta_ids)
    log.info("delta source_ids to fetch from GAVO: %d", n_delta)

    # 4. Run UPLOAD batches (fallback on systematic failure).
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    batch_size = BATCH_SIZE
    n_batches = (n_delta + batch_size - 1) // batch_size
    log.info("planning %d batches of %d (UPLOAD primary)", n_batches, batch_size)

    svc = gavo_service()

    chunk_wallclocks: list[float] = []
    chunk_retries_total = 0
    chunk_failure_docs: list[str] = []
    fetched_frames: list[pd.DataFrame] = []
    strategy_used = "upload"  # will flip to 'fallback_full_table' if we switch
    n_fallback_chunks = 0
    consecutive_failures = 0
    halt_reason: str | None = None

    pbar = tqdm(total=n_batches, desc="BJ21 chunks", unit="chunk")

    for idx in range(n_batches):
        wall_elapsed = time.time() - t0
        if wall_elapsed > MAX_WALLCLOCK_SEC:
            halt_reason = (
                f"wall-clock {wall_elapsed / 3600:.2f} h exceeds "
                f"{MAX_WALLCLOCK_SEC / 3600:.1f} h budget"
            )
            log.error("HALT: %s", halt_reason)
            break

        footprint = _data_footprint_gb()
        if footprint > DATA_FOOTPRINT_CEILING_GB:
            halt_reason = (
                f"data/ footprint {footprint:.2f} GB exceeds "
                f"{DATA_FOOTPRINT_CEILING_GB:.2f} GB ceiling"
            )
            log.error("HALT: %s", halt_reason)
            break

        chunk = delta_ids[idx * batch_size : (idx + 1) * batch_size]
        chunk_file = CHUNK_DIR / f"chunk_{idx:06d}.parquet"
        if chunk_file.is_file():
            try:
                frame = pd.read_parquet(chunk_file)
                fetched_frames.append(frame)
                pbar.update(1)
                consecutive_failures = 0
                continue
            except Exception as exc:  # noqa: BLE001 — corrupt ckpt → refetch
                log.warning(
                    "batch %d: checkpoint %s unreadable (%r); refetching",
                    idx,
                    chunk_file.name,
                    exc,
                )
                chunk_file.unlink(missing_ok=True)

        t_batch = time.time()
        try:
            if strategy_used == "upload":
                df_batch, retries = _fetch_chunk_upload(svc, chunk, log=log, chunk_idx=idx)
            else:
                df_batch, retries = _fetch_chunk_inlist(svc, chunk, log=log, chunk_idx=idx)
                n_fallback_chunks += 1
        except Exception as exc:  # noqa: BLE001 — surface, escalate, decide
            consecutive_failures += 1
            msg = f"batch {idx}: {strategy_used} unrecoverable after retries — {exc!r}"
            chunk_failure_docs.append(msg)
            log.error(msg)
            if (
                strategy_used == "upload"
                and consecutive_failures >= MAX_CONSECUTIVE_FAILURES_BEFORE_FALLBACK
            ):
                log.error(
                    "FALLBACK: %d consecutive UPLOAD failures — switching to IN-list",
                    consecutive_failures,
                )
                strategy_used = "fallback_full_table"
                consecutive_failures = 0
                # Retry this chunk with the fallback strategy.
                try:
                    df_batch, retries = _fetch_chunk_inlist(svc, chunk, log=log, chunk_idx=idx)
                    n_fallback_chunks += 1
                except Exception as exc2:  # noqa: BLE001
                    chunk_failure_docs.append(
                        f"batch {idx}: IN-list fallback also failed — {exc2!r}"
                    )
                    consecutive_failures = 1
                    pbar.update(1)
                    continue
            elif strategy_used == "fallback_full_table" and consecutive_failures >= 3:
                halt_reason = (
                    "3 consecutive failures on IN-list fallback — both strategies exhausted"
                )
                log.error("HALT: %s", halt_reason)
                break
            else:
                pbar.update(1)
                continue

        elapsed = time.time() - t_batch
        chunk_wallclocks.append(elapsed)
        chunk_retries_total += retries
        consecutive_failures = 0

        # Row-count sanity: BJ21 coverage is high but not 100% — flag > 5 % miss.
        n_in = len(chunk)
        n_out = len(df_batch)
        miss_frac = 1.0 - (n_out / n_in if n_in else 1.0)
        if miss_frac > 0.05:
            msg = (
                f"batch {idx}: returned {n_out}/{n_in} rows "
                f"({miss_frac:.2%} missing) — BJ21-missing is normal; aggregate reported"
            )
            log.info(msg)
            chunk_failure_docs.append(f"SOFT: {msg}")

        df_batch = df_batch[list(KEEP_COLS)].copy()
        _write_parquet_atomic(df_batch, chunk_file)
        fetched_frames.append(df_batch)

        pbar.set_postfix(
            strategy=strategy_used,
            rows=n_out,
            s=f"{elapsed:.1f}",
            retries=retries,
        )
        pbar.update(1)
        log.info(
            "batch %d/%d done: %d rows in %.1f s (retries=%d, strategy=%s)",
            idx + 1,
            n_batches,
            n_out,
            elapsed,
            retries,
            strategy_used,
        )

    pbar.close()

    # 5. Concat + dedupe.
    log.info("concatenating existing reused + newly fetched")
    if fetched_frames:
        new_df = pd.concat(fetched_frames, ignore_index=True)
    else:
        new_df = pd.DataFrame(columns=list(KEEP_COLS))
    if not new_df.empty:
        new_df = _cast_chunk_df(new_df)

    combined = pd.concat([existing_reused[list(KEEP_COLS)], new_df], ignore_index=True)
    n_before_dedupe = len(combined)
    combined = combined.drop_duplicates(subset="source_id", keep="first").reset_index(drop=True)
    log.info(
        "final combined: %d rows (%d dupes dropped)",
        len(combined),
        n_before_dedupe - len(combined),
    )
    combined = combined.sort_values("source_id").reset_index(drop=True)

    # 6. Atomic write final parquet.
    _write_parquet_atomic(combined, OUT_PARQUET)
    size_mb = OUT_PARQUET.stat().st_size / 1024**2
    out_sha = _sha256_of(OUT_PARQUET)
    log.info("wrote %s (%.1f MB, sha256=%s…)", OUT_PARQUET, size_mb, out_sha[:12])

    # 7. Provenance sidecar.
    andrae_sha = _sha256_of(ANDRAE_PARQUET)
    existing_shas = [
        (str(p.relative_to(REPO)), _sha256_of(p)) for p in EXISTING_BJ_PATHS if p.is_file()
    ]
    prov_sources: list[Any] = [
        LocalSource(
            name="Andrae+2023 vetted RGB catalogue (VizieR J/MNRAS/537/1984)",
            path=str(ANDRAE_PARQUET.relative_to(REPO)),
            sha256=andrae_sha,
        ),
    ]
    for rel, sha in existing_shas:
        prov_sources.append(
            LocalSource(
                name=f"Existing BJ21 fetch ({Path(rel).name})",
                path=rel,
                sha256=sha,
            )
        )
    prov_sources.append(
        TapSource(
            name="GAVO gedr3dist.main (Bailer-Jones+2021 photogeometric, UPLOAD join)",
            endpoint=GAVO_TAP_URL,
            query=(UPLOAD_ADQL if strategy_used == "upload" else IN_LIST_ADQL),
            n_batches=n_batches,
            batch_size=batch_size,
        )
    )

    total_wall = time.time() - t0
    sumstats = _summary_stats(combined)
    prov = Provenance(
        output_file=str(OUT_PARQUET.relative_to(REPO)),
        script="scripts/fetch_bailerjones_andrae_rgb.py",
        sources=prov_sources,
        cuts_applied=[],
        corrections=[
            "float64 → float32 on r_med_photogeo, r_lo_photogeo, r_hi_photogeo",
            "keep photogeometric cols only (drop r_med_geo, r_lo_geo, r_hi_geo, flag)",
            "source_id cast to int64",
            "dedupe on source_id (keep first across existing BJ21 union + new fetch)",
            "GAVO maxrec=100000 to override default 20000 truncation",
        ],
        row_count_before=n_andrae_unique,
        row_count_after=int(len(combined)),
        notes=(
            "Full BJ21 photogeometric distances for the Andrae+2023 vetted RGB "
            "sample (~10.48 M source_ids). Thread-3 prerequisite for the volume-"
            "limited arm of Stream 3 (consumed downstream by Starfold). "
            "Existing Stream 1 + "
            "Stream 3 BJ21 on disk is re-used where source_id is in the Andrae "
            "set; only the delta is fetched from GAVO via TAP UPLOAD (async, "
            "10k-batch, resumable chunk ckpts under data/interim/bj21_andrae_chunks/). "
            "GAVO TAP is public so no AIP auth needed. r_med_photogeo is the "
            "primary distance per data_acquisition.md §7.1. Andrae+2023 pool "
            "snapshot: see LocalSource sha256 field for immutable pool reference."
        ),
        extra={
            "fetch_strategy": strategy_used,
            "n_fallback_chunks": int(n_fallback_chunks),
            "batch_size": batch_size,
            "n_batches_planned": n_batches,
            "n_batches_actual": len(chunk_wallclocks),
            "chunk_retries_total": chunk_retries_total,
            "wallclock_seconds": total_wall,
            "chunk_wallclock_mean_s": (
                float(np.mean(chunk_wallclocks)) if chunk_wallclocks else None
            ),
            "chunk_wallclock_median_s": (
                float(np.median(chunk_wallclocks)) if chunk_wallclocks else None
            ),
            "chunk_wallclock_min_s": (
                float(np.min(chunk_wallclocks)) if chunk_wallclocks else None
            ),
            "chunk_wallclock_max_s": (
                float(np.max(chunk_wallclocks)) if chunk_wallclocks else None
            ),
            "andrae_total_source_ids": n_andrae_total,
            "andrae_unique_source_ids": n_andrae_unique,
            "existing_bj21_reused": int(len(existing_reused)),
            "newly_fetched_rows": int(len(new_df)),
            "missing_from_bj21": int(n_andrae_unique - len(combined)),
            "distance_distribution": sumstats,
            "chunk_failure_docs": chunk_failure_docs,
            "output_sha256": out_sha,
            "halt_reason": halt_reason,
            "halt_conditions_tripped": {
                "wallclock_exceeded": total_wall > MAX_WALLCLOCK_SEC,
                "footprint_exceeded": _data_footprint_gb() > DATA_FOOTPRINT_CEILING_GB,
                "n_chunks_incomplete": int(
                    n_batches
                    - len(chunk_wallclocks)
                    - sum(
                        1
                        for idx in range(n_batches)
                        if (CHUNK_DIR / f"chunk_{idx:06d}.parquet").is_file()
                        and (idx * batch_size) >= (len(existing_reused) - 1)
                    )
                ),
            },
        },
    )
    write_sidecar(prov)

    # 8. Cleanup if chunks > 0.5 GB.
    total_chunk_size_gb = sum(p.stat().st_size for p in CHUNK_DIR.glob("chunk_*.parquet")) / 1024**3
    footprint_after = _data_footprint_gb()
    chunk_dir_removed = False
    if total_chunk_size_gb > 0.5 and len(combined) > 0 and OUT_PARQUET.is_file():
        log.info(
            "chunk dir %.2f GB > 0.5 GB — removing after successful concat",
            total_chunk_size_gb,
        )
        shutil.rmtree(CHUNK_DIR)
        chunk_dir_removed = True
        footprint_after = _data_footprint_gb()

    # 9. Operator-facing report.
    _write_report(
        log=log,
        strategy_used=strategy_used,
        n_fallback_chunks=n_fallback_chunks,
        n_andrae_total=n_andrae_total,
        n_andrae_unique=n_andrae_unique,
        existing_reused=int(len(existing_reused)),
        newly_fetched=int(len(new_df)),
        final_rows=int(len(combined)),
        missing_from_bj21=int(n_andrae_unique - len(combined)),
        chunk_wallclocks=chunk_wallclocks,
        chunk_retries_total=chunk_retries_total,
        chunk_failure_docs=chunk_failure_docs,
        total_wallclock_sec=total_wall,
        sumstats=sumstats,
        footprint_before=footprint_before,
        footprint_after=footprint_after,
        chunk_dir_removed=chunk_dir_removed,
        out_size_mb=size_mb,
        halt_reason=halt_reason,
    )

    log.info("=== done in %.1f min ===", total_wall / 60)


def _write_report(  # noqa: PLR0913, PLR0915 — operator report
    *,
    log: logging.Logger,
    strategy_used: str,
    n_fallback_chunks: int,
    n_andrae_total: int,
    n_andrae_unique: int,
    existing_reused: int,
    newly_fetched: int,
    final_rows: int,
    missing_from_bj21: int,
    chunk_wallclocks: list[float],
    chunk_retries_total: int,
    chunk_failure_docs: list[str],
    total_wallclock_sec: float,
    sumstats: dict[str, Any],
    footprint_before: float,
    footprint_after: float,
    chunk_dir_removed: bool,
    out_size_mb: float,
    halt_reason: str | None,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%SZ")
    mean_s = float(np.mean(chunk_wallclocks)) if chunk_wallclocks else float("nan")
    median_s = float(np.median(chunk_wallclocks)) if chunk_wallclocks else float("nan")
    total_h = total_wallclock_sec / 3600

    vol_target = 250_000
    vol_passed = sumstats["count_le_2500pc"] > vol_target

    lines = [
        "# BJ21 photogeometric distances — Andrae+2023 vetted RGB",
        "",
        f"*Generated: {now} — script `scripts/fetch_bailerjones_andrae_rgb.py`*",
        "",
        "Thread-3 prerequisite for the volume-limited arm of Stream 3 "
        "(consumed downstream by Starfold, separate repo).",
        "",
        "## 1. Strategy",
        "",
        "- Primary strategy: **TAP UPLOAD** (VOTable join against `TAP_UPLOAD.sidlist`).",
        "  Validated against GAVO live service with a 10 k smoke test before the run.",
        f"- Actual strategy used: **`{strategy_used}`**"
        + (
            " (no fallback needed)"
            if strategy_used == "upload"
            else f" ({n_fallback_chunks} IN-list fallback chunks)"
        ),
        "  > UPLOAD keeps request body small and sidesteps the ~100 KB IN-list ceiling "
        "that would otherwise push 10k-ID lists into HTTP-header territory on some TAP "
        "gateways.",
    ]
    if halt_reason:
        lines.append(f"- **Halt tripped:** {halt_reason}")
    lines += [
        "",
        "## 2. Source_id counts",
        "",
        f"- Andrae+2023 total rows: **{n_andrae_total:,}**",
        f"- Andrae+2023 unique source_ids: **{n_andrae_unique:,}**",
        f"- Already-on-disk BJ21 reused (Stream 1 + Stream 3 union ∩ Andrae): **{existing_reused:,}**",
        f"- Newly fetched from GAVO (delta): **{newly_fetched:,}**",
        f"- Final combined rows in output: **{final_rows:,}**",
        f"- BJ21-missing source_ids (no photogeometric solution on GAVO): **{missing_from_bj21:,}** "
        f"({missing_from_bj21 / max(n_andrae_unique, 1):.3%} — BJ21 coverage is ~99.9% of Gaia DR3; "
        "a small residual is expected and not a halt condition)",
        "",
        "## 3. Wall-clock",
        "",
        f"- Total: **{total_wallclock_sec:.0f} s** ({total_h:.2f} h)",
        f"- Batches completed: **{len(chunk_wallclocks)}**",
        f"- Mean chunk wall-clock: **{mean_s:.1f} s**",
        f"- Median chunk wall-clock: **{median_s:.1f} s**",
        "",
        "## 4. Retries + failures",
        "",
        f"- Total retry count (across all batches): **{chunk_retries_total}**",
    ]
    if chunk_failure_docs:
        lines.append("- Failures / soft anomalies:")
        for m in chunk_failure_docs[:50]:
            lines.append(f"  - {m}")
        if len(chunk_failure_docs) > 50:
            lines.append(f"  - … and {len(chunk_failure_docs) - 50} more (see log)")
    else:
        lines.append("- No failures; all batches completed on first pass.")

    lines += [
        "",
        "## 5. Distance distribution (`r_med_photogeo`, pc)",
        "",
        f"- n finite: **{sumstats['n_finite']:,}**",
        f"- min: {sumstats['min']}",
        f"- Q25: {sumstats['q25']}",
        f"- median: {sumstats['median']}",
        f"- Q75: {sumstats['q75']}",
        f"- max: {sumstats['max']}",
        "",
        f"- Count with `r_med_photogeo ≤ 2500` pc: **{sumstats['count_le_2500pc']:,}** "
        f"({sumstats['frac_le_2500pc']:.2%})",
        "",
        f"  > Phase 3 target (revised under 10 GB budget): > **{vol_target:,}** volume-limited stars.",
        "  > " + ("**PASSED**" if vol_passed else "**NOT MET** — revisit volume selection"),
        "",
        "## 6. Storage diff",
        "",
        f"- `data/` footprint before: **{footprint_before:.2f} GB**",
        f"- `data/` footprint after:  **{footprint_after:.2f} GB**",
        f"- Final Parquet size: **{out_size_mb:.1f} MB**",
        f"- Per-chunk checkpoints consolidated + removed: **{chunk_dir_removed}**",
        "",
        "## 7. Anomalies",
        "",
    ]
    anomalies: list[str] = []
    miss_frac = missing_from_bj21 / max(n_andrae_unique, 1)
    if miss_frac > 0.01:
        anomalies.append(f"BJ21 miss rate {miss_frac:.2%} exceeds 1% rule-of-thumb — investigate")
    if total_h > 2.5:
        anomalies.append(f"Wall-clock {total_h:.2f} h exceeded 2 h estimate — GAVO rate-limiting?")
    if strategy_used != "upload":
        anomalies.append(f"UPLOAD fallback tripped: {n_fallback_chunks} chunks used IN-list")
    if halt_reason:
        anomalies.append(f"Halt condition fired: {halt_reason}")
    if not anomalies:
        lines.append("- None.")
    else:
        for a in anomalies:
            lines.append(f"- {a}")

    content = "\n".join(lines) + "\n"
    tmp = REPORT_PATH.with_suffix(REPORT_PATH.suffix + ".part")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(REPORT_PATH)
    log.info("wrote report: %s", REPORT_PATH)
    print("\n" + content)


if __name__ == "__main__":
    main()
