"""Pyvo TAP wrappers for ArqueoGal data acquisition.

All TAP queries go through this module. Never import ``astroquery.gaia`` —
it has been unstable in recent months (see ``docs/data_acquisition.md`` §14.3).
Astroquery.vizier is last-resort only and belongs in call-site code, not here.

Endpoints are listed in ``docs/data_acquisition.md`` §1.

Usage
-----
.. code-block:: python

    from arqueogal.data.tap import aip_service, run_async, batched_in_query

    aip = aip_service()  # loads ~/.arqueogal/credentials.yaml
    table = run_async(aip, "SELECT TOP 100 source_id FROM gaiadr3.gaia_source")

    # Batched IN-query — substitute __batch__ with comma-separated source_ids.
    template = "SELECT source_id, ra, dec FROM gaiadr3.gaia_source WHERE source_id IN (__batch__)"
    for batch_table in batched_in_query(aip, template, source_ids, batch_size=10_000):
        ...
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

import pandas as pd
import pyvo
import requests
from astropy.table import Table
from pyvo.auth import authsession
from pyvo.dal.tap import AsyncTAPJob, TAPService

from arqueogal.data.credentials import (
    AIP_TOKEN_ENV_VAR,
    Credentials,
    load_aip_token_from_env,
    load_credentials,
)

logger = logging.getLogger(__name__)

AIP_TAP_URL = "https://gaia.aip.de/tap"
ESA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap"
GAVO_TAP_URL = "https://dc.g-vo.org/tap"
VIZIER_TAP_URL = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"

SYNC_ROW_THRESHOLD = 5_000
"""Queries expected to return more than this go async (`submit_job`).

AIP sync queries time out at ~90 s (data_acquisition.md §13.11).
"""

BATCH_PLACEHOLDER = "__batch__"

DEFAULT_ASYNC_TIMEOUT_SEC = 3600
"""Hard ceiling on async job wait; raise rather than block forever."""


# -----------------------------------------------------------------------------
# service constructors
# -----------------------------------------------------------------------------


def aip_service(credentials: Credentials | None = None) -> TAPService:
    """AIP authenticated TAP service.

    Hosts Gaia DR3 (``gaiadr3.*``) and Queiroz+2023 StarHorse2 v2 tables
    (``gaiadr3_contrib.aqueiroz2023_*_v2``). Authenticated access gives higher
    quotas and async support.

    Auth precedence
    ---------------
    1. ``credentials.aip`` (YAML) → HTTP Basic via ``pyvo.auth.AuthSession``.
    2. ``GAIA_AIP_TOKEN`` env var → ``Authorization: Token <token>`` on a
       plain ``requests.Session``.
    3. Otherwise, :class:`RuntimeError`.
    """
    if credentials is None:
        try:
            credentials = load_credentials()
        except FileNotFoundError:
            credentials = Credentials()

    if credentials.aip is not None:
        session = authsession.AuthSession()
        session.credentials.set_password(credentials.aip.user, credentials.aip.password)
        return pyvo.dal.TAPService(AIP_TAP_URL, session=session)

    token_creds = load_aip_token_from_env()
    if token_creds is not None:
        logger.info("AIP: using %s env-var token (YAML aip block absent)", AIP_TOKEN_ENV_VAR)
        session = requests.Session()
        session.headers["Authorization"] = f"Token {token_creds.token}"
        return pyvo.dal.TAPService(AIP_TAP_URL, session=session)

    raise RuntimeError(
        "AIP credentials missing: no 'aip' block in credentials.yaml and no "
        f"{AIP_TOKEN_ENV_VAR} environment variable set."
    )


def esa_service(credentials: Credentials | None = None) -> TAPService:
    """ESA Gaia Archive TAP (backup endpoint).

    Auth is optional — public access works without. Supply credentials for
    per-user quotas if you have a Gaia Archive account.
    """
    if credentials is None:
        try:
            credentials = load_credentials()
        except FileNotFoundError:
            credentials = Credentials()
    if credentials.esa is None:
        return pyvo.dal.TAPService(ESA_TAP_URL)
    session = authsession.AuthSession()
    session.credentials.set_password(credentials.esa.user, credentials.esa.password)
    return pyvo.dal.TAPService(ESA_TAP_URL, session=session)


def gavo_service() -> TAPService:
    """GAVO TAP — Bailer-Jones+2021 distances at ``gedr3dist.main``."""
    return pyvo.dal.TAPService(GAVO_TAP_URL)


def vizier_service() -> TAPService:
    """VizieR TAP — published catalogues (e.g. Hon+2021 at J/ApJ/919/131)."""
    return pyvo.dal.TAPService(VIZIER_TAP_URL)


# -----------------------------------------------------------------------------
# query runners
# -----------------------------------------------------------------------------


def run_sync(
    service: TAPService,
    adql: str,
    *,
    response_format: str | None = None,
    language: str = "ADQL",
) -> Table:
    """Synchronous query. Use only for ≤ ``SYNC_ROW_THRESHOLD`` rows.

    AIP sync queries time out at ~90 s — for anything larger use
    :func:`run_async` instead.

    ``response_format`` maps to the TAP ``RESPONSEFORMAT`` parameter.
    ``None`` uses the service default (usually VOTable/TABLEDATA — XML).
    ``"votable/binary2"`` is 3–5× faster to parse for tables above ~100 k
    rows because the payload is BASE64-encoded binary rather than XML text.
    """
    logger.info("sync TAP query on %s (format=%s)", service.baseurl, response_format or "default")
    kwargs: dict[str, str] = {"language": language}
    if response_format is not None:
        kwargs["responseformat"] = response_format
    result = service.search(adql, **kwargs)
    return result.to_table()


def run_async(
    service: TAPService,
    adql: str,
    *,
    timeout_sec: float | None = DEFAULT_ASYNC_TIMEOUT_SEC,
    response_format: str | None = None,
    queue: str | None = None,
    runid: str | None = None,
    language: str = "ADQL",
) -> Table:
    """Asynchronous query (submit_job + wait). Required for queries > 5 000 rows.

    ``response_format`` maps to the TAP ``RESPONSEFORMAT`` parameter.
    ``None`` uses the service default (usually VOTable/TABLEDATA — XML).
    For multi-million-row pulls, passing ``"votable/binary2"`` can cut wall
    time by an order of magnitude (TABLEDATA XML parsing is CPU-bound at
    roughly 10–20 k rows/s; binary2 is O(stream-decode)). See
    ``scripts/fetch_andrae2023_vizier.py`` for the motivating case.

    ``queue`` selects the AIP server-side queue (``"2h"``, ``"5m"``, etc.).
    Without an explicit long-queue pick, AIP's gateway 504s large-IN async
    submissions because the default queue's time budget is too small. Pass
    ``queue="2h"`` for Gaia enrichment with ~10 k IDs/batch. See
    :func:`aip_service` for auth.

    ``runid`` is a free-form job identifier echoed in AIP's job listing —
    useful when several scripts share the same token.

    Raises
    ------
    RuntimeError
        If the job finishes in the ``ERROR`` phase. The UWS error message is
        included where available.
    TimeoutError
        If ``timeout_sec`` elapses before the job reaches a terminal phase.
    """
    logger.info(
        "async TAP submit on %s (format=%s, queue=%s)",
        service.baseurl, response_format or "default", queue or "default",
    )
    submit_kwargs: dict[str, str] = {"language": language}
    if response_format is not None:
        submit_kwargs["responseformat"] = response_format
    if queue is not None:
        submit_kwargs["queue"] = queue
    if runid is not None:
        submit_kwargs["runid"] = runid
    job: AsyncTAPJob = service.submit_job(adql, **submit_kwargs)
    try:
        job.run()
        # pyvo's AsyncTAPJob.wait does not accept an ``interval`` kwarg; it
        # uses an internal 1 s→exponential-backoff poller. We just wait for
        # a terminal phase within ``timeout_sec``.
        job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=timeout_sec)
        final = job.phase
        if final != "COMPLETED":
            message = _safe_error_message(job)
            raise RuntimeError(f"async TAP job finished in phase {final!r}: {message}")
        logger.info("async TAP job completed: %s", job.url)
        return job.fetch_result().to_table()
    finally:
        _safe_delete(job)


def _safe_error_message(job: AsyncTAPJob) -> str:
    # This pyvo build exposes errors via ``raise_if_error()`` — no
    # ``error_summary`` attr. Catch the raised DALQueryError and stringify it.
    try:
        job.raise_if_error()
    except Exception as exc:
        return repr(exc)
    return "(no UWS error message)"


def _safe_delete(job: AsyncTAPJob) -> None:
    try:
        job.delete()
    except Exception as exc:
        logger.warning("failed to delete async job %s: %r", getattr(job, "url", "?"), exc)


# -----------------------------------------------------------------------------
# batched IN-list helper
# -----------------------------------------------------------------------------


def batched_in_query(  # noqa: PLR0913 — keyword-only tuning knobs with safe defaults
    service: TAPService,
    adql_template: str,
    source_ids: Iterable[int],
    *,
    batch_size: int = 10_000,
    mode: Literal["async", "sync", "auto"] = "auto",
    timeout_sec: float | None = DEFAULT_ASYNC_TIMEOUT_SEC,
    response_format: str | None = None,
    queue: str | None = None,
    runid: str | None = None,
) -> Iterator[Table]:
    """Yield one `astropy.Table` per batch of source_ids.

    The template must contain exactly one ``__batch__`` placeholder, which will
    be substituted with a comma-separated integer list per batch, e.g.::

        "SELECT ... FROM gaiadr3.gaia_source WHERE source_id IN (__batch__)"

    ``mode="auto"`` picks async when the batch size exceeds
    :data:`SYNC_ROW_THRESHOLD`, else sync. XP-coefficient queries should use
    ``batch_size=5_000`` per data_acquisition.md §6.3; everything else
    ``batch_size=10_000`` per §3.6.
    """
    if adql_template.count(BATCH_PLACEHOLDER) != 1:
        raise ValueError(
            f"adql_template must contain exactly one {BATCH_PLACEHOLDER!r} placeholder"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    use_async = {
        "async": True,
        "sync": False,
        "auto": batch_size > SYNC_ROW_THRESHOLD,
    }[mode]

    ids_iter = iter(source_ids)
    batch_idx = 0
    while True:
        batch = _take(ids_iter, batch_size)
        if not batch:
            return
        adql = adql_template.replace(BATCH_PLACEHOLDER, ",".join(str(i) for i in batch))
        logger.info(
            "batch %d: %d ids, %s", batch_idx, len(batch), "async" if use_async else "sync"
        )
        yield (
            run_async(
                service, adql, timeout_sec=timeout_sec,
                response_format=response_format, queue=queue, runid=runid,
            )
            if use_async
            else run_sync(service, adql, response_format=response_format)
        )
        batch_idx += 1


def _take(iterator: Iterator[int], n: int) -> list[int]:
    """Pull up to ``n`` items, casting to int (rejects non-castable early)."""
    batch: list[int] = []
    for item in iterator:
        batch.append(int(item))
        if len(batch) == n:
            return batch
    return batch


# -----------------------------------------------------------------------------
# batched, resumable DataFrame fetch
# -----------------------------------------------------------------------------


def batched_fetch_df(  # noqa: PLR0913 — keyword-only tuning knobs with safe defaults
    service: TAPService,
    source_ids: Iterable[int],
    adql_template: str,
    *,
    batch_size: int,
    mode: Literal["async", "sync", "auto"] = "async",
    checkpoint_dir: Path | str | None = None,
    checkpoint_prefix: str = "batch",
    timeout_sec: float | None = DEFAULT_ASYNC_TIMEOUT_SEC,
    response_format: str | None = None,
    queue: str | None = None,
    runid: str | None = None,
) -> pd.DataFrame:
    """Run an ``IN (__batch__)`` query in chunks and return a single DataFrame.

    Unlike :func:`batched_in_query` (streaming iterator of astropy Tables,
    no checkpointing), this helper **materialises** every batch to pandas
    and optionally writes each chunk to ``{checkpoint_prefix}_{NNNN}.parquet``
    for resumable multi-hour ingestions. On rerun, existing checkpoint files
    are loaded and the TAP query is skipped.

    Parameters
    ----------
    service
        TAP service (e.g. ``aip_service()``, ``gavo_service()``).
    source_ids
        Iterable of int64 Gaia DR3 source_ids. Duplicates are kept as-is —
        dedupe upstream if needed.
    adql_template
        Query template containing exactly one ``__batch__`` placeholder,
        substituted per batch with a comma-separated integer list.
    batch_size
        Chunk size for the ``IN (__batch__)`` query. 10 000 is the §3.6
        default for main Gaia enrichment; 5 000 for XP coefficients (§6.3).
    mode
        ``"async"`` (default), ``"sync"``, or ``"auto"`` (picks async when
        ``batch_size > SYNC_ROW_THRESHOLD``).
    checkpoint_dir
        If set, each batch is written atomically (``.part`` → rename) to
        ``{checkpoint_prefix}_{NNNN}.parquet`` in this directory.
    checkpoint_prefix
        Filename prefix for per-batch checkpoints. Callers use distinct
        prefixes (``"batch"``, ``"xp_batch"``, ``"bj_batch"``, …) so that
        different query kinds can share one directory without collision.
    timeout_sec
        Forwarded to :func:`run_async`.

    Returns
    -------
    pd.DataFrame
        Concatenated results across all batches. Empty input → empty frame.
    """
    if adql_template.count(BATCH_PLACEHOLDER) != 1:
        raise ValueError(
            f"adql_template must contain exactly one {BATCH_PLACEHOLDER!r} placeholder"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    ids = [int(x) for x in source_ids]
    if not ids:
        return pd.DataFrame()

    ckpt = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if ckpt is not None:
        ckpt.mkdir(parents=True, exist_ok=True)

    use_async = {
        "async": True,
        "sync": False,
        "auto": batch_size > SYNC_ROW_THRESHOLD,
    }[mode]

    n_batches = (len(ids) + batch_size - 1) // batch_size
    frames: list[pd.DataFrame] = []

    for idx in range(n_batches):
        chunk = ids[idx * batch_size : (idx + 1) * batch_size]
        batch_file = (
            ckpt / f"{checkpoint_prefix}_{idx:04d}.parquet" if ckpt is not None else None
        )

        if batch_file is not None and batch_file.is_file():
            logger.info("batch %d/%d: reusing %s", idx + 1, n_batches, batch_file)
            frames.append(pd.read_parquet(batch_file))
            continue

        batch_adql = adql_template.replace(
            BATCH_PLACEHOLDER, ",".join(str(i) for i in chunk)
        )
        logger.info(
            "batch %d/%d: %d ids (%s)",
            idx + 1,
            n_batches,
            len(chunk),
            "async" if use_async else "sync",
        )
        table = (
            run_async(
                service, batch_adql, timeout_sec=timeout_sec,
                response_format=response_format, queue=queue, runid=runid,
            )
            if use_async
            else run_sync(service, batch_adql, response_format=response_format)
        )
        frame = table.to_pandas()

        if batch_file is not None:
            tmp = batch_file.with_suffix(batch_file.suffix + ".part")
            frame.to_parquet(tmp, index=False)
            tmp.replace(batch_file)

        frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# -----------------------------------------------------------------------------
# batched TAP UPLOAD helper — for large ID joins that 504 an inline IN clause
# -----------------------------------------------------------------------------


def batched_upload_fetch_df(  # noqa: PLR0913 — keyword-only tuning knobs with safe defaults
    service: TAPService,
    source_ids: Iterable[int],
    adql_template: str,
    *,
    upload_name: str = "ids",
    batch_size: int = 10_000,
    checkpoint_dir: Path | str | None = None,
    checkpoint_prefix: str = "batch",
    timeout_sec: float | None = DEFAULT_ASYNC_TIMEOUT_SEC,
    queue: str | None = None,
    runid: str | None = None,
) -> pd.DataFrame:
    """Chunked TAP UPLOAD fetch — each batch POSTs a VOTable instead of IN (…).

    Large inline ``IN (id1, id2, …)`` lists hit a hard ceiling on AIP: queries
    over ~100 KB body size return ``504 Gateway Timeout`` before the backend
    ever sees them. TAP UPLOAD sidesteps that by attaching the IDs as a
    multipart VOTable parameter, keeping the ADQL body tiny.

    Parameters
    ----------
    service
        TAP service (e.g. ``aip_service()``).
    source_ids
        Iterable of int64 Gaia DR3 source_ids.
    adql_template
        ADQL that JOINs against ``tap_upload.<upload_name>`` and has **no**
        ``__batch__`` placeholder. Must reference the upload table literally.
        Example::

            SELECT g.source_id, g.ra, g.dec
            FROM gaiadr3.gaia_source AS g
            JOIN tap_upload.ids AS u ON g.source_id = u.source_id
    upload_name
        VOTable resource name exposed as ``tap_upload.<name>``. Default
        ``"ids"`` matches the example above.
    batch_size
        IDs per upload batch. Safe up to at least 10 000 on AIP.
    checkpoint_dir, checkpoint_prefix
        Same resumable-checkpoint semantics as :func:`batched_fetch_df`.
    timeout_sec
        Forwarded to ``job.wait``.
    queue, runid
        Forwarded to ``submit_job``. ``queue="2h"`` recommended on AIP.

    Returns
    -------
    pd.DataFrame
        Concatenated results across all batches. Empty input → empty frame.
    """
    from astropy.table import Table as _Table  # local import: keeps import graph thin

    if BATCH_PLACEHOLDER in adql_template:
        raise ValueError(
            f"adql_template must not contain {BATCH_PLACEHOLDER!r} — "
            f"batched_upload_fetch_df joins against tap_upload.{upload_name}"
        )
    if f"tap_upload.{upload_name}" not in adql_template.lower().replace(" ", ""):
        # Gentle guard against the common mistake of forgetting the JOIN.
        logger.warning(
            "adql_template does not reference tap_upload.%s; "
            "are you sure it uses the uploaded table?", upload_name,
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    ids = [int(x) for x in source_ids]
    if not ids:
        return pd.DataFrame()

    ckpt = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if ckpt is not None:
        ckpt.mkdir(parents=True, exist_ok=True)

    n_batches = (len(ids) + batch_size - 1) // batch_size
    frames: list[pd.DataFrame] = []

    submit_kwargs: dict[str, str] = {"language": "ADQL"}
    if queue is not None:
        submit_kwargs["queue"] = queue
    if runid is not None:
        submit_kwargs["runid"] = runid

    for idx in range(n_batches):
        chunk = ids[idx * batch_size : (idx + 1) * batch_size]
        batch_file = (
            ckpt / f"{checkpoint_prefix}_{idx:04d}.parquet" if ckpt is not None else None
        )
        if batch_file is not None and batch_file.is_file():
            logger.info("batch %d/%d: reusing %s", idx + 1, n_batches, batch_file)
            frames.append(pd.read_parquet(batch_file))
            continue

        upload = _Table({"source_id": chunk})
        logger.info(
            "batch %d/%d: upload %d ids to tap_upload.%s",
            idx + 1, n_batches, len(chunk), upload_name,
        )
        job: AsyncTAPJob = service.submit_job(
            adql_template, uploads={upload_name: upload}, **submit_kwargs
        )
        try:
            job.run()
            job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=timeout_sec)
            if job.phase != "COMPLETED":
                raise RuntimeError(
                    f"async TAP upload job ended in {job.phase!r}: "
                    f"{_safe_error_message(job)}"
                )
            logger.info("batch %d/%d completed: %s", idx + 1, n_batches, job.url)
            frame = job.fetch_result().to_table().to_pandas()
        finally:
            _safe_delete(job)

        if batch_file is not None:
            tmp = batch_file.with_suffix(batch_file.suffix + ".part")
            frame.to_parquet(tmp, index=False)
            tmp.replace(batch_file)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
