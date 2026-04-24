"""2MASS + AllWISE infrared photometry cross-match for ArqueoGal.

Pipeline 1 (``xp_abundances``) uses infrared photometry (2MASS J/H/K and
AllWISE W1/W2) as auxiliary features alongside the Gaia XP coefficients.
Pre-flight zero-imputation diagnostics on Stream 3 showed that dropping IR
degrades all five v1 label predictions by 28–130 % RMSE, and NaN-imputation
crashes the adapter — so per-star IR is non-negotiable for inference.

This module does the cross-match via the *Gaia-hosted* external-source join
tables, not by downloading the full 2MASS PSC or AllWISE catalogues. The
official cross-match was computed by Marrese et al. (2017, 2019) for the
Gaia DR3 release and is exposed as ``gaiadr3.tmass_psc_xsc_best_neighbour``
/ ``gaiadr3.allwise_best_neighbour`` — one "best" external counterpart per
Gaia source. Magnitudes live in the ``gaiadr1.tmass_original_valid`` /
``gaiadr1.allwise_original_valid`` mirror tables and are joined in.

The queries use TAP UPLOAD (not inline ``IN (…)``) so the ADQL body stays
tiny and the gateway never times out. ``batched_upload_fetch_df`` handles
chunking, async submission, per-chunk checkpointing, and resumability.

Endpoints
---------
- ESA Gaia Archive TAP (default here): ``https://gea.esac.esa.int/tap-server/tap``.
  Anonymous works for TAP UPLOAD jobs up to 30 min runtime; authenticated
  gives per-user queues. Chosen here because AIP's queue is currently busy
  serving the BJ21 Stream 3 fetch (GAVO) + Stream-3 Gaia enrichment — ESA
  decouples us completely.
- AIP fallback: same schema, same table names. Swap via ``aip_service()``.

Scope
-----
One function per catalogue plus :func:`assemble_ir_photometry` that combines
both into a single per-source_id row with an ``ir_missing_flag`` = True when
*any* of {J, H, K, W1, W2} is missing. Downstream Pipeline 1 inference gates
on this flag and drops (or flags) the star before feeding the network.

References
----------
- Marrese et al. 2019, A&A 621 A144 (arXiv:1808.09151) — the method and
  the best-neighbour table schema.
- Gaia DR3 data-model: ``gaiadr3.tmass_psc_xsc_best_neighbour``,
  ``gaiadr3.tmass_psc_xsc_join``, ``gaiadr3.allwise_best_neighbour``.
  https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/
- 2MASS PSC: Cutri et al. 2003 (VII/233). j_m / h_m / ks_m are the default
  magnitudes (Vega); ``_msigcom`` is the combined uncertainty.
- AllWISE: Cutri et al. 2013 (II/328). w1mpro / w2mpro are the profile-fit
  magnitudes; ``_error`` is the 1-σ uncertainty.

Usage
-----
.. code-block:: python

    from arqueogal.data.ir_photometry import (
        assemble_ir_photometry, crossmatch_2mass, crossmatch_allwise,
    )
    from arqueogal.data.tap import esa_service

    svc = esa_service()
    df = assemble_ir_photometry(svc, source_ids, checkpoint_dir=...)
    # df is a polars DataFrame with one row per input source_id.

    # Or run them independently:
    t2 = crossmatch_2mass(svc, source_ids)
    aw = crossmatch_allwise(svc, source_ids)
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import polars as pl
from pyvo.dal.tap import TAPService

from arqueogal.data.tap import DEFAULT_ASYNC_TIMEOUT_SEC, batched_upload_fetch_df

logger = logging.getLogger(__name__)

TMASS_BATCH_SIZE: Final[int] = 10_000
ALLWISE_BATCH_SIZE: Final[int] = 10_000

TMASS_SCHEMA: Final[tuple[str, ...]] = (
    "source_id",
    "j_mag",
    "e_j_mag",
    "h_mag",
    "e_h_mag",
    "k_mag",
    "e_k_mag",
    "tmass_source_id",
    "tmass_angular_distance",
    "tmass_xm_quality_flag",
)

ALLWISE_SCHEMA: Final[tuple[str, ...]] = (
    "source_id",
    "w1_mag",
    "e_w1_mag",
    "w2_mag",
    "e_w2_mag",
    "allwise_source_id",
    "allwise_angular_distance",
    "allwise_xm_quality_flag",
)

# --- ADQL ---------------------------------------------------------------------
#
# LEFT JOIN from the uploaded id table preserves rows for stars without a
# 2MASS / AllWISE counterpart — those land with NaN mag columns and we flag
# them downstream via :func:`assemble_ir_photometry`. Aliases renamed to the
# module-internal schema (TMASS_SCHEMA, ALLWISE_SCHEMA) at fetch time.
#
# The 2MASS path is 2-table on ESA: we go from upload → best_neighbour →
# ``gaiadr1.tmass_original_valid`` directly via the PSC ``designation``,
# which best_neighbour exposes as ``original_ext_source_id``. This skips the
# otherwise-natural ``gaiadr3.tmass_psc_xsc_join`` hop because that table is
# currently broken on ESA's TAP backend — joining against it triggers
# ``java.sql.SQLException: PooledConnection has already been closed``
# deterministically on ~60% of attempts (see ``batched_upload_fetch_df``'s
# retry loop). The direct 2-join via ``designation`` is equivalent: for PSC
# counterparts (which is all we need — RGB giants never fall into 2MASS XSC),
# ``best_neighbour.original_ext_source_id`` equals the PSC designation. If
# we ever need the rare XSC matches, add the 3rd join back.
#
# The AllWISE path used to JOIN ``gaiadr1.allwise_original_valid`` on
# ``allwise_oid``, but that join deterministically triggers
# ``java.sql.SQLException: PooledConnection has already been closed`` on
# ESA's TAP backend (reproduced 3/3 under TAP UPLOAD, 2026-04-19). The
# ``designation`` join that works for 2MASS works identically here: AllWISE
# ``best_neighbour.original_ext_source_id`` is the AllWISE PSC designation
# (``Jhhmmss.ss±ddmmss.s`` format) and matches ``original_valid.designation``
# directly. We use it for AllWISE too.
#
# AIP hosts the magnitude catalogues under the ``catalogs.*`` schema with
# DR1-era column naming that differs slightly (``k_m`` not ``ks_m``,
# ``w1sigmpro`` not ``w1mpro_error``). The AIP variant joins via the
# ``designation`` field in the ``best_neighbour.original_ext_source_id``
# column, which is 2MASS's or AllWISE's native PSC designation. This avoids
# the ``tmass_psc_xsc_join`` detour entirely on AIP.

# ------------------ ESA variants (primary, Gaia Archive hosts gaiadr1) ---------

TMASS_ADQL_UPLOAD_ESA: Final[str] = """\
SELECT
    u.source_id,
    t.j_m            AS j_mag,
    t.j_msigcom      AS e_j_mag,
    t.h_m            AS h_mag,
    t.h_msigcom      AS e_h_mag,
    t.ks_m           AS k_mag,
    t.ks_msigcom     AS e_k_mag,
    t.designation    AS tmass_source_id,
    bn.angular_distance AS tmass_angular_distance,
    bn.xm_flag       AS tmass_xm_quality_flag
FROM tap_upload.ids AS u
LEFT JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS bn
    ON bn.source_id = u.source_id
LEFT JOIN gaiadr1.tmass_original_valid AS t
    ON t.designation = bn.original_ext_source_id
"""

ALLWISE_ADQL_UPLOAD_ESA: Final[str] = """\
SELECT
    u.source_id,
    a.w1mpro         AS w1_mag,
    a.w1mpro_error   AS e_w1_mag,
    a.w2mpro         AS w2_mag,
    a.w2mpro_error   AS e_w2_mag,
    a.designation    AS allwise_source_id,
    bn.angular_distance AS allwise_angular_distance,
    bn.xm_flag       AS allwise_xm_quality_flag
FROM tap_upload.ids AS u
LEFT JOIN gaiadr3.allwise_best_neighbour AS bn
    ON bn.source_id = u.source_id
LEFT JOIN gaiadr1.allwise_original_valid AS a
    ON a.designation = bn.original_ext_source_id
"""

# ------------------ AIP variants (fallback, catalogs.* schema) -----------------
#
# Column rename map vs ESA's gaiadr1.*_original_valid:
#   2MASS:   k_m ← ks_m            (no "s")
#            (j_m, j_msigcom, h_m, h_msigcom, designation identical)
#   AllWISE: w1sigmpro ← w1mpro_error; w2sigmpro ← w2mpro_error
#            (w1mpro, w2mpro, designation identical)
# AIP exposes the 2MASS PSC rows keyed on ``designation`` directly; the
# ``tmass_psc_xsc_join`` detour is avoided entirely.

# NOTE on AIP query shape (2026-04-20):
#
# The natural two-LEFT-JOIN form (upload → best_neighbour → catalogs.tmass)
# was the previous template here and succeeded on 2026-04-19 for the 164 k
# existing-Stream-3 IR fetch. It stopped planning on AIP some time before
# 2026-04-20 16:00 UTC — every async job ends in phase ``ERROR`` with a
# bare ``DALQueryError: Query Error: <No useful error from server>``
# (verified via direct submissions at 2026-04-20 17:10–17:40 UTC). The
# underlying tables are fine (direct ``WHERE designation = '…'`` queries
# on ``catalogs.tmass`` / ``catalogs.allwise`` still return rows), and the
# best-neighbour LEFT JOIN alone still works — AIP's planner just can't
# handle the cascaded upload→best_neighbour→catalogs pattern today.
#
# The form that does work is a materialised subquery (upload JOIN
# best_neighbour) followed by an INNER JOIN to ``catalogs.tmass`` /
# ``catalogs.allwise``. INNER JOIN drops no-match ids, but
# :func:`_finalise_tmass` / :func:`_finalise_allwise` already left-merge
# the TAP result onto the caller's ``expected_ids`` list and fill
# non-matches with NaN/null — so the caller-visible semantics are
# identical to the old LEFT-JOIN form. Empirically this recovers the full
# counterpart rate observed on 2026-04-19 (99.5 % 2MASS, 100 % AllWISE on
# RGB stars).

TMASS_ADQL_UPLOAD_AIP: Final[str] = """\
SELECT
    u.source_id,
    t.j_m            AS j_mag,
    t.j_msigcom      AS e_j_mag,
    t.h_m            AS h_mag,
    t.h_msigcom      AS e_h_mag,
    t.k_m            AS k_mag,
    t.k_msigcom      AS e_k_mag,
    t.designation    AS tmass_source_id,
    u.angular_distance AS tmass_angular_distance,
    u.xm_flag        AS tmass_xm_quality_flag
FROM (
    SELECT u2.source_id,
           bn.original_ext_source_id AS dsg,
           bn.angular_distance,
           bn.xm_flag
    FROM tap_upload.ids AS u2
    JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS bn
        ON bn.source_id = u2.source_id
) AS u
JOIN catalogs.tmass AS t ON t.designation = u.dsg
"""

ALLWISE_ADQL_UPLOAD_AIP: Final[str] = """\
SELECT
    u.source_id,
    a.w1mpro         AS w1_mag,
    a.w1sigmpro      AS e_w1_mag,
    a.w2mpro         AS w2_mag,
    a.w2sigmpro      AS e_w2_mag,
    a.designation    AS allwise_source_id,
    u.angular_distance AS allwise_angular_distance,
    u.xm_flag        AS allwise_xm_quality_flag
FROM (
    SELECT u2.source_id,
           bn.original_ext_source_id AS dsg,
           bn.angular_distance,
           bn.xm_flag
    FROM tap_upload.ids AS u2
    JOIN gaiadr3.allwise_best_neighbour AS bn
        ON bn.source_id = u2.source_id
) AS u
JOIN catalogs.allwise AS a ON a.designation = u.dsg
"""

# Backwards-compatible default: the ESA variant is the "canonical" ADQL that
# the docs reference. Test assertions on module-level constants read these.
TMASS_ADQL_UPLOAD: Final[str] = TMASS_ADQL_UPLOAD_ESA
ALLWISE_ADQL_UPLOAD: Final[str] = ALLWISE_ADQL_UPLOAD_ESA


def _adql_for_service(service_flavor: str) -> tuple[str, str]:
    """Return (tmass_adql, allwise_adql) for the given service flavour."""
    sf = service_flavor.lower()
    if sf in {"esa", "gaia_esa", "esac"}:
        return TMASS_ADQL_UPLOAD_ESA, ALLWISE_ADQL_UPLOAD_ESA
    if sf in {"aip", "gaia_aip"}:
        return TMASS_ADQL_UPLOAD_AIP, ALLWISE_ADQL_UPLOAD_AIP
    raise ValueError(f"unknown service_flavor {service_flavor!r}; expected 'esa' or 'aip'")


# -----------------------------------------------------------------------------
# per-catalogue fetches
# -----------------------------------------------------------------------------


def crossmatch_2mass(  # noqa: PLR0913 — keyword-only knobs with safe defaults
    service: TAPService,
    source_ids: Iterable[int],
    *,
    batch_size: int = TMASS_BATCH_SIZE,
    checkpoint_dir: Path | str | None = None,
    checkpoint_prefix: str = "tmass_batch",
    timeout_sec: float | None = DEFAULT_ASYNC_TIMEOUT_SEC,
    queue: str | None = None,
    runid: str | None = None,
    service_flavor: str = "esa",
) -> pl.DataFrame:
    """Cross-match ``source_ids`` against 2MASS PSC via Gaia's best-neighbour table.

    One row per input ``source_id``. Stars without a 2MASS counterpart return
    NaN magnitudes, null ``tmass_source_id``, and NaN ``tmass_angular_distance``.

    Parameters
    ----------
    service
        TAP service (usually :func:`arqueogal.data.tap.esa_service` or
        :func:`arqueogal.data.tap.aip_service`).
    source_ids
        Iterable of int64 Gaia DR3 source_ids. Duplicates are kept as-is;
        dedupe upstream.
    batch_size
        IDs per upload batch. 10 000 is safe on AIP and ESA.
    checkpoint_dir, checkpoint_prefix
        Same resumable semantics as :func:`batched_upload_fetch_df`.
    timeout_sec, queue, runid
        Forwarded to ``submit_job`` / ``job.wait``.
    service_flavor
        Selects the ADQL schema variant. ``"esa"`` (default) joins against
        ``gaiadr1.tmass_original_valid``; ``"aip"`` joins against
        ``catalogs.tmass`` instead (AIP does not host the gaiadr1 schema).

    Returns
    -------
    pl.DataFrame
        Columns: ``source_id, j_mag, e_j_mag, h_mag, e_h_mag, k_mag, e_k_mag,
        tmass_source_id, tmass_angular_distance, tmass_xm_quality_flag``.
        Mags and angular distance are float32; ``source_id`` int64;
        ``tmass_source_id`` string (nullable); ``tmass_xm_quality_flag``
        int8 (nullable).
    """
    ids = [int(x) for x in source_ids]
    if not ids:
        return _empty_polars(TMASS_SCHEMA)

    tmass_adql, _ = _adql_for_service(service_flavor)
    pdf = batched_upload_fetch_df(
        service,
        ids,
        tmass_adql,
        upload_name="ids",
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix=checkpoint_prefix,
        timeout_sec=timeout_sec,
        queue=queue,
        runid=runid,
    )
    return _finalise_tmass(pdf, expected_ids=ids)


def crossmatch_allwise(  # noqa: PLR0913 — keyword-only knobs with safe defaults
    service: TAPService,
    source_ids: Iterable[int],
    *,
    batch_size: int = ALLWISE_BATCH_SIZE,
    checkpoint_dir: Path | str | None = None,
    checkpoint_prefix: str = "allwise_batch",
    timeout_sec: float | None = DEFAULT_ASYNC_TIMEOUT_SEC,
    queue: str | None = None,
    runid: str | None = None,
    service_flavor: str = "esa",
) -> pl.DataFrame:
    """Cross-match ``source_ids`` against AllWISE via Gaia's best-neighbour table.

    One row per input ``source_id``. Stars without an AllWISE counterpart
    return NaN magnitudes, null ``allwise_source_id``, and NaN
    ``allwise_angular_distance``.

    Columns: ``source_id, w1_mag, e_w1_mag, w2_mag, e_w2_mag,
    allwise_source_id, allwise_angular_distance, allwise_xm_quality_flag``.

    See :func:`crossmatch_2mass` for the ``service_flavor`` semantics.
    """
    ids = [int(x) for x in source_ids]
    if not ids:
        return _empty_polars(ALLWISE_SCHEMA)

    _, allwise_adql = _adql_for_service(service_flavor)
    pdf = batched_upload_fetch_df(
        service,
        ids,
        allwise_adql,
        upload_name="ids",
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix=checkpoint_prefix,
        timeout_sec=timeout_sec,
        queue=queue,
        runid=runid,
    )
    return _finalise_allwise(pdf, expected_ids=ids)


# -----------------------------------------------------------------------------
# combined assembly
# -----------------------------------------------------------------------------


def assemble_ir_photometry(  # noqa: PLR0913 — keyword-only knobs with safe defaults
    service: TAPService,
    source_ids: Iterable[int],
    *,
    batch_size: int = TMASS_BATCH_SIZE,
    checkpoint_dir: Path | str | None = None,
    timeout_sec: float | None = DEFAULT_ASYNC_TIMEOUT_SEC,
    queue: str | None = None,
    runid: str | None = None,
    service_flavor: str = "esa",
) -> pl.DataFrame:
    """Cross-match ``source_ids`` against 2MASS *and* AllWISE, return one combined frame.

    Adds ``ir_missing_flag`` — True if *any* of {j, h, k, w1, w2} is missing
    (NaN) for that star. Stars with no 2MASS neighbour *or* no AllWISE
    neighbour get the flag set.

    Parameters
    ----------
    service
        TAP service hosting the Gaia DR3 + DR1 tables (ESA or AIP).
    source_ids
        Iterable of int64 Gaia DR3 source_ids.
    batch_size
        Upload chunk size for both queries (default 10 000).
    checkpoint_dir
        If provided, 2MASS checkpoints land under
        ``<checkpoint_dir>/tmass/`` and AllWISE under ``<checkpoint_dir>/allwise/``,
        so reruns skip completed batches independently per catalogue.

    Returns
    -------
    pl.DataFrame
        Columns are the union of :data:`TMASS_SCHEMA` ∪ :data:`ALLWISE_SCHEMA`
        (source_id appears once) plus ``ir_missing_flag`` (bool).
        Row count == len(unique source_ids).
    """
    ids = [int(x) for x in source_ids]
    ckpt = Path(checkpoint_dir) if checkpoint_dir is not None else None
    tmass_ckpt = ckpt / "tmass" if ckpt is not None else None
    allwise_ckpt = ckpt / "allwise" if ckpt is not None else None

    logger.info(
        "assemble_ir_photometry: %d source_ids → 2MASS + AllWISE (flavour=%s)",
        len(ids),
        service_flavor,
    )
    t2 = crossmatch_2mass(
        service,
        ids,
        batch_size=batch_size,
        checkpoint_dir=tmass_ckpt,
        timeout_sec=timeout_sec,
        queue=queue,
        runid=runid,
        service_flavor=service_flavor,
    )
    aw = crossmatch_allwise(
        service,
        ids,
        batch_size=batch_size,
        checkpoint_dir=allwise_ckpt,
        timeout_sec=timeout_sec,
        queue=queue,
        runid=runid,
        service_flavor=service_flavor,
    )

    # Outer join on source_id so a star missing in either table is preserved.
    combined = t2.join(aw, on="source_id", how="full", coalesce=True)

    # ir_missing_flag: True when ANY of the 5 IR magnitudes is missing (NaN).
    combined = combined.with_columns(
        (
            pl.col("j_mag").is_null()
            | pl.col("h_mag").is_null()
            | pl.col("k_mag").is_null()
            | pl.col("w1_mag").is_null()
            | pl.col("w2_mag").is_null()
            | pl.col("j_mag").is_nan()
            | pl.col("h_mag").is_nan()
            | pl.col("k_mag").is_nan()
            | pl.col("w1_mag").is_nan()
            | pl.col("w2_mag").is_nan()
        ).alias("ir_missing_flag")
    )
    return combined


# -----------------------------------------------------------------------------
# finalisation helpers (pandas-in, polars-out; isolated for testability)
# -----------------------------------------------------------------------------


def _finalise_tmass(pdf: pd.DataFrame, *, expected_ids: list[int]) -> pl.DataFrame:
    """Normalise a pandas frame from the 2MASS TAP UPLOAD job into polars.

    - Ensures all :data:`TMASS_SCHEMA` columns are present (creates missing
      ones filled with NaN / null).
    - Casts magnitudes and angular distance to float32 for disk budget.
    - Casts ``source_id`` to int64, ``tmass_xm_quality_flag`` to Int8.
    - Reindexes to ``expected_ids`` so every requested source_id appears
      exactly once — safety net against TAP silently dropping unmatched
      IDs despite the LEFT JOIN (observed on some ADQL implementations).
    """
    return _finalise_generic(
        pdf,
        expected_ids=expected_ids,
        schema=TMASS_SCHEMA,
        float32_cols=(
            "j_mag",
            "e_j_mag",
            "h_mag",
            "e_h_mag",
            "k_mag",
            "e_k_mag",
            "tmass_angular_distance",
        ),
        string_cols=("tmass_source_id",),
        int8_cols=("tmass_xm_quality_flag",),
    )


def _finalise_allwise(pdf: pd.DataFrame, *, expected_ids: list[int]) -> pl.DataFrame:
    """AllWISE analogue of :func:`_finalise_tmass`."""
    return _finalise_generic(
        pdf,
        expected_ids=expected_ids,
        schema=ALLWISE_SCHEMA,
        float32_cols=(
            "w1_mag",
            "e_w1_mag",
            "w2_mag",
            "e_w2_mag",
            "allwise_angular_distance",
        ),
        string_cols=("allwise_source_id",),
        int8_cols=("allwise_xm_quality_flag",),
    )


def _finalise_generic(  # noqa: PLR0913 — keyword-only, one arg per schema knob
    pdf: pd.DataFrame,
    *,
    expected_ids: list[int],
    schema: tuple[str, ...],
    float32_cols: tuple[str, ...],
    string_cols: tuple[str, ...],
    int8_cols: tuple[str, ...],
) -> pl.DataFrame:
    # Rebuild expected schema: create missing cols as NaN / None, drop
    # extras, cast dtypes. Reindex to expected_ids to guarantee one row
    # per input source_id.
    if pdf is None or len(pdf) == 0:
        # Empty frame: build a NaN/null row for every expected id.
        out = pd.DataFrame({"source_id": np.asarray(expected_ids, dtype=np.int64)})
        for col in schema:
            if col == "source_id":
                continue
            out[col] = np.nan if col in float32_cols else pd.NA
    else:
        out = pdf.copy()
        if "source_id" not in out.columns:
            raise ValueError("TAP result missing 'source_id' column")
        out["source_id"] = out["source_id"].astype(np.int64)
        for col in schema:
            if col not in out.columns:
                out[col] = np.nan if col in float32_cols else pd.NA

        # Deduplicate — best_neighbour is 1:1 by construction, but a
        # conservative drop_duplicates guards against accidental
        # multi-match rows if the Gaia mirror behaves unexpectedly.
        out = out.drop_duplicates(subset=["source_id"], keep="first")

        # Left-merge onto the full expected id list so non-matches
        # resurface as NaN rows. This is defensive vs. TAP backends that
        # silently drop LEFT-JOIN-no-match rows under upload joins.
        anchor = pd.DataFrame({"source_id": np.asarray(expected_ids, dtype=np.int64)})
        out = anchor.merge(out, on="source_id", how="left", validate="one_to_one")

    # Dtype casts on the pandas frame before handing to polars. Doing this
    # in pandas is cheaper than repeated polars cast() calls, and avoids
    # pyarrow conversion issues with mixed-type object columns.
    for col in float32_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float32)
    for col in string_cols:
        # pandas → polars: leave as object dtype; polars infers Utf8.
        out[col] = out[col].astype("object").where(out[col].notna(), None)
    for col in int8_cols:
        # Nullable integer so we can keep NaN for missing counterparts.
        out[col] = pd.array(pd.to_numeric(out[col], errors="coerce"), dtype="Int8")

    # Project to the declared schema order.
    out = out[list(schema)]

    pdf_final = out.reset_index(drop=True)
    return pl.from_pandas(pdf_final)


def _empty_polars(schema: tuple[str, ...]) -> pl.DataFrame:
    """Build an empty polars frame with the right column names and sensible dtypes."""
    dtypes: dict[str, pl.DataType] = {}
    for col in schema:
        if col == "source_id":
            dtypes[col] = pl.Int64
        elif col.endswith("_source_id"):
            dtypes[col] = pl.Utf8
        elif col.endswith("_xm_quality_flag"):
            dtypes[col] = pl.Int8
        else:
            dtypes[col] = pl.Float32
    return pl.DataFrame(schema=dtypes)
