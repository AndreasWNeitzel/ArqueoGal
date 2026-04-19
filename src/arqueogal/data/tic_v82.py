"""TIC v8.2 → Gaia DR2 lookup via VizieR — §4.3 Step 1.

The TESS Input Catalog (Paegert+2021; Stassun+2019) is the DR2-era bridge
between Hon+2021 ``TIC`` ids and Gaia. VizieR mirrors TIC v8.2 as table
``IV/39/tic82``. For each TIC in our Hon+2021 high-confidence cut we pull
back the DR2 ``GAIA`` source_id (and a few coordinates/magnitudes for
sanity checks) so the DR2→DR3 cross-match in
:mod:`arqueogal.data.crossmatch` can resolve each star to a DR3 id.

This module is **just the TAP fetch** — the DR2→DR3 mapping and §4.3 cuts
live in :mod:`arqueogal.data.crossmatch`.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd
from pyvo.dal.tap import TAPService

from arqueogal.data.tap import BATCH_PLACEHOLDER, DEFAULT_ASYNC_TIMEOUT_SEC, batched_fetch_df

logger = logging.getLogger(__name__)

TIC_V82_VIZIER_TABLE: Final[str] = '"IV/39/tic82"'
"""§4.3: VizieR catalogue identifier for the TESS Input Catalog v8.2."""

TIC_V82_COLUMNS: Final[tuple[str, ...]] = (
    "TIC", "GAIA", "RAJ2000", "DEJ2000", "Tmag", "plx",
)
"""§4.3: minimal set we ingest. ``GAIA`` is the Gaia DR2 source_id."""

TIC_V82_ADQL = f"""
SELECT {", ".join(TIC_V82_COLUMNS)}
FROM {TIC_V82_VIZIER_TABLE}
WHERE TIC IN ({BATCH_PLACEHOLDER})
"""
"""Parameterised over a comma-joined batch of TIC ids."""


def fetch_tic_v82(  # noqa: PLR0913 — keyword-only tuning knobs with safe defaults
    service: TAPService,
    tic_ids: list[int] | np.ndarray,
    *,
    batch_size: int = 5_000,
    checkpoint_dir=None,  # noqa: ANN001 — Path | None
    mode: str = "async",
    timeout_sec: int = DEFAULT_ASYNC_TIMEOUT_SEC,
) -> pd.DataFrame:
    """Fetch TIC v8.2 rows for the given TIC ids from VizieR.

    Parameters
    ----------
    service
        VizieR TAP service (use :func:`arqueogal.data.tap.vizier_service`).
    tic_ids
        Iterable of integer TIC ids (typically from
        :func:`arqueogal.data.tess_hon2021.fetch_hon2021`).
    batch_size
        ``IN (...)`` chunk size. Kept modest (5 000) because VizieR's sync
        limit is tighter than AIP's.
    checkpoint_dir
        If set, each batch is written to ``tic_v82_{NNNN}.parquet`` for
        resumable ingestion.
    mode, timeout_sec
        Forwarded to :func:`batched_fetch_df`.

    Returns
    -------
    pd.DataFrame
        Columns per :data:`TIC_V82_COLUMNS`. ``GAIA`` is the DR2 source_id;
        rows without a DR2 counterpart (``GAIA IS NULL``) are returned as
        ``NaN`` and must be dropped before cross-matching.
    """
    logger.info("fetching TIC v8.2 rows for %d TIC ids", len(list(tic_ids)))
    return batched_fetch_df(
        service,
        tic_ids,
        TIC_V82_ADQL,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix="tic_v82",
        mode=mode,
        timeout_sec=timeout_sec,
    )


__all__ = [
    "TIC_V82_ADQL",
    "TIC_V82_COLUMNS",
    "TIC_V82_VIZIER_TABLE",
    "fetch_tic_v82",
]
