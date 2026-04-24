"""Stream 2 — Hon+2021 TESS ν_max catalogue via VizieR — §4.1.

Fetches the 158 505-row red-giant ν_max catalogue (Hon, Huber, Kuszlewicz
et al. 2021, ApJ 919, 131) from VizieR TAP (J/ApJ/919/131). Applies the
authors' recommended ``Prob > 0.95`` high-confidence cut (§4.1: yields
~120 k stars).

Downstream chain: the catalogue's ``TIC`` column → a separate TIC v8.2
lookup for ``GAIA`` (DR2 id) → :mod:`arqueogal.data.crossmatch` resolves
to DR3. Only the Hon+2021 fetch lives here.

This module only contains the VizieR ADQL and a thin :func:`fetch_hon2021`
wrapper around :func:`run_async`/:func:`run_sync`; no local file download
helper, because VizieR returns the full ~120 k rows in a single async
call comfortably.
"""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd
from pyvo.dal.tap import TAPService

from arqueogal.data import tap as tap_mod
from arqueogal.data.tap import DEFAULT_ASYNC_TIMEOUT_SEC, SYNC_ROW_THRESHOLD

logger = logging.getLogger(__name__)

HON2021_VIZIER_TABLE: Final[str] = '"J/ApJ/919/131/table1"'
"""§4.1: VizieR catalogue identifier for Hon+2021 Table 1."""

HON2021_DEFAULT_PROB_THRESHOLD: Final[float] = 0.95
"""§4.1: authors' recommended high-confidence detection threshold."""

HON2021_COLUMNS: Final[tuple[str, ...]] = (
    "TIC",
    "RAJ2000",
    "DEJ2000",
    "numax",
    "e_numax",
    "Teff",
    "R",
    "Prob",
)
"""§4.1: minimal column set we ingest. ``R`` is radius in R_sun. ``Sector``
flags are excluded — they're multi-column and not needed for Task 4 yet."""


def build_hon2021_adql(prob_threshold: float = HON2021_DEFAULT_PROB_THRESHOLD) -> str:
    """Return the ADQL for the Hon+2021 fetch with a ``Prob`` cut."""
    if not 0.0 <= prob_threshold <= 1.0:
        raise ValueError(f"prob_threshold must be in [0, 1]; got {prob_threshold}")
    cols = ", ".join(HON2021_COLUMNS)
    return f"SELECT {cols} FROM {HON2021_VIZIER_TABLE} WHERE Prob > {prob_threshold}"


def fetch_hon2021(
    service: TAPService,
    *,
    prob_threshold: float = HON2021_DEFAULT_PROB_THRESHOLD,
    mode: str = "async",
    timeout_sec: float | None = DEFAULT_ASYNC_TIMEOUT_SEC,
) -> pd.DataFrame:
    """Fetch the Hon+2021 ν_max catalogue from VizieR.

    Parameters
    ----------
    service
        VizieR TAP service (use :func:`arqueogal.data.tap.vizier_service`).
    prob_threshold
        Strictly greater-than threshold on ``Prob``. Default 0.95 per §4.1.
    mode
        ``"async"`` (default, safe for 120 k rows) or ``"sync"``. §4.1 notes
        the full cut comfortably fits an async VizieR job. Sync is available
        for testing or restricted-subset queries.
    timeout_sec
        Async job timeout in seconds.

    Returns
    -------
    pd.DataFrame
        Columns per :data:`HON2021_COLUMNS`, one row per high-confidence
        oscillating giant.
    """
    adql = build_hon2021_adql(prob_threshold=prob_threshold)
    logger.info("fetching Hon+2021 from VizieR with Prob > %s", prob_threshold)
    if mode == "sync":
        table = tap_mod.run_sync(service, adql)
    elif mode == "async":
        table = tap_mod.run_async(service, adql, timeout_sec=timeout_sec)
    else:
        raise ValueError(f"mode must be 'async' or 'sync'; got {mode!r}")
    df = table.to_pandas()
    if len(df) > SYNC_ROW_THRESHOLD and mode == "sync":
        logger.warning(
            "Hon+2021 sync query returned %d rows (> %d) — consider mode='async'",
            len(df),
            SYNC_ROW_THRESHOLD,
        )
    logger.info("Hon+2021: %d rows after Prob > %s cut", len(df), prob_threshold)
    return df


__all__ = [
    "HON2021_COLUMNS",
    "HON2021_DEFAULT_PROB_THRESHOLD",
    "HON2021_VIZIER_TABLE",
    "build_hon2021_adql",
    "fetch_hon2021",
]
