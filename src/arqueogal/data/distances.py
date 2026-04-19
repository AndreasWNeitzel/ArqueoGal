"""Distance catalogues for Pipeline 1/2 — Bailer-Jones+2021 and StarHorse2 v2.

§7 of data_acquisition.md. Two independent TAP queries followed by a
reconciliation step:

- ``fetch_bailerjones`` → GAVO ``gedr3dist.main`` (primary photogeometric).
- ``fetch_starhorse2`` → AIP ``gaiadr3_contrib.aqueiroz2023_apogee_dr17_v2``
  (spectroscopic-informed cross-check for the Stream 1 training overlap).
- ``merge_distances`` → joins the two on ``source_id``, flags factor-of-2
  disagreements per §7.3.

Only photogeometric Bailer-Jones distances are treated as primary — §7.1
cites Bailer-Jones+2021 Fig. 10 for why geometric alone underperforms at
parallax S/N < 10.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd
from pyvo.dal.tap import TAPService

from arqueogal.data.tap import (
    BATCH_PLACEHOLDER,
    DEFAULT_ASYNC_TIMEOUT_SEC,
    batched_fetch_df,
)

logger = logging.getLogger(__name__)

BAILERJONES_TABLE: Final[str] = "gedr3dist.main"
BAILERJONES_ADQL: Final[str] = f"""\
SELECT source_id,
       r_med_geo, r_lo_geo, r_hi_geo,
       r_med_photogeo, r_lo_photogeo, r_hi_photogeo,
       flag
FROM {BAILERJONES_TABLE}
WHERE source_id IN ({BATCH_PLACEHOLDER})
"""

STARHORSE2_SAMPLE_TABLES: Final[dict[str, str]] = {
    "apogee_dr17": "gaiadr3_contrib.aqueiroz2023_apogee_dr17_v2",
    "gaia_rvs": "gaiadr3_contrib.aqueiroz2023_gaia_rvs_v2",
}
"""§7.2: only v2 files are correct. v1 had a biased piecewise age prior."""

STARHORSE2_ADQL_TEMPLATE: Final[str] = """\
SELECT source_id,
       dist16, dist50, dist84,
       av16, av50, av84,
       mass16, mass50, mass84,
       age16, age50, age84,
       starhorse_outputflag, starhorse_ageflag
FROM {table}
WHERE source_id IN ({placeholder})
"""

DIST_CONFLICT_LOG_THRESHOLD: Final[float] = 0.3
"""§7.3: |log10(r_BJ / d_SH2)| > 0.3 → flag as ``dist_conflict`` (factor of 2)."""


def fetch_bailerjones(  # noqa: PLR0913 — keyword-only tuning knobs with safe defaults
    service: TAPService,
    source_ids: Iterable[int],
    *,
    batch_size: int = 10_000,
    mode: Literal["async", "sync", "auto"] = "async",
    checkpoint_dir: Path | str | None = None,
    adql: str = BAILERJONES_ADQL,
    timeout_sec: float | None = DEFAULT_ASYNC_TIMEOUT_SEC,
) -> pd.DataFrame:
    """Fetch Bailer-Jones+2021 photogeometric distances from GAVO.

    ``service`` must point at GAVO (use :func:`gavo_service`). §7.1 — per-
    star cost is small, so the default batch_size matches the general Gaia
    enrichment size.
    """
    return batched_fetch_df(
        service,
        source_ids,
        adql,
        batch_size=batch_size,
        mode=mode,
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix="bj_batch",
        timeout_sec=timeout_sec,
    )


def fetch_starhorse2(  # noqa: PLR0913 — keyword-only tuning knobs with safe defaults
    service: TAPService,
    source_ids: Iterable[int],
    *,
    sample: str = "apogee_dr17",
    batch_size: int = 10_000,
    mode: Literal["async", "sync", "auto"] = "async",
    checkpoint_dir: Path | str | None = None,
    timeout_sec: float | None = DEFAULT_ASYNC_TIMEOUT_SEC,
) -> pd.DataFrame:
    """Fetch StarHorse2 v2 posteriors from AIP.

    ``service`` must point at AIP (use :func:`aip_service`). ``sample``
    selects the parent spectroscopic survey — only ``"apogee_dr17"`` and
    ``"gaia_rvs"`` are wired up here (§7.2). The ADQL is built from
    :data:`STARHORSE2_ADQL_TEMPLATE` + the sample-specific table name.
    """
    if sample not in STARHORSE2_SAMPLE_TABLES:
        raise ValueError(
            f"unknown StarHorse2 sample {sample!r}; "
            f"supported: {sorted(STARHORSE2_SAMPLE_TABLES)}"
        )
    adql = STARHORSE2_ADQL_TEMPLATE.format(
        table=STARHORSE2_SAMPLE_TABLES[sample], placeholder=BATCH_PLACEHOLDER
    )
    return batched_fetch_df(
        service,
        source_ids,
        adql,
        batch_size=batch_size,
        mode=mode,
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix=f"sh2_{sample}_batch",
        timeout_sec=timeout_sec,
    )


def merge_distances(
    bailerjones: pd.DataFrame,
    starhorse2: pd.DataFrame | None = None,
    *,
    log_threshold: float = DIST_CONFLICT_LOG_THRESHOLD,
) -> pd.DataFrame:
    """Join BJ21 + SH2 on source_id; flag factor-of-2 disagreements.

    Output columns:

    - All BJ21 columns (``r_med_photogeo`` is the primary distance).
    - ``dist_primary_pc`` — convenience alias, equal to ``r_med_photogeo``.
    - ``dist_sigma_sym_pc`` — ``(r_hi_photogeo - r_lo_photogeo) / 2``, the
      symmetric σ approximation (§7.1).
    - If ``starhorse2`` is given: its ``dist50`` plus ``dist_conflict``
      (bool), ``True`` where
      ``|log10(r_med_photogeo / sh2.dist50)| > log_threshold`` and both
      distances are positive / finite.

    ``starhorse2 is None`` is a legitimate call path for Streams without
    SH2 overlap (e.g. Stream 3 field stars) — returned frame has no
    SH2-derived columns and no ``dist_conflict`` column.
    """
    if "r_med_photogeo" not in bailerjones.columns:
        raise KeyError("bailerjones frame missing 'r_med_photogeo'")
    out = bailerjones.copy()
    out["dist_primary_pc"] = out["r_med_photogeo"].astype(float)
    out["dist_sigma_sym_pc"] = (
        (out["r_hi_photogeo"].astype(float) - out["r_lo_photogeo"].astype(float)) / 2.0
    )

    if starhorse2 is None or starhorse2.empty:
        return out

    if "source_id" not in starhorse2.columns or "dist50" not in starhorse2.columns:
        raise KeyError("starhorse2 frame missing 'source_id' or 'dist50'")

    sh2 = starhorse2[["source_id", "dist50"]].rename(columns={"dist50": "dist_sh2_pc"})
    merged = out.merge(sh2, on="source_id", how="left")

    ratio = _safe_log10_ratio(
        merged["dist_primary_pc"].to_numpy(dtype=float),
        merged["dist_sh2_pc"].to_numpy(dtype=float),
    )
    merged["dist_conflict"] = np.abs(ratio) > log_threshold
    n_conflict = int(merged["dist_conflict"].sum())
    if n_conflict:
        logger.info(
            "%d/%d stars flagged dist_conflict (|Δlog10| > %.2f)",
            n_conflict, len(merged), log_threshold,
        )
    return merged


# -----------------------------------------------------------------------------
# internals
# -----------------------------------------------------------------------------


def _safe_log10_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``log10(a / b)`` element-wise, NaN where either input is ≤ 0 or NaN."""
    out = np.full_like(a, np.nan, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    out[mask] = np.log10(a[mask] / b[mask])
    return out


__all__ = [
    "BAILERJONES_ADQL",
    "BAILERJONES_TABLE",
    "DIST_CONFLICT_LOG_THRESHOLD",
    "STARHORSE2_ADQL_TEMPLATE",
    "STARHORSE2_SAMPLE_TABLES",
    "fetch_bailerjones",
    "fetch_starhorse2",
    "merge_distances",
]
