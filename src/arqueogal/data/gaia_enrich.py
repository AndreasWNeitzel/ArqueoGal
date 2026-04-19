"""Gaia DR3 enrichment via AIP TAP — fetches astrometry, photometry, and
GSP-Phot / GSP-Spec astrophysical parameters for a pre-selected list of
``source_id``\\ s.

Shared by all three data streams:
- Stream 1: post-cut APOGEE DR19 source_ids → Gaia DR3 enrichment.
- Stream 2: TESS Hon+2021 × TIC → DR2→DR3 cross-match → Gaia DR3 enrichment.
- Stream 3: stratified Andrae+2023 RGB source_ids → Gaia DR3 enrichment.

Does **not** fetch XP coefficients — those live in ``gaia_xp.py`` (§6) and
use a different TAP table + smaller batch size.

The query template is pinned to :data:`ENRICHMENT_ADQL` for reproducibility.
data_acquisition.md §3.6.

Usage
-----
.. code-block:: python

    from arqueogal.data.tap import aip_service
    from arqueogal.data.gaia_enrich import enrich_source_ids

    aip = aip_service()
    df = enrich_source_ids(
        aip,
        source_ids=post_cut_dr19["source_id"],
        checkpoint_dir="data/interim/enrich_batches/stream1/",
    )
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import pandas as pd
from pyvo.dal.tap import TAPService

from arqueogal.data.tap import (
    DEFAULT_ASYNC_TIMEOUT_SEC,
    batched_fetch_df,
)

_ENRICHMENT_SELECT = """\
SELECT
    g.source_id, g.ra, g.dec,
    g.parallax, g.parallax_error,
    g.pmra, g.pmra_error, g.pmdec, g.pmdec_error,
    g.ra_dec_corr, g.ra_parallax_corr, g.ra_pmra_corr, g.ra_pmdec_corr,
    g.dec_parallax_corr, g.dec_pmra_corr, g.dec_pmdec_corr,
    g.parallax_pmra_corr, g.parallax_pmdec_corr, g.pmra_pmdec_corr,
    g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag,
    g.phot_g_mean_flux_over_error, g.bp_rp, g.bp_g, g.g_rp,
    g.ruwe, g.visibility_periods_used, g.astrometric_excess_noise,
    g.astrometric_params_solved, g.ipd_gof_harmonic_amplitude,
    g.ipd_frac_multi_peak,
    g.has_xp_continuous, g.has_rvs,
    g.radial_velocity, g.radial_velocity_error,
    g.nu_eff_used_in_astrometry, g.pseudocolour,
    g.ecl_lat,
    ap.teff_gspphot, ap.teff_gspphot_lower, ap.teff_gspphot_upper,
    ap.logg_gspphot, ap.logg_gspphot_lower, ap.logg_gspphot_upper,
    ap.mh_gspphot,   ap.mh_gspphot_lower,   ap.mh_gspphot_upper,
    ap.ag_gspphot,   ap.ag_gspphot_lower,   ap.ag_gspphot_upper,
    ap.ebpminrp_gspphot,
    ap.distance_gspphot, ap.distance_gspphot_lower, ap.distance_gspphot_upper,
    ap.teff_gspspec, ap.logg_gspspec, ap.mh_gspspec, ap.alphafe_gspspec,
    ap.flags_gspspec"""

ENRICHMENT_ADQL = f"""\
{_ENRICHMENT_SELECT}
FROM gaiadr3.gaia_source AS g
LEFT JOIN gaiadr3.astrophysical_parameters AS ap ON ap.source_id = g.source_id
WHERE g.source_id IN (__batch__)
"""

ENRICHMENT_ADQL_UPLOAD = f"""\
{_ENRICHMENT_SELECT}
FROM gaiadr3.gaia_source AS g
JOIN tap_upload.ids AS u ON g.source_id = u.source_id
LEFT JOIN gaiadr3.astrophysical_parameters AS ap ON ap.source_id = g.source_id
"""
"""TAP UPLOAD variant of :data:`ENRICHMENT_ADQL` — use with
:func:`arqueogal.data.tap.batched_upload_fetch_df`. AIP's gateway 504s on
inline ``IN (...)`` payloads above ~100 KB (≈5 k IDs); UPLOAD sidesteps that.
Note explicit ``LEFT JOIN ... ON`` — AIP's ADQL parser rejects ``USING``."""


def enrich_source_ids(  # noqa: PLR0913 — keyword-only tuning knobs with safe defaults
    service: TAPService,
    source_ids: Iterable[int],
    *,
    batch_size: int = 10_000,
    mode: Literal["async", "sync", "auto"] = "async",
    checkpoint_dir: Path | str | None = None,
    adql: str = ENRICHMENT_ADQL,
    timeout_sec: float | None = DEFAULT_ASYNC_TIMEOUT_SEC,
    queue: str | None = None,
    runid: str | None = None,
) -> pd.DataFrame:
    """Fetch Gaia DR3 enrichment for ``source_ids``; return one DataFrame.

    Parameters
    ----------
    service
        Authenticated TAP service (usually ``aip_service()``).
    source_ids
        Iterable of int64 Gaia DR3 source_ids. Duplicates are kept as-is —
        dedupe upstream if needed.
    batch_size
        Chunk size for the ``IN (__batch__)`` query. 10 000 is the §3.6
        default; drop to 5 000 for XP queries (use ``gaia_xp.py`` instead).
    mode
        ``"async"`` (default, required above 5 000 rows — sync TAP times
        out on AIP at ~90 s), ``"sync"`` for small tests, ``"auto"`` picks
        async if ``batch_size > SYNC_ROW_THRESHOLD``.
    checkpoint_dir
        When set, each batch is written to ``batch_NNNN.parquet`` inside this
        directory. On rerun, existing batch files are loaded and the TAP
        query is skipped — resumability for multi-hour ingestions.
    adql
        Query template. Must contain exactly one ``__batch__`` placeholder.

    Returns
    -------
    pd.DataFrame
        Concatenated results. Empty input → empty DataFrame.
    """
    return batched_fetch_df(
        service,
        source_ids,
        adql,
        batch_size=batch_size,
        mode=mode,
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix="batch",
        timeout_sec=timeout_sec,
        queue=queue,
        runid=runid,
    )
