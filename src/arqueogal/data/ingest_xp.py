"""Gaia DR3 XP extraction + Ye+2024 flux-correction orchestrator — §11 Level 3.

Ties the single-purpose primitives in :mod:`arqueogal.data.gaia_xp` into the
§6.3 + §6.4-step-1 pipeline:

    source_id list + coord frame (from Stream 1 ∪ Stream 3 enrichment)
        → AIP ``gaiadr3.xp_continuous_mean_spectrum`` (async TAP, checkpointed)
        → §6.5 pre-correction sanity check on raw Hermite coeffs
        → §6.4 step 1: Ye+2024 NN flux-correction (gaiaxpy.calibrate →
          CCM89 + SFD deredden → 14-feature NN → dereddened corrected flux
          on :data:`YE2024_SAMPLING_NM`)
        → write ``xp_sampled_corrected.parquet``
        → write ``xp_sampled_corrected.provenance.json``

Output schema is per-star *sampled corrected flux* (length-330 float32 array),
**not** Hermite coefficients. §6.4 steps 2–5 (normalise by c_0, log+zscore
c_0, error propagation, float32) operate on Hermite coefficients and are
therefore no longer chained into this orchestrator: Ye+2024 transforms the
data into sampled space, and the downstream Hermite-basis ``normalise_xp`` /
``zscore_c0`` primitives no longer apply after this step. Pipeline-1 feature-
matrix code chooses whether to consume the sampled flux directly or re-
project onto the Hermite basis and then apply §6.4 steps 2–5 — see the
``apply_ye2024_correction`` docstring for the architectural note.

References
----------
data_acquisition.md §6, §11 (Level 3), §14 (provenance).
Ye et al. 2025 (A&A 695, A75; arXiv:2411.19105); Zenodo 10.5281/zenodo.14028588.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from pyvo.dal.tap import TAPService

from arqueogal.data.gaia_xp import (
    XP_BATCH_SIZE,
    XP_QUERY_ADQL,
    XP_TABLE,
    YE2024_FLAG_CALIBRATE_FAIL,
    YE2024_FLAG_NO_SYNTH_PHOT,
    YE2024_FLAG_OK,
    YE2024_SAMPLING_NM,
    apply_ye2024_correction,
    fetch_xp_coefficients,
    xp_sanity_check,
)
from arqueogal.data.provenance import Provenance, TapSource, write_sidecar
from arqueogal.data.tap import AIP_TAP_URL, DEFAULT_ASYNC_TIMEOUT_SEC, aip_service
from arqueogal.utils.io import save_parquet

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_FILENAME = "xp_sampled_corrected.parquet"


def ingest_xp(  # noqa: PLR0913 — keyword-only tuning knobs with safe defaults
    data_dir: Path | str,
    source_ids: Iterable[int],
    coords_df: pd.DataFrame,
    *,
    service: TAPService | None = None,
    batch_size: int = XP_BATCH_SIZE,
    model_dir: Path | str | None = None,
    device: str | None = None,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
    timeout_sec: float | None = DEFAULT_ASYNC_TIMEOUT_SEC,
) -> tuple[Path, dict[str, int]]:
    """Run the full XP fetch + Ye+2024 correction pipeline.

    Parameters
    ----------
    data_dir
        Repository ``data/`` root. Creates
        ``{data_dir}/interim/enrich_batches/xp/`` for TAP checkpoints and
        writes ``xp_sampled_corrected.parquet`` + provenance to
        ``{data_dir}/interim/``.
    source_ids
        Iterable of Gaia DR3 source_ids to fetch. Typically drawn from the
        Level-2 stream parquets filtered by ``has_xp_continuous == True``.
    coords_df
        DataFrame with columns ``(source_id, ra, dec)`` — ICRS degrees at
        Ep=2016.0. Required for Ye+2024 SFD A_V dereddening. Every fetched
        source_id should be present; any missing is dropped with a warning
        by :func:`apply_ye2024_correction`.
    service
        Authenticated AIP TAP service. ``None`` calls :func:`aip_service`
        lazily.
    batch_size
        ``IN (...)`` chunk size; default :data:`gaia_xp.XP_BATCH_SIZE`
        (~5 000 — larger times out on AIP for XP arrays).
    model_dir
        Directory holding Ye+2024 weights + scaler. ``None`` uses the
        vendored copy under ``data/external/ye2024/``.
    device
        ``"cuda"`` / ``"cpu"`` / ``None`` (auto).
    output_filename
        Override the default Parquet filename.
    timeout_sec
        Async TAP per-batch timeout.

    Returns
    -------
    tuple[Path, dict[str, int]]
        The output Parquet path and a summary of Ye+2024 flag counts
        ``{"n_ok", "n_no_synth_phot", "n_calibrate_fail"}`` for provenance
        cross-reference.
    """
    data_dir = Path(data_dir)
    interim_dir = data_dir / "interim"
    checkpoint_dir = interim_dir / "enrich_batches" / "xp"
    output_path = interim_dir / output_filename

    interim_dir.mkdir(parents=True, exist_ok=True)

    for col in ("source_id", "ra", "dec"):
        if col not in coords_df.columns:
            raise KeyError(f"coords_df must include a {col!r} column")

    ids = np.asarray(list(source_ids), dtype=np.int64)
    n_requested = int(ids.size)
    logger.info("Level-3: XP fetch for %d source_ids (batch=%d)", n_requested, batch_size)

    tap = service if service is not None else aip_service()
    raw = fetch_xp_coefficients(
        tap,
        ids,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
        timeout_sec=timeout_sec,
    )
    n_fetched = len(raw)
    logger.info("Level-3: XP TAP returned %d rows", n_fetched)

    logger.info("Level-3: §6.5 sanity check (raw Hermite coeffs)")
    sanity_counts = xp_sanity_check(raw)

    logger.info("Level-3: §6.4 step 1 — Ye+2024 flux correction (→ sampled flux)")
    corrected = apply_ye2024_correction(
        raw,
        coords_df,
        model_dir=model_dir,
        batch_size=batch_size,
        device=device,
    )
    flag_counts = {
        "n_ok": int((corrected["ye2024_flag"] == YE2024_FLAG_OK).sum()),
        "n_no_synth_phot": int((corrected["ye2024_flag"] == YE2024_FLAG_NO_SYNTH_PHOT).sum()),
        "n_calibrate_fail": int((corrected["ye2024_flag"] == YE2024_FLAG_CALIBRATE_FAIL).sum()),
    }

    logger.info("Level-3: writing %s", output_path)
    save_parquet(corrected, output_path)

    n_batches = (n_requested + batch_size - 1) // batch_size
    prov = Provenance(
        output_file=str(output_path),
        script="src/arqueogal/data/ingest_xp.py",
        sources=[
            TapSource(
                name=f"AIP Gaia DR3 XP continuous mean spectrum ({XP_TABLE})",
                endpoint=AIP_TAP_URL,
                query=XP_QUERY_ADQL,
                n_batches=n_batches,
                batch_size=batch_size,
            ),
        ],
        cuts_applied=[],
        corrections=[
            "§6.4 step 1: Ye+2024 NN flux-correction on gaiaxpy-calibrated "
            "sampled spectra (CCM89 + SFD deredden, 14-feature NN)",
            "SFD E(B-V) × 2.742 → A_V (Schlafly & Finkbeiner 2011, Rv=3.1)",
        ],
        row_count_before=n_requested,
        row_count_after=len(corrected),
        notes=(
            "Level-3 XP extraction: Ye+2024-corrected *sampled flux* (length "
            f"{len(YE2024_SAMPLING_NM)} on np.geomspace(360, 990, 330) nm), "
            "not Hermite coefficients. §6.4 steps 2–5 (normalise by c_0, "
            "log+zscore c_0, error propagation, float32 downcast) do NOT run "
            "here — they operate on Hermite coefficients and are no longer "
            "applicable after Ye+2024. Pipeline-1 feature-matrix code either "
            "(a) consumes this sampled flux directly, or (b) re-projects onto "
            "the Hermite basis and then runs normalise_xp / zscore_c0."
        ),
        extra={
            "source_ids_requested": n_requested,
            "xp_rows_returned": n_fetched,
            "xp_checkpoint_dir": str(checkpoint_dir),
            "sanity_counts_raw": sanity_counts,
            "ye2024_flag_counts": flag_counts,
            "ye2024_sampling_n": int(len(YE2024_SAMPLING_NM)),
            "ye2024_sampling_nm_min": float(YE2024_SAMPLING_NM[0]),
            "ye2024_sampling_nm_max": float(YE2024_SAMPLING_NM[-1]),
            "ye2024_model_dir": str(model_dir) if model_dir is not None else None,
        },
    )
    write_sidecar(prov)
    logger.info(
        "Level-3: done (%d rows → %s; Ye flags: %s)",
        len(corrected),
        output_path,
        flag_counts,
    )
    return output_path, flag_counts


__all__ = ["DEFAULT_OUTPUT_FILENAME", "ingest_xp"]
