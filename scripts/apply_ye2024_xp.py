"""Apply Ye+2024 NN flux-correction to the fetched raw XP coefficients.

Consumes the Level-2 products:
- ``data/interim/xp_coeffs_raw.parquet`` (from ``scripts/fetch_gaia_xp.py``,
  ~457 k rows)
- ``data/interim/stream1_gaia_dr3_corrected.parquet`` — Stream 1 (ra, dec).
- ``data/interim/stream3_gaia_dr3_corrected.parquet`` — Stream 3 (ra, dec).

Runs :func:`arqueogal.data.gaia_xp.apply_ye2024_correction` in resumable
mega-batches of :data:`MEGA_BATCH` (~20 k) stars. Each mega-batch writes a
checkpoint to ``data/interim/enrich_batches/ye2024/ye_NNNN.parquet``; reruns
skip completed checkpoints. Final concatenation lands at
``data/interim/xp_sampled_corrected.parquet`` with a provenance sidecar.

Expected wall-time: ~25-40 min on the RTX 3060. Most time is gaiaxpy CPU
calibrate+generate, not NN inference.

See ``data_acquisition.md`` §6.4 step 1 for the scientific contract
(including the n_parameters=55 / zero-correlations substitution).
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from arqueogal.data.gaia_xp import (
    YE2024_FLAG_CALIBRATE_FAIL,
    YE2024_FLAG_NO_SYNTH_PHOT,
    YE2024_FLAG_OK,
    YE2024_N_OUTPUT,
    YE2024_SAMPLING_NM,
    apply_ye2024_correction,
)
from arqueogal.data.provenance import LocalSource, Provenance, write_sidecar

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("apply_ye2024_xp")

MEGA_BATCH = 20_000
YE_BATCH = 5_000  # inner gaiaxpy batch


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


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    xp_path = repo / "data" / "interim" / "xp_coeffs_raw.parquet"
    s1_path = repo / "data" / "interim" / "stream1_gaia_dr3_corrected.parquet"
    s3_path = repo / "data" / "interim" / "stream3_gaia_dr3_corrected.parquet"
    out_path = repo / "data" / "interim" / "xp_sampled_corrected.parquet"
    ckpt_dir = repo / "data" / "interim" / "enrich_batches" / "ye2024"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for p in (xp_path, s1_path, s3_path):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    logger.info("loading raw XP: %s", xp_path)
    xp = pd.read_parquet(xp_path)
    logger.info("  %d rows", len(xp))

    logger.info("loading Stream 1 + Stream 3 coords")
    s1_coords = pd.read_parquet(s1_path, columns=["source_id", "ra", "dec"])
    s3_coords = pd.read_parquet(s3_path, columns=["source_id", "ra", "dec"])
    coords = pd.concat([s1_coords, s3_coords], ignore_index=True).drop_duplicates("source_id")
    logger.info("  coords: %d unique source_ids", len(coords))

    # Inner join to drop any XP rows not present in either stream's coord set
    # (should be zero given xp_coeffs_raw.parquet was built from Stream 1 ∪ 3).
    joined = xp.merge(coords, on="source_id", how="inner")
    missing = len(xp) - len(joined)
    if missing:
        logger.warning("%d XP rows had no Stream 1/3 coord match (dropped)", missing)
    # Sort for deterministic mega-batch partitioning.
    joined = joined.sort_values("source_id").reset_index(drop=True)
    n = len(joined)
    n_mega = (n + MEGA_BATCH - 1) // MEGA_BATCH
    logger.info("Ye+2024: %d rows → %d mega-batches of %d", n, n_mega, MEGA_BATCH)

    flag_totals = {"n_ok": 0, "n_no_synth_phot": 0, "n_calibrate_fail": 0}

    for b in range(n_mega):
        lo = b * MEGA_BATCH
        hi = min(lo + MEGA_BATCH, n)
        ckpt = ckpt_dir / f"ye_{b:04d}.parquet"
        if ckpt.exists():
            df_ck = pd.read_parquet(ckpt, columns=["ye2024_flag"])
            flag_totals["n_ok"] += int((df_ck["ye2024_flag"] == YE2024_FLAG_OK).sum())
            flag_totals["n_no_synth_phot"] += int(
                (df_ck["ye2024_flag"] == YE2024_FLAG_NO_SYNTH_PHOT).sum()
            )
            flag_totals["n_calibrate_fail"] += int(
                (df_ck["ye2024_flag"] == YE2024_FLAG_CALIBRATE_FAIL).sum()
            )
            logger.info("[%d/%d] resume: %s (%d rows)", b + 1, n_mega, ckpt.name, hi - lo)
            continue

        logger.info("[%d/%d] processing rows %d..%d", b + 1, n_mega, lo, hi)
        chunk = joined.iloc[lo:hi].reset_index(drop=True)
        out = apply_ye2024_correction(
            chunk[[c for c in chunk.columns if c not in ("ra", "dec")]],
            chunk[["source_id", "ra", "dec"]],
            batch_size=YE_BATCH,
        )
        _write_parquet_atomic(out, ckpt)
        flag_totals["n_ok"] += int((out["ye2024_flag"] == YE2024_FLAG_OK).sum())
        flag_totals["n_no_synth_phot"] += int(
            (out["ye2024_flag"] == YE2024_FLAG_NO_SYNTH_PHOT).sum()
        )
        flag_totals["n_calibrate_fail"] += int(
            (out["ye2024_flag"] == YE2024_FLAG_CALIBRATE_FAIL).sum()
        )
        logger.info(
            "  wrote %s (%d rows); running flags: %s",
            ckpt.name,
            len(out),
            flag_totals,
        )

    logger.info("concatenating %d checkpoints", n_mega)
    parts = [pd.read_parquet(ckpt_dir / f"ye_{b:04d}.parquet") for b in range(n_mega)]
    corrected = pd.concat(parts, ignore_index=True)
    logger.info("final: %d rows; flags: %s", len(corrected), flag_totals)

    _write_parquet_atomic(corrected, out_path)
    size_mb = out_path.stat().st_size / 1024**2
    logger.info("wrote %s (%.1f MB, %d cols)", out_path, size_mb, len(corrected.columns))

    # §12 disk budget: drop the per-mega-batch checkpoints once the final
    # concat has landed atomically. Reruns regenerate them on demand.
    removed = 0
    for b in range(n_mega):
        ckpt = ckpt_dir / f"ye_{b:04d}.parquet"
        if ckpt.exists():
            ckpt.unlink()
            removed += 1
    logger.info("removed %d checkpoint parquets under %s", removed, ckpt_dir)

    prov = Provenance(
        output_file=str(out_path.relative_to(repo)),
        script="scripts/apply_ye2024_xp.py",
        sources=[
            LocalSource(
                name="raw XP coefficients (Stream 1 ∪ Stream 3)",
                path=str(xp_path.relative_to(repo)),
                sha256=_sha256_of(xp_path),
            ),
            LocalSource(
                name="Stream 1 Gaia DR3 corrected coords",
                path=str(s1_path.relative_to(repo)),
                sha256=_sha256_of(s1_path),
            ),
            LocalSource(
                name="Stream 3 Gaia DR3 corrected coords",
                path=str(s3_path.relative_to(repo)),
                sha256=_sha256_of(s3_path),
            ),
        ],
        cuts_applied=[
            "INNER JOIN raw XP × (Stream 1 ∪ Stream 3) coords on source_id",
        ],
        corrections=[
            "§6.4 step 1: Ye+2024 NN flux-correction on gaiaxpy-calibrated "
            "sampled spectra (CCM89 + SFD deredden, 14-feature NN)",
            "SFD E(B-V) × 2.742 → A_V (Schlafly & Finkbeiner 2011, Rv=3.1)",
            "Injected bp/rp_n_parameters=55 and zero-vector "
            "bp/rp_coefficient_correlations (length 1485) as gaiaxpy-required "
            "inputs dropped per §6.1 budget; flux_error from gaiaxpy is "
            "consequently uncalibrated and NOT used by Ye's NN (flux only).",
        ],
        row_count_before=n,
        row_count_after=int(len(corrected)),
        notes=(
            "Ye+2024-corrected *sampled flux* on np.geomspace(360, 990, 330) "
            "nm. §6.4 steps 2-5 (Hermite normalise + log+zscore + error "
            "propagation + float32 downcast) are NOT chained — they operate "
            "on Hermite coefficients and no longer apply after Ye. Pipeline-1 "
            "feature-matrix code consumes this sampled flux directly; a "
            "Hermite re-projection path is available if Pipeline 1 opts for "
            "it (not used in the main pipeline per TESS_ML prototype lineage)."
        ),
        extra={
            "ye2024_flag_counts": flag_totals,
            "ye2024_sampling_n": int(len(YE2024_SAMPLING_NM)),
            "ye2024_sampling_nm_min": float(YE2024_SAMPLING_NM[0]),
            "ye2024_sampling_nm_max": float(YE2024_SAMPLING_NM[-1]),
            "mega_batch_size": MEGA_BATCH,
            "inner_ye_batch_size": YE_BATCH,
            "ye2024_model_dir": "data/external/ye2024/GaiaXP-correction_V0/model",
            "ye2024_zenodo_concept_doi": "10.5281/zenodo.14028588",
            "ye2024_zenodo_record_id": "14712749",
        },
    )
    write_sidecar(prov)
    logger.info("wrote provenance sidecar")


if __name__ == "__main__":
    main()
