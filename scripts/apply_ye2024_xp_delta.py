"""Phase 3a — apply Ye+2024 NN flux-correction on the XP delta.

Delta-aware variant of ``scripts/apply_ye2024_xp.py``. Reads the raw XP
coefficients for the Phase 3a delta source_ids (output of
``scripts/fetch_gaia_xp_delta.py``) and applies the Ye+2024 NN
flux-correction, writing the corrected sampled flux to
``data/interim/xp_sampled_corrected_delta.parquet`` plus a provenance
sidecar.

Coordinates (ra, dec in ICRS deg) are sourced from the Andrae+2023
catalogue at ``data/raw/andrae2023/andrae2023_rgb.parquet`` (10.48 M
rows, ra_deg/dec_deg float32) — the Phase 3a delta source_ids are a
subset of Andrae so no extra fetch is needed.

Halt condition
--------------
The delta ``NO_SYNTH_PHOT`` rate must not deviate from the Stream 1
baseline (2.60 %) by more than ±2 pp.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import time
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

REPO = Path(__file__).resolve().parents[1]

MEGA_BATCH = 20_000
YE_BATCH = 5_000
STREAM1_BASELINE_NO_SYNTH_PHOT_RATE = 0.0260
HALT_DELTA_PP = 0.02  # absolute deviation from baseline that triggers HALT


def _setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("apply_ye2024_xp_delta")
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--xp-parquet",
        type=Path,
        default=REPO / "data" / "interim" / "xp_coeffs_raw_delta.parquet",
    )
    p.add_argument(
        "--andrae-parquet",
        type=Path,
        default=REPO / "data" / "raw" / "andrae2023" / "andrae2023_rgb.parquet",
    )
    p.add_argument(
        "--output-parquet",
        type=Path,
        default=REPO / "data" / "interim" / "xp_sampled_corrected_delta.parquet",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO / "data" / "interim" / "enrich_batches" / "ye2024_delta",
    )
    p.add_argument(
        "--log-path",
        type=Path,
        default=REPO / "logs" / "stream3_delta_ye2024_20260419.log",
    )
    return p.parse_args()


def main() -> None:  # noqa: PLR0912, PLR0915 — linear driver
    args = _parse_args()
    log = _setup_logging(args.log_path)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log.info("=== Phase 3a Ye+2024 delta application — start ===")

    for p in (args.xp_parquet, args.andrae_parquet):
        if not p.is_file():
            raise SystemExit(f"missing input: {p}")

    log.info("loading raw XP delta: %s", args.xp_parquet)
    xp = pd.read_parquet(args.xp_parquet)
    log.info("  %d rows, %d cols", len(xp), len(xp.columns))

    log.info("loading Andrae coords (ra_deg, dec_deg)")
    coords = pd.read_parquet(
        args.andrae_parquet,
        columns=["source_id", "ra_deg", "dec_deg"],
    )
    coords["source_id"] = coords["source_id"].astype("int64")
    coords = coords.rename(columns={"ra_deg": "ra", "dec_deg": "dec"})
    log.info("  Andrae coords: %d unique source_ids", len(coords))

    # Inner join XP × coords — every XP row should have a coord row since the
    # delta source_ids all came from Andrae.
    joined = xp.merge(coords, on="source_id", how="inner")
    missing = len(xp) - len(joined)
    if missing:
        log.warning("%d XP rows had no Andrae coord match (dropped)", missing)
    joined = joined.sort_values("source_id").reset_index(drop=True)
    n = len(joined)
    n_mega = (n + MEGA_BATCH - 1) // MEGA_BATCH
    log.info("Ye+2024: %d rows → %d mega-batches of %d", n, n_mega, MEGA_BATCH)

    flag_totals = {"n_ok": 0, "n_no_synth_phot": 0, "n_calibrate_fail": 0}

    for b in range(n_mega):
        lo = b * MEGA_BATCH
        hi = min(lo + MEGA_BATCH, n)
        ckpt = args.checkpoint_dir / f"ye_{b:04d}.parquet"
        if ckpt.exists():
            df_ck = pd.read_parquet(ckpt, columns=["ye2024_flag"])
            flag_totals["n_ok"] += int((df_ck["ye2024_flag"] == YE2024_FLAG_OK).sum())
            flag_totals["n_no_synth_phot"] += int(
                (df_ck["ye2024_flag"] == YE2024_FLAG_NO_SYNTH_PHOT).sum()
            )
            flag_totals["n_calibrate_fail"] += int(
                (df_ck["ye2024_flag"] == YE2024_FLAG_CALIBRATE_FAIL).sum()
            )
            log.info("[%d/%d] resume ckpt %s", b + 1, n_mega, ckpt.name)
            continue

        log.info("[%d/%d] rows %d..%d", b + 1, n_mega, lo, hi)
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
        log.info("  wrote %s; running flags: %s", ckpt.name, flag_totals)

    log.info("concatenating %d checkpoints", n_mega)
    parts = [pd.read_parquet(args.checkpoint_dir / f"ye_{b:04d}.parquet") for b in range(n_mega)]
    corrected = pd.concat(parts, ignore_index=True)
    n_total = len(corrected)
    n_ok = flag_totals["n_ok"]
    n_no_synth = flag_totals["n_no_synth_phot"]
    n_cal_fail = flag_totals["n_calibrate_fail"]
    no_synth_rate = n_no_synth / max(n_total, 1)
    log.info(
        "Ye+2024 flag totals: OK=%d NO_SYNTH_PHOT=%d CAL_FAIL=%d (rate=%.4f%%)",
        n_ok, n_no_synth, n_cal_fail, no_synth_rate * 100.0,
    )

    # Halt check.
    halt_reason: str | None = None
    delta_pp = abs(no_synth_rate - STREAM1_BASELINE_NO_SYNTH_PHOT_RATE)
    if delta_pp > HALT_DELTA_PP:
        halt_reason = (
            f"Ye NO_SYNTH_PHOT rate {no_synth_rate:.4%} deviates from "
            f"baseline {STREAM1_BASELINE_NO_SYNTH_PHOT_RATE:.4%} by "
            f"{delta_pp * 100:.2f} pp > {HALT_DELTA_PP * 100:.2f} pp threshold"
        )
        log.error("HALT: %s", halt_reason)
    else:
        log.info(
            "Ye NO_SYNTH_PHOT rate within ±%.1f pp of baseline (delta=%.3f pp)",
            HALT_DELTA_PP * 100, delta_pp * 100,
        )

    _write_parquet_atomic(corrected, args.output_parquet)
    size_mb = args.output_parquet.stat().st_size / 1024**2
    out_sha = _sha256_of(args.output_parquet)
    log.info(
        "wrote %s (%.1f MB, %d cols, sha256=%s…)",
        args.output_parquet, size_mb, len(corrected.columns), out_sha[:12],
    )

    # Drop checkpoints on success (§12 budget).
    if halt_reason is None:
        removed = 0
        for b in range(n_mega):
            ckpt = args.checkpoint_dir / f"ye_{b:04d}.parquet"
            if ckpt.exists():
                ckpt.unlink()
                removed += 1
        log.info("removed %d checkpoints under %s", removed, args.checkpoint_dir)
    else:
        log.warning("halt fired; keeping checkpoints for debugging")

    # Provenance
    total_wall = time.time() - t0
    prov = Provenance(
        output_file=str(args.output_parquet.relative_to(REPO)),
        script="scripts/apply_ye2024_xp_delta.py",
        sources=[
            LocalSource(
                name="raw XP coefficients (Phase 3a delta)",
                path=str(args.xp_parquet.relative_to(REPO)),
                sha256=_sha256_of(args.xp_parquet),
            ),
            LocalSource(
                name="Andrae+2023 coords (ra_deg, dec_deg)",
                path=str(args.andrae_parquet.relative_to(REPO)),
                sha256=_sha256_of(args.andrae_parquet),
            ),
        ],
        cuts_applied=[
            "INNER JOIN raw XP delta × Andrae coords on source_id",
        ],
        corrections=[
            "§6.4 step 1: Ye+2024 NN flux-correction on gaiaxpy-calibrated "
            "sampled spectra (CCM89 + SFD deredden, 14-feature NN)",
            "SFD E(B-V) × 2.742 → A_V (Schlafly & Finkbeiner 2011, Rv=3.1)",
            "Injected bp/rp_n_parameters=55 and zero-vector "
            "bp/rp_coefficient_correlations (length 1485) as gaiaxpy-required "
            "inputs per §6.1 budget; flux_error from gaiaxpy is consequently "
            "uncalibrated and NOT used by Ye's NN (flux only).",
        ],
        row_count_before=n,
        row_count_after=int(n_total),
        notes=(
            "Ye+2024-corrected sampled flux (np.geomspace(360, 990, 330) nm) "
            "for the Phase 3a delta. Mirrors scripts/apply_ye2024_xp.py for "
            "the existing Stream 1 ∪ Stream 3 artefact; the two are kept as "
            "separate parquets so the existing 457k-row file is untouched."
        ),
        extra={
            "ye2024_flag_counts": flag_totals,
            "ye2024_no_synth_phot_rate": no_synth_rate,
            "stream1_baseline_no_synth_phot_rate": STREAM1_BASELINE_NO_SYNTH_PHOT_RATE,
            "halt_threshold_abs_pp": HALT_DELTA_PP,
            "halt_reason": halt_reason,
            "halt_delta_pp_actual": delta_pp * 100,
            "ye2024_sampling_n": int(len(YE2024_SAMPLING_NM)),
            "ye2024_sampling_nm_min": float(YE2024_SAMPLING_NM[0]),
            "ye2024_sampling_nm_max": float(YE2024_SAMPLING_NM[-1]),
            "mega_batch_size": MEGA_BATCH,
            "inner_ye_batch_size": YE_BATCH,
            "wallclock_seconds": total_wall,
            "output_sha256": out_sha,
            "ye2024_model_dir": "data/external/ye2024/GaiaXP-correction_V0/model",
            "ye2024_zenodo_concept_doi": "10.5281/zenodo.14028588",
            "ye2024_zenodo_record_id": "14712749",
        },
    )
    write_sidecar(prov)
    log.info("wrote provenance sidecar; done in %.1f min", total_wall / 60)


if __name__ == "__main__":
    main()
