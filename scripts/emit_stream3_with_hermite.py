"""Re-emit Pipeline-1 Stream 3 features with Hermite coefficients (stage B).

Stream-3 analogue of ``scripts/emit_stream1_with_hermite.py`` — same Hermite
basis fingerprint and per-Teff-bin p99 residual thresholds, but emits
**raw** ``bp_c0_log`` / ``rp_c0_log`` and raw ``bp_coef_norm_{i}`` = c_i/c_0
ratios INSTEAD of the z-scored scalars. The inference driver detects the
raw schema (presence of ``*_c0_log``) and applies :func:`apply_frozen_zscore`
from Stream-1 provenance (``data/processed/pipeline1_features_stream1
.provenance.json``) at assembly time — Stream-3 MUST use training-set stats
verbatim, not refit its own.

Outputs
-------

Same path (``data/processed/pipeline1_features_stream3.parquet``) with
``corrected_flux`` replaced by:

- Raw Hermite coefficients: ``bp_coef_0..54`` / ``rp_coef_0..54``
  (diagnostic only; retained for §9.2 LOOCO audit).
- Raw shape ratios: ``bp_coef_norm_1..54`` / ``rp_coef_norm_1..54``
  = ``c_i / c_0`` on the normal-population subset (Ye-ok AND fit-residual
  normal AND c0 > 0); NaN elsewhere. NOT z-scored.
- Raw absolute-scale log: ``bp_c0_log`` / ``rp_c0_log`` = ``log10(c_0)`` on
  the normal-population subset; NaN elsewhere. NOT z-scored.
- Reprojection residuals (ML-input features): ``reprojection_residual_rms``,
  ``..._bp``, ``..._rp``.
- Fit flags: ``xp_fit_flag_residual_high`` (Teff-stratified p99 from
  Stream-1 pre_emit_decisions.json), ``xp_fit_flag_residual_high_global``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from arqueogal.data.downloads import sha256_file
from arqueogal.data.gaia_xp import (
    HERMITE_N_BASIS,
    HERMITE_REPROJECTION_VERSION,
    XP_FIT_FLAG_OK,
    XP_FIT_FLAG_RESIDUAL_HIGH,
    reproject_ye_to_hermite,
)
from arqueogal.data.provenance import LocalSource, Provenance, write_sidecar

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("emit_stream3_hermite")

CHUNK_ROWS = 50_000
TEFF_BIN_EDGES_K = (4000.0, 4500.0, 5000.0, 5500.0, 6000.0)
TEFF_BIN_LABELS = (
    "lt_4000",
    "4000_4500",
    "4500_5000",
    "5000_5500",
    "5500_6000",
    "ge_6000",
)


def _teff_bin_label(teff: np.ndarray) -> np.ndarray:
    labels = np.full(teff.shape, "no_teff", dtype=object)
    finite = np.isfinite(teff)
    edges = np.asarray(TEFF_BIN_EDGES_K, dtype=np.float64)
    idx = np.digitize(teff[finite], edges, right=False)
    labels_finite = np.array(TEFF_BIN_LABELS, dtype=object)[idx]
    labels[finite] = labels_finite
    return labels


def _load_thresholds(decisions_path: Path) -> dict:
    with decisions_path.open("r", encoding="utf-8") as fh:
        obj = json.load(fh)
    bins = obj["thresholds"]["per_teff_bin"]
    p99_by_label = {label: float(bins[label]["p99"]) for label in bins}
    global_p99 = float(obj["thresholds"]["global_p99_normal"])
    return {
        "p99_by_label": p99_by_label,
        "global_p99": global_p99,
        "catastrophic": float(obj["thresholds"]["catastrophic_threshold"]),
        "basis_version": obj.get("basis_version"),
    }


def _reproject_all(flux_matrix: np.ndarray) -> dict:
    n_total = flux_matrix.shape[0]
    n_basis = HERMITE_N_BASIS
    bp = np.full((n_total, n_basis), np.nan, dtype=np.float32)
    rp = np.full((n_total, n_basis), np.nan, dtype=np.float32)
    rms_all = np.full(n_total, np.nan, dtype=np.float32)
    rms_bp = np.full(n_total, np.nan, dtype=np.float32)
    rms_rp = np.full(n_total, np.nan, dtype=np.float32)
    valid = np.isfinite(flux_matrix).all(axis=1)
    fingerprint = None
    basis_version = HERMITE_REPROJECTION_VERSION
    valid_idx = np.where(valid)[0]
    for start in range(0, valid_idx.size, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, valid_idx.size)
        sel = valid_idx[start:stop]
        logger.info(
            "reprojecting valid rows [%d, %d) / %d (of %d total)",
            start,
            stop,
            valid_idx.size,
            n_total,
        )
        out = reproject_ye_to_hermite(flux_matrix[sel])
        bp[sel] = out["bp_coeffs"]
        rp[sel] = out["rp_coeffs"]
        rms_all[sel] = out["reprojection_residual_rms"]
        rms_bp[sel] = out["reprojection_residual_rms_bp"]
        rms_rp[sel] = out["reprojection_residual_rms_rp"]
        fingerprint = out["basis_fingerprint_sha256"]
        basis_version = out["basis_version"]
    return {
        "bp": bp,
        "rp": rp,
        "rms": rms_all,
        "rms_bp": rms_bp,
        "rms_rp": rms_rp,
        "valid": valid,
        "fingerprint": fingerprint,
        "basis_version": basis_version,
    }


def _compute_flags(
    rms: np.ndarray,
    valid: np.ndarray,
    teff: np.ndarray,
    thresholds: dict,
) -> tuple[pd.arrays.IntegerArray, pd.arrays.IntegerArray]:
    n = rms.size
    mask_na = ~valid
    labels = _teff_bin_label(teff)
    p99_by_label = thresholds["p99_by_label"]
    global_p99 = thresholds["global_p99"]
    thr_by_label = dict(p99_by_label)
    if "no_teff" not in thr_by_label:
        thr_by_label["no_teff"] = global_p99
    per_star_threshold = np.full(n, np.nan, dtype=np.float64)
    for label, thr in thr_by_label.items():
        per_star_threshold[labels == label] = thr
    strat = np.where(
        valid & (rms > per_star_threshold),
        XP_FIT_FLAG_RESIDUAL_HIGH,
        XP_FIT_FLAG_OK,
    ).astype(np.int8)
    glob = np.where(
        valid & (rms > global_p99),
        XP_FIT_FLAG_RESIDUAL_HIGH,
        XP_FIT_FLAG_OK,
    ).astype(np.int8)
    strat_arr = pd.array(strat, dtype="Int8")
    glob_arr = pd.array(glob, dtype="Int8")
    strat_arr[mask_na] = pd.NA
    glob_arr[mask_na] = pd.NA
    return strat_arr, glob_arr


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    src = repo / "data" / "processed" / "pipeline1_features_stream3.parquet"
    dst = src
    decisions_path = (
        repo / "reports" / "figures" / "hermite_smoke" / "pre_emit" / "pre_emit_decisions.json"
    )
    stream1_prov = repo / "data" / "processed" / "pipeline1_features_stream1.provenance.json"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    if not decisions_path.exists():
        raise SystemExit(f"missing {decisions_path}")
    if not stream1_prov.exists():
        raise SystemExit(f"missing {stream1_prov}")

    logger.info("loading %s", src)
    df = pd.read_parquet(src)
    logger.info("loaded %d rows × %d cols", len(df), len(df.columns))
    src_sha = sha256_file(src)

    thresholds = _load_thresholds(decisions_path)
    for label, thr in thresholds["p99_by_label"].items():
        logger.info("  bin %-12s p99 = %.3e", label, thr)
    logger.info("  global p99 (fallback) = %.3e", thresholds["global_p99"])

    logger.info("stacking sampled flux to (N, 330) matrix")
    flux = np.stack(
        [np.asarray(row, dtype=np.float32) for row in df["corrected_flux"]],
        axis=0,
    )
    logger.info("flux matrix shape: %s", flux.shape)

    logger.info("reprojecting onto Hermite basis (%s)", HERMITE_REPROJECTION_VERSION)
    proj = _reproject_all(flux)
    logger.info("basis fingerprint: %s", proj["fingerprint"])

    teff = df["teff_gspphot"].to_numpy(dtype=np.float64, na_value=np.nan)
    strat_flag, global_flag = _compute_flags(
        rms=proj["rms"],
        valid=proj["valid"],
        teff=teff,
        thresholds=thresholds,
    )
    n_valid = int(proj["valid"].sum())
    n_high_strat = int(
        (strat_flag == XP_FIT_FLAG_RESIDUAL_HIGH).to_numpy(dtype=bool, na_value=False).sum()
    )
    n_high_global = int(
        (global_flag == XP_FIT_FLAG_RESIDUAL_HIGH).to_numpy(dtype=bool, na_value=False).sum()
    )
    logger.info(
        "flagging: n_valid=%d, n_strat_high=%d (%.3f%%), n_global_high=%d (%.3f%%)",
        n_valid,
        n_high_strat,
        100.0 * n_high_strat / max(n_valid, 1),
        n_high_global,
        100.0 * n_high_global / max(n_valid, 1),
    )

    logger.info("computing raw c0-normalized shape ratios + log10(c0) on normal-population subset")
    ye_ok = (df["ye2024_flag"] == 0).to_numpy(dtype=bool)
    strat_ok = (strat_flag == XP_FIT_FLAG_OK).to_numpy(dtype=bool, na_value=False)
    normal_pop = ye_ok & strat_ok

    bp_c0 = proj["bp"][:, 0].astype(np.float64)
    rp_c0 = proj["rp"][:, 0].astype(np.float64)
    c0_ok = normal_pop & np.isfinite(bp_c0) & (bp_c0 > 0.0) & np.isfinite(rp_c0) & (rp_c0 > 0.0)
    n_c0_ok = int(c0_ok.sum())
    n_c0_rejected_from_normal = int(normal_pop.sum() - n_c0_ok)
    logger.info(
        "normal-population: %d rows; c0>0 finite in both bands: %d (rejected %d)",
        int(normal_pop.sum()),
        n_c0_ok,
        n_c0_rejected_from_normal,
    )

    # Raw ratios c_i / c_0 (NOT z-scored). inference driver handles z-score.
    bp_norm = np.full((len(df), HERMITE_N_BASIS), np.nan, dtype=np.float32)
    rp_norm = np.full((len(df), HERMITE_N_BASIS), np.nan, dtype=np.float32)
    for n in range(1, HERMITE_N_BASIS):
        bp_norm[c0_ok, n] = (proj["bp"][c0_ok, n] / bp_c0[c0_ok]).astype(np.float32)
        rp_norm[c0_ok, n] = (proj["rp"][c0_ok, n] / rp_c0[c0_ok]).astype(np.float32)

    # Raw log10(c_0) (NOT z-scored). inference driver applies frozen stats.
    bp_c0_log = np.full(len(df), np.nan, dtype=np.float32)
    rp_c0_log = np.full(len(df), np.nan, dtype=np.float32)
    bp_c0_log[c0_ok] = np.log10(bp_c0[c0_ok]).astype(np.float32)
    rp_c0_log[c0_ok] = np.log10(rp_c0[c0_ok]).astype(np.float32)

    logger.info("assembling output DataFrame")
    df_out = df.drop(columns=["corrected_flux"]).copy()
    bp_cols = {f"bp_coef_{n}": proj["bp"][:, n] for n in range(HERMITE_N_BASIS)}
    rp_cols = {f"rp_coef_{n}": proj["rp"][:, n] for n in range(HERMITE_N_BASIS)}
    bp_norm_cols = {f"bp_coef_norm_{n}": bp_norm[:, n] for n in range(1, HERMITE_N_BASIS)}
    rp_norm_cols = {f"rp_coef_norm_{n}": rp_norm[:, n] for n in range(1, HERMITE_N_BASIS)}
    df_out = df_out.assign(
        **bp_cols,
        **rp_cols,
        **bp_norm_cols,
        **rp_norm_cols,
        bp_c0_log=bp_c0_log,
        rp_c0_log=rp_c0_log,
        reprojection_residual_rms=proj["rms"],
        reprojection_residual_rms_bp=proj["rms_bp"],
        reprojection_residual_rms_rp=proj["rms_rp"],
        xp_fit_flag_residual_high=strat_flag,
        xp_fit_flag_residual_high_global=global_flag,
    )
    logger.info("output cols: %d", len(df_out.columns))

    tmp = dst.with_suffix(dst.suffix + ".part")
    logger.info("writing %s", dst)
    df_out.to_parquet(tmp, index=False)
    os.replace(tmp, dst)
    size_mb = dst.stat().st_size / 1024**2
    logger.info("wrote %s (%.1f MB, %d cols)", dst, size_mb, len(df_out.columns))

    prov = Provenance(
        output_file=str(dst.relative_to(repo)),
        script="scripts/emit_stream3_with_hermite.py",
        sources=[
            LocalSource(
                name="Pipeline-1 Stream-3 stage-A features (Ye-corrected sampled flux)",
                path=str(src.relative_to(repo)),
                sha256=src_sha,
            ),
            LocalSource(
                name="Pre-emit decision record (Stream-1 p99 thresholds)",
                path=str(decisions_path.relative_to(repo)),
                sha256=sha256_file(decisions_path),
            ),
            LocalSource(
                name="Stream-1 provenance (frozen z-score stats used at inference)",
                path=str(stream1_prov.relative_to(repo)),
                sha256=sha256_file(stream1_prov),
            ),
        ],
        cuts_applied=[],
        corrections=[
            f"Hermite re-projection {HERMITE_REPROJECTION_VERSION} onto 55+55 "
            "physicist-Hermite orthonormal basis",
            "xp_fit_flag_residual_high: Teff-stratified p99 from Stream-1 normal-population",
            "c0 normalization: bp_coef_norm_{1..54} = bp_coef_{i} / bp_coef_0 "
            "on normal-population subset (raw ratios; NOT z-scored)",
            "bp_c0_log/rp_c0_log = log10(c_0) on normal-population subset "
            "(raw; NOT z-scored; inference driver applies frozen Stream-1 z-score)",
        ],
        row_count_before=len(df),
        row_count_after=len(df_out),
        notes=(
            "Stage B of Stream-3 emit — raw schema. Produces bp_coef_norm_{i} as "
            "raw c_i/c_0 and bp_c0_log/rp_c0_log as raw log10(c_0). The "
            "run_pipeline1_inference.py driver detects presence of bp_c0_log + "
            "rp_c0_log and applies apply_frozen_zscore using stats from the "
            "Stream-1 provenance sidecar (extra.coef_norm_zscore_frozen / "
            "extra.c0_zscore_frozen). This guarantees Stream-3 uses the exact "
            "training-set reference distribution — refitting would mask drift."
        ),
        extra={
            "basis_version": proj["basis_version"],
            "basis_fingerprint_sha256": proj["fingerprint"],
            "hermite_n_basis": HERMITE_N_BASIS,
            "teff_bin_edges_K": list(TEFF_BIN_EDGES_K),
            "teff_bin_labels": list(TEFF_BIN_LABELS) + ["no_teff"],
            "p99_thresholds_by_teff_bin": thresholds["p99_by_label"],
            "global_p99_normal": thresholds["global_p99"],
            "n_residual_high_stratified": n_high_strat,
            "n_residual_high_global": n_high_global,
            "n_hermite_valid": n_valid,
            "n_hermite_no_fit": int(len(df) - n_valid),
            "n_c0_ok_normal_population": n_c0_ok,
            "n_c0_rejected_from_normal": n_c0_rejected_from_normal,
            "emit_schema": "raw (bp_c0_log/rp_c0_log + raw c_i/c_0 ratios); "
            "inference driver applies frozen z-score from Stream-1",
        },
    )
    write_sidecar(prov)
    logger.info("wrote provenance sidecar")


if __name__ == "__main__":
    main()
