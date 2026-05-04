"""Re-emit Pipeline-1 Stream 1 features with Hermite coefficients (stage B).

Stage B of the two-stage emit pipeline (stage A is
``scripts/build_pipeline1_features_stream1.py``). Replaces the
``corrected_flux`` (list<float>, 330) column of
``data/processed/pipeline1_features_stream1.parquet`` with the 3-tier XP
column convention from ``src/arqueogal/xp_abundances/main/DESIGN.md``:

- **Raw diagnostic (not ML input)**: ``bp_coef_0..54``, ``rp_coef_0..54`` —
  float32 Hermite coefficients.
- **ML-input shape features**: ``bp_coef_norm_1..54``, ``rp_coef_norm_1..54``
  — per-coefficient z-scored shape coefficients
  ``(c_i/c_0 - mu_i) / sigma_i``, populated only on the normal-population
  subset (``ye2024_flag == 0 AND xp_fit_flag_residual_high == 0``); NaN
  elsewhere. Frozen ``(mu_i, sigma_i)`` persisted in provenance at
  ``extra.coef_norm_zscore_frozen`` for Stream-3 inference reproducibility.
  Raw ``c_i/c_0`` is recoverable as ``stored * sigma_i + mu_i``; raw
  unnormalised ``c_i`` remains on disk as ``bp_coef_{i}``/``rp_coef_{i}``.
  The trivial ``*_coef_norm_0 ≡ 1`` (pre-zscore) is NOT stored.
- **ML-input absolute-scale scalars**: ``bp_c0_z``, ``rp_c0_z`` —
  ``z_score(log10(c0))`` on the normal-population subset. The frozen
  population ``(mu, sigma)`` is persisted in provenance as
  ``extra.c0_zscore_frozen`` so Stream-3 inference applies training-set
  statistics, not its own.
- **Reprojection residuals (ML-input features)**:
  ``reprojection_residual_rms``, ``..._bp``, ``..._rp``.
- **Flags**: ``xp_fit_flag_residual_high`` (Teff-stratified p99),
  ``xp_fit_flag_residual_high_global`` (flat global p99, auxiliary).

Ye flag=1 rows (NaN sampled flux → no Hermite fit possible) carry NaN
coefficients / NaN residuals / <NA> flags; the pre-existing ``ye2024_flag``
stays in the file unchanged so downstream users disambiguate the two failure
modes via (ye2024_flag, xp_fit_flag_residual_high) jointly.

Thresholds and the NO_SYNTH_PHOT × Hermite-catastrophic contingency table come
from ``reports/figures/hermite_smoke/pre_emit/pre_emit_decisions.json``
(produced by ``scripts/analyze_hermite_pre_emit.py``).
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
logger = logging.getLogger("emit_stream1_hermite")

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
    """Return bin-label string per row; 'no_teff' when ``teff`` is NaN."""
    labels = np.full(teff.shape, "no_teff", dtype=object)
    finite = np.isfinite(teff)
    edges = np.asarray(TEFF_BIN_EDGES_K, dtype=np.float64)
    # np.digitize right=False -> idx in [0, len(edges)] inclusive
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
        "contingency": obj["contingency"],
        "pca_comparison": obj["pca_comparison"],
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
    # rows with any NaN in the sampled flux cannot be reprojected
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
    # Int8 nullable: <NA> where there is no Hermite fit (Ye flag=1 rows)
    strat = np.full(n, 0, dtype=np.int8)
    glob = np.full(n, 0, dtype=np.int8)
    mask_na = ~valid

    # Stratified: per-Teff-bin p99 thresholds; no-Teff rows use global_p99 fallback
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
    src = repo / "data" / "processed" / "pipeline1_features_stream1.parquet"
    dst = src
    decisions_path = (
        repo / "reports" / "figures" / "hermite_smoke" / "pre_emit" / "pre_emit_decisions.json"
    )
    if not src.exists():
        raise SystemExit(f"missing {src}")
    if not decisions_path.exists():
        raise SystemExit(f"missing {decisions_path}; run scripts/analyze_hermite_pre_emit.py first")

    logger.info("loading %s", src)
    df = pd.read_parquet(src)
    logger.info("loaded %d rows × %d cols", len(df), len(df.columns))
    src_sha = sha256_file(src)

    # Memory-efficient corrected_flux retrieval. As of 2026-04-29 the
    # build_pipeline1_features_stream1 stage drops corrected_flux from its
    # output (330 floats × ~330k rows ~5 GB pandas blow-up). Stream the
    # column from xp_sampled_corrected.parquet for the build's source_ids
    # only.
    if "corrected_flux" not in df.columns:
        import gc
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as _pq
        xp_path = repo / "data" / "interim" / "xp_sampled_corrected.parquet"
        logger.info("streaming corrected_flux from %s", xp_path)
        wanted = pa.array(df["source_id"].to_numpy())
        opts = pc.SetLookupOptions(value_set=wanted)
        pf = _pq.ParquetFile(xp_path)
        kept = []
        for rg_idx in range(pf.metadata.num_row_groups):
            rg = pf.read_row_group(rg_idx, columns=["source_id", "corrected_flux"])
            mask = pc.is_in(rg.column("source_id"), options=opts)
            chunk = rg.filter(mask)
            if chunk.num_rows:
                kept.append(chunk)
            del rg, mask, chunk
            gc.collect()
        flux_table = pa.concat_tables(kept)
        del kept
        gc.collect()
        flux_df = flux_table.to_pandas()
        del flux_table
        gc.collect()
        logger.info("  retrieved corrected_flux for %d rows", len(flux_df))
        df = df.merge(flux_df[["source_id", "corrected_flux"]], on="source_id", how="left")
        del flux_df
        gc.collect()
        n_missing = df["corrected_flux"].isna().sum()
        if n_missing:
            logger.warning("  %d rows have no corrected_flux match (will skip Hermite)", n_missing)
            df = df[df["corrected_flux"].notna()].reset_index(drop=True)
        logger.info("  post-merge df: %d rows", len(df))

    thresholds = _load_thresholds(decisions_path)
    logger.info("loaded per-Teff-bin thresholds from %s", decisions_path.name)
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
        "flagging: n_valid=%d, n_residual_high_strat=%d (%.3f%%), "
        "n_residual_high_global=%d (%.3f%%)",
        n_valid,
        n_high_strat,
        100.0 * n_high_strat / max(n_valid, 1),
        n_high_global,
        100.0 * n_high_global / max(n_valid, 1),
    )

    # --- c0 normalization + z-scoring (DESIGN §XP Hermite coefficients) ---
    #
    # Normal-population mask: Ye-ok AND Hermite-fit-residual-normal (stratified).
    # Rows failing either condition get NaN in both the normalized shape
    # coefficients and the absolute-scale scalars. The frozen z-score stats
    # below are computed on that subset ONLY — they represent the training-set
    # reference distribution and must be re-used verbatim on Stream-3 inference
    # to expose distribution drift.
    logger.info("computing c0-normalized XP coefficients on normal-population subset")
    ye_ok = (df["ye2024_flag"] == 0).to_numpy(dtype=bool)
    strat_ok = (strat_flag == XP_FIT_FLAG_OK).to_numpy(dtype=bool, na_value=False)
    normal_pop = ye_ok & strat_ok

    bp_c0 = proj["bp"][:, 0].astype(np.float64)
    rp_c0 = proj["rp"][:, 0].astype(np.float64)
    # Positive-c0 is a physical sanity check: c0 encodes absolute flux. Negative
    # or zero c0 indicates a degenerate fit; such rows fall out of the normal
    # population for the purposes of normalization and the log10 z-score.
    c0_ok = normal_pop & np.isfinite(bp_c0) & (bp_c0 > 0.0) & np.isfinite(rp_c0) & (rp_c0 > 0.0)
    n_c0_ok = int(c0_ok.sum())
    n_c0_rejected_from_normal = int(normal_pop.sum() - n_c0_ok)
    logger.info(
        "normal-population: %d rows; of which c0>0 finite in both bands: %d "
        "(rejected %d for nonpositive/nonfinite c0)",
        int(normal_pop.sum()),
        n_c0_ok,
        n_c0_rejected_from_normal,
    )

    # Normalized shape coefficients: bp_coef_norm_{1..54} = bp_coef_{i} / bp_coef_0.
    # Index 0 is trivially 1.0 by construction; not stored per DESIGN.
    bp_norm = np.full((len(df), HERMITE_N_BASIS), np.nan, dtype=np.float32)
    rp_norm = np.full((len(df), HERMITE_N_BASIS), np.nan, dtype=np.float32)
    for n in range(1, HERMITE_N_BASIS):
        bp_norm[c0_ok, n] = (proj["bp"][c0_ok, n] / bp_c0[c0_ok]).astype(np.float32)
        rp_norm[c0_ok, n] = (proj["rp"][c0_ok, n] / rp_c0[c0_ok]).astype(np.float32)

    # Frozen c0 z-score stats — computed on c0_ok subset, re-applied verbatim
    # at Stream-3 inference time via checkpoint-embedded (mu, sigma).
    bp_c0_log = np.log10(bp_c0[c0_ok])
    rp_c0_log = np.log10(rp_c0[c0_ok])
    bp_mu = float(np.mean(bp_c0_log))
    bp_sigma = float(np.std(bp_c0_log))
    rp_mu = float(np.mean(rp_c0_log))
    rp_sigma = float(np.std(rp_c0_log))
    logger.info(
        "frozen c0 z-score stats: BP mu=%.6f sigma=%.6f (log10 space) | "
        "RP mu=%.6f sigma=%.6f | n_reference=%d",
        bp_mu,
        bp_sigma,
        rp_mu,
        rp_sigma,
        n_c0_ok,
    )

    bp_c0_z = np.full(len(df), np.nan, dtype=np.float32)
    rp_c0_z = np.full(len(df), np.nan, dtype=np.float32)
    bp_c0_z[c0_ok] = ((bp_c0_log - bp_mu) / bp_sigma).astype(np.float32)
    rp_c0_z[c0_ok] = ((rp_c0_log - rp_mu) / rp_sigma).astype(np.float32)

    c0_zscore_frozen = {
        "bp": {"mu_log10": bp_mu, "sigma_log10": bp_sigma},
        "rp": {"mu_log10": rp_mu, "sigma_log10": rp_sigma},
        "n_reference_population": n_c0_ok,
        "log_base": 10,
        "reference_population": (
            "ye2024_flag == 0 AND xp_fit_flag_residual_high == 0 "
            "AND bp_coef_0 > 0 AND rp_coef_0 > 0"
        ),
    }

    # --- Per-coefficient z-scoring of the normalised Hermite ratios ---
    #
    # Without this, the 110-D network input spans ~17 orders of magnitude (the
    # c_i/c_0 ratio is O(0.1-1) at low order and O(1e-17) near the noise floor),
    # which a Glorot-init ReLU/GELU network cannot use — high-order coefficients
    # contribute effectively zero to the first linear layer and receive
    # effectively zero gradient. Per-coefficient z-scoring on the same
    # reference population as c0_z brings every coefficient to O(1) and lets
    # the network learn *deviations from the star-type-expected pattern* rather
    # than re-discovering which coefficients are naturally large or small.
    # Column names stay the same (``bp_coef_norm_{i}`` / ``rp_coef_norm_{i}``);
    # the semantic shift from raw-ratio → z-scored-ratio is recorded in
    # provenance at ``extra.coef_norm_zscore_frozen`` and the raw unnormalised
    # coefficients (``bp_coef_{i}`` / ``rp_coef_{i}``) remain on disk for audit.
    logger.info("computing per-coefficient z-score stats on c0_ok subset")
    bp_norm_mu = np.zeros(HERMITE_N_BASIS, dtype=np.float64)
    bp_norm_sigma = np.ones(HERMITE_N_BASIS, dtype=np.float64)
    rp_norm_mu = np.zeros(HERMITE_N_BASIS, dtype=np.float64)
    rp_norm_sigma = np.ones(HERMITE_N_BASIS, dtype=np.float64)
    # Floor to guard against degenerate σ=0 dims. 1e-30 is well below the
    # Ye-era coefficient noise floor (~1e-17) so it only triggers on a truly
    # constant column, which would be a data pathology worth surfacing.
    sigma_floor = 1e-30
    tiny_sigma_bp: list[int] = []
    tiny_sigma_rp: list[int] = []
    for i in range(1, HERMITE_N_BASIS):
        bp_col = bp_norm[c0_ok, i]
        bp_col = bp_col[np.isfinite(bp_col)]
        rp_col = rp_norm[c0_ok, i]
        rp_col = rp_col[np.isfinite(rp_col)]
        if bp_col.size < 8 or rp_col.size < 8:
            raise RuntimeError(
                f"coefficient {i}: insufficient finite values in normal pop "
                f"(bp={bp_col.size}, rp={rp_col.size}); cannot fit z-score.",
            )
        bp_norm_mu[i] = float(bp_col.mean())
        bp_norm_sigma[i] = float(bp_col.std())
        rp_norm_mu[i] = float(rp_col.mean())
        rp_norm_sigma[i] = float(rp_col.std())
        if bp_norm_sigma[i] < sigma_floor:
            tiny_sigma_bp.append(i)
            bp_norm_sigma[i] = 1.0
        if rp_norm_sigma[i] < sigma_floor:
            tiny_sigma_rp.append(i)
            rp_norm_sigma[i] = 1.0
    logger.info(
        "BP σ range: min=%.3e max=%.3e  |  RP σ range: min=%.3e max=%.3e",
        float(bp_norm_sigma[1:].min()),
        float(bp_norm_sigma[1:].max()),
        float(rp_norm_sigma[1:].min()),
        float(rp_norm_sigma[1:].max()),
    )
    if tiny_sigma_bp or tiny_sigma_rp:
        logger.warning(
            "degenerate σ (<%g) substituted with 1.0: BP %s, RP %s",
            sigma_floor,
            tiny_sigma_bp,
            tiny_sigma_rp,
        )

    bp_norm_z = np.full_like(bp_norm, np.nan, dtype=np.float32)
    rp_norm_z = np.full_like(rp_norm, np.nan, dtype=np.float32)
    for i in range(1, HERMITE_N_BASIS):
        bp_norm_z[:, i] = (
            (bp_norm[:, i].astype(np.float64) - bp_norm_mu[i]) / bp_norm_sigma[i]
        ).astype(np.float32)
        rp_norm_z[:, i] = (
            (rp_norm[:, i].astype(np.float64) - rp_norm_mu[i]) / rp_norm_sigma[i]
        ).astype(np.float32)
    # Replace normalised ratios with z-scored ratios for downstream emit.
    bp_norm = bp_norm_z
    rp_norm = rp_norm_z

    coef_norm_zscore_frozen = {
        "bp": {
            str(i): {"mu": float(bp_norm_mu[i]), "sigma": float(bp_norm_sigma[i])}
            for i in range(1, HERMITE_N_BASIS)
        },
        "rp": {
            str(i): {"mu": float(rp_norm_mu[i]), "sigma": float(rp_norm_sigma[i])}
            for i in range(1, HERMITE_N_BASIS)
        },
        "n_reference_population": n_c0_ok,
        "sigma_floor": sigma_floor,
        "tiny_sigma_substituted_bp": tiny_sigma_bp,
        "tiny_sigma_substituted_rp": tiny_sigma_rp,
        "reference_population": (
            "same normal-population subset as c0_zscore_frozen "
            "(ye2024_flag == 0 AND xp_fit_flag_residual_high == 0 "
            "AND bp_coef_0 > 0 AND rp_coef_0 > 0)"
        ),
        "stored_column_semantic": (
            "bp_coef_norm_{i} / rp_coef_norm_{i} now carry "
            "(c_i/c_0 - mu_i) / sigma_i; raw c_i/c_0 is recoverable as "
            "stored * sigma_i + mu_i; raw unnormalised c_i is in "
            "bp_coef_{i} / rp_coef_{i}."
        ),
    }

    # Assemble output DataFrame
    logger.info("assembling output DataFrame (raw diagnostic + norm ML + c0_z scalars)")
    df_out = df.drop(columns=["corrected_flux"]).copy()
    # Raw coefficients (diagnostic-only; kept for §9.2 LOOCO audit).
    bp_cols = {f"bp_coef_{n}": proj["bp"][:, n] for n in range(HERMITE_N_BASIS)}
    rp_cols = {f"rp_coef_{n}": proj["rp"][:, n] for n in range(HERMITE_N_BASIS)}
    # Normalized shape coefficients (ML inputs); index 0 trivially 1.0, not stored.
    bp_norm_cols = {f"bp_coef_norm_{n}": bp_norm[:, n] for n in range(1, HERMITE_N_BASIS)}
    rp_norm_cols = {f"rp_coef_norm_{n}": rp_norm[:, n] for n in range(1, HERMITE_N_BASIS)}
    df_out = df_out.assign(
        **bp_cols,
        **rp_cols,
        **bp_norm_cols,
        **rp_norm_cols,
        bp_c0_z=bp_c0_z,
        rp_c0_z=rp_c0_z,
        reprojection_residual_rms=proj["rms"],
        reprojection_residual_rms_bp=proj["rms_bp"],
        reprojection_residual_rms_rp=proj["rms_rp"],
        xp_fit_flag_residual_high=strat_flag,
        xp_fit_flag_residual_high_global=global_flag,
    )
    logger.info(
        "output cols: %d  (drop corrected_flux; +110 raw coeffs; +108 norm coeffs; "
        "+2 c0_z scalars; +3 residual rms; +2 flags)",
        len(df_out.columns),
    )

    tmp = dst.with_suffix(dst.suffix + ".part")
    logger.info("writing %s (tmp: %s)", dst, tmp.name)
    df_out.to_parquet(tmp, index=False)
    os.replace(tmp, dst)
    size_mb = dst.stat().st_size / 1024**2
    logger.info("wrote %s (%.1f MB, %d cols)", dst, size_mb, len(df_out.columns))

    # Provenance
    cont = thresholds["contingency"]
    pca = thresholds["pca_comparison"]
    prov = Provenance(
        output_file=str(dst.relative_to(repo)),
        script="scripts/emit_stream1_with_hermite.py",
        sources=[
            LocalSource(
                name="Pipeline-1 Stream-1 features (Ye-corrected sampled flux)",
                path=str(src.relative_to(repo)),
                sha256=src_sha,
            ),
            LocalSource(
                name="Pre-emit decision record (thresholds + contingency + PCA)",
                path=str(decisions_path.relative_to(repo)),
                sha256=sha256_file(decisions_path),
            ),
        ],
        cuts_applied=[],
        corrections=[
            f"Hermite re-projection {HERMITE_REPROJECTION_VERSION} onto 55+55 "
            "physicist-Hermite orthonormal basis (BP 360-660 nm, RP 660-990 nm, "
            "hard grid split), positive-diagonal-R sign convention",
            "xp_fit_flag_residual_high: Teff-stratified p99 per-bin thresholds "
            "computed on normal-population residuals (Ye flag=0, "
            "residual_rms < 1e-10); no-Teff rows fall back to global p99",
            "xp_fit_flag_residual_high_global: flat global p99 of normal "
            "population, retained as auxiliary diagnostic column",
            "c0 normalization: bp_coef_norm_{1..54} = bp_coef_{i} / bp_coef_0 "
            "on rows where ye2024_flag == 0 AND xp_fit_flag_residual_high == 0 "
            "AND bp_coef_0 > 0 AND rp_coef_0 > 0; NaN elsewhere. Trivial "
            "bp_coef_norm_0 ≡ 1 is NOT stored.",
            "Per-coefficient z-scoring: bp_coef_norm_{i} / rp_coef_norm_{i} "
            "are replaced in place with (c_i/c_0 - mu_i) / sigma_i using "
            "frozen (mu_i, sigma_i) fit on the same normal-population subset. "
            "Persisted in extra.coef_norm_zscore_frozen for Stream-3 inference "
            "reproducibility. Rationale: the raw 110-D c_i/c_0 vector spans "
            "~17 orders of magnitude (low-order O(0.1-1), noise-floor ~1e-17), "
            "which a Glorot-init network cannot use — high-order dims receive "
            "effectively zero gradient. Z-scoring brings every dim to O(1) so "
            "the network learns cell-conditional deviations instead of "
            "rediscovering the coefficient-magnitude hierarchy.",
            "c0 z-scoring: bp_c0_z = (log10(bp_coef_0) - mu) / sigma using "
            "frozen stats from the same normal-population subset; persisted in "
            "extra.c0_zscore_frozen for Stream-3 inference reproducibility.",
        ],
        row_count_before=len(df),
        row_count_after=len(df_out),
        notes=(
            "Post-Ye Hermite basis materialisation. `corrected_flux` (N, 330) "
            "dropped in favour of the 3-tier XP column convention: raw "
            "bp_coef_{0..54}/rp_coef_{0..54} (diagnostic only, retained for the "
            "§9.2 LOOCO attribution audit), normalized bp_coef_norm_{1..54}/"
            "rp_coef_norm_{1..54} (ML-input shape features), bp_c0_z/rp_c0_z "
            "(ML-input absolute-scale scalars with frozen z-score stats "
            "persisted in extra.c0_zscore_frozen), plus per-band + combined "
            "residual RMS (ML features) and two fit flags. The 43-D noise-"
            "floor truncation (BP[1:20] + RP[1:23]) is an ML-input-layer choice "
            "made at model-training time via FeatureLayout.truncated_43d(); the "
            "parquet preserves the full 108-D normalized basis. Ye flag=1 rows "
            "(NaN flux) carry NaN coefficients and <NA> flags; downstream users "
            "disambiguate failure modes via (ye2024_flag, "
            "xp_fit_flag_residual_high) jointly. See research_brief.md §3.1 "
            "for the empirical noise-floor / PCA finding and §14 items 11-12 "
            "for the low-b/high-Av and Teff-edge incompleteness caveats."
        ),
        extra={
            "basis_version": proj["basis_version"],
            "basis_fingerprint_sha256": proj["fingerprint"],
            "hermite_n_basis": HERMITE_N_BASIS,
            "teff_bin_edges_K": list(TEFF_BIN_EDGES_K),
            "teff_bin_labels": list(TEFF_BIN_LABELS) + ["no_teff"],
            "p99_thresholds_by_teff_bin": thresholds["p99_by_label"],
            "global_p99_normal": thresholds["global_p99"],
            "catastrophic_residual_threshold": thresholds["catastrophic"],
            "contingency_counts": cont["counts"],
            "contingency_population_medians": cont["population_stats"],
            "pca_variance_full_110D": pca["full_110"],
            "pca_variance_truncated_43D": pca["truncated_43"],
            "n_residual_high_stratified": n_high_strat,
            "n_residual_high_global": n_high_global,
            "n_hermite_valid": n_valid,
            "n_hermite_no_fit": int(len(df) - n_valid),
            "c0_zscore_frozen": c0_zscore_frozen,
            "coef_norm_zscore_frozen": coef_norm_zscore_frozen,
            "n_c0_rejected_from_normal": n_c0_rejected_from_normal,
        },
    )
    write_sidecar(prov)
    logger.info("wrote provenance sidecar")


if __name__ == "__main__":
    main()
