"""Latent-support OOD gate (convex-hull surrogate) for Pipeline-1 Stream-3.

Cache the 32-D trunk latent ``h`` of every Stream-1 training star, compute
kNN-mean-distance of each Stream-3 star to that cached manifold, and flag
stars whose distance exceeds the 99th percentile of the Stream-1 *val*-split
kNN-distance distribution.

Rationale
---------
The 108-D XP Mahalanobis gate is an **ellipsoidal** approximation of the
training manifold and misses non-convex concavities. A kNN-distance gate in
the SupCon-trained latent is the non-parametric equivalent of a convex hull
(Sun et al. 2022, "Out-of-Distribution Detection with Deep Nearest Neighbors"),
and is SOTA-comparable for backbones trained without explicit distance-aware
constraints (DUQ / SNGP). This script is the preliminary step (option A in
the upgrade ladder); a full bi-Lipschitz retrain is a separate decision.

Outputs
-------
- ``<output>.parquet`` — one row per Stream-3 star:
  ``{source_id, latent_knn_dist, latent_support_flag, sample}``
- ``<output>.parquet.provenance.json`` — input SHAs, encoder ckpt SHA,
  threshold value, val-split summary stats.

Usage
-----
    PYTHONPATH=src python scripts/build_latent_support_gate.py \\
        --train-features data/processed/pipeline1_features_stream1.parquet \\
        --stream3-features data/processed/pipeline1_features_stream3.parquet \\
        --checkpoint models/main/xp_abundances/<ensemble>/member_seed0/xp_abundances_main_joint_seed0_best.pt \\
        --frozen-stats data/processed/pipeline1_features_stream1.provenance.json \\
        --output data/processed/pipeline1_latent_support_stream3.parquet
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from arqueogal.data.frozen_stats import apply_frozen_zscore, load_frozen_zscore_stats
from arqueogal.xp_abundances.main.data import (
    FeatureLayout,
    LabelTiers,
    stratified_split_ids,
)
from arqueogal.xp_abundances.main.inference import load_ensemble

_LOG = logging.getLogger("latent_gate")

REPO_ROOT = Path(__file__).resolve().parents[1]

KNN_K_DEFAULT: int = 10
QUERY_CHUNK_DEFAULT: int = 2048
THRESHOLD_QUANTILE_DEFAULT: float = 0.99
S3_ROW_CHUNK_DEFAULT: int = 50_000  # rows per Stream-3 assemble+encode+kNN pass


# --- schema detection (mirrors scripts/run_pipeline1_inference.py) ------------

def _detect_schema(cols: set[str]) -> str:
    has_z = "bp_c0_z" in cols and "rp_c0_z" in cols
    has_log = "bp_c0_log" in cols and "rp_c0_log" in cols
    if has_z and has_log:
        raise ValueError("both z-scored and log c0 columns present")
    if has_z:
        return "zscored"
    if has_log:
        return "raw"
    raise ValueError("neither z-scored nor log c0 columns present")


def _assemble_features(
    df: pd.DataFrame, layout: FeatureLayout, stats, schema: str,
) -> np.ndarray:
    """Build the (N, input_dim) feature matrix in layout order.

    Mirrors ``scripts/run_pipeline1_inference.py::_assemble_feature_matrix``.
    Raw-schema inputs (Stream 3) get frozen z-scored here; z-scored inputs
    (Stream 1) pass through unchanged.
    """
    bp_cols = list(layout.bp_coef_cols)
    rp_cols = list(layout.rp_coef_cols)

    if schema == "raw":
        bp_all = sorted(
            (c for c in df.columns if c.startswith("bp_coef_norm_")),
            key=lambda s: int(s.removeprefix("bp_coef_norm_")),
        )
        rp_all = sorted(
            (c for c in df.columns if c.startswith("rp_coef_norm_")),
            key=lambda s: int(s.removeprefix("rp_coef_norm_")),
        )
        bp_full = df[bp_all].to_numpy(dtype=np.float64)
        rp_full = df[rp_all].to_numpy(dtype=np.float64)
        bp_c0_log = df["bp_c0_log"].to_numpy(dtype=np.float64)
        rp_c0_log = df["rp_c0_log"].to_numpy(dtype=np.float64)
        bp_z, rp_z, bp_c0_z, rp_c0_z = apply_frozen_zscore(
            bp_full, rp_full, bp_c0_log, rp_c0_log, stats,
        )
        bp_idx = {int(c.removeprefix("bp_coef_norm_")): k for k, c in enumerate(bp_all)}
        rp_idx = {int(c.removeprefix("rp_coef_norm_")): k for k, c in enumerate(rp_all)}
        bp_coef = bp_z[:, [bp_idx[i] for i in layout.xp_bp_indices]]
        rp_coef = rp_z[:, [rp_idx[i] for i in layout.xp_rp_indices]]
        c0 = {"bp_c0_z": bp_c0_z, "rp_c0_z": rp_c0_z}
    else:
        bp_coef = df[bp_cols].to_numpy(dtype=np.float64)
        rp_coef = df[rp_cols].to_numpy(dtype=np.float64)
        c0 = {n: df[n].to_numpy(dtype=np.float64) for n in layout.xp_scalar_cols}

    parts: list[np.ndarray] = [bp_coef, rp_coef]
    for n in layout.xp_scalar_cols:
        parts.append(c0[n][:, None])
    for col in (*layout.residual_cols, *layout.aux_cols):
        parts.append(df[col].to_numpy(dtype=np.float64)[:, None])
    X = np.concatenate(parts, axis=1).astype(np.float32)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if X.shape[1] != layout.input_dim:
        raise RuntimeError(
            f"assembled matrix width {X.shape[1]} != layout.input_dim {layout.input_dim}",
        )
    return X


# --- embedding -----------------------------------------------------------------

@torch.no_grad()
def _encode(
    X: np.ndarray, model, device: torch.device, batch: int = 8192,
) -> np.ndarray:
    """Push rows of ``X`` through ``model.encoder`` → trunk latent ``h``.

    Some training rows have extreme z-scored coefficient outliers
    (|x| > 10²¹, ~0.05% of Stream-1) that overflow the Linear+LayerNorm
    math and produce NaN latents. Training masked these via NaN-safe loss
    arithmetic; the gate doesn't have that path, so we leave NaN in the
    output and let callers filter reference rows / flag query rows as OOD.
    """
    model.eval()
    n, _ = X.shape
    out = np.empty((n, model.config.latent_dim), dtype=np.float32)
    for i in range(0, n, batch):
        xb = torch.as_tensor(X[i:i + batch], device=device)
        h, _z = model.encoder(xb)
        out[i:i + batch] = h.cpu().numpy()
    return out


def _filter_finite_rows(H: np.ndarray, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (H_clean, finite_mask) — drop rows that contain any non-finite entry."""
    mask = np.isfinite(H).all(axis=1)
    n_bad = int((~mask).sum())
    if n_bad:
        _LOG.warning(
            "%s: %d / %d rows (%.3f%%) have non-finite latents — "
            "dropping from reference",
            tag, n_bad, len(H), 100.0 * n_bad / len(H),
        )
    return H[mask], mask


# --- kNN distance -------------------------------------------------------------

@torch.no_grad()
def _knn_mean_distance(
    H_query: np.ndarray,
    ref: torch.Tensor,
    ref_sq: torch.Tensor,
    *,
    k: int,
    chunk: int,
    device: torch.device,
) -> np.ndarray:
    """Per-row mean Euclidean distance to the ``k`` nearest rows of the ref set.

    ``ref`` and ``ref_sq`` are pre-uploaded to ``device`` by the caller so we
    don't re-allocate the 239k×32 reference on every invocation.
    """
    n = H_query.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(0, n, chunk):
        q_np = H_query[i:i + chunk]
        q = torch.as_tensor(q_np, device=device, dtype=torch.float32)
        # Manual squared-Euclidean via (a-b)² = ‖a‖² - 2 a·b + ‖b‖², with
        # clamp to ≥0 before sqrt. torch.cdist's mm-accelerated default can
        # produce tiny negative residuals from float32 fused-multiply
        # cancellation, which then NaN-out in sqrt. Clamping + NaN-to-inf
        # replacement below makes this robust to both.
        q_sq = (q * q).sum(dim=1, keepdim=True)  # (chunk, 1)
        d2 = q_sq + ref_sq[None, :] - 2.0 * (q @ ref.T)
        d = torch.sqrt(d2.clamp_min(0.0))  # (chunk, N_ref)
        # Query rows with NaN latents → all distances NaN → mark as inf so
        # topk picks inf values and the query's kNN-mean distance comes out
        # as +inf (guaranteed OOD).
        d = torch.where(torch.isnan(d), torch.full_like(d, float("inf")), d)
        dk, _ = torch.topk(d, k=k, dim=1, largest=False)
        out[i:i + chunk] = dk.mean(dim=1).cpu().numpy()
    return out


def _free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- provenance --------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


@dataclass
class GateRun:
    train_sha: str
    s3_sha: str
    ckpt_sha: str
    frozen_sha: str
    n_train_ref: int
    n_val_ref: int
    n_s3: int
    k: int
    threshold_quantile: float
    threshold_value: float
    val_stats: dict
    s3_stats: dict


# --- main --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--stream3-features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="single ensemble-member checkpoint (.pt)")
    parser.add_argument("--frozen-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=KNN_K_DEFAULT)
    parser.add_argument("--threshold-quantile", type=float, default=THRESHOLD_QUANTILE_DEFAULT)
    parser.add_argument("--chunk", type=int, default=QUERY_CHUNK_DEFAULT)
    parser.add_argument("--s3-row-chunk", type=int, default=S3_ROW_CHUNK_DEFAULT,
                        help="Stream-3 rows processed per assemble+encode+kNN pass")
    parser.add_argument("--split-seed", type=int, default=0,
                        help="must match the TrainingConfig.split_seed used at training")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOG.info("device=%s", device)

    # --- Load encoder ---------------------------------------------------------
    _LOG.info("loading checkpoint: %s", args.checkpoint)
    members = load_ensemble([args.checkpoint], device=device)
    if len(members) != 1:
        raise RuntimeError(f"expected 1 checkpoint, loaded {len(members)}")
    model = members[0].model
    latent_dim = model.config.latent_dim
    _LOG.info("model loaded, latent_dim=%d", latent_dim)

    # --- Frozen stats ---------------------------------------------------------
    stats = load_frozen_zscore_stats(args.frozen_stats)
    _LOG.info("frozen stats loaded, basis_fp=%s…", stats.basis_fingerprint[:16])

    layout = FeatureLayout()

    # --- Stream-1 train/val reference ----------------------------------------
    _LOG.info("loading Stream-1 features: %s", args.train_features)
    df_s1 = pd.read_parquet(args.train_features)
    _LOG.info("  Stream-1 rows: %d", len(df_s1))

    schema_s1 = _detect_schema(set(df_s1.columns))
    _LOG.info("  Stream-1 schema: %s", schema_s1)

    # Reproduce the training train/val split exactly.
    fracs = (1.0 - args.val_frac - args.val_frac, args.val_frac, args.val_frac)
    split_ids = stratified_split_ids(df_s1, fracs=fracs, seed=args.split_seed)
    train_ids = set(int(x) for x in split_ids["train"])
    val_ids = set(int(x) for x in split_ids["val"])
    _LOG.info("  train=%d val=%d test=%d",
              len(split_ids["train"]), len(split_ids["val"]), len(split_ids["test"]))

    df_s1_train = df_s1[df_s1["source_id"].isin(train_ids)].reset_index(drop=True)
    df_s1_val = df_s1[df_s1["source_id"].isin(val_ids)].reset_index(drop=True)
    _LOG.info("  df_s1_train=%d df_s1_val=%d", len(df_s1_train), len(df_s1_val))

    # --- Assemble + encode train reference ------------------------------------
    _LOG.info("assembling + encoding Stream-1 train reference ...")
    X_train = _assemble_features(df_s1_train, layout, stats, schema_s1)
    H_train_raw = _encode(X_train, model, device)
    H_train, _train_finite = _filter_finite_rows(H_train_raw, "H_train")
    _LOG.info("  H_train shape=%s (%d dropped)",
              H_train.shape, len(H_train_raw) - len(H_train))

    # --- Encode val split -----------------------------------------------------
    _LOG.info("assembling + encoding Stream-1 val ...")
    X_val = _assemble_features(df_s1_val, layout, stats, schema_s1)
    H_val_raw = _encode(X_val, model, device)
    H_val, _val_finite = _filter_finite_rows(H_val_raw, "H_val")

    # Upload reference to GPU once and reuse across val + Stream-3 passes.
    ref = torch.as_tensor(H_train, device=device, dtype=torch.float32)
    ref_sq = (ref * ref).sum(dim=1)

    _LOG.info("kNN (k=%d) val → train ...", args.k)
    d_val = _knn_mean_distance(
        H_val, ref, ref_sq, k=args.k, chunk=args.chunk, device=device,
    )
    d_val_finite = d_val[np.isfinite(d_val)]
    tau = float(np.quantile(d_val_finite, args.threshold_quantile))
    val_stats = {
        "n": int(len(d_val)),
        "n_finite": int(len(d_val_finite)),
        "median": float(np.median(d_val_finite)), "mean": float(np.mean(d_val_finite)),
        "std": float(np.std(d_val_finite)),
        "p50": float(np.quantile(d_val_finite, 0.50)),
        "p95": float(np.quantile(d_val_finite, 0.95)),
        "p99": float(np.quantile(d_val_finite, 0.99)),
        "p99.5": float(np.quantile(d_val_finite, 0.995)),
    }
    _LOG.info("  val stats: %s", val_stats)
    _LOG.info("  threshold (q=%.3f) = %.6f", args.threshold_quantile, tau)

    # Free Stream-1 state before Stream-3. We keep ``ref`` / ``ref_sq`` on GPU
    # (≈30 MiB) + ``H_train`` on host (for the provenance row-count only), and
    # drop everything else so the Stream-3 pass has headroom.
    n_train_ref_kept = int(len(H_train))
    n_val_ref_kept = int(len(H_val))
    del H_train, H_train_raw, H_val, H_val_raw, X_train, X_val
    del df_s1, df_s1_train, df_s1_val, split_ids
    del d_val, d_val_finite
    _free_cuda()

    # --- Stream 3 (chunked assemble+encode+kNN) -------------------------------
    _LOG.info("loading Stream-3 features: %s", args.stream3_features)
    df_s3 = pd.read_parquet(args.stream3_features)
    n_s3 = len(df_s3)
    _LOG.info("  Stream-3 rows: %d", n_s3)
    schema_s3 = _detect_schema(set(df_s3.columns))
    _LOG.info("  Stream-3 schema: %s", schema_s3)

    source_ids_s3 = df_s3["source_id"].to_numpy()
    sample_col = (
        df_s3["sample"].to_numpy() if "sample" in df_s3.columns else None
    )

    _LOG.info(
        "assemble+encode+kNN Stream-3 in row-chunks (size=%d) ...",
        args.s3_row_chunk,
    )
    d_s3 = np.empty(n_s3, dtype=np.float32)
    for i in range(0, n_s3, args.s3_row_chunk):
        j = min(i + args.s3_row_chunk, n_s3)
        df_chunk = df_s3.iloc[i:j]
        X_chunk = _assemble_features(df_chunk, layout, stats, schema_s3)
        H_chunk = _encode(X_chunk, model, device)
        d_s3[i:j] = _knn_mean_distance(
            H_chunk, ref, ref_sq, k=args.k, chunk=args.chunk, device=device,
        )
        del X_chunk, H_chunk, df_chunk
        _free_cuda()
        _LOG.info(
            "  [%d/%d] chunk rows=%d d_med=%.3f flag_rate=%.4f",
            j, n_s3, j - i,
            float(np.median(d_s3[i:j][np.isfinite(d_s3[i:j])])),
            float((d_s3[i:j] > tau).mean()),
        )

    # df_s3 no longer needed except for provenance row count; drop before output.
    del df_s3
    _free_cuda()
    # Non-finite distances ⇒ encoder emitted NaN latents for those rows
    # (extreme aux / coef values). Those rows are flagged OOD automatically
    # because inf > τ, and the finite subset drives the summary stats.
    d_s3_finite = d_s3[np.isfinite(d_s3)]
    flag = d_s3 > tau
    s3_stats = {
        "n": int(len(d_s3)),
        "n_finite": int(len(d_s3_finite)),
        "n_infinite_latent": int(len(d_s3) - len(d_s3_finite)),
        "median": float(np.median(d_s3_finite)),
        "mean": float(np.mean(d_s3_finite)),
        "p50": float(np.quantile(d_s3_finite, 0.50)),
        "p95": float(np.quantile(d_s3_finite, 0.95)),
        "p99": float(np.quantile(d_s3_finite, 0.99)),
        "flag_rate": float(flag.mean()),
        "flag_count": int(flag.sum()),
    }
    _LOG.info("  s3 stats: %s", s3_stats)

    # --- Output parquet -------------------------------------------------------
    out_cols = {
        "source_id": source_ids_s3,
        "latent_knn_dist": d_s3.astype(np.float32),
        "latent_support_flag": flag,
    }
    if sample_col is not None:
        out_cols["sample"] = sample_col
    df_out = pd.DataFrame(out_cols)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(args.output, index=False)
    _LOG.info("wrote %s (%d rows)", args.output, len(df_out))

    # --- Provenance -----------------------------------------------------------
    run = GateRun(
        train_sha=_sha256(args.train_features),
        s3_sha=_sha256(args.stream3_features),
        ckpt_sha=_sha256(args.checkpoint),
        frozen_sha=_sha256(args.frozen_stats),
        n_train_ref=n_train_ref_kept,
        n_val_ref=n_val_ref_kept,
        n_s3=int(n_s3),
        k=int(args.k),
        threshold_quantile=float(args.threshold_quantile),
        threshold_value=tau,
        val_stats=val_stats,
        s3_stats=s3_stats,
    )
    prov = {
        "script": "scripts/build_latent_support_gate.py",
        "output_file": str(args.output),
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_sha": _git_sha(),
        "device": str(device),
        "inputs": {
            "train_features": {"path": str(args.train_features), "sha256": run.train_sha},
            "stream3_features": {"path": str(args.stream3_features), "sha256": run.s3_sha},
            "checkpoint": {"path": str(args.checkpoint), "sha256": run.ckpt_sha},
            "frozen_stats": {"path": str(args.frozen_stats), "sha256": run.frozen_sha},
        },
        "config": {
            "k": run.k,
            "threshold_quantile": run.threshold_quantile,
            "threshold_value": run.threshold_value,
            "latent_dim": int(latent_dim),
            "split_seed": int(args.split_seed),
            "val_frac": float(args.val_frac),
        },
        "reference": {
            "n_train_ref": run.n_train_ref,
            "n_val_ref": run.n_val_ref,
            "val_knn_dist_stats": run.val_stats,
        },
        "stream3": {
            "n_rows": run.n_s3,
            "knn_dist_stats": run.s3_stats,
        },
        "frozen_stats_basis_fingerprint_sha256": stats.basis_fingerprint,
    }
    prov_path = args.output.with_suffix(args.output.suffix + ".provenance.json")
    with prov_path.open("w") as f:
        json.dump(prov, f, indent=2, default=str)
    _LOG.info("wrote provenance: %s", prov_path)

    # Suppress the reference to LabelTiers import so the linter doesn't complain;
    # kept around because downstream callers often expect it available from this
    # module's imports.
    _ = LabelTiers  # noqa: F841


if __name__ == "__main__":
    main()
