"""Run latent-kNN rescue on a Pipeline-1 inference parquet.

For each star in the inference parquet, find its K nearest neighbours in the
encoder-projection (z) space within the training parquet, then write a
companion parquet with per-element neighbour-label summaries.

Typical usage::

    python scripts/run_knn_rescue.py \\
        --ensemble-dir models/main/xp_abundances/20260425_6b96c06_cd1cbb9_ensemble_5label \\
        --train-parquet data/processed/pipeline1_features_stream1.parquet \\
        --infer-parquet data/processed/pipeline1_features_stream3.parquet \\
        --frozen-stats data/processed/pipeline1_features_stream1.provenance.json \\
        --output data/processed/pipeline1_knn_rescue_stream3.parquet

The output parquet has columns ``source_id`` plus the schema-aligned kNN
columns (``knn_<elem>_med/p25/p75/iqr/std``, ``knn_top_distance``,
``knn_median_distance``) — see ``master_schema._PIPELINE1_KNN_RESCUE_COLS``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from arqueogal.data.frozen_stats import apply_frozen_zscore, load_frozen_zscore_stats
from arqueogal.xp_abundances.main.data import FeatureLayout, LabelTiers, load_arrays
from arqueogal.xp_abundances.main.inference import load_ensemble
from arqueogal.xp_abundances.main.knn_rescue import (
    compute_latents,
    gpu_knn_search,
    summarize_neighbors,
    write_artifact,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("run_knn_rescue")

REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ENSEMBLE = REPO_ROOT / "models/main/xp_abundances/20260425_6b96c06_cd1cbb9_ensemble_5label"
_DEFAULT_TRAIN = REPO_ROOT / "data/processed/pipeline1_features_stream1.parquet"
_DEFAULT_INFER = REPO_ROOT / "data/processed/pipeline1_features_stream3.parquet"
_DEFAULT_FROZEN = REPO_ROOT / "data/processed/pipeline1_features_stream1.provenance.json"
_DEFAULT_OUTPUT = REPO_ROOT / "data/processed/pipeline1_knn_rescue.parquet"


def _load_inference_features(
    parquet: Path, layout: FeatureLayout, frozen_stats_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Load an inference parquet and apply frozen z-scoring to the c0 columns.

    Stream 3 features carry ``bp_c0_log`` / ``rp_c0_log`` rather than the
    z-scored ``bp_c0_z`` / ``rp_c0_z`` (which are training-time only). We
    reproduce the training-time z-score using the frozen v1 stats so that
    the encoder sees inputs in the same scale as training.

    Returns
    -------
    X : np.ndarray
        ``(N, F)`` feature matrix in the layout's expected column order.
    source_id : np.ndarray
        ``(N,)`` source identifiers as ``int64``.
    """
    feature_cols = list(layout.all_required_columns)
    raw_cols = ["source_id"] + [
        c.replace("bp_c0_z", "bp_c0_log").replace("rp_c0_z", "rp_c0_log") for c in feature_cols
    ]
    df = pd.read_parquet(parquet, columns=list(dict.fromkeys(raw_cols)))
    sid = df["source_id"].to_numpy(dtype=np.int64)

    stats = load_frozen_zscore_stats(frozen_stats_path)
    bp_norm_cols = [f"bp_coef_norm_{i}" for i in range(1, 55)]
    rp_norm_cols = [f"rp_coef_norm_{i}" for i in range(1, 55)]
    bp_norm = df[bp_norm_cols].to_numpy(dtype=np.float64)
    rp_norm = df[rp_norm_cols].to_numpy(dtype=np.float64)
    bp_c0_log = df["bp_c0_log"].to_numpy(dtype=np.float64)
    rp_c0_log = df["rp_c0_log"].to_numpy(dtype=np.float64)
    bp_z, rp_z, bp_c0_z, rp_c0_z = apply_frozen_zscore(
        bp_norm, rp_norm, bp_c0_log, rp_c0_log, stats
    )

    df_z = df.copy()
    for i in range(1, 55):
        df_z[f"bp_coef_norm_{i}"] = bp_z[:, i - 1]
        df_z[f"rp_coef_norm_{i}"] = rp_z[:, i - 1]
    df_z["bp_c0_z"] = bp_c0_z
    df_z["rp_c0_z"] = rp_c0_z
    X = np.column_stack([df_z[c].to_numpy(dtype=np.float32) for c in feature_cols])
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X, sid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-dir", type=Path, default=_DEFAULT_ENSEMBLE)
    parser.add_argument(
        "--member", type=int, default=0, help="Ensemble member index used for kNN encoder."
    )
    parser.add_argument("--train-parquet", type=Path, default=_DEFAULT_TRAIN)
    parser.add_argument("--infer-parquet", type=Path, default=_DEFAULT_INFER)
    parser.add_argument("--frozen-stats", type=Path, default=_DEFAULT_FROZEN)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument(
        "--device", default=None, help="Override torch device (e.g. 'cuda:0', 'cpu')."
    )
    args = parser.parse_args()

    # load_ensemble does flat glob *.pt; production checkpoints live at
    # ensemble_dir/member_seed*/best.pt. Resolve here so both layouts work.
    ckpt_paths = sorted(args.ensemble_dir.glob("*.pt"))
    if not ckpt_paths:
        ckpt_paths = sorted(args.ensemble_dir.glob("member_seed*/*_best.pt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"no checkpoints under {args.ensemble_dir}")
    members = load_ensemble(ckpt_paths)
    if not 0 <= args.member < len(members):
        raise ValueError(f"--member {args.member} out of range [0, {len(members)})")
    model = members[args.member].model
    device = (
        torch.device(args.device)
        if args.device
        else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    )
    model.to(device)
    _LOG.info("loaded ensemble member %d from %s on %s", args.member, args.ensemble_dir, device)

    layout = FeatureLayout()
    tiers = LabelTiers.five_label()

    _LOG.info("loading training arrays from %s", args.train_parquet)
    train = load_arrays(args.train_parquet, layout, tiers, include_label_errors=False)
    X_tr = np.asarray(train["X"])
    Y_tr = np.asarray(train["Y"])
    sid_tr = np.asarray(train["source_id"])
    np.nan_to_num(X_tr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    keep = np.isfinite(Y_tr).all(axis=1)
    X_tr, Y_tr, sid_tr = X_tr[keep], Y_tr[keep], sid_tr[keep]
    _, first_idx = np.unique(sid_tr, return_index=True)
    first_idx = np.sort(first_idx)
    X_tr, Y_tr, sid_tr = X_tr[first_idx], Y_tr[first_idx], sid_tr[first_idx]
    _LOG.info("training set: %d stars (post-dedup)", len(X_tr))

    _LOG.info("computing training latents...")
    z_tr = compute_latents(model, X_tr, device=device)

    _LOG.info("loading inference features from %s", args.infer_parquet)
    X_q, sid_q = _load_inference_features(args.infer_parquet, layout, args.frozen_stats)
    _LOG.info("inference set: %d stars", len(X_q))

    _LOG.info("computing inference latents...")
    z_q = compute_latents(model, X_q, device=device)

    _LOG.info("running GPU kNN: K=%d", args.k)
    distances, indices = gpu_knn_search(
        z_tr, z_q, k=args.k, device=device, batch=args.batch, progress=True
    )

    artefact = summarize_neighbors(Y_tr, indices, distances, source_id=sid_q)
    out_path = write_artifact(artefact, args.output)
    _LOG.info("wrote %s (%d rows)", out_path, len(artefact.source_id))

    sidecar = out_path.with_suffix(".knn_rescue.json")
    sidecar.write_text(
        json.dumps(
            {
                "ensemble_dir": str(args.ensemble_dir),
                "member": args.member,
                "train_parquet": str(args.train_parquet),
                "infer_parquet": str(args.infer_parquet),
                "k": int(args.k),
                "n_train": int(len(X_tr)),
                "n_query": int(len(X_q)),
                "label_order": ["teff", "logg", "mh", "alpha_m", "mg_h"],
                "summary_columns": ["med", "p25", "p75", "iqr", "std"],
            },
            indent=2,
        )
    )
    _LOG.info("wrote sidecar %s", sidecar)


if __name__ == "__main__":
    main()
