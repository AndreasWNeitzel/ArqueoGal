"""Emit per-epoch predictions from a training cadence directory.

Walks ``<run_dir>/cadence/*.pt`` (one weight checkpoint per training epoch),
loads each as a 1-member ensemble, runs inference on a fixed evaluation
subset of the Stream-1 features parquet (the val + test splits at
seed=0 — deterministic across re-runs), and writes one parquet per epoch
to ``data/processed/cadence_predictions/<run_id>/epoch_####.parquet``.

These per-epoch parquets are the data layer the cadence-animation gallery
(Y31+) consumes. Source ID, predicted means, and predicted σ are stored;
truth labels live in the Stream-1 features parquet and are joined in by
the animation script — they do not need to be duplicated per epoch.

Storage cost per epoch on the canonical Stream-1 val + test cohort
(~88k stars × 5 labels × 2 floats + source_id) ≈ 4 MB.
At checkpoint_every_n_epochs=1 over 200 pretrain + 20 finetune epochs,
the on-disk footprint is ~0.9 GB. The cadence weight checkpoints
themselves are ~5 MB each → ~1.1 GB additional; both can coexist on
the project's disk budget.

Usage:

    uv run python scripts/emit_cadence_predictions.py \\
        --run-dir models/main/xp_abundances/<DATE_SHA_CFGHASH>_finetune \\
        --output-root data/processed/cadence_predictions
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from torch.utils.data import DataLoader

from arqueogal.xp_abundances.main.data import (
    FeatureLayout, FeatureScaler, LabelTiers, XpAbundanceDataset,
    load_arrays, stratified_split_ids,
)
from arqueogal.xp_abundances.main.inference import load_ensemble, predict_ensemble

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("emit_cadence_predictions")


def _layout_and_tiers(blob: dict):
    """Pick the (FeatureLayout, LabelTiers) pair the checkpoint was built on.

    FeatureLayout is the default 139-D production layout — matches what the
    inference driver reconstructs (`scripts/run_pipeline1_inference.py:1239`).
    LabelTiers is selected by the n_labels recorded in the checkpoint blob.
    """
    n_labels = int(blob.get("n_labels", 0))
    layout = FeatureLayout()  # default production layout, 139-D
    if n_labels == 5:
        return layout, LabelTiers.five_label()
    if n_labels == 21:
        return layout, LabelTiers()
    if n_labels == 2:
        return layout, LabelTiers.two_label()
    raise ValueError(f"unrecognised n_labels={n_labels} in checkpoint")


def _scaler_from_blob(blob: dict) -> FeatureScaler:
    """Reconstruct the FeatureScaler from a checkpoint blob.

    Only the ``_best.pt`` carries ``feature_scaler``; cadence checkpoints
    don't (``_save_cadence_checkpoint`` doesn't pass it). The scaler is
    fit at start-of-training and frozen — same blob across all epochs of
    one run — so the caller looks it up once from ``_best.pt`` and
    reuses for every cadence ckpt.
    """
    fs = blob["feature_scaler"]
    return FeatureScaler(
        mean=np.asarray(fs["mean"], dtype=np.float32),
        scale=np.asarray(fs["scale"], dtype=np.float32),
        feature_names=tuple(fs["feature_names"]),
        log10_mask=np.asarray(fs["log10_mask"], dtype=bool),
        apply_mask=np.asarray(fs["apply_mask"], dtype=bool),
    )


def _load_run_feature_scaler(run_dir: Path) -> FeatureScaler:
    """Pick the run's _best.pt and pull the FeatureScaler from it."""
    best = sorted(run_dir.glob("*_best.pt"))
    if not best:
        raise FileNotFoundError(
            f"no _best.pt in {run_dir} — needed to recover the FeatureScaler"
        )
    blob = torch.load(best[0], map_location="cpu", weights_only=False)
    return _scaler_from_blob(blob)


def _refit_feature_scaler(parquet: Path, layout: FeatureLayout,
                            tiers: LabelTiers) -> FeatureScaler:
    """Re-fit the FeatureScaler from the training parquet's seed=0 partition.

    Mirrors the recipe in ``training.build_dataloaders`` exactly so the
    re-fit scaler is bit-identical to what the model trained with.

    Used as a fallback when the checkpoint blob doesn't carry the scaler
    (the supervised finetune driver omits ``feature_scaler=`` from
    ``save_checkpoint``; see scripts/run_supervised_finetune.py:222-238).
    """
    df = pd.read_parquet(parquet,
                         columns=["source_id", "fe_h_apogee",
                                  "teff_apogee", "b_deg"])
    df = df.drop_duplicates("source_id", keep="first").reset_index(drop=True)
    split = stratified_split_ids(df, seed=0)
    arrs = load_arrays(parquet, layout, tiers,
                       include_label_errors=False, include_source_id=True)
    train_mask = np.isin(arrs["source_id"], split["train"])
    xp_passthrough_cols = (
        *layout.bp_coef_cols, *layout.rp_coef_cols, *layout.xp_scalar_cols,
    )
    return FeatureScaler.fit(
        arrs["X"][train_mask],
        feature_names=layout.all_required_columns,
        residual_cols=layout.residual_cols,
        xp_already_scaled_cols=xp_passthrough_cols,
    )


def _build_eval_loader(parquet: Path, blob: dict, feature_scaler: FeatureScaler,
                       batch_size: int = 1024):
    """Build a deterministic val + test loader on the Stream-1 parquet.

    Uses ``stratified_split_ids(seed=0)`` — the same split the training
    pipeline carves, so the eval cohort is exactly the held-out stars
    every other diagnostic touches. Applies the run's FeatureScaler
    (sourced once from _best.pt) to X — without this, predictions
    collapse near the training-set mean regardless of input.
    """
    layout, tiers = _layout_and_tiers(blob)

    df = pd.read_parquet(parquet, columns=["source_id", "fe_h_apogee",
                                            "teff_apogee", "b_deg"])
    df = df.drop_duplicates("source_id", keep="first").reset_index(drop=True)
    split = stratified_split_ids(df, seed=0)
    keep_ids = set(np.concatenate([split["val"], split["test"]]).tolist())

    # Load full arrays (raw, may include per-visit duplicates) then dedup on
    # source_id keeping the first occurrence. Match the convention used by
    # build_dataloaders in training.py — first-row-wins is deterministic on
    # the parquet's stored order.
    arrs = load_arrays(parquet, layout, tiers,
                       include_label_errors=False, include_source_id=True)
    src_all = arrs["source_id"]
    _, first_idx = np.unique(src_all, return_index=True)
    first_idx = np.sort(first_idx)  # preserve original parquet ordering
    src_dedup = src_all[first_idx]
    X_dedup = arrs["X"][first_idx]

    keep_mask = np.array([sid in keep_ids for sid in src_dedup], dtype=bool)
    X = X_dedup[keep_mask]
    src = src_dedup[keep_mask]

    # Apply the training-time FeatureScaler — production inference does this
    # before the encoder sees the input. Without it, predictions collapse to
    # near-mean across the cohort.
    X = feature_scaler.transform(X)
    # NaN-impute after scaling — log10 produces NaN on non-positive residuals
    # and aux columns can be NaN by design. Production inference does this
    # immediately after FeatureScaler (run_pipeline1_inference.py:1062).
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    # Match _build_loader: zero-filled Y so the dataset's __getitem__ contract
    # holds even when truth labels are not needed at inference time.
    Y = np.zeros((X.shape[0], tiers.n_labels), dtype=X.dtype)
    ds = XpAbundanceDataset(X, Y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader, src, tiers


def _epoch_from_path(p: Path) -> int:
    stem = p.stem
    if "_epoch" not in stem:
        return -1
    try:
        return int(stem.split("_epoch")[-1].split("_")[0])
    except ValueError:
        return -1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Run directory containing a cadence/ subdir of *.pt files.")
    ap.add_argument("--features-parquet", type=Path,
                    default=REPO / "data/processed/pipeline1_features_stream1_kiel.parquet")
    ap.add_argument("--output-root", type=Path,
                    default=REPO / "data/processed/cadence_predictions")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-emit even if the per-epoch parquet already exists.")
    args = ap.parse_args()

    cadence_dir = args.run_dir / "cadence"
    if not cadence_dir.is_dir():
        logger.error("missing cadence dir: %s", cadence_dir)
        return 1

    ckpts = sorted(cadence_dir.glob("*_epoch*.pt"), key=_epoch_from_path)
    if not ckpts:
        logger.error("no cadence checkpoints in %s", cadence_dir)
        return 1
    logger.info("found %d cadence checkpoints in %s", len(ckpts), cadence_dir)

    out_dir = args.output_root / args.run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Recover the FeatureScaler. Try _best.pt first; if absent (driver bug,
    # see scripts/run_supervised_finetune.py:222), re-fit from the training
    # parquet — bit-identical to training-time fit because the recipe is
    # deterministic on the seed=0 train partition.
    first = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    layout, _tiers = _layout_and_tiers(first)
    try:
        feature_scaler = _load_run_feature_scaler(args.run_dir)
        logger.info("feature scaler loaded from _best.pt")
    except (FileNotFoundError, KeyError):
        logger.warning("no feature_scaler in _best.pt; re-fitting from "
                       "training parquet (driver omitted feature_scaler=)")
        feature_scaler = _refit_feature_scaler(args.features_parquet,
                                                layout, _tiers)
    logger.info("scaler: %d aux/residual columns scaled, %d passthrough",
                int(feature_scaler.apply_mask.sum()),
                int((~feature_scaler.apply_mask).sum()))

    loader, src_ids, tiers = _build_eval_loader(args.features_parquet, first,
                                                  feature_scaler,
                                                  batch_size=args.batch_size)
    label_names = list(tiers.all_labels)
    label_keys = [n.replace("_apogee", "") for n in label_names]
    logger.info("eval cohort: %d stars × %d labels (%s)",
                len(src_ids), len(label_names), ",".join(label_keys))

    n_done = 0
    for ckpt in ckpts:
        epoch = _epoch_from_path(ckpt)
        out = out_dir / f"epoch_{epoch:04d}.parquet"
        if out.exists() and not args.overwrite:
            logger.info("epoch %04d: parquet exists, skipping", epoch)
            n_done += 1
            continue

        members = load_ensemble([ckpt], device=device)
        pred = predict_ensemble(members, loader, device=device,
                                amp_dtype=torch.bfloat16
                                if device.type == "cuda" else None)

        # CRITICAL: pred.mu / pred.sigma_total are in BLOCK order (the
        # Cholesky block layout's slot order), but label_scaler_mean and
        # label_scaler_scale are in HUMAN order. Apply the same permutation
        # production inference does (run_pipeline1_inference.py:738-769).
        ckpt_blob = members[0].blob
        bl = ckpt_blob["block_layout"]  # stored as dict
        block_order = list(bl["label_order_block"])
        human_order = list(bl["label_order_human"])
        perm = np.asarray([block_order.index(n) for n in human_order],
                          dtype=np.int64)
        ls_mean = np.asarray(ckpt_blob["label_scaler_mean"], dtype=np.float64)
        ls_scale = np.asarray(ckpt_blob["label_scaler_scale"], dtype=np.float64)
        mu_raw = pred.mu[:, perm] * ls_scale[None, :] + ls_mean[None, :]
        sigma_raw = pred.sigma_total[:, perm] * ls_scale[None, :]

        # Map human-order labels (e.g. "teff_apogee") to short keys ("teff").
        human_keys = [n.replace("_apogee", "") for n in human_order]

        cols: dict[str, np.ndarray] = {"source_id": src_ids,
                                       "epoch": np.full(len(src_ids), epoch,
                                                        dtype=np.int32)}
        for j, k in enumerate(human_keys):
            cols[f"{k}_pred"] = mu_raw[:, j].astype(np.float32)
            cols[f"{k}_sigma"] = sigma_raw[:, j].astype(np.float32)
        df = pd.DataFrame(cols)
        df.to_parquet(out, index=False)
        size_kb = out.stat().st_size / 1024.0
        logger.info("epoch %04d → %s  (%.0f KB, n=%d)",
                    epoch, out.relative_to(REPO), size_kb, len(df))
        n_done += 1

    logger.info("emitted %d / %d epochs into %s",
                n_done, len(ckpts), out_dir.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
