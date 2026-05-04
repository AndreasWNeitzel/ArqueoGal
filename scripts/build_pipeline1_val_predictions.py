"""Dump val-partition ensemble predictions + truth as a parquet for diagnostics.

Runs the 5-member v1.1 ensemble across the val split of Stream 1 and emits

    reports/pipeline1/run_a_v11/val_predictions.parquet

with one row per val star and columns

    teff_truth, logg_truth, mh_truth, alpha_m_truth, mg_h_truth
    teff_pred,  logg_pred,  mh_pred,  alpha_m_pred,  mg_h_pred
    teff_sigma, logg_sigma, mh_sigma, alpha_m_sigma, mg_h_sigma  (aleatoric, diag(Σ̄))
    teff_epi,   logg_epi,   mh_epi,   alpha_m_epi,   mg_h_epi    (epistemic std)

All abundance-style labels are in dex; Teff in K; logg in dex.

The harness mirrors ``diagnose_alpha_m_by_mh_bin.py`` / ``run_calibration.py``
and reuses ``_build_cfg_for_val_loader``, ``_reconstruct_model``,
``_collect_member_preds``. No retraining.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from arqueogal.xp_abundances.main.data import FeatureLayout, LabelScaler, LabelTiers
from arqueogal.xp_abundances.main.model import CovarianceBlockLayout
from arqueogal.xp_abundances.main.training import build_dataloaders, load_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_calibration import (  # noqa: E402
    _build_cfg_for_val_loader,
    _collect_member_preds,
    _reconstruct_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("val_predictions")

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ENSEMBLE = REPO / "models/main/xp_abundances/20260421_38a993e_cdc55be_ensemble_5label"
DEFAULT_PARQUET = REPO / "data/processed/pipeline1_features_stream1.parquet"
DEFAULT_OUT = REPO / "reports/pipeline1/run_a_v11/val_predictions.parquet"

LABEL_ORDER = ("teff", "logg", "mh", "alpha_m", "mg_h")


def _ensemble_predict(ensemble_dir: Path, parquet: Path, device: torch.device):
    tiers = LabelTiers.five_label()
    layout = FeatureLayout()

    member_ckpts = sorted(ensemble_dir.glob("member_seed*/*_best.pt"))
    if not member_ckpts:
        raise FileNotFoundError(f"no member checkpoints under {ensemble_dir}")
    _LOG.info("ensemble=%s  n_members=%d", ensemble_dir.name, len(member_ckpts))

    first = load_checkpoint(member_ckpts[0], map_location="cpu")
    first_cfg = json.loads(first["config_yaml"])
    split_seed = int(first_cfg.get("split_seed", 0))

    _pretrained_raw = first_cfg.get("pretrained_encoder_ckpt")
    cfg = _build_cfg_for_val_loader(
        parquet=parquet,
        pretrained_ckpt=Path(_pretrained_raw) if _pretrained_raw else None,
        batch_size=1024,
        seed=split_seed,
    )
    _, val_loader, _, _, _ = build_dataloaders(cfg, layout, tiers, seed=split_seed)

    per_mu: list[np.ndarray] = []
    per_L: list[np.ndarray] = []
    y_human_scaled: np.ndarray | None = None
    scaler: LabelScaler | None = None
    block_to_human = None
    for ck in member_ckpts:
        blob = load_checkpoint(ck, map_location=device)
        ckpt_layout = CovarianceBlockLayout.from_dict(blob["block_layout"])
        model, adapter = _reconstruct_model(blob, layout, ckpt_layout, device)
        mu, L, y_h = _collect_member_preds(model, adapter, val_loader, device)
        if block_to_human is None:
            block_to_human = ckpt_layout.block_to_human_perm.cpu().numpy()
        per_mu.append(mu[:, block_to_human])
        # Permute L rows/cols from block to human: L_human = P L P^T where P is perm
        # For a square L in block basis: Sigma_block = L L^T; Sigma_human = P Sigma_block P^T
        Sigma_block = np.einsum("bij,bkj->bik", L, L)  # L L^T
        Sigma_human = Sigma_block[:, block_to_human][:, :, block_to_human]
        per_L.append(Sigma_human)  # per-member Σ in human order
        if y_human_scaled is None:
            y_human_scaled = y_h
            scaler = LabelScaler(
                mean=np.asarray(blob["label_scaler_mean"], dtype=np.float32),
                scale=np.asarray(blob["label_scaler_scale"], dtype=np.float32),
                label_names=tuple(blob["label_names"]),
            )

    # ensemble moment-match in SCALED HUMAN coords
    mus = np.stack(per_mu)  # (K, N, D) scaled
    Sigmas = np.stack(per_L)  # (K, N, D, D) scaled
    mu_bar = mus.mean(axis=0)
    mean_sigma = Sigmas.mean(axis=0)
    diff = mus - mu_bar[None]
    between = np.einsum("kbi,kbj->bij", diff, diff) / mus.shape[0]
    sigma_scaled = mean_sigma + between  # (N, D, D)

    # diagonals → aleatoric std (post-combine), epistemic std = sqrt(diag between)
    diag_total = np.maximum(np.einsum("bii->bi", sigma_scaled), 0.0)
    diag_between = np.maximum(np.einsum("bii->bi", between), 0.0)
    std_scaled = np.sqrt(diag_total)
    epi_scaled = np.sqrt(diag_between)

    # Back to human units. scale[d] is the std used during training (per-label).
    mu_bar_human = scaler.inverse_mean(mu_bar)
    y_truth_human = scaler.inverse_mean(y_human_scaled)
    scale_vec = np.asarray(scaler.scale, dtype=np.float32)  # (D,)
    std_human = std_scaled * scale_vec[None]
    epi_human = epi_scaled * scale_vec[None]

    labels = list(scaler.label_names)
    # labels are of the form e.g. "teff_apogee", "mh_apogee", "alpha_m_apogee", "mg_h_apogee"
    # Map to our short names
    short_to_col: dict[str, int] = {}
    for short in LABEL_ORDER:
        # Match teff_apogee, logg_apogee, mh_apogee, alpha_m_apogee, mg_h_apogee
        if short == "teff":
            key = "teff_apogee"
        elif short == "logg":
            key = "logg_apogee"
        elif short == "mh":
            key = "mh_apogee"
        elif short == "alpha_m":
            key = "alpha_m_apogee"
        elif short == "mg_h":
            key = "mg_h_apogee"
        else:
            raise KeyError(short)
        short_to_col[short] = labels.index(key)

    data = {}
    for short, col in short_to_col.items():
        data[f"{short}_truth"] = y_truth_human[:, col].astype(np.float32)
        data[f"{short}_pred"] = mu_bar_human[:, col].astype(np.float32)
        data[f"{short}_sigma"] = std_human[:, col].astype(np.float32)
        data[f"{short}_epi"] = epi_human[:, col].astype(np.float32)
    return pd.DataFrame(data)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ensemble", type=Path, default=DEFAULT_ENSEMBLE)
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOG.info("device=%s", device)

    df = _ensemble_predict(args.ensemble, args.parquet, device)
    _LOG.info("val predictions: n=%d cols=%d", len(df), df.shape[1])
    for c in df.columns:
        x = df[c].to_numpy()
        _LOG.info(
            "  %-18s  mean=%+.4f  std=%.4f  p05=%+.4f  p95=%+.4f",
            c,
            np.nanmean(x),
            np.nanstd(x),
            np.nanpercentile(x, 5),
            np.nanpercentile(x, 95),
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(args.out)
    _LOG.info("wrote %s (%d rows)", args.out, len(df))


if __name__ == "__main__":
    main()
