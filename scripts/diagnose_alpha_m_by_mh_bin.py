"""Stratified [α/M] bias audit — v1 vs v1.1 ensemble on val partition.

The v1 ensemble showed metal-poor regression-to-mean: [α/M] predictions at
[M/H]<-1 clustered near the disc mean (≈ +0.10) instead of the training-truth
mean (≈ +0.23). #198 patched this with inverse-frequency [M/H]-bin weighting
in the β-NLL. This script computes the per-[M/H]-bin mean-pred vs mean-truth
for both ensembles so the user can see the fix before committing to release.

Outputs a table to stdout and a JSON next to ``reports/pipeline1/run_a_v11/``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from arqueogal.xp_abundances.main.data import FeatureLayout, LabelScaler, LabelTiers
from arqueogal.xp_abundances.main.model import CovarianceBlockLayout
from arqueogal.xp_abundances.main.training import build_dataloaders, load_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_calibration import (  # noqa: E402
    _build_cfg_for_val_loader,
    _collect_member_preds,
    _moment_match,
    _reconstruct_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("alpha_m_bias")

REPO = Path(__file__).resolve().parent.parent
V1_ENSEMBLE = REPO / "models/main/xp_abundances/20260419_nogit_a0e10aa_ensemble_5label"
V11_ENSEMBLE = REPO / "models/main/xp_abundances/20260421_38a993e_cdc55be_ensemble_5label"
PARQUET = REPO / "data/processed/pipeline1_features_stream1.parquet"
OUT_JSON = REPO / "reports/pipeline1/run_a_v11/alpha_m_bias_by_mh_bin.json"

MH_EDGES = (-1.5, -1.0, -0.5, 0.0)
BIN_LABELS = ["(-inf, -1.50)", "[-1.50, -1.00)", "[-1.00, -0.50)", "[-0.50, 0.00)", "[0.00, +inf)"]


def _ensemble_predict(ensemble_dir: Path, device: torch.device) -> dict:
    """Run ensemble on val partition, return predictions + truth in human units."""
    tiers = LabelTiers.five_label()
    layout = FeatureLayout()

    member_ckpts = sorted(ensemble_dir.glob("member_seed*/*_best.pt"))
    if not member_ckpts:
        raise FileNotFoundError(f"no member checkpoints under {ensemble_dir}")
    _LOG.info("%s: %d members", ensemble_dir.name, len(member_ckpts))

    first = load_checkpoint(member_ckpts[0], map_location="cpu")
    first_cfg = json.loads(first["config_yaml"])
    split_seed = int(first_cfg.get("split_seed", 0))

    cfg = _build_cfg_for_val_loader(
        parquet=PARQUET,
        pretrained_ckpt=Path(first_cfg["pretrained_encoder_ckpt"]),
        batch_size=1024,
        seed=split_seed,
    )
    _, val_loader, _, _ = build_dataloaders(cfg, layout, tiers, seed=split_seed)

    per_mu: list[np.ndarray] = []
    y_human_scaled: np.ndarray | None = None
    scaler: LabelScaler | None = None
    block_to_human = None
    for ck in member_ckpts:
        blob = load_checkpoint(ck, map_location=device)
        ckpt_layout = CovarianceBlockLayout.from_dict(blob["block_layout"])
        model, adapter = _reconstruct_model(blob, layout, ckpt_layout, device)
        mu, L, y_h = _collect_member_preds(model, adapter, val_loader, device)
        # mu is in BLOCK order; y_h is in HUMAN order. Permute mu to human.
        if block_to_human is None:
            block_to_human = ckpt_layout.block_to_human_perm.cpu().numpy()
        per_mu.append(mu[:, block_to_human])
        if y_human_scaled is None:
            y_human_scaled = y_h
            scaler = LabelScaler(
                mean=np.asarray(blob["label_scaler_mean"], dtype=np.float32),
                scale=np.asarray(blob["label_scaler_scale"], dtype=np.float32),
                label_names=tuple(blob["label_names"]),
            )

    mu_bar_scaled = np.mean(np.stack(per_mu), axis=0)  # (N, K) in scaled HUMAN units
    mu_bar = scaler.inverse_mean(mu_bar_scaled)  # back to human units
    y_truth = scaler.inverse_mean(y_human_scaled)  # human units

    labels = list(scaler.label_names)
    mh_idx = labels.index("mh_apogee")
    alpha_idx = labels.index("alpha_m_apogee")
    return {
        "mh_pred": mu_bar[:, mh_idx],
        "mh_truth": y_truth[:, mh_idx],
        "alpha_m_pred": mu_bar[:, alpha_idx],
        "alpha_m_truth": y_truth[:, alpha_idx],
        "n_members": len(member_ckpts),
    }


def _stratify_by_mh_bin(preds: dict, edges: tuple[float, ...]) -> list[dict]:
    """Bin val stars by *truth* [M/H] and compute per-bin alpha_m stats."""
    mh_t = preds["mh_truth"]
    bin_idx = np.digitize(mh_t, edges)
    rows = []
    n_bins = len(edges) + 1
    for b in range(n_bins):
        m = bin_idx == b
        finite_a = np.isfinite(preds["alpha_m_truth"]) & np.isfinite(preds["alpha_m_pred"]) & m
        n = int(finite_a.sum())
        if n == 0:
            rows.append(
                {
                    "bin": BIN_LABELS[b],
                    "n": 0,
                    "alpha_pred_mean": None,
                    "alpha_truth_mean": None,
                    "bias": None,
                    "abs_bias": None,
                }
            )
            continue
        pred = float(preds["alpha_m_pred"][finite_a].mean())
        truth = float(preds["alpha_m_truth"][finite_a].mean())
        rows.append(
            {
                "bin": BIN_LABELS[b],
                "n": n,
                "alpha_pred_mean": pred,
                "alpha_truth_mean": truth,
                "bias": pred - truth,
                "abs_bias": abs(pred - truth),
            }
        )
    return rows


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOG.info("device=%s", device)

    v1 = _ensemble_predict(V1_ENSEMBLE, device)
    v11 = _ensemble_predict(V11_ENSEMBLE, device)

    v1_bins = _stratify_by_mh_bin(v1, MH_EDGES)
    v11_bins = _stratify_by_mh_bin(v11, MH_EDGES)

    print()
    print(
        f"{'bin':<20} {'n':>6}  "
        f"{'v1_pred':>8} {'v1_truth':>9} {'v1_bias':>9}   "
        f"{'v11_pred':>9} {'v11_truth':>10} {'v11_bias':>10}   {'Δ|bias|':>8}"
    )
    print("-" * 120)
    for r1, r11 in zip(v1_bins, v11_bins, strict=True):
        if r1["n"] == 0:
            continue
        print(
            f"{r1['bin']:<20} {r1['n']:>6d}  "
            f"{r1['alpha_pred_mean']:>8.4f} {r1['alpha_truth_mean']:>9.4f} "
            f"{r1['bias']:>+9.4f}   "
            f"{r11['alpha_pred_mean']:>9.4f} {r11['alpha_truth_mean']:>10.4f} "
            f"{r11['bias']:>+10.4f}   "
            f"{r11['abs_bias'] - r1['abs_bias']:>+8.4f}"
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(
            {
                "mh_bin_edges": list(MH_EDGES),
                "v1_ensemble": str(V1_ENSEMBLE),
                "v11_ensemble": str(V11_ENSEMBLE),
                "v1": v1_bins,
                "v1_1": v11_bins,
            },
            f,
            indent=2,
        )
    _LOG.info("wrote %s", OUT_JSON)


if __name__ == "__main__":
    main()
