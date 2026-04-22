"""Stratified [α/M] bias + collapse audit — v1 vs v1.1 vs v2 ensembles on val partition.

Extends ``diagnose_alpha_m_by_mh_bin.py`` with:
- v2 ensemble (supcon_label_n_first=None, β=0, NaN-safe SupCon)
- per-bin pred_std / truth_std ratio (prototype-collapse signature: ratio ≪ 1)

The user's complaint after v2: the prototype at [α/M]≈+0.11 is still present in
Stream-3 inference, just relocated in [M/H]. This audit quantifies *on the val
partition* whether the regression-to-mean and intra-bin collapse have been
cured, partially cured, or merely relocated.
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
    _reconstruct_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("alpha_m_v2")

REPO = Path(__file__).resolve().parent.parent
V1_ENSEMBLE = REPO / "models/main/xp_abundances/20260419_nogit_a0e10aa_ensemble_5label"
V11_ENSEMBLE = REPO / "models/main/xp_abundances/20260421_38a993e_cdc55be_ensemble_5label"
V2_ENSEMBLE = REPO / "models/main/xp_abundances/20260421_38a993e_712b774_ensemble_5label"
PARQUET = REPO / "data/processed/pipeline1_features_stream1.parquet"
OUT_JSON = REPO / "reports/pipeline1/run_a_v2/alpha_m_bias_by_mh_bin.json"

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

    mu_bar_scaled = np.mean(np.stack(per_mu), axis=0)
    mu_bar = scaler.inverse_mean(mu_bar_scaled)
    y_truth = scaler.inverse_mean(y_human_scaled)

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


def _stratify(preds: dict, edges: tuple[float, ...]) -> list[dict]:
    mh_t = preds["mh_truth"]
    bin_idx = np.digitize(mh_t, edges)
    rows = []
    n_bins = len(edges) + 1
    for b in range(n_bins):
        m = bin_idx == b
        finite = np.isfinite(preds["alpha_m_truth"]) & np.isfinite(preds["alpha_m_pred"]) & m
        n = int(finite.sum())
        if n == 0:
            rows.append({
                "bin": BIN_LABELS[b], "n": 0,
                "pred_mean": None, "truth_mean": None, "bias": None,
                "pred_std": None, "truth_std": None, "std_ratio": None,
            })
            continue
        pred = preds["alpha_m_pred"][finite]
        truth = preds["alpha_m_truth"][finite]
        pred_std = float(pred.std())
        truth_std = float(truth.std())
        rows.append({
            "bin": BIN_LABELS[b],
            "n": n,
            "pred_mean": float(pred.mean()),
            "truth_mean": float(truth.mean()),
            "bias": float(pred.mean() - truth.mean()),
            "pred_std": pred_std,
            "truth_std": truth_std,
            "std_ratio": pred_std / truth_std if truth_std > 0 else None,
        })
    return rows


def _print_table(v1, v11, v2):
    print()
    hdr = (
        f"{'bin':<18} {'n':>6}  "
        f"{'v1.bias':>8} {'v11.bias':>9} {'v2.bias':>8}   "
        f"{'v1.ratio':>9} {'v11.ratio':>10} {'v2.ratio':>9}   "
        f"{'truth':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r1, r11, r2 in zip(v1, v11, v2, strict=True):
        if r1["n"] == 0:
            continue
        print(
            f"{r1['bin']:<18} {r1['n']:>6d}  "
            f"{r1['bias']:>+8.4f} {r11['bias']:>+9.4f} {r2['bias']:>+8.4f}   "
            f"{r1['std_ratio']:>9.3f} {r11['std_ratio']:>10.3f} {r2['std_ratio']:>9.3f}   "
            f"{r1['truth_mean']:>+7.4f}"
        )
    print()
    print("Legend: bias = pred_mean - truth_mean (per [M/H] bin)")
    print("        ratio = pred_std / truth_std (prototype collapse → ratio ≪ 1)")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOG.info("device=%s", device)

    v1 = _ensemble_predict(V1_ENSEMBLE, device)
    v11 = _ensemble_predict(V11_ENSEMBLE, device)
    v2 = _ensemble_predict(V2_ENSEMBLE, device)

    v1_bins = _stratify(v1, MH_EDGES)
    v11_bins = _stratify(v11, MH_EDGES)
    v2_bins = _stratify(v2, MH_EDGES)

    _print_table(v1_bins, v11_bins, v2_bins)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump({
            "mh_bin_edges": list(MH_EDGES),
            "v1_ensemble": str(V1_ENSEMBLE),
            "v11_ensemble": str(V11_ENSEMBLE),
            "v2_ensemble": str(V2_ENSEMBLE),
            "v1": v1_bins,
            "v1_1": v11_bins,
            "v2": v2_bins,
        }, f, indent=2)
    _LOG.info("wrote %s", OUT_JSON)


if __name__ == "__main__":
    main()
