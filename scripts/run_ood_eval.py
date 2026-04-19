"""OOD flag distribution on the val split for the 5-label production ensemble (#136).

Fits ``MahalanobisOODBundle`` on the training XP 108-D feature block
``(bp_coef_norm_1..54, rp_coef_norm_1..54)``, scores the val split, and
computes the ensemble-disagreement OOD flag from per-member (μ, L).
Writes an ``ood_distribution.json`` summary containing:

- Mahalanobis flag rate and score-quantile histogram.
- Ensemble-disagreement ratio histogram and flag rate.
- Joint ``combined_ood_status`` level counts (0 green, 1 yellow, 2 red).
- Release-gate context: how many released, excluded by Regime B, or OOD.

No model retraining. Reuses the ensemble in ``--ensemble`` and the val
split defined by the checkpoint's ``split_seed`` — same loader as
:mod:`run_calibration`.

Run: ``PYTHONPATH=src python scripts/run_ood_eval.py --ensemble <dir>``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from arqueogal.xp_abundances.main.adapter import XpFeatureAdapter
from arqueogal.xp_abundances.main.data import FeatureLayout, LabelTiers
from arqueogal.xp_abundances.main.model import (
    CovarianceBlockLayout,
    ModelConfig,
    XpAbundanceModel,
)
from arqueogal.xp_abundances.main.ood import (
    combined_ood_status,
    ensemble_disagreement_ratio,
    fit_mahalanobis_ood,
    flag_ensemble_ood,
    flag_mahalanobis_ood,
    score_mahalanobis_ood,
)
from arqueogal.xp_abundances.main.training import build_dataloaders, load_checkpoint
from arqueogal.xp_abundances.main.uncertainty import RegimeBEnvelope
from arqueogal.xp_abundances.main.config import LossWeights, TrainingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("run_ood_eval")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_DIR = REPO_ROOT / "reports/pipeline1/run_a"
DEFAULT_PARQUET = REPO_ROOT / "data/processed/pipeline1_features_stream1.parquet"


def _build_cfg_for_loader(
    parquet: Path, pretrained_ckpt: Path, batch_size: int, seed: int,
) -> TrainingConfig:
    return TrainingConfig(
        train_parquet=parquet,
        output_dir=REPO_ROOT / "tmp_ood",
        epochs=1, batch_size=batch_size, num_workers=2,
        amp_dtype="bfloat16",
        use_c0_scalars=True, encoder_lr_ratio=0.1,
        pretrained_encoder_ckpt=pretrained_ckpt,
        reload_head_from_pretrained=False, split_seed=seed,
        output_prefix="xp_abundances_main_oodeval",
        loss_weights=LossWeights(supcon=0.0, beta_nll=1.0, beta=0.5),
        temperature_init=0.10, ensemble_seeds=(0,),
    )


def _reconstruct_model(
    blob: dict, layout: FeatureLayout, block_layout: CovarianceBlockLayout,
    device: torch.device,
) -> tuple[XpAbundanceModel, XpFeatureAdapter]:
    cfg_yaml = json.loads(blob["config_yaml"])
    use_c0 = bool(cfg_yaml.get("use_c0_scalars", True))
    latent_dim = int(cfg_yaml.get("latent_dim", 32))
    trunk_hidden = tuple(cfg_yaml.get("trunk_hidden", (256, 128)))
    head_hidden = int(cfg_yaml.get("head_hidden", 128))
    dropout = float(cfg_yaml.get("dropout", 0.10))

    model = XpAbundanceModel(
        ModelConfig(
            input_dim=layout.input_dim, block_layout=block_layout,
            latent_dim=latent_dim, trunk_hidden=trunk_hidden,
            head_hidden=head_hidden, dropout=dropout,
        ),
    ).to(device)
    model.encoder.load_state_dict(blob["encoder"])
    model.head.load_state_dict(blob["regressor"])
    adapter = XpFeatureAdapter(layout, use_c0_scalars=use_c0).to(device)
    return model, adapter


def _collect_member_preds(
    model: XpAbundanceModel, adapter: XpFeatureAdapter,
    loader, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    mus: list[np.ndarray] = []
    Ls: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            mu, L, _h, _z = model(adapter(x))
            mus.append(mu.float().cpu().numpy())
            Ls.append(L.float().cpu().numpy())
    return (
        np.concatenate(mus, axis=0).astype(np.float32),
        np.concatenate(Ls, axis=0).astype(np.float32),
    )


def _xp_feature_block(df: pd.DataFrame, layout: FeatureLayout) -> np.ndarray:
    """Pull the 108-D ``(bp_coef_norm_1..54, rp_coef_norm_1..54)`` block."""
    bp = df[list(layout.bp_coef_cols)].to_numpy(dtype=np.float32)
    rp = df[list(layout.rp_coef_cols)].to_numpy(dtype=np.float32)
    return np.concatenate([bp, rp], axis=1)


def _histogram_counts(values: np.ndarray, bin_edges: np.ndarray) -> dict[str, int]:
    counts, _ = np.histogram(values[np.isfinite(values)], bins=bin_edges)
    return {
        f"[{bin_edges[i]:.3g}, {bin_edges[i+1]:.3g})": int(counts[i])
        for i in range(len(counts))
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ensemble", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--p-threshold", type=float, default=0.99,
                   help="Mahalanobis distance quantile for the training-set"
                        " OOD threshold.")
    p.add_argument("--ensemble-threshold", type=float, default=0.5,
                   help="epistemic/total σ ratio cutoff for ensemble disagreement.")
    p.add_argument("--tag", type=str, default="ensemble_5label")
    args = p.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    member_ckpts = sorted(
        [p for p in args.ensemble.glob(
            "member_seed*/xp_abundances_main_ensemble*_seed*_best.pt",
        )],
    )
    if not member_ckpts:
        raise FileNotFoundError(f"no member checkpoints under {args.ensemble}")
    _LOG.info("device=%s ensemble=%s members=%d",
              device, args.ensemble, len(member_ckpts))

    layout = FeatureLayout()
    first_blob = load_checkpoint(member_ckpts[0], map_location="cpu")
    first_cfg_yaml = json.loads(first_blob["config_yaml"])
    pretrained_ckpt = Path(first_cfg_yaml["pretrained_encoder_ckpt"])
    split_seed = int(first_cfg_yaml.get("split_seed", 0))
    ckpt_label_names = tuple(first_blob["label_names"])
    tier_map = first_blob.get("tier_map", {})
    tiers = LabelTiers(
        tier1=tuple(n for n in ckpt_label_names if tier_map.get(n) == 1),
        tier2=tuple(n for n in ckpt_label_names if tier_map.get(n) == 2),
        tier3=tuple(n for n in ckpt_label_names if tier_map.get(n) == 3),
    )
    block_layout = CovarianceBlockLayout.from_dict(first_blob["block_layout"])

    cfg = _build_cfg_for_loader(
        parquet=args.parquet, pretrained_ckpt=pretrained_ckpt,
        batch_size=args.batch_size, seed=split_seed,
    )
    train_loader, val_loader, _, _ = build_dataloaders(
        cfg, layout, tiers, seed=split_seed,
    )
    _LOG.info("train batches=%d val batches=%d", len(train_loader), len(val_loader))

    # Pull XP feature blocks from the parquet by source_id (the loader stores them).
    full_df = pd.read_parquet(
        args.parquet,
        columns=[
            "source_id", *layout.bp_coef_cols, *layout.rp_coef_cols,
        ],
    ).drop_duplicates(subset="source_id", keep="first")
    sid_to_xp = full_df.set_index("source_id")

    train_sids = np.asarray(train_loader.dataset.source_id)
    val_sids = np.asarray(val_loader.dataset.source_id)

    X_train = sid_to_xp.loc[train_sids][list(layout.bp_coef_cols) + list(layout.rp_coef_cols)]
    X_train = X_train.to_numpy(dtype=np.float32)
    X_val = sid_to_xp.loc[val_sids][list(layout.bp_coef_cols) + list(layout.rp_coef_cols)]
    X_val = X_val.to_numpy(dtype=np.float32)
    _LOG.info("X_train=%s X_val=%s (108-D XP feature block)", X_train.shape, X_val.shape)

    # Fit Mahalanobis OOD on training, flag val.
    ood_bundle = fit_mahalanobis_ood(
        X_train, p_threshold=args.p_threshold, regularization=1e-6,
    )
    _LOG.info(
        "Mahalanobis bundle: n_train=%d threshold=%.3f p=%.3f",
        ood_bundle.n_training, ood_bundle.threshold, ood_bundle.p_threshold,
    )

    mahal_scores = score_mahalanobis_ood(X_val, ood_bundle)
    mahal_flags = flag_mahalanobis_ood(X_val, ood_bundle)
    mahal_flag_rate = float(mahal_flags.mean())
    _LOG.info("Mahalanobis flag rate on val: %.4f", mahal_flag_rate)

    # Collect per-member predictions.
    per_mu: list[np.ndarray] = []
    per_L: list[np.ndarray] = []
    for ckpt in member_ckpts:
        blob = load_checkpoint(ckpt, map_location=device)
        ckpt_block = CovarianceBlockLayout.from_dict(blob["block_layout"])
        model, adapter = _reconstruct_model(blob, layout, ckpt_block, device)
        mu, L = _collect_member_preds(model, adapter, val_loader, device)
        per_mu.append(mu)
        per_L.append(L)
    mus = np.stack(per_mu, axis=0)
    Ls = np.stack(per_L, axis=0)
    sigma_diag_per_member = np.sqrt(np.einsum("kbij,kbij->kbi", Ls, Ls))
    ens_ratio = ensemble_disagreement_ratio(mus, sigma_diag_per_member)
    ens_flags = flag_ensemble_ood(
        mus, sigma_diag_per_member, threshold=args.ensemble_threshold,
    )
    ens_flag_rate = float(ens_flags.mean())
    _LOG.info("Ensemble disagreement flag rate on val: %.4f", ens_flag_rate)

    # Combined status.
    status = combined_ood_status(mahal_flags, ens_flags)
    status_counts = {
        "0_green": int((status == 0).sum()),
        "1_yellow": int((status == 1).sum()),
        "2_red": int((status == 2).sum()),
    }
    _LOG.info("combined status: %s", status_counts)

    # Regime B envelope on predicted Teff / log g + val b_deg.
    b_deg_df = pd.read_parquet(
        args.parquet, columns=["source_id", "b_deg"],
    ).drop_duplicates(subset="source_id", keep="first")
    b_deg_by_sid = dict(
        zip(b_deg_df["source_id"].to_numpy(), b_deg_df["b_deg"].to_numpy()),
    )
    b_deg_val = np.asarray(
        [b_deg_by_sid.get(int(sid), np.nan) for sid in val_sids], dtype=np.float64,
    )
    mu_bar_scaled = mus.mean(axis=0)
    mean_block = np.asarray(first_blob["label_scaler_mean"], dtype=np.float64)
    scale_block = np.asarray(first_blob["label_scaler_scale"], dtype=np.float64)
    perm = block_layout.human_to_block_perm.cpu().numpy()
    mean_block = mean_block[perm]; scale_block = scale_block[perm]
    mu_bar = (mu_bar_scaled * scale_block[None] + mean_block[None]).astype(np.float64)
    envelope = RegimeBEnvelope()
    inside_envelope = envelope.mask(mu_bar[:, 0], mu_bar[:, 1], b_deg_val)
    tier1_release = ~inside_envelope

    # Cross-tab: release × status.
    release_by_status = {}
    for s_code, s_name in ((0, "0_green"), (1, "1_yellow"), (2, "2_red")):
        mask = status == s_code
        release_by_status[s_name] = {
            "total": int(mask.sum()),
            "released": int((mask & tier1_release).sum()),
            "excluded_regime_b": int((mask & ~tier1_release).sum()),
        }

    # Mahalanobis score distribution.
    mahal_bin_edges = np.linspace(
        float(np.nanmin(mahal_scores)),
        float(np.nanmax(mahal_scores)),
        21,
    )
    ens_bin_edges = np.linspace(0.0, 1.0, 21)

    summary = {
        "ensemble": str(args.ensemble),
        "n_train_for_mahalanobis": int(ood_bundle.n_training),
        "n_val_stars": int(X_val.shape[0]),
        "mahalanobis": {
            "p_threshold": float(ood_bundle.p_threshold),
            "threshold_distance": float(ood_bundle.threshold),
            "regularization": float(ood_bundle.regularization),
            "flag_rate": mahal_flag_rate,
            "score_mean": float(np.nanmean(mahal_scores)),
            "score_median": float(np.nanmedian(mahal_scores)),
            "score_p95": float(np.nanquantile(mahal_scores, 0.95)),
            "score_p99": float(np.nanquantile(mahal_scores, 0.99)),
            "score_histogram": _histogram_counts(mahal_scores, mahal_bin_edges),
        },
        "ensemble_disagreement": {
            "ratio_threshold": float(args.ensemble_threshold),
            "flag_rate": ens_flag_rate,
            "ratio_mean": float(np.nanmean(ens_ratio)),
            "ratio_median": float(np.nanmedian(ens_ratio)),
            "ratio_p95": float(np.nanquantile(ens_ratio, 0.95)),
            "ratio_p99": float(np.nanquantile(ens_ratio, 0.99)),
            "ratio_histogram": _histogram_counts(ens_ratio, ens_bin_edges),
        },
        "combined_status_counts": status_counts,
        "regime_b": {
            "envelope": envelope.to_dict(),
            "n_excluded": int(inside_envelope.sum()),
            "n_released": int(tier1_release.sum()),
            "frac_excluded": float(inside_envelope.mean()),
        },
        "release_by_status": release_by_status,
    }
    out_path = args.report_dir / f"ood_distribution_{args.tag}.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    _LOG.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
