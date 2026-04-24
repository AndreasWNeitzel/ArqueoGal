"""Per-label calibration diagnostic — #140 follow-up.

Runs on a trained ensemble directory (member_seed*/*.pt) and produces three
tables the user needs to disambiguate the three hypotheses after a failed
#135 calibration gate:

A. per-label marginal variance miscalibration → std(z) ≠ 1
B. per-label residual bias → mean residual ≠ 0 (or mean(z) ≠ 0)
C. concentration of failure in a subset of labels → heterogeneous per-label
   reliability error

Tables:
1. per-label residual mean / std, with |mean residual| / std(truth).
2. per-label z-statistic: mean(z), std(z), 68%/95% empirical coverage.
3. per-label reliability error |Var(z) − 1| (unconditional).

All metrics are in raw physical units, computed on the val split rebuilt with
the same split_seed that trained the ensemble. Reuses the moment-match and
un-scaling logic from :mod:`run_calibration` to ensure apples-to-apples.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from arqueogal.xp_abundances.main.data import FeatureLayout, LabelScaler, LabelTiers
from arqueogal.xp_abundances.main.model import default_pipeline1_layout
from arqueogal.xp_abundances.main.training import build_dataloaders, load_checkpoint

# Reuse the heavy lifting from run_calibration.
import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_calibration import (  # noqa: E402
    _build_cfg_for_val_loader,
    _collect_member_preds,
    _moment_match,
    _reconstruct_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("diagnose_per_label_calibration")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET = REPO_ROOT / "data/processed/pipeline1_features_stream1.parquet"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports/pipeline1/run_a"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ensemble", type=Path, required=True)
    p.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = p.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    member_ckpts = sorted(
        args.ensemble.glob("member_seed*/xp_abundances_main_ensemble_seed*_best.pt")
    )
    if not member_ckpts:
        raise FileNotFoundError(f"no members under {args.ensemble}")
    _LOG.info("found %d members", len(member_ckpts))

    layout = FeatureLayout()
    tiers = LabelTiers()

    first_blob = load_checkpoint(member_ckpts[0], map_location="cpu")
    first_cfg = json.loads(first_blob["config_yaml"])
    pretrained_ckpt = Path(first_cfg["pretrained_encoder_ckpt"])
    split_seed = int(first_cfg.get("split_seed", 0))

    cfg = _build_cfg_for_val_loader(
        parquet=args.parquet,
        pretrained_ckpt=pretrained_ckpt,
        batch_size=args.batch_size,
        seed=split_seed,
    )
    _, val_loader, _, _ = build_dataloaders(cfg, layout, tiers, seed=split_seed)
    _LOG.info("val loader built, batches=%d", len(val_loader))

    block_layout = default_pipeline1_layout()

    per_mu: list[np.ndarray] = []
    per_L: list[np.ndarray] = []
    y_human: np.ndarray | None = None
    scaler_block: LabelScaler | None = None
    scaler_human: LabelScaler | None = None
    for ck in member_ckpts:
        blob = load_checkpoint(ck, map_location=device)
        model, adapter = _reconstruct_model(blob, layout, device)
        mu, L, y_h = _collect_member_preds(model, adapter, val_loader, device)
        per_mu.append(mu)
        per_L.append(L)
        if y_human is None:
            y_human = y_h
            scaler_human = LabelScaler(
                mean=np.asarray(blob["label_scaler_mean"], dtype=np.float32),
                scale=np.asarray(blob["label_scaler_scale"], dtype=np.float32),
                label_names=tuple(blob["label_names"]),
            )
            scaler_block = scaler_human.reorder_to(block_layout.label_order_block)

    mus = np.stack(per_mu, axis=0)
    Ls = np.stack(per_L, axis=0)
    mu_bar_block_scaled, L_bar_block_scaled = _moment_match(mus, Ls)

    # Un-scale to raw units, still in block order.
    mu_bar_block = scaler_block.inverse_mean(mu_bar_block_scaled)
    L_bar_block = scaler_block.inverse_L(L_bar_block_scaled)

    # y_human is in human (tier) order, scaled → un-scale via the human-order scaler.
    y_human_raw = scaler_human.inverse_mean(y_human)

    # For per-label diagnostics we iterate in block order and look up names.
    # Reorder y to block order for apples-to-apples per-label arrays.
    perm = block_layout.human_to_block_perm.cpu().numpy()
    y_block = y_human_raw[:, perm]

    # Residual and σ per label, in block order.
    resid = mu_bar_block - y_block  # (B, 21)
    sigma = np.sqrt(np.einsum("bij,bij->bi", L_bar_block, L_bar_block)).clip(1e-8, None)
    z = resid / sigma  # (B, 21)

    # Per-label stats.
    rows: list[dict] = []
    names = list(block_layout.label_order_block)
    for j, name in enumerate(names):
        r = resid[:, j]
        y_col = y_block[:, j]
        z_col = z[:, j]
        mask = np.isfinite(r) & np.isfinite(z_col)
        if mask.sum() < 8:
            rows.append({"label": name, "n": int(mask.sum())})
            continue
        rj = r[mask]
        zc = z_col[mask]
        yc = y_col[mask]
        rows.append(
            {
                "label": name,
                "n": int(mask.sum()),
                "y_std": float(yc.std()),
                "resid_mean": float(rj.mean()),
                "resid_std": float(rj.std()),
                "abs_bias_over_ystd": float(abs(rj.mean()) / (yc.std() + 1e-12)),
                "z_mean": float(zc.mean()),
                "z_std": float(zc.std()),
                "cov68": float(np.mean(np.abs(zc) < 1.0)),
                "cov95": float(np.mean(np.abs(zc) < 1.96)),
                "rel_err": float(abs(zc.var() - 1.0)),
            }
        )

    # Pretty-print.
    def _fmt(v, width, prec=4):
        if v is None:
            return " " * width
        return f"{v:>{width}.{prec}g}"

    # Table 1: residuals.
    print("\n# Table 1 — per-label residual bias (raw units)")
    print(
        f"{'label':<17} {'n':>7} {'y_std':>9} {'resid_mean':>12} {'resid_std':>10} {'|bias|/y_std':>13}"
    )
    for r in rows:
        if r.get("n", 0) < 8:
            print(f"{r['label']:<17} {r['n']:>7}  (insufficient data)")
            continue
        print(
            f"{r['label']:<17} {r['n']:>7} "
            f"{_fmt(r['y_std'], 9)} {_fmt(r['resid_mean'], 12)} "
            f"{_fmt(r['resid_std'], 10)} {_fmt(r['abs_bias_over_ystd'], 13)}",
        )

    # Table 2: z-statistic and coverage.
    print(
        "\n# Table 2 — per-label z-statistic and coverage (target: z_mean=0, z_std=1, cov68=0.68, cov95=0.95)"
    )
    print(f"{'label':<17} {'z_mean':>9} {'z_std':>9} {'cov68':>9} {'cov95':>9}")
    for r in rows:
        if r.get("n", 0) < 8:
            continue
        print(
            f"{r['label']:<17} "
            f"{_fmt(r['z_mean'], 9)} {_fmt(r['z_std'], 9)} "
            f"{_fmt(r['cov68'], 9, 3)} {_fmt(r['cov95'], 9, 3)}",
        )

    # Table 3: reliability error sorted worst → best.
    print(
        "\n# Table 3 — per-label reliability error |Var(z) − 1| (target ≤ 0.10), sorted worst → best"
    )
    print(f"{'label':<17} {'rel_err':>9}")
    sorted_rows = [r for r in rows if r.get("n", 0) >= 8]
    sorted_rows.sort(key=lambda r: -r["rel_err"])
    for r in sorted_rows:
        print(f"{r['label']:<17} {_fmt(r['rel_err'], 9)}")

    # Summary hypothesis signals.
    valid = [r for r in rows if r.get("n", 0) >= 8]
    max_abs_bias = max((r["abs_bias_over_ystd"] for r in valid), default=float("nan"))
    max_zstd_dev = max((abs(r["z_std"] - 1.0) for r in valid), default=float("nan"))
    max_zmean = max((abs(r["z_mean"]) for r in valid), default=float("nan"))
    bad_labels = [r["label"] for r in valid if r["rel_err"] > 0.10]
    good_labels = [r["label"] for r in valid if r["rel_err"] <= 0.10]

    print("\n# Summary")
    print(f"- max |bias|/y_std across labels: {max_abs_bias:.4g}")
    print(f"  (Hypothesis B flag if > ~0.10)")
    print(f"- max |z_std − 1| across labels: {max_zstd_dev:.4g}")
    print(f"  (Hypothesis A flag — direct measure of per-label marginal miscalibration)")
    print(f"- max |z_mean| across labels: {max_zmean:.4g}")
    print(f"  (Hypothesis B flag if > ~0.2)")
    print(f"- labels failing rel_err ≤ 0.10: {len(bad_labels)}/{len(valid)}")
    print(f"  failing: {bad_labels}")
    print(f"  passing: {good_labels}")

    # Persist as JSON.
    out = args.report_dir / "per_label_calibration_diagnostic.json"
    with out.open("w") as f:
        json.dump(
            {
                "ensemble_dir": str(args.ensemble),
                "n_members": len(member_ckpts),
                "n_val_stars": int(y_block.shape[0]),
                "rows": rows,
                "summary": {
                    "max_abs_bias_over_ystd": max_abs_bias,
                    "max_zstd_dev_from_1": max_zstd_dev,
                    "max_abs_zmean": max_zmean,
                    "failing_labels": bad_labels,
                    "passing_labels": good_labels,
                },
            },
            f,
            indent=2,
            default=str,
        )
    _LOG.info("wrote %s", out)


if __name__ == "__main__":
    main()
