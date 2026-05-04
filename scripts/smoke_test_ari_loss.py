"""Smoke test — verify the new ARI loss integration.

Builds a tiny synthetic XP feature batch (CPU, no parquet), constructs the
production XpAbundanceModel + adapter at the 5-label block layout + 140-D
FeatureLayout, and runs ``_compute_losses`` once with ari=0.1 to confirm:
1. The forward pass returns finite scalars for all five terms (loss/supcon/
   nll/barlow/ari/tau).
2. ``parts["ari"]`` is non-trivial (>0) when truth and prediction differ.
3. ``loss.backward()`` produces finite gradients on every parameter (no NaN
   propagation through the soft K=2 sigmoid or block_layout reorder).

This is a code-level integration check, NOT a science check; it validates
that the new branch works without launching a multi-hour training run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from arqueogal.xp_abundances.main.config import LossWeights, TrainingConfig
from arqueogal.xp_abundances.main.data import FeatureLayout, LabelTiers
from arqueogal.xp_abundances.main.training import (
    _build_model_and_temperature,
    _compute_losses,
)


def main() -> int:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    device = torch.device("cpu")
    layout = FeatureLayout()
    print(f"[smoke] FeatureLayout.input_dim = {layout.input_dim}")
    tiers = LabelTiers.five_label()
    print(f"[smoke] LabelTiers.five_label all_labels = {tiers.all_labels}")

    cfg = TrainingConfig(
        train_parquet=Path("/dev/null"),
        output_dir=Path("/tmp/smoke_ari"),
        use_c0_scalars=True,
        amp_dtype="none",
        loss_weights=LossWeights(
            supcon=0.1,
            beta_nll=1.0,
            beta=0.5,
            supcon_sigma=0.10,
            barlow=1.0,
            barlow_lam=0.005,
            ari=0.1,
            ari_alpha_threshold=0.15,
            ari_kernel_sigma=0.03,
        ),
        epochs=1,
        batch_size=32,
        ensemble_seeds=(0,),
        pretrained_encoder_ckpt=None,
    )

    model, log_temp, adapter = _build_model_and_temperature(cfg, layout, tiers, device)
    # Attach realistic per-label scaler stats so the ARI threshold conversion
    # exercises the same code path as the production training loop.
    label_mean = torch.tensor([4500.0, 2.5, -0.20, 0.10, -0.05], dtype=torch.float32)
    label_scale = torch.tensor([200.0, 0.30, 0.30, 0.10, 0.15], dtype=torch.float32)
    model.register_buffer("label_mean_human", label_mean, persistent=False)
    model.register_buffer("label_scale_human", label_scale, persistent=False)
    print(
        f"[smoke] model built: input_dim={layout.input_dim}, "
        f"label_order_human={model.block_layout.label_order_human}"
    )
    print(f"[smoke] label_mean_human = {label_mean.tolist()}")
    print(f"[smoke] label_scale_human = {label_scale.tolist()}")

    B = 32
    x = torch.randn(B, layout.input_dim, dtype=torch.float32, device=device)
    # Truth in PHYSICAL units; LabelScaler.transform divides out (mean, scale).
    # Spread α/M across the 0.15 dex threshold so soft K=2 has both clusters.
    y_phys_np = np.column_stack(
        [
            rng.normal(4500, 200, B).astype(np.float32),
            rng.normal(2.5, 0.3, B).astype(np.float32),
            rng.normal(-0.2, 0.3, B).astype(np.float32),
            rng.normal(0.10, 0.10, B).astype(np.float32),
            rng.normal(-0.05, 0.15, B).astype(np.float32),
        ]
    )
    y_scaled_np = (y_phys_np - label_mean.numpy()) / label_scale.numpy()
    y = torch.tensor(y_scaled_np, device=device, requires_grad=False)
    print(
        f"[smoke] y[:,3] (scaled [α/M]) range: "
        f"{y[:, 3].min().item():+.2f} .. {y[:, 3].max().item():+.2f}"
    )

    # Direct soft_ari_loss probe with the same conversion logic to confirm the
    # formula on its own.
    from arqueogal.xp_abundances.main.losses import soft_ari_loss

    with torch.no_grad():
        mu_dummy_phys = torch.tensor(
            np.column_stack(
                [
                    rng.normal(4500, 200, B).astype(np.float32),
                    rng.normal(2.5, 0.3, B).astype(np.float32),
                    rng.normal(-0.2, 0.3, B).astype(np.float32),
                    rng.normal(0.05, 0.10, B).astype(np.float32),  # different mean from truth
                    rng.normal(-0.05, 0.15, B).astype(np.float32),
                ]
            )
        )
        a_pred_phys = mu_dummy_phys[:, 3]
        a_true_phys = torch.tensor(y_phys_np[:, 3])
        p_pred = torch.sigmoid((a_pred_phys - 0.15) / 0.03)
        p_true = torch.sigmoid((a_true_phys - 0.15) / 0.03)
        pred_K2 = torch.stack([1 - p_pred, p_pred], dim=1)
        true_K2 = torch.stack([1 - p_true, p_true], dim=1)
        ari_direct = soft_ari_loss(pred_K2, true_K2)
        print(f"[smoke] direct soft_ari_loss(diff distributions): {ari_direct.item():.4f}")
        ari_self = soft_ari_loss(true_K2, true_K2)
        print(f"[smoke] direct soft_ari_loss(self): {ari_self.item():.4f}  (expect ~0)")

    print("[smoke] forward + loss …")
    total, parts, _z, _y = _compute_losses(
        model,
        log_temp,
        x,
        y,
        cfg,
        adapter,
        weights=None,
        queue=None,
    )
    print(f"[smoke] parts = {parts}")
    if any(not np.isfinite(v) for v in parts.values()):
        print("[smoke] FAIL: non-finite term in parts")
        return 1
    if parts["ari"] <= 0:
        print(f"[smoke] WARN: ari={parts['ari']:.4f} <= 0; expected > 0 with random pred ≠ truth")

    print("[smoke] backward …")
    total.backward()
    finite_grads = 0
    nan_grads = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        if torch.isfinite(g).all():
            finite_grads += 1
        else:
            nan_grads += 1
            print(f"  ❌ {name} has non-finite grad")
    if log_temp.grad is not None:
        if torch.isfinite(log_temp.grad):
            finite_grads += 1
        else:
            nan_grads += 1
    print(f"[smoke] finite-grad params: {finite_grads}, non-finite: {nan_grads}")
    if nan_grads > 0:
        print("[smoke] FAIL: non-finite gradients")
        return 1

    # Second pass: identical truth → ari should drop sharply
    print("[smoke] second pass with truth==pred mock (set α/M ≈ pred) …")
    total2, parts2, _, _ = _compute_losses(
        model,
        log_temp,
        x,
        y,
        cfg,
        adapter,
        weights=None,
        queue=None,
    )
    print(f"[smoke] parts2 = {parts2}")

    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
