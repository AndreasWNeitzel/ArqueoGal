"""Profile per-batch cost of the joint-loss training step.

Builds the production joint-training config (139-D, queue=8192, β=0, barlow),
runs ``warmup`` + ``measure`` steps on the real training parquet, and prints:

1. Macro per-component wall-clock breakdown via cuda.Event timing
   (data-fetch, adapter, forward, supcon, beta_nll, barlow, backward,
   optimizer, queue.enqueue).
2. ``torch.profiler`` table: top-30 ops by self-CUDA time.

Run::

    PYTHONPATH=src python scripts/profile_joint_step.py

No checkpoint is saved.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, schedule

from arqueogal.xp_abundances.main.adapter import XpFeatureAdapter
from arqueogal.xp_abundances.main.config import LossWeights, TrainingConfig
from arqueogal.xp_abundances.main.data import FeatureLayout, LabelTiers
from arqueogal.xp_abundances.main.losses import (
    ContrastiveQueue,
    barlow_twins_loss,
    beta_nll_block_cholesky,
    supcon_soft_positive,
)
from arqueogal.xp_abundances.main.model import (
    ModelConfig,
    XpAbundanceModel,
    five_label_block_layout,
)
from arqueogal.xp_abundances.main.training import build_dataloaders

REPO = Path(__file__).resolve().parent.parent


def build_cfg() -> TrainingConfig:
    return TrainingConfig(
        train_parquet=REPO / "data/processed/pipeline1_features_stream1.parquet",
        output_dir=REPO / "tmp_profile",
        epochs=1,
        batch_size=512,
        num_workers=2,
        amp_dtype="bfloat16",
        max_lr=1e-3,
        pct_start=0.3,
        weight_decay=1e-4,
        grad_clip_norm=1.0,
        use_c0_scalars=True,
        encoder_lr_ratio=1.0,
        stage_dataset_on_gpu=True,
        output_prefix="xp_abundances_main_joint_profile",
        loss_weights=LossWeights(
            supcon=1.0,
            beta_nll=1.0,
            beta=0.0,
            barlow=1.0,
            barlow_lam=0.005,
            supcon_sigma=0.10,
            supcon_label_n_first=None,
        ),
        temperature_init=0.10,
        queue_size=8192,
        queue_warm_start=True,
        ensemble_seeds=(0,),
    )


def _clamped_temperature(log_temp: torch.nn.Parameter, bounds: tuple[float, float]) -> torch.Tensor:
    lo, hi = bounds
    return torch.clamp(log_temp.exp(), min=lo, max=hi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--measure", type=int, default=60)
    parser.add_argument("--profiler-steps", type=int, default=15)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    cfg = build_cfg()
    layout = FeatureLayout()
    tiers = LabelTiers.five_label()
    train_loader, _val_loader, _split_ids, label_scaler = build_dataloaders(
        cfg,
        layout,
        tiers,
        seed=0,
    )
    print(f"train batches/epoch={len(train_loader)}")

    block_layout = five_label_block_layout()
    model = XpAbundanceModel(
        ModelConfig(
            input_dim=layout.input_dim,
            block_layout=block_layout,
            latent_dim=32,
            trunk_hidden=(256, 128),
            head_hidden=128,
            dropout=0.10,
        ),
    ).to(device)
    log_temp = torch.nn.Parameter(
        torch.log(torch.tensor(cfg.temperature_init)).to(device),
    )
    adapter = XpFeatureAdapter(layout, use_c0_scalars=cfg.use_c0_scalars).to(device)
    queue = ContrastiveQueue(
        size=cfg.queue_size,
        latent_dim=32,
        n_labels=tiers.n_labels,
        device=device,
        warm_start=cfg.queue_warm_start,
    )
    optim = torch.optim.AdamW(
        [*model.parameters(), log_temp],
        lr=cfg.max_lr,
        weight_decay=cfg.weight_decay,
        fused=torch.cuda.is_available(),
    )

    # Macro timing via cuda.Event pairs
    components = (
        "data",
        "adapter",
        "forward",
        "supcon",
        "beta_nll",
        "barlow",
        "backward",
        "optimizer",
        "enqueue",
    )
    start_events: dict[str, list[torch.cuda.Event]] = {k: [] for k in components}
    end_events: dict[str, list[torch.cuda.Event]] = {k: [] for k in components}

    lw = cfg.loss_weights
    model.train()

    def tick(name: str) -> torch.cuda.Event:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        start_events[name].append(e)
        return e

    def tock(name: str) -> None:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        end_events[name].append(e)

    # --- warmup ---
    step_iter = iter(train_loader)
    for _ in range(args.warmup):
        try:
            batch = next(step_iter)
        except StopIteration:
            step_iter = iter(train_loader)
            batch = next(step_iter)
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            x_a = adapter(x)
            mu, L, h, z = model(x_a)
            tau = _clamped_temperature(log_temp, cfg.temperature_bounds)
            qz, qy = queue.get()
            zk = torch.cat([z, qz], dim=0)
            yk = torch.cat([y, qy], dim=0)
            supcon = supcon_soft_positive(z, y, zk, yk, temperature=tau, sigma=lw.supcon_sigma)
            y_block = model.block_layout.reorder_human_to_block(y)
            finite = torch.isfinite(y_block)
            y_clean = torch.where(finite, y_block, mu.detach())
            nll = beta_nll_block_cholesky(mu, L, y_clean, beta=lw.beta, mask=finite.float())
            bt = barlow_twins_loss(h, lam=lw.barlow_lam)
            total = lw.supcon * supcon + lw.beta_nll * nll + lw.barlow * bt
        total.backward()
        torch.nn.utils.clip_grad_norm_([*model.parameters(), log_temp], cfg.grad_clip_norm)
        optim.step()
        queue.enqueue(z.detach(), y.detach())
    torch.cuda.synchronize()
    print(f"warmup done ({args.warmup} steps)")

    # --- measured (macro timing) ---
    step_iter = iter(train_loader)
    for _ in range(args.measure):
        try:
            tick("data")
            batch = next(step_iter)
            tock("data")
        except StopIteration:
            step_iter = iter(train_loader)
            tick("data")
            batch = next(step_iter)
            tock("data")
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            tick("adapter")
            x_a = adapter(x)
            tock("adapter")
            tick("forward")
            mu, L, h, z = model(x_a)
            tock("forward")
            tau = _clamped_temperature(log_temp, cfg.temperature_bounds)
            qz, qy = queue.get()
            zk = torch.cat([z, qz], dim=0)
            yk = torch.cat([y, qy], dim=0)
            tick("supcon")
            supcon = supcon_soft_positive(z, y, zk, yk, temperature=tau, sigma=lw.supcon_sigma)
            tock("supcon")
            y_block = model.block_layout.reorder_human_to_block(y)
            finite = torch.isfinite(y_block)
            y_clean = torch.where(finite, y_block, mu.detach())
            tick("beta_nll")
            nll = beta_nll_block_cholesky(mu, L, y_clean, beta=lw.beta, mask=finite.float())
            tock("beta_nll")
            tick("barlow")
            bt = barlow_twins_loss(h, lam=lw.barlow_lam)
            tock("barlow")
            total = lw.supcon * supcon + lw.beta_nll * nll + lw.barlow * bt

        tick("backward")
        total.backward()
        tock("backward")
        tick("optimizer")
        torch.nn.utils.clip_grad_norm_([*model.parameters(), log_temp], cfg.grad_clip_norm)
        optim.step()
        tock("optimizer")
        tick("enqueue")
        queue.enqueue(z.detach(), y.detach())
        tock("enqueue")
    torch.cuda.synchronize()

    # Collect timings
    elapsed: dict[str, list[float]] = {}
    for name in components:
        ss = start_events[name]
        es = end_events[name]
        elapsed[name] = [s.elapsed_time(e) for s, e in zip(ss, es, strict=True)]

    print(f"\n--- macro wall-clock breakdown (ms per batch, n={args.measure}) ---")
    total_ms = 0.0
    for name in components:
        if not elapsed[name]:
            continue
        med = statistics.median(elapsed[name])
        total_ms += med
    print(f"{'component':12s} {'median_ms':>10s} {'pct':>7s} {'p90_ms':>10s}")
    for name in components:
        if not elapsed[name]:
            continue
        med = statistics.median(elapsed[name])
        p90 = np.percentile(elapsed[name], 90)
        pct = 100.0 * med / max(total_ms, 1e-9)
        print(f"{name:12s} {med:10.3f} {pct:6.1f}% {p90:10.3f}")
    print(f"{'total':12s} {total_ms:10.3f}")

    # --- torch.profiler sweep ---
    print(f"\n--- torch.profiler ({args.profiler_steps} steps) ---")
    step_iter = iter(train_loader)
    sched = schedule(wait=1, warmup=2, active=args.profiler_steps, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(args.profiler_steps + 3):
            try:
                batch = next(step_iter)
            except StopIteration:
                step_iter = iter(train_loader)
                batch = next(step_iter)
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                x_a = adapter(x)
                mu, L, h, z = model(x_a)
                tau = _clamped_temperature(log_temp, cfg.temperature_bounds)
                qz, qy = queue.get()
                zk = torch.cat([z, qz], dim=0)
                yk = torch.cat([y, qy], dim=0)
                supcon = supcon_soft_positive(z, y, zk, yk, temperature=tau, sigma=lw.supcon_sigma)
                y_block = model.block_layout.reorder_human_to_block(y)
                finite = torch.isfinite(y_block)
                y_clean = torch.where(finite, y_block, mu.detach())
                nll = beta_nll_block_cholesky(mu, L, y_clean, beta=lw.beta, mask=finite.float())
                bt = barlow_twins_loss(h, lam=lw.barlow_lam)
                total = lw.supcon * supcon + lw.beta_nll * nll + lw.barlow * bt
            total.backward()
            torch.nn.utils.clip_grad_norm_([*model.parameters(), log_temp], cfg.grad_clip_norm)
            optim.step()
            queue.enqueue(z.detach(), y.detach())
            prof.step()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))


if __name__ == "__main__":
    main()
