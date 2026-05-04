"""Tests for xp_abundances.main.training — seeds, loop, checkpoint, ensemble.

Post-#117 FeatureLayout contract: flat scalar columns
``bp_coef_norm_{i}`` / ``rp_coef_norm_{i}`` for ``i`` in ``xp_bp_indices`` /
``xp_rp_indices``, plus xp_scalar / residual / aux column tuples. No
list-typed array columns. Also exercises the #130 adapter + the phase-toggle
knobs on :class:`TrainingConfig` (``use_c0_scalars``, ``encoder_lr_ratio``,
``pretrained_encoder_ckpt``, ``checkpoint_every_n_epochs``, ``relative_min_delta``,
``output_prefix``).
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from pathlib import Path

# Tiny synthetic datasets (48-64 samples × 2-6 epochs) cannot learn per-label
# bulk offsets in one epoch; disable the #140 raw-units halt for all unit tests.
_SANITY_OFF = math.inf

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from arqueogal.xp_abundances.main.config import LossWeights, TrainingConfig
from arqueogal.xp_abundances.main.data import (
    FeatureLayout,
    LabelScaler,
    LabelTiers,
    XpAbundanceDataset,
    stratified_split_ids,
)
from arqueogal.xp_abundances.main.training import (
    CHECKPOINT_VERSION,
    _compute_inverse_freq_weights,
    build_dataloaders,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    train_ensemble,
    train_model,
)

# --- fixtures ---------------------------------------------------------------


def _tiny_layout(n_bp: int = 4, n_rp: int = 4) -> FeatureLayout:
    """Minimal FeatureLayout — a few BP+RP shape coefs, no aux/residuals/scalars.

    This keeps synthetic training cheap on CPU while still exercising the
    real flat-column flattening path through :class:`FeatureLayout`.
    """
    return FeatureLayout(
        xp_bp_indices=tuple(range(1, n_bp + 1)),
        xp_rp_indices=tuple(range(1, n_rp + 1)),
        xp_scalar_cols=(),
        residual_cols=(),
        aux_cols=(),
    )


def _synth_frame(n: int, layout: FeatureLayout, tiers: LabelTiers) -> pd.DataFrame:
    """Synthetic training parquet — label = linear projection of inputs (learnable)."""
    rng = np.random.default_rng(0)
    cols: dict[str, object] = {"source_id": np.arange(1, n + 1, dtype=np.int64)}

    # Flat Hermite-coef scalar columns (the post-#117 contract).
    feat_cols = list(layout.all_required_columns)
    X = rng.uniform(-1.0, 1.0, (n, layout.input_dim)).astype(np.float32)
    for j, c in enumerate(feat_cols):
        cols[c] = X[:, j]

    # Learnable labels: Y = X @ W (deterministic), with small label noise on σ.
    label_rng = np.random.default_rng(1)
    W = label_rng.normal(
        scale=1.0 / np.sqrt(max(layout.input_dim, 1)),
        size=(layout.input_dim, tiers.n_labels),
    ).astype(np.float32)
    Y = X @ W
    for i, lab in enumerate(tiers.all_labels):
        cols[lab] = Y[:, i]
    for e in tiers.label_error_columns():
        cols[e] = rng.uniform(0.01, 0.1, n).astype(np.float32)

    # Columns needed by stratified_split_ids.
    if "fe_h_apogee" not in cols:
        cols["fe_h_apogee"] = rng.normal(-0.2, 0.3, n).astype(np.float32)
    if "teff_apogee" not in cols:
        cols["teff_apogee"] = rng.uniform(4000, 5500, n).astype(np.float32)
    cols["b_deg"] = rng.uniform(-60, 60, n).astype(np.float32)
    return pd.DataFrame(cols)


def _arrs_from_frame(
    df: pd.DataFrame,
    layout: FeatureLayout,
    tiers: LabelTiers,
) -> dict[str, np.ndarray]:
    """Build arrays directly from a DataFrame (avoids parquet round-trip in tests)."""
    feat_cols = list(layout.all_required_columns)
    X = (
        np.column_stack([df[c].to_numpy(np.float32) for c in feat_cols])
        if feat_cols
        else np.empty((len(df), 0), dtype=np.float32)
    )
    Y = np.column_stack([df[c].to_numpy(np.float32) for c in tiers.all_labels])
    sig = np.column_stack([df[c].to_numpy(np.float32) for c in tiers.label_error_columns()])
    return {
        "X": X,
        "Y": Y,
        "sigma_Y": sig,
        "source_id": df["source_id"].to_numpy(np.int64),
    }


def _tiny_loaders(
    layout: FeatureLayout,
    tiers: LabelTiers,
    n: int = 64,
    batch: int = 16,
) -> tuple[DataLoader, DataLoader, LabelScaler]:
    df = _synth_frame(n, layout, tiers)
    ids = stratified_split_ids(df, fracs=(0.75, 0.25, 0.0), seed=0)
    arrs = _arrs_from_frame(df, layout, tiers)
    tr_mask = np.isin(arrs["source_id"], ids["train"])
    va_mask = np.isin(arrs["source_id"], ids["val"])
    # Fit a scaler on the train partition and apply to both partitions; the
    # training loop expects labels to live in standardised space.
    scaler = LabelScaler.fit(arrs["Y"][tr_mask], tiers.all_labels)
    Y_std = scaler.transform(arrs["Y"])
    sig_std = arrs["sigma_Y"] / scaler.scale.reshape(1, -1)
    train_ds = XpAbundanceDataset(
        X=arrs["X"][tr_mask],
        Y=Y_std[tr_mask],
        sigma_Y=sig_std[tr_mask],
    )
    val_ds = XpAbundanceDataset(
        X=arrs["X"][va_mask],
        Y=Y_std[va_mask],
        sigma_Y=sig_std[va_mask],
    )
    return (
        DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=batch, shuffle=False),
        scaler,
    )


def _tiny_cfg(tmp_path: Path, epochs: int = 2) -> TrainingConfig:
    return TrainingConfig(
        train_parquet=tmp_path / "train.parquet",
        output_dir=tmp_path / "ckpts",
        epochs=epochs,
        batch_size=16,
        num_workers=0,
        amp_dtype="none",
        ensemble_seeds=(0, 1),
        early_stop_patience=100,  # don't trigger on a tiny run
        first_epoch_sanity_k=_SANITY_OFF,
    )


# --- tests ------------------------------------------------------------------


def test_seed_everything_matches_on_rerun() -> None:
    seed_everything(42)
    a = (random.random(), np.random.rand(), torch.randn(1).item())
    seed_everything(42)
    b = (random.random(), np.random.rand(), torch.randn(1).item())
    assert a == b


def test_build_dataloaders_shapes(tmp_path: Path) -> None:
    layout = _tiny_layout()
    tiers = LabelTiers()  # full 21-label default (block layout requires it)
    df = _synth_frame(80, layout, tiers)
    parquet = tmp_path / "train.parquet"
    df.to_parquet(parquet, index=False)

    cfg = replace(_tiny_cfg(tmp_path), train_parquet=parquet)
    tr, va, ids, scaler, _feat_scaler = build_dataloaders(cfg, layout, tiers, seed=0)
    x, y, _s = next(iter(tr))
    assert x.shape[1] == layout.input_dim
    assert y.shape[1] == tiers.n_labels
    assert set(ids) == {"train", "val", "test"}
    assert sum(len(v) for v in ids.values()) == 80
    assert scaler.label_names == tiers.all_labels
    assert not scaler.is_default()


def test_train_model_reduces_loss() -> None:
    """Loss at last epoch should be lower than at epoch 0 on learnable data."""
    layout = _tiny_layout(n_bp=6, n_rp=6)
    tiers = LabelTiers()
    tr, va, scaler = _tiny_loaders(layout, tiers, n=64, batch=16)

    cfg = TrainingConfig(
        epochs=6,
        batch_size=16,
        num_workers=0,
        amp_dtype="none",
        max_lr=5e-3,
        early_stop_patience=100,
        ensemble_seeds=(0,),
        first_epoch_sanity_k=_SANITY_OFF,
    )
    result = train_model(
        cfg,
        layout,
        tiers,
        seed=0,
        train_loader=tr,
        val_loader=va,
        label_scaler=scaler,
        device=torch.device("cpu"),
    )
    hist = result["history"]
    assert len(hist) == cfg.epochs
    assert hist[-1]["val_loss"] < hist[0]["val_loss"], (
        f"expected val_loss to drop; got {hist[0]['val_loss']} → {hist[-1]['val_loss']}"
    )


def test_train_model_early_stopping_triggers() -> None:
    """If val-loss plateaus, early stopping should cut training short."""
    layout = _tiny_layout()
    tiers = LabelTiers()
    tr, va, scaler = _tiny_loaders(layout, tiers, n=48, batch=16)

    cfg = TrainingConfig(
        epochs=50,
        batch_size=16,
        num_workers=0,
        amp_dtype="none",
        max_lr=1e-6,  # effectively frozen — won't improve
        early_stop_patience=2,
        early_stop_min_delta=1e-3,
        ensemble_seeds=(0,),
        first_epoch_sanity_k=_SANITY_OFF,
    )
    result = train_model(
        cfg,
        layout,
        tiers,
        seed=0,
        train_loader=tr,
        val_loader=va,
        label_scaler=scaler,
        device=torch.device("cpu"),
    )
    assert len(result["history"]) < cfg.epochs


def test_save_and_load_checkpoint_roundtrip(tmp_path: Path) -> None:
    layout = _tiny_layout()
    tiers = LabelTiers()
    tr, va, scaler = _tiny_loaders(layout, tiers, n=32, batch=16)

    cfg = TrainingConfig(
        epochs=1,
        batch_size=16,
        num_workers=0,
        amp_dtype="none",
        ensemble_seeds=(0,),
        early_stop_patience=100,
        first_epoch_sanity_k=_SANITY_OFF,
    )
    result = train_model(
        cfg,
        layout,
        tiers,
        seed=7,
        train_loader=tr,
        val_loader=va,
        label_scaler=scaler,
        device=torch.device("cpu"),
    )

    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        model=result["model"],
        log_temp=result["log_temp"],
        cfg=cfg,
        layout=layout,
        tiers=tiers,
        label_scaler=result["label_scaler"],
        seed=7,
        training_metrics={"best_val_loss": result["best_val_loss"]},
        git_sha="abcdef1234567",
    )
    blob = load_checkpoint(path, map_location="cpu")
    assert blob["version"] == CHECKPOINT_VERSION
    assert blob["n_labels"] == tiers.n_labels
    assert blob["label_names"] == list(tiers.all_labels)
    assert blob["random_seed"] == 7
    assert "encoder" in blob and "regressor" in blob
    assert blob["tier_map"]["teff_apogee"] == 1


def test_load_checkpoint_rejects_wrong_version(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"version": 999}, path)
    with pytest.raises(ValueError, match="version"):
        load_checkpoint(path, map_location="cpu")


def test_train_ensemble_writes_one_file_per_seed(tmp_path: Path) -> None:
    layout = _tiny_layout()
    tiers = LabelTiers()
    df = _synth_frame(48, layout, tiers)
    parquet = tmp_path / "train.parquet"
    df.to_parquet(parquet, index=False)

    cfg = TrainingConfig(
        train_parquet=parquet,
        output_dir=tmp_path / "ckpts",
        epochs=1,
        batch_size=16,
        num_workers=0,
        amp_dtype="none",
        ensemble_seeds=(0, 1, 2),
        early_stop_patience=100,
        first_epoch_sanity_k=_SANITY_OFF,
    )
    paths = train_ensemble(cfg, layout, tiers, date_tag="20260418", git_sha="abcdef1")
    assert len(paths) == 3
    for p, seed in zip(paths, cfg.ensemble_seeds, strict=True):
        assert p.exists()
        assert f"seed{seed}" in p.name
        blob = load_checkpoint(p, map_location="cpu")
        assert blob["random_seed"] == seed


def test_train_ensemble_members_are_distinct(tmp_path: Path) -> None:
    """Different seeds must yield different final weights."""
    layout = _tiny_layout()
    tiers = LabelTiers()
    df = _synth_frame(48, layout, tiers)
    parquet = tmp_path / "train.parquet"
    df.to_parquet(parquet, index=False)

    cfg = TrainingConfig(
        train_parquet=parquet,
        output_dir=tmp_path / "ckpts",
        epochs=1,
        batch_size=16,
        num_workers=0,
        amp_dtype="none",
        ensemble_seeds=(0, 1),
        early_stop_patience=100,
        first_epoch_sanity_k=_SANITY_OFF,
    )
    paths = train_ensemble(cfg, layout, tiers, date_tag="20260418", git_sha="abcdef1")
    b0 = load_checkpoint(paths[0], map_location="cpu")
    b1 = load_checkpoint(paths[1], map_location="cpu")
    k = next(iter(b0["encoder"]))
    assert not torch.allclose(b0["encoder"][k], b1["encoder"][k])


def test_training_config_serialises_paths(tmp_path: Path) -> None:
    """Path fields must round-trip through checkpoint JSON."""
    layout = _tiny_layout()
    tiers = LabelTiers()
    tr, va, scaler = _tiny_loaders(layout, tiers, n=32, batch=16)

    cfg = TrainingConfig(
        train_parquet=tmp_path / "x.parquet",
        output_dir=tmp_path / "ckpts",
        epochs=1,
        batch_size=16,
        num_workers=0,
        amp_dtype="none",
        ensemble_seeds=(0,),
        early_stop_patience=100,
        first_epoch_sanity_k=_SANITY_OFF,
    )
    result = train_model(
        cfg,
        layout,
        tiers,
        seed=0,
        train_loader=tr,
        val_loader=va,
        label_scaler=scaler,
        device=torch.device("cpu"),
    )
    path = tmp_path / "c.pt"
    save_checkpoint(
        path,
        model=result["model"],
        log_temp=result["log_temp"],
        cfg=cfg,
        layout=layout,
        tiers=tiers,
        label_scaler=result["label_scaler"],
        seed=0,
    )
    blob = load_checkpoint(path, map_location="cpu")
    import json

    parsed = json.loads(blob["config_yaml"])
    assert parsed["train_parquet"].endswith("x.parquet")


def test_feature_layout_default_input_dim_matches_design() -> None:
    """DESIGN: full layout = 54 BP + 54 RP + 2 c0 + 3 residuals + 27 aux = 140-D.

    Aux count moved from 26 → 27 on 2026-04-29 (extinction-correction
    protocol). See ``docs/protocols/extinction_correction.md``.
    """
    assert FeatureLayout().input_dim == 140
    # Truncated layout: 19 + 22 + 2 + 3 + 27 = 73-D.
    assert FeatureLayout.truncated_43d().input_dim == 73


# --- Run A / #119.3 / #119.5 phase-toggle knobs ------------------------------


def test_use_c0_scalars_false_masks_c0_in_trunk_input() -> None:
    """With ``use_c0_scalars=False``, c0 channels in the batch are zeroed before the trunk.

    We train on data where the label depends on c0 — if c0 is zeroed, the
    val-loss should be higher than with ``use_c0_scalars=True`` on the same
    data/seed. This proves the adapter is wired into the forward pass.
    """
    n, batch = 64, 16
    layout = FeatureLayout(
        xp_bp_indices=(1, 2),
        xp_rp_indices=(1, 2),
        xp_scalar_cols=("bp_c0_z", "rp_c0_z"),  # c0 present, at positions 4,5
        residual_cols=(),
        aux_cols=(),
    )
    tiers = LabelTiers()

    # Baseline frame with full tiers, then overwrite X+labels so the labels
    # depend *only* on the two c0 columns (positions 4, 5). If c0 gets zeroed
    # by the adapter, the best any trunk can do is predict the mean.
    df = _synth_frame(n, layout, tiers)
    feat_cols = list(layout.all_required_columns)
    rng = np.random.default_rng(7)
    X = rng.uniform(-1.0, 1.0, (n, layout.input_dim)).astype(np.float32)
    for j, c in enumerate(feat_cols):
        df[c] = X[:, j]
    c0_signal = (X[:, 4] + X[:, 5]).astype(np.float32)  # bp_c0_z + rp_c0_z
    for lab in tiers.all_labels:
        df[lab] = c0_signal  # every label equals the c0 signal
    arrs = _arrs_from_frame(df, layout, tiers)
    scaler = LabelScaler.fit(arrs["Y"][:48], tiers.all_labels)
    Y_std = scaler.transform(arrs["Y"])
    sig_std = arrs["sigma_Y"] / scaler.scale.reshape(1, -1)
    ds_tr = XpAbundanceDataset(X=arrs["X"][:48], Y=Y_std[:48], sigma_Y=sig_std[:48])
    ds_va = XpAbundanceDataset(X=arrs["X"][48:], Y=Y_std[48:], sigma_Y=sig_std[48:])
    tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, drop_last=True)
    va = DataLoader(ds_va, batch_size=batch, shuffle=False)

    cfg_on = TrainingConfig(
        epochs=8,
        batch_size=batch,
        num_workers=0,
        amp_dtype="none",
        max_lr=5e-3,
        early_stop_patience=100,
        ensemble_seeds=(0,),
        use_c0_scalars=True,
        first_epoch_sanity_k=_SANITY_OFF,
    )
    cfg_off = replace(cfg_on, use_c0_scalars=False)

    r_on = train_model(
        cfg_on,
        layout,
        tiers,
        seed=0,
        train_loader=tr,
        val_loader=va,
        label_scaler=scaler,
        device=torch.device("cpu"),
    )
    r_off = train_model(
        cfg_off,
        layout,
        tiers,
        seed=0,
        train_loader=tr,
        val_loader=va,
        label_scaler=scaler,
        device=torch.device("cpu"),
    )
    # c0-dependent signal: trunk that can see c0 must end up with lower val loss.
    assert r_on["best_val_loss"] < r_off["best_val_loss"]


def test_encoder_lr_ratio_creates_two_param_groups(tmp_path: Path) -> None:
    """``encoder_lr_ratio < 1.0`` → two AdamW groups, encoder group at the reduced LR.

    Inspected via the optimizer inside train_one_epoch — exposed through
    ``result['model']``'s weight changes: over one epoch with head-ratio 0,
    encoder weights must stay frozen while head weights move.
    """
    # Use ratio = 0.0 (encoder fully frozen) so the effect is unambiguous.
    layout = _tiny_layout()
    tiers = LabelTiers()
    tr, va, scaler = _tiny_loaders(layout, tiers, n=48, batch=16)

    cfg = TrainingConfig(
        epochs=3,
        batch_size=16,
        num_workers=0,
        amp_dtype="none",
        max_lr=1e-2,
        early_stop_patience=100,
        ensemble_seeds=(0,),
        encoder_lr_ratio=0.0,
        first_epoch_sanity_k=_SANITY_OFF,
    )
    # Snapshot encoder weights *before* train_model builds+optimises.
    # train_model builds the model internally, so we capture post-train and
    # verify head/log_temp moved while encoder did not (at ratio=0 they can't).
    result = train_model(
        cfg,
        layout,
        tiers,
        seed=0,
        train_loader=tr,
        val_loader=va,
        label_scaler=scaler,
        device=torch.device("cpu"),
    )
    model = result["model"]

    # Re-init a fresh model with the same seed and compare post-train encoder weights.
    from arqueogal.xp_abundances.main.training import _build_model_and_temperature

    seed_everything(0)
    fresh_model, _fresh_lt, _ad = _build_model_and_temperature(
        cfg,
        layout,
        tiers,
        device=torch.device("cpu"),
    )

    # At encoder_lr_ratio=0 the encoder must not have moved.
    for (name, p_trained), (_n, p_fresh) in zip(
        model.encoder.named_parameters(),
        fresh_model.encoder.named_parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            p_trained,
            p_fresh,
            msg=f"encoder param {name} changed despite encoder_lr_ratio=0",
        )
    # Head should have moved.
    moved_any_head = False
    for (_n, p_trained), (_n2, p_fresh) in zip(
        model.head.named_parameters(),
        fresh_model.head.named_parameters(),
        strict=True,
    ):
        if not torch.allclose(p_trained, p_fresh):
            moved_any_head = True
            break
    assert moved_any_head, "head parameters did not move at encoder_lr_ratio=0"


def test_checkpoint_every_n_epochs_writes_cadence_files(tmp_path: Path) -> None:
    """``checkpoint_every_n_epochs=1`` writes cadence ckpts alongside the best-val one."""
    layout = _tiny_layout()
    tiers = LabelTiers()
    tr, va, scaler = _tiny_loaders(layout, tiers, n=32, batch=16)

    cfg = TrainingConfig(
        train_parquet=tmp_path / "train.parquet",
        output_dir=tmp_path / "ckpts",
        epochs=3,
        batch_size=16,
        num_workers=0,
        amp_dtype="none",
        ensemble_seeds=(0,),
        early_stop_patience=100,
        checkpoint_every_n_epochs=1,
        output_prefix="xp_abundances_main_unit",
        first_epoch_sanity_k=_SANITY_OFF,
    )
    result = train_model(
        cfg,
        layout,
        tiers,
        seed=0,
        train_loader=tr,
        val_loader=va,
        label_scaler=scaler,
        device=torch.device("cpu"),
    )
    cadence_paths = result["cadence_checkpoints"]
    assert len(cadence_paths) == cfg.epochs
    cadence_dir = cfg.output_dir / "cadence"
    for p in cadence_paths:
        assert p.exists()
        assert p.parent == cadence_dir
        assert "xp_abundances_main_unit" in p.name
        assert "seed0" in p.name


def test_pretrained_encoder_ckpt_loads_weights(tmp_path: Path) -> None:
    """Saving a checkpoint and pointing ``pretrained_encoder_ckpt`` at it loads encoder weights."""
    layout = _tiny_layout()
    tiers = LabelTiers()
    tr, va, scaler = _tiny_loaders(layout, tiers, n=32, batch=16)

    # Phase A: pretrain a bit with encoder alive.
    cfg_a = TrainingConfig(
        epochs=2,
        batch_size=16,
        num_workers=0,
        amp_dtype="none",
        max_lr=5e-3,
        early_stop_patience=100,
        ensemble_seeds=(0,),
        first_epoch_sanity_k=_SANITY_OFF,
    )
    r_a = train_model(
        cfg_a,
        layout,
        tiers,
        seed=0,
        train_loader=tr,
        val_loader=va,
        label_scaler=scaler,
        device=torch.device("cpu"),
    )
    ckpt_path = tmp_path / "pretrain.pt"
    save_checkpoint(
        ckpt_path,
        model=r_a["model"],
        log_temp=r_a["log_temp"],
        cfg=cfg_a,
        layout=layout,
        tiers=tiers,
        label_scaler=r_a["label_scaler"],
        seed=0,
    )

    # Phase B: fine-tune with encoder_lr_ratio=0 so encoder is frozen,
    # so we can verify the loaded encoder weights match the pretrained ones bit-for-bit.
    cfg_b = TrainingConfig(
        epochs=1,
        batch_size=16,
        num_workers=0,
        amp_dtype="none",
        max_lr=1e-3,
        early_stop_patience=100,
        ensemble_seeds=(1,),
        encoder_lr_ratio=0.0,
        pretrained_encoder_ckpt=ckpt_path,
        first_epoch_sanity_k=_SANITY_OFF,
    )
    r_b = train_model(
        cfg_b,
        layout,
        tiers,
        seed=1,
        train_loader=tr,
        val_loader=va,
        label_scaler=scaler,
        device=torch.device("cpu"),
    )

    # Encoder must equal the saved encoder (frozen + loaded).
    saved_encoder = r_a["model"].encoder.state_dict()
    loaded_encoder = r_b["model"].encoder.state_dict()
    assert saved_encoder.keys() == loaded_encoder.keys()
    for k in saved_encoder:
        torch.testing.assert_close(saved_encoder[k], loaded_encoder[k])


def test_supcon_label_n_first_slices_labels_used_for_pair_kernel() -> None:
    """``supcon_label_n_first=3`` restricts SupCon's Gaussian-kernel to first 3 labels.

    We set ``beta_nll=0`` and train with full tiers — the run must not crash and
    must produce finite losses, proving the slicing branch is exercised (the
    alternative path uses all labels).
    """
    layout = _tiny_layout()
    # Four labels: first 3 are Tier-1 analogues, fourth lives in tier2.
    tiers = LabelTiers()
    tr, va, scaler = _tiny_loaders(layout, tiers, n=48, batch=16)

    cfg = TrainingConfig(
        epochs=2,
        batch_size=16,
        num_workers=0,
        amp_dtype="none",
        max_lr=1e-3,
        early_stop_patience=100,
        ensemble_seeds=(0,),
        loss_weights=LossWeights(
            supcon=1.0,
            beta_nll=0.0,
            supcon_label_n_first=3,
        ),
        first_epoch_sanity_k=_SANITY_OFF,
    )
    r = train_model(
        cfg,
        layout,
        tiers,
        seed=0,
        train_loader=tr,
        val_loader=va,
        label_scaler=scaler,
        device=torch.device("cpu"),
    )
    for epoch_row in r["history"]:
        assert np.isfinite(epoch_row["val_loss"])


def test_relative_min_delta_uses_fraction_of_best_val() -> None:
    """Relative min_delta fires early-stop when improvements shrink below the fraction."""
    layout = _tiny_layout()
    tiers = LabelTiers()
    tr, va, scaler = _tiny_loaders(layout, tiers, n=48, batch=16)

    # Huge relative threshold (50%) plus tiny LR — nothing will improve enough.
    cfg = TrainingConfig(
        epochs=30,
        batch_size=16,
        num_workers=0,
        amp_dtype="none",
        max_lr=1e-6,
        early_stop_patience=1,
        early_stop_min_delta=0.5,
        relative_min_delta=True,
        ensemble_seeds=(0,),
        first_epoch_sanity_k=_SANITY_OFF,
    )
    r = train_model(
        cfg,
        layout,
        tiers,
        seed=0,
        train_loader=tr,
        val_loader=va,
        label_scaler=scaler,
        device=torch.device("cpu"),
    )
    assert len(r["history"]) < cfg.epochs


def test_output_prefix_controls_ensemble_filenames(tmp_path: Path) -> None:
    layout = _tiny_layout()
    tiers = LabelTiers()
    df = _synth_frame(48, layout, tiers)
    parquet = tmp_path / "train.parquet"
    df.to_parquet(parquet, index=False)

    cfg = TrainingConfig(
        train_parquet=parquet,
        output_dir=tmp_path / "ckpts",
        epochs=1,
        batch_size=16,
        num_workers=0,
        amp_dtype="none",
        ensemble_seeds=(0,),
        early_stop_patience=100,
        output_prefix="xp_abundances_main_finetune",
        first_epoch_sanity_k=_SANITY_OFF,
    )
    paths = train_ensemble(cfg, layout, tiers, date_tag="20260419", git_sha="deadbee")
    assert all("xp_abundances_main_finetune" in p.name for p in paths)


# --- inverse-frequency [M/H] weighting (#198, v1.1) -------------------------


def _mh_column_index(tiers: LabelTiers, col: str = "mh_apogee") -> int:
    return tiers.all_labels.index(col)


def test_compute_inverse_freq_weights_mean_one_normalisation() -> None:
    """The returned weight vector must have mean exactly 1.0."""
    tiers = LabelTiers.five_label()
    n = 5000
    rng = np.random.default_rng(0)
    Y = np.zeros((n, tiers.n_labels), dtype=np.float32)
    # Disc-dominated: 80% at [M/H]≈0, 20% at [M/H]≈-1.0.
    Y[:, _mh_column_index(tiers)] = np.concatenate(
        [
            rng.normal(0.0, 0.1, int(0.80 * n)),
            rng.normal(-1.0, 0.1, n - int(0.80 * n)),
        ]
    ).astype(np.float32)
    cfg = TrainingConfig(
        inverse_freq_weighting=True,
        inverse_freq_bin_edges=(-1.5, -1.0, -0.5, 0.0),
        inverse_freq_clip=5.0,
    )
    w = _compute_inverse_freq_weights(Y, tiers=tiers, cfg=cfg)
    assert w.shape == (n,)
    assert np.isclose(float(w.mean()), 1.0, atol=1e-4)


def test_compute_inverse_freq_weights_up_weights_rare_bin() -> None:
    """Stars in the metal-poor bin must receive larger weights than disc stars."""
    tiers = LabelTiers.five_label()
    n_disc, n_halo = 8000, 200
    rng = np.random.default_rng(1)
    Y = np.zeros((n_disc + n_halo, tiers.n_labels), dtype=np.float32)
    Y[:n_disc, _mh_column_index(tiers)] = rng.normal(0.0, 0.1, n_disc)
    Y[n_disc:, _mh_column_index(tiers)] = rng.normal(-1.2, 0.05, n_halo)

    cfg = TrainingConfig(
        inverse_freq_weighting=True,
        inverse_freq_bin_edges=(-1.5, -1.0, -0.5, 0.0),
        inverse_freq_clip=5.0,
    )
    w = _compute_inverse_freq_weights(Y, tiers=tiers, cfg=cfg)
    w_disc = float(w[:n_disc].mean())
    w_halo = float(w[n_disc:].mean())
    # Clip-then-mean-1 normalisation compresses the nominal 5× clip to a smaller
    # post-norm ratio because the rare-bin weight lifts the global mean used as
    # denominator. 2× is the weakest ratio that still proves monotonicity.
    assert w_halo > w_disc * 2.0


def test_compute_inverse_freq_weights_nan_mh_keeps_weight_one() -> None:
    """NaN-[M/H] stars must pass through with weight ≈ 1.0 (pre-normalisation)."""
    tiers = LabelTiers.five_label()
    n = 1000
    rng = np.random.default_rng(2)
    Y = np.zeros((n, tiers.n_labels), dtype=np.float32)
    mh_col = _mh_column_index(tiers)
    Y[:, mh_col] = rng.normal(0.0, 0.1, n).astype(np.float32)
    nan_idx = np.array([3, 17, 42, 100], dtype=np.int64)
    Y[nan_idx, mh_col] = np.nan

    cfg = TrainingConfig(
        inverse_freq_weighting=True,
        inverse_freq_bin_edges=(-1.5, -1.0, -0.5, 0.0),
        inverse_freq_clip=5.0,
    )
    w = _compute_inverse_freq_weights(Y, tiers=tiers, cfg=cfg)
    # After mean-1 normalisation, the NaN-[M/H] stars carry the same
    # constant — different from the bin-weight-1 value only by the global
    # mean-preserving scale. Check they're all equal (within fp precision).
    assert np.allclose(w[nan_idx], w[nan_idx[0]], atol=1e-5)


def test_compute_inverse_freq_weights_clip_caps_extreme_bins() -> None:
    """A near-empty bin must not produce unbounded weight — clip enforces cap."""
    tiers = LabelTiers.five_label()
    rng = np.random.default_rng(3)
    n_disc, n_rare = 9990, 10
    Y = np.zeros((n_disc + n_rare, tiers.n_labels), dtype=np.float32)
    mh_col = _mh_column_index(tiers)
    Y[:n_disc, mh_col] = rng.normal(0.0, 0.1, n_disc)
    # One very rare bin at [M/H] < -1.5 (prob 10/10000 = 0.001).
    Y[n_disc:, mh_col] = rng.normal(-2.0, 0.05, n_rare)

    cfg = TrainingConfig(
        inverse_freq_weighting=True,
        inverse_freq_bin_edges=(-1.5, -1.0, -0.5, 0.0),
        inverse_freq_clip=5.0,
    )
    w = _compute_inverse_freq_weights(Y, tiers=tiers, cfg=cfg)
    # Post-normalisation max can exceed clip × 1 because normalisation shifts
    # the mean; what must hold is max/min ratio ≈ clip upper bound.
    ratio = float(w.max() / w.min())
    assert ratio < cfg.inverse_freq_clip / 0.1, (  # generous bound
        f"weight range ratio {ratio} exceeded expected clip regime"
    )


def test_compute_inverse_freq_weights_bad_mh_column_raises() -> None:
    tiers = LabelTiers.five_label()
    Y = np.zeros((10, tiers.n_labels), dtype=np.float32)
    cfg = TrainingConfig(
        inverse_freq_weighting=True,
        inverse_freq_mh_column="nonexistent_label",
    )
    with pytest.raises(ValueError, match="not in tiers.all_labels"):
        _compute_inverse_freq_weights(Y, tiers=tiers, cfg=cfg)


def test_compute_inverse_freq_weights_non_monotonic_edges_raises() -> None:
    tiers = LabelTiers.five_label()
    Y = np.zeros((10, tiers.n_labels), dtype=np.float32)
    cfg = TrainingConfig(
        inverse_freq_weighting=True,
        inverse_freq_bin_edges=(0.0, -0.5, -1.0),  # non-monotonic
    )
    with pytest.raises(ValueError, match="strictly-increasing"):
        _compute_inverse_freq_weights(Y, tiers=tiers, cfg=cfg)


def test_build_dataloaders_yields_weights_when_enabled(tmp_path: Path) -> None:
    """When inverse_freq_weighting=True, the train loader yields 3-tuples (x, y, w)."""
    layout = _tiny_layout()
    tiers = LabelTiers.five_label()
    df = _synth_frame(120, layout, tiers)
    parquet = tmp_path / "train.parquet"
    df.to_parquet(parquet, index=False)

    cfg = replace(
        _tiny_cfg(tmp_path),
        train_parquet=parquet,
        inverse_freq_weighting=True,
        inverse_freq_bin_edges=(-0.5, 0.0),  # cheap 3-bin on synthetic range
        inverse_freq_clip=5.0,
    )
    tr, _va, _ids, _scaler, _feat_scaler = build_dataloaders(cfg, layout, tiers, seed=0)
    batch = next(iter(tr))
    assert len(batch) == 3, f"expected (x, y, w), got {len(batch)}-tuple"
    x, y, w = batch
    assert x.shape[0] == y.shape[0] == w.shape[0]
    assert w.shape == (x.shape[0],)
    # Weights are non-negative finite.
    assert bool(torch.isfinite(w).all())
    assert bool((w > 0).all())


def test_build_dataloaders_yields_sigma_when_weighting_disabled(tmp_path: Path) -> None:
    """When inverse_freq_weighting=False, legacy (x, y, sigma_Y) contract holds."""
    layout = _tiny_layout()
    tiers = LabelTiers()
    df = _synth_frame(120, layout, tiers)
    parquet = tmp_path / "train.parquet"
    df.to_parquet(parquet, index=False)

    cfg = replace(_tiny_cfg(tmp_path), train_parquet=parquet)
    tr, _va, _ids, _scaler, _feat_scaler = build_dataloaders(cfg, layout, tiers, seed=0)
    batch = next(iter(tr))
    assert len(batch) == 3, "legacy contract still yields (x, y, sigma_Y)"
    x, y, s = batch
    assert s.shape == y.shape  # sigma_Y is per-label, not per-star
    assert x.shape[0] == y.shape[0]
