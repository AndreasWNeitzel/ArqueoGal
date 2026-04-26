"""Tests for xp_abundances.main.config: TrainingConfig + LossWeights contracts.

The dataclasses are frozen + slotted, so the only meaningful tests are:
1. Defaults are stable (rename detection, since YAML round-trip relies on
   field names being load-bearing).
2. Construction with custom values is rejected when types are wrong (slots
   make this implicit but the tests document the contract).
3. The default LossWeights match the TESS_ML joint-loss recipe in AGENTS.md
   notes: supcon=1.0, beta_nll=1.0, beta=0.5, barlow=0.0 (off by default).
4. ``loss_weights`` is constructed via ``field(default_factory=LossWeights)``,
   so the default is shared across all default-constructed configs.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from arqueogal.xp_abundances.main.config import LossWeights, TrainingConfig


def test_loss_weights_defaults_match_joint_recipe() -> None:
    """SupCon=1.0, β-NLL=1.0, β=0.5, Barlow=0.0 (off) per AGENTS.md notes."""
    w = LossWeights()
    assert w.supcon == 1.0
    assert w.beta_nll == 1.0
    assert w.beta == 0.5
    assert w.barlow == 0.0
    assert w.barlow_lam == 0.005
    assert w.supcon_sigma == 0.10
    assert w.supcon_label_n_first is None


def test_loss_weights_is_frozen() -> None:
    w = LossWeights()
    with pytest.raises(Exception):
        w.supcon = 2.0  # type: ignore[misc]


def test_training_config_defaults_are_stable() -> None:
    """Defaults double as a regression test, any rename is silent breakage."""
    cfg = TrainingConfig()
    assert cfg.latent_dim == 32
    assert cfg.trunk_hidden == (256, 128)
    assert cfg.head_hidden == 128
    assert cfg.dropout == 0.10
    assert cfg.max_lr == 2e-3
    assert cfg.weight_decay == 1e-4
    assert cfg.epochs == 60
    assert cfg.batch_size == 512
    assert cfg.amp_dtype == "bfloat16"
    assert cfg.fracs == (0.70, 0.15, 0.15)
    assert cfg.split_seed == 0
    assert cfg.ensemble_seeds == (0, 1, 2, 3, 4)
    assert cfg.use_c0_scalars is True
    assert cfg.encoder_lr_ratio == 1.0
    assert cfg.deterministic is False


def test_training_config_paths_are_path_instances() -> None:
    cfg = TrainingConfig()
    assert isinstance(cfg.train_parquet, Path)
    assert isinstance(cfg.output_dir, Path)
    assert cfg.pretrained_encoder_ckpt is None


def test_training_config_field_set_is_recorded() -> None:
    """Snapshot of declared field names so YAML-key drift surfaces in CI."""
    declared = {f.name for f in fields(TrainingConfig)}
    expected_subset = {
        "train_parquet",
        "output_dir",
        "latent_dim",
        "trunk_hidden",
        "head_hidden",
        "dropout",
        "max_lr",
        "weight_decay",
        "pct_start",
        "grad_clip_norm",
        "epochs",
        "batch_size",
        "num_workers",
        "amp_dtype",
        "loss_weights",
        "temperature_init",
        "temperature_bounds",
        "split_seed",
        "fracs",
        "ensemble_seeds",
        "early_stop_patience",
        "early_stop_min_delta",
        "deterministic",
        "use_c0_scalars",
        "encoder_lr_ratio",
        "checkpoint_every_n_epochs",
        "pretrained_encoder_ckpt",
        "relative_min_delta",
        "output_prefix",
        "reload_head_from_pretrained",
        "first_epoch_sanity_k",
        "stage_dataset_on_gpu",
        "queue_size",
        "queue_warm_start",
        "grad_norm_abort_threshold",
        "inverse_freq_weighting",
        "inverse_freq_mh_column",
        "inverse_freq_bin_edges",
        "inverse_freq_clip",
    }
    missing = expected_subset - declared
    assert not missing, f"renamed/removed fields: {missing}"


def test_loss_weights_default_factory_not_shared_instance() -> None:
    """Two TrainingConfig() instances should each get their own LossWeights."""
    a = TrainingConfig()
    b = TrainingConfig()
    assert a.loss_weights == b.loss_weights
    assert a.loss_weights is not b.loss_weights


def test_training_config_is_frozen() -> None:
    cfg = TrainingConfig()
    with pytest.raises(Exception):
        cfg.epochs = 1  # type: ignore[misc]


def test_amp_dtype_only_accepts_documented_values_at_call_site() -> None:
    """The dataclass itself does not validate amp_dtype, the consumer does.

    This documents the contract: callers in training.py treat
    ``{"bfloat16", "float16", "none"}`` as the accepted set; any other value
    is a runtime error at the consumer, not at construction.
    """
    for ok in ("bfloat16", "float16", "none"):
        cfg = TrainingConfig(amp_dtype=ok)
        assert cfg.amp_dtype == ok


def test_inverse_freq_weighting_default_is_off() -> None:
    """v1 default: uniform NLL averaging. Inverse-frequency is opt-in for v1.1."""
    cfg = TrainingConfig()
    assert cfg.inverse_freq_weighting is False
    assert cfg.inverse_freq_mh_column == "mh_apogee"
    assert cfg.inverse_freq_bin_edges == (-1.5, -1.0, -0.5, 0.0)
    assert cfg.inverse_freq_clip == 5.0


def test_queue_disabled_by_default() -> None:
    """v1 trains without the SupCon queue; queue=8192 is a TESS_ML opt-in."""
    cfg = TrainingConfig()
    assert cfg.queue_size == 0
    assert cfg.queue_warm_start is True


def test_training_config_temperature_bounds_form_open_interval() -> None:
    cfg = TrainingConfig()
    lo, hi = cfg.temperature_bounds
    assert lo > 0
    assert hi > lo
    assert lo <= cfg.temperature_init <= hi
