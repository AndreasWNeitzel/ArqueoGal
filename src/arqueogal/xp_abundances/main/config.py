"""Pipeline 1 training-run configuration — serialisable knobs in one place.

A single :class:`TrainingConfig` captures every dial that differs between
experiments. It's intentionally flat (no nested configs for the model/loss
subsystems) because the downstream checkpoint schema (DESIGN §v2) wants one
config blob it can round-trip to YAML.

Keep field names stable across runs: they become keys in
``training_metrics["config_yaml"]`` and any rename silently breaks reload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LossWeights:
    """Scalar weights for the combined training objective.

    ``supcon=0`` disables the contrastive term (pure supervised fine-tune);
    ``beta_nll=0`` disables the regression term (contrastive pretrain only).
    Defaults train both jointly, which is what TESS_ML's prototype does.

    ``supcon_label_n_first`` restricts the SupCon Gaussian-kernel pair weighting
    to the first N label columns. Use ``3`` to weight on Tier-1 atmospheric
    labels ``(teff, logg, [M/H])`` only — the correct choice for contrastive
    pretraining, where aligning on chemistry would double-count what the
    supervised fine-tune is about to learn. ``None`` uses all labels (joint).
    """

    supcon: float = 1.0
    beta_nll: float = 1.0
    beta: float = 0.5  # Seitzer β in β-NLL — 0.5 per DESIGN.
    supcon_sigma: float = 0.10  # label-space Gaussian-kernel bandwidth.
    supcon_label_n_first: int | None = None


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Top-level training configuration.

    ``ensemble_seeds`` drives :func:`~.training.train_ensemble` — its length
    determines ensemble size. ``amp_dtype`` is "bfloat16", "float16", or
    "none"; bfloat16 is the RTX 3060 default (no GradScaler needed).
    """

    # --- data ---
    train_parquet: Path = Path("data/processed/pipeline1_training.parquet")
    output_dir: Path = Path("models/main/xp_abundances")

    # --- model ---
    latent_dim: int = 32
    trunk_hidden: tuple[int, ...] = (256, 128)
    head_hidden: int = 128
    dropout: float = 0.10

    # --- optimizer ---
    max_lr: float = 2e-3
    weight_decay: float = 1e-4
    pct_start: float = 0.15  # OneCycleLR warmup fraction
    grad_clip_norm: float = 1.0

    # --- training loop ---
    epochs: int = 60
    batch_size: int = 512
    num_workers: int = 2
    amp_dtype: str = "bfloat16"  # "bfloat16" | "float16" | "none"

    # --- loss ---
    loss_weights: LossWeights = field(default_factory=LossWeights)
    temperature_init: float = 0.10
    temperature_bounds: tuple[float, float] = (1e-3, 0.5)

    # --- split ---
    split_seed: int = 0
    fracs: tuple[float, float, float] = (0.70, 0.15, 0.15)

    # --- ensemble ---
    ensemble_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)

    # --- early stopping ---
    early_stop_patience: int = 10
    early_stop_min_delta: float = 1e-4

    # --- reproducibility ---
    deterministic: bool = False  # True forces CUDA determinism (slower).

    # --- phase-selecting knobs (contrastive pretrain vs supervised fine-tune) ---
    use_c0_scalars: bool = True
    """``False`` zeroes ``bp_c0_z`` / ``rp_c0_z`` via the adapter (contrastive
    pretrain, where c0_z confounds cell geometry with distance/extinction).
    ``True`` passes them through (supervised fine-tune, where the head gains a
    real luminosity/extinction degree of freedom)."""

    encoder_lr_ratio: float = 1.0
    """Learning-rate multiplier on encoder (trunk + projection) parameters
    relative to head parameters. ``1.0`` = single-group AdamW.
    ``0.1`` = encoder trains at 10× lower LR than the head — the supervised
    fine-tune default, per DESIGN: pretrained trunk needs gentle adjustment
    while the Cholesky head trains from scratch."""

    checkpoint_every_n_epochs: int = 0
    """If > 0, save a checkpoint every N epochs (in addition to the best-val
    model kept in memory). ``10`` is the contrastive-pretrain cadence;
    ``1`` is the supervised-fine-tune cadence. ``0`` disables."""

    pretrained_encoder_ckpt: Path | None = None
    """If set, load encoder (trunk + projection) weights from this checkpoint
    before training. Head + log_temp train from scratch. Used at the start of
    supervised fine-tune and for every ensemble member."""

    relative_min_delta: bool = False
    """Interpret ``early_stop_min_delta`` as a relative fraction of the current
    best val loss (e.g. ``0.01`` = 1% relative) rather than an absolute delta.
    Used by contrastive pretrain where SupCon loss magnitudes drift over
    training."""

    output_prefix: str = "xp_abundances_main"
    """Filename prefix for checkpoints written by :func:`train_ensemble`.
    Stage drivers override to ``"xp_abundances_main_contrastive"``,
    ``"xp_abundances_main_finetune"``, etc. so stages don't clobber each other."""

    reload_head_from_pretrained: bool = False
    """If True and ``pretrained_encoder_ckpt`` is set, also load the Cholesky
    head weights. Ensemble members do NOT use this — they reinit the head per
    seed. The joint-train workflow would."""

    first_epoch_sanity_k: float = 2.0
    """Multiplier on ``std(truth)`` for the epoch-0 mean-bias halt in
    :func:`~.training._first_epoch_sanity_check`. ``2.0`` catches the broken-
    label-scaler class of failure that surfaced in #140 without false-firing
    on normal fine-tune starts. Tests on tiny synthetic datasets override this
    upward (e.g. ``1e6``) since one epoch on 48 samples cannot learn bulk
    offsets. Setting ``inf`` disables the check."""

    grad_norm_abort_threshold: float = float("inf")
    """Hard upper bound on pre-clip gradient norm in any single training
    batch. If exceeded, training aborts with a diagnostic RuntimeError. Used
    by the β=0 retrain canary (#135 escalation): pure Gaussian NLL can
    explode on high-σ samples. Typical β=0.5 grads land in 0.5–5; ``500.0``
    is a safe β=0 canary. ``inf`` (default) disables the check."""


__all__ = ["LossWeights", "TrainingConfig"]
