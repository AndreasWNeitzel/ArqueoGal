"""Pipeline 1 training orchestrator — AdamW + OneCycleLR + AMP + ensemble.

This module wires the pieces already in place (``data.py``, ``model.py``,
``losses.py``) into a reproducible training loop that can be called either as
a single-seed run (:func:`train_model`) or as a sequential ensemble
(:func:`train_ensemble`, 5–10 seeds per DESIGN §Ensemble).

Design decisions made intentionally narrow for this slice:

- **Combined loss, single stage.** SupCon soft-positive + β-NLL summed with
  configurable weights, trained together from scratch. DESIGN talks about
  two-stage (pretrain then fine-tune) — deferred; the combined loss matches
  TESS_ML's prototype and avoids a second checkpoint schema.
- **No momentum queue, no Barlow-Twins.** The TESS_ML prototype uses both
  for dense-negatives and trunk regularisation; we want the slice small so
  we use in-batch negatives for SupCon and let LayerNorm/dropout handle
  trunk regularisation. Both extensions live in ``experimental/`` when we
  need them.
- **Self-referential contrastive batch.** Anchors and keys are the same
  batch's ``z``; self-pairs are masked out inside ``supcon_soft_positive``.
- **Learnable temperature** is a bare :class:`torch.nn.Parameter` clamped
  on every forward via ``torch.clamp`` rather than a custom Module — this
  keeps the checkpoint blob flat (one extra tensor, no extra state dict).

Everything else (evol-stage head, post-hoc calibration, conformal prediction,
§9.2 audit) lands in subsequent slices.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from arqueogal.xp_abundances.main.adapter import XpFeatureAdapter
from arqueogal.xp_abundances.main.config import TrainingConfig
from arqueogal.xp_abundances.main.data import (
    FeatureLayout,
    FeatureScaler,
    LabelScaler,
    LabelTiers,
    XpAbundanceDataset,
    load_arrays,
    stratified_split_ids,
)
from arqueogal.xp_abundances.main.losses import (
    ContrastiveQueue,
    barlow_twins_loss,
    beta_nll_block_cholesky,
    soft_ari_loss,
    supcon_soft_positive,
)
from arqueogal.xp_abundances.main.model import (
    CovarianceBlockLayout,
    ModelConfig,
    XpAbundanceModel,
    default_pipeline1_layout,
    five_label_block_layout,
    two_label_block_layout,
)

_LOG = logging.getLogger(__name__)

CHECKPOINT_VERSION: int = 2
"""DESIGN §Checkpoint schema: v2 = heteroscedastic + evol-stage + ensemble."""

_AMP_DTYPES: dict[str, torch.dtype | None] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "none": None,
}


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and Torch RNGs (CPU + CUDA) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_dataloaders(
    cfg: TrainingConfig,
    layout: FeatureLayout,
    tiers: LabelTiers,
    *,
    seed: int,
) -> tuple[DataLoader, DataLoader, dict[str, np.ndarray], LabelScaler]:
    """Load the training parquet, run the stratified split, wrap as DataLoaders.

    The label scaler is fit on the **train partition only** (finite values per
    label, ignoring NaN) and applied to every partition before the Datasets are
    constructed — so models train, validate, and infer in a single consistent
    standardised label space. The fit scaler is returned for the caller to
    persist in the checkpoint so inference can invert the transform.

    Returns
    -------
    ``(train_loader, val_loader, split_ids, label_scaler)`` — ``split_ids`` has
    keys ``"train" / "val" / "test"`` holding ``source_id`` arrays (test held
    back). ``label_scaler`` carries ``mean`` / ``scale`` in ``tiers.all_labels``
    order.
    """
    df = pd.read_parquet(
        cfg.train_parquet,
        columns=["source_id", *_strat_columns_available(cfg.train_parquet)],
    )
    # Dedup on source_id before splitting so repeat APOGEE visits can't leak
    # a star across train/val/test — see task #113. Keeps the first row per
    # source_id (arbitrary but deterministic on the parquet's stored order).
    df = df.drop_duplicates(subset="source_id", keep="first").reset_index(drop=True)
    split_ids = stratified_split_ids(df, fracs=cfg.fracs, seed=cfg.split_seed)
    arrs = load_arrays(cfg.train_parquet, layout, tiers, include_label_errors=True)

    # Apply the same dedup to the feature arrays: keep the first occurrence
    # of each source_id so arrs["source_id"] matches df["source_id"] post-dedup.
    _, first_idx = np.unique(arrs["source_id"], return_index=True)
    first_idx = np.sort(first_idx)
    for k in ("X", "Y", "sigma_Y", "source_id"):
        if k in arrs:
            arrs[k] = arrs[k][first_idx]

    # Drop rows with NaN in core XP features (BP + RP coefs + c0 scalars).
    # A NaN in a coefficient indicates XP reprojection failed for that star;
    # the encoder can't learn from it. Extinction priors (av_edenhofer, etc.)
    # are allowed to be NaN — those rows represent stars outside a given map's
    # coverage — and we impute to 0 so the encoder sees "no prior signal".
    n_xp = len(layout.bp_coef_cols) + len(layout.rp_coef_cols) + len(layout.xp_scalar_cols)
    xp_finite = np.isfinite(arrs["X"][:, :n_xp]).all(axis=1)
    if not xp_finite.all():
        _LOG.info(
            "dropping %d/%d rows with NaN in XP features",
            int((~xp_finite).sum()),
            len(xp_finite),
        )
        for k in ("X", "Y", "sigma_Y", "source_id"):
            if k in arrs:
                arrs[k] = arrs[k][xp_finite]

    train_mask = np.isin(arrs["source_id"], split_ids["train"])
    val_mask = np.isin(arrs["source_id"], split_ids["val"])

    # Fit the feature scaler on the train partition only (NaN-aware,
    # log10-aware on residual columns). Apply to every partition before
    # the encoder sees the input. The XP block (Hermite z-scored coefs +
    # c0 scalars) is left passthrough — it's already standardised by the
    # frozen Hermite z-score basis at parquet-build time, and downstream
    # consumers depend on that.
    xp_passthrough_cols = (
        *layout.bp_coef_cols, *layout.rp_coef_cols, *layout.xp_scalar_cols,
    )
    feature_scaler = FeatureScaler.fit(
        arrs["X"][train_mask],
        feature_names=layout.all_required_columns,
        residual_cols=layout.residual_cols,
        xp_already_scaled_cols=xp_passthrough_cols,
    )
    n_aux_scaled = int(feature_scaler.apply_mask.sum())
    _LOG.info(
        "fit feature scaler on %d train stars: %d aux/residual columns "
        "scaled, %d XP columns passthrough",
        int(train_mask.sum()),
        n_aux_scaled,
        len(layout.all_required_columns) - n_aux_scaled,
    )
    arrs["X"] = feature_scaler.transform(arrs["X"])

    # Impute remaining feature NaN (in residuals / aux priors) to 0.
    # After standardisation, x=0 corresponds to the per-feature mean —
    # the least-disruptive imputation in standardised space.
    np.nan_to_num(arrs["X"], copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Fit label scaler on the train partition only (NaN-aware per label),
    # then apply to every partition's Y so the network trains in a
    # standardised label space. Without this, the single-layer head cannot
    # learn the raw-physical-unit offsets for Teff / log g / [M/H] — see
    # task #140 and the Run A calibration findings.
    label_scaler = LabelScaler.fit(arrs["Y"][train_mask], tiers.all_labels)
    # Per-label mean/scale logging is generic over arbitrary label sets
    # (5-label, 2-label, or 21-label configurations all reach this point).
    _scaler_summary = ", ".join(
        f"{name} mean={label_scaler.mean[i]:+.3f} scale={label_scaler.scale[i]:.3f}"
        for i, name in enumerate(tiers.all_labels)
    )
    _LOG.info(
        "fit label scaler on %d train stars — %s",
        int(train_mask.sum()),
        _scaler_summary,
    )
    # ``transform`` is NaN-preserving; rescaled uncertainties scale by 1/s too
    # (only σ_Y is also divided, so its meaning relative to the standardised
    # target is preserved for downstream calibration consumers).
    # Compute inverse-frequency [M/H] weights on the train partition in raw
    # units, BEFORE the label scaler transforms Y. See #198 — v1's uniform
    # per-star NLL let [α/M] regress to the disc mean at [M/H]<-0.5.
    if cfg.inverse_freq_weighting:
        train_weights = _compute_inverse_freq_weights(
            arrs["Y"][train_mask],
            tiers=tiers,
            cfg=cfg,
        )
    else:
        train_weights = None

    arrs["Y"] = label_scaler.transform(arrs["Y"])
    if "sigma_Y" in arrs:
        arrs["sigma_Y"] = arrs["sigma_Y"] / label_scaler.scale.reshape(1, -1)

    stage_gpu = cfg.stage_dataset_on_gpu and torch.cuda.is_available()
    ds_device = "cuda" if stage_gpu else "cpu"
    train_ds = XpAbundanceDataset(
        X=arrs["X"][train_mask],
        Y=arrs["Y"][train_mask],
        sigma_Y=arrs["sigma_Y"][train_mask],
        source_id=arrs["source_id"][train_mask],
        weights=train_weights,
        device=ds_device,
    )
    val_ds = XpAbundanceDataset(
        X=arrs["X"][val_mask],
        Y=arrs["Y"][val_mask],
        sigma_Y=arrs["sigma_Y"][val_mask],
        source_id=arrs["source_id"][val_mask],
        device=ds_device,
    )

    # When tensors already live on the GPU, DataLoader workers (fork) can't
    # share CUDA tensors and pin_memory is meaningless — force num_workers=0
    # and pin_memory=False in that regime.
    loader_workers = 0 if stage_gpu else cfg.num_workers
    loader_pin = (not stage_gpu) and torch.cuda.is_available()
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=loader_workers,
        pin_memory=loader_pin,
        generator=g,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=loader_pin,
    )
    return train_loader, val_loader, split_ids, label_scaler, feature_scaler


def _strat_columns_available(parquet_path: Path | str) -> list[str]:
    """Return the subset of stratification columns actually present in the parquet."""
    import pyarrow.parquet as pq

    schema = pq.read_schema(parquet_path)
    present = set(schema.names)
    wanted = ["fe_h_apogee", "teff_apogee", "b_deg", "dec"]
    return [c for c in wanted if c in present]


def _compute_inverse_freq_weights(
    Y_train_raw: np.ndarray,  # noqa: N803 — matrix convention
    *,
    tiers: LabelTiers,
    cfg: TrainingConfig,
) -> np.ndarray:
    """Per-star inverse-frequency weight from raw [M/H] bin counts.

    Weight is ``1 / p(bin_i)`` clipped at ``cfg.inverse_freq_clip``, then
    normalised so ``mean(w) = 1`` over the train partition. NaN-[M/H] stars
    retain weight 1.0 (no adjustment); they count toward the mean-1
    normalisation denominator via the full array.

    Rank-invariant against vanilla unweighted training — when
    ``cfg.inverse_freq_weighting`` is ``False``, the caller skips this and
    leaves ``weights=None`` on the Dataset, which the loss interprets as
    uniform (identical to v1 semantics). Enabling this knob is the v1.1
    fix for the metal-poor [α/M] regression-to-mean reported at #198.
    """
    try:
        mh_idx = tiers.all_labels.index(cfg.inverse_freq_mh_column)
    except ValueError as e:
        raise ValueError(
            f"inverse_freq_mh_column={cfg.inverse_freq_mh_column!r} "
            f"not in tiers.all_labels={tiers.all_labels}",
        ) from e

    mh_raw = Y_train_raw[:, mh_idx].astype(np.float64, copy=False)
    finite = np.isfinite(mh_raw)

    edges = np.asarray(cfg.inverse_freq_bin_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 1 or not np.all(np.diff(edges) > 0):
        raise ValueError(
            f"inverse_freq_bin_edges must be a strictly-increasing 1-D tuple; "
            f"got {cfg.inverse_freq_bin_edges}",
        )

    bin_idx = np.digitize(mh_raw, edges)  # 0..len(edges); 0 = below first edge
    n_bins = edges.size + 1
    counts = np.zeros(n_bins, dtype=np.int64)
    for k in range(n_bins):
        counts[k] = int(((bin_idx == k) & finite).sum())
    n_finite = int(finite.sum())
    if n_finite == 0:
        raise RuntimeError(
            "no finite [M/H] values in train partition — cannot compute inverse-frequency weights",
        )
    probs = counts / float(n_finite)
    inv_freq = np.where(counts > 0, 1.0 / np.maximum(probs, 1e-12), 1.0)
    inv_freq = np.minimum(inv_freq, cfg.inverse_freq_clip)

    w = np.ones_like(mh_raw, dtype=np.float32)
    w[finite] = inv_freq[bin_idx[finite]].astype(np.float32)
    # Normalise to mean=1 so the overall loss scale matches unweighted v1.
    w *= float(len(w)) / float(w.sum())

    # Per-bin diagnostics for the training log.
    edge_repr = [f"(-inf, {edges[0]:.2f})"]
    edge_repr.extend(f"[{edges[i]:.2f}, {edges[i + 1]:.2f})" for i in range(len(edges) - 1))
    edge_repr.append(f"[{edges[-1]:.2f}, +inf)")
    for k in range(n_bins):
        bin_mean_w = float(w[(bin_idx == k) & finite].mean()) if counts[k] > 0 else 1.0
        _LOG.info(
            "inverse-freq bin %d %s: n=%d (p=%.4f) → w_mean=%.3f",
            k,
            edge_repr[k],
            counts[k],
            probs[k],
            bin_mean_w,
        )
    _LOG.info(
        "inverse-freq weights: n_finite=%d, n_nan=%d, clip=%.1f, w range [%.3f, %.3f], mean=%.4f",
        n_finite,
        len(w) - n_finite,
        cfg.inverse_freq_clip,
        float(w.min()),
        float(w.max()),
        float(w.mean()),
    )
    return w


def _block_layout_for(tiers: LabelTiers) -> CovarianceBlockLayout:
    """Return the block layout matching ``tiers``.

    For the 21-label production tiers returns :func:`default_pipeline1_layout`
    (4-block + diagonal-only). For the reduced :meth:`LabelTiers.five_label`
    variant returns :func:`five_label_block_layout` (single full 5×5 block).
    Other tier sets fall back to a single-block layout with all labels in one
    dense Cholesky factor — sensible default, still guaranteed block-valid.
    """
    default = default_pipeline1_layout()
    if default.label_order_human == tiers.all_labels:
        return default
    five = five_label_block_layout()
    if five.label_order_human == tiers.all_labels:
        return five
    two = two_label_block_layout()
    if two.label_order_human == tiers.all_labels:
        return two
    # Fallback: one dense block over every label in tier order.
    labels = tiers.all_labels
    return CovarianceBlockLayout(
        block_sizes=(len(labels),),
        n_diagonal_only=0,
        label_order_block=labels,
        label_order_human=labels,
        block_names=("all",),
    )


def _build_model_and_temperature(
    cfg: TrainingConfig,
    layout: FeatureLayout,
    tiers: LabelTiers,
    device: torch.device,
) -> tuple[XpAbundanceModel, nn.Parameter, XpFeatureAdapter]:
    """Construct model + learnable temperature + adapter on ``device``.

    The adapter is built from ``layout`` and the ``cfg.use_c0_scalars`` flag;
    it is a zero-parameter shape-and-mask operation that the training loop
    pipes the flat feature vector through before the encoder. Keeping it
    external to :class:`XpAbundanceModel` preserves the model's input contract
    while still letting us flip the flag between stages without reloading
    weights.

    If ``cfg.pretrained_encoder_ckpt`` is set, encoder (trunk + projection)
    weights are loaded before returning; head weights are optionally loaded
    per ``cfg.reload_head_from_pretrained``.
    """
    block_layout = _block_layout_for(tiers)
    model = XpAbundanceModel(
        ModelConfig(
            input_dim=layout.input_dim,
            block_layout=block_layout,
            latent_dim=cfg.latent_dim,
            trunk_hidden=cfg.trunk_hidden,
            head_hidden=cfg.head_hidden,
            dropout=cfg.dropout,
        ),
    ).to(device)
    log_temp = nn.Parameter(
        torch.tensor(math.log(cfg.temperature_init), device=device),
    )
    adapter = XpFeatureAdapter(layout, use_c0_scalars=cfg.use_c0_scalars).to(device)

    if cfg.pretrained_encoder_ckpt is not None:
        _load_pretrained_weights(
            model,
            cfg.pretrained_encoder_ckpt,
            reload_head=cfg.reload_head_from_pretrained,
            device=device,
        )

    return model, log_temp, adapter


def _load_pretrained_weights(
    model: XpAbundanceModel,
    ckpt_path: Path | str,
    *,
    reload_head: bool,
    device: torch.device,
) -> None:
    """Load encoder (and optionally head) weights from a v2 checkpoint."""
    blob = load_checkpoint(ckpt_path, map_location=device)
    model.encoder.load_state_dict(blob["encoder"])
    if reload_head and blob.get("regressor"):
        model.head.load_state_dict(blob["regressor"])
    _LOG.info(
        "loaded pretrained encoder from %s (reload_head=%s)",
        ckpt_path,
        reload_head,
    )


def _clamped_temperature(
    log_temp: torch.Tensor,
    bounds: tuple[float, float],
) -> torch.Tensor:
    """Exponentiate + clamp to physically sensible range — in-graph."""
    return torch.exp(log_temp).clamp(min=bounds[0], max=bounds[1])


def _compute_losses(  # noqa: PLR0913 — loss accountancy keeps all knobs explicit
    model: XpAbundanceModel,
    log_temp: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: TrainingConfig,
    adapter: XpFeatureAdapter,
    weights: torch.Tensor | None = None,
    queue: ContrastiveQueue | None = None,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor]:
    """Forward pass + weighted SupCon + β-NLL loss.

    ``y`` arrives in human (tier) order from the loader; ``mu`` and ``L`` live
    in block order. SupCon is order-invariant in its label arg (only distances
    matter for the Gaussian kernel), so we reorder ``y`` once for the NLL and
    leave SupCon on human-ordered labels.

    When ``cfg.loss_weights.supcon_label_n_first`` is set, SupCon pair weighting
    uses only the leading ``N`` label columns — ``3`` restricts to Tier-1
    atmospherics (the contrastive-pretrain regime). β-NLL always trains on
    every label.

    The ``adapter`` preprocesses ``x`` (e.g. zero out c0 scalars during
    contrastive pretrain). With ``use_c0_scalars=True`` it is an identity.

    Skip-computation optimisation: ``supcon=0`` / ``beta_nll=0`` skips the
    corresponding forward/backward path. Keeps contrastive-only pretrain from
    paying the Cholesky-head cost and supervised-only fine-tune from paying
    SupCon's B×K similarity matrix.
    """
    x_adapted = adapter(x)
    tau = _clamped_temperature(log_temp, cfg.temperature_bounds)
    lw = cfg.loss_weights

    supcon_active = lw.supcon != 0.0
    nll_active = lw.beta_nll != 0.0
    barlow_active = lw.barlow != 0.0

    # Forward: always call model (cheap relative to losses) so trunk grads flow
    # through either term — we don't gain by skipping it.
    mu, L, h, z = model(x_adapted)

    if supcon_active:
        y_for_supcon = y[:, : lw.supcon_label_n_first] if lw.supcon_label_n_first is not None else y
        if queue is not None:
            qz, qy = queue.get()
            # Label dims must agree: queue was sized for the full y; slice to
            # match supcon_label_n_first if in use.
            if lw.supcon_label_n_first is not None:
                qy = qy[:, : lw.supcon_label_n_first]
            zk = torch.cat([z, qz], dim=0)
            yk = torch.cat([y_for_supcon, qy], dim=0)
        else:
            zk, yk = z, y_for_supcon

        # Per-label kernel bandwidth in raw label units (production path),
        # falling back to the legacy isotropic scalar when no per-label
        # vector is configured or no scaler buffers exist on the model.
        sigma_arg: torch.Tensor | float
        label_scale_arg: torch.Tensor | None = None
        n_first = lw.supcon_label_n_first
        if (
            lw.supcon_sigma_raw is not None
            and hasattr(model, "label_scale_human")
        ):
            sigma_vec = torch.as_tensor(
                lw.supcon_sigma_raw,
                dtype=torch.float32,
                device=x.device,
            )
            scale_vec = model.label_scale_human.to(torch.float32)
            if n_first is not None:
                sigma_vec = sigma_vec[:n_first]
                scale_vec = scale_vec[:n_first]
            if sigma_vec.shape[0] != y_for_supcon.shape[1]:
                raise ValueError(
                    f"supcon_sigma_raw length {sigma_vec.shape[0]} does not match"
                    f" label dim {y_for_supcon.shape[1]}",
                )
            sigma_arg = sigma_vec
            label_scale_arg = scale_vec
        else:
            sigma_arg = lw.supcon_sigma

        supcon = supcon_soft_positive(
            z,
            y_for_supcon,
            zk,
            yk,
            temperature=tau,
            sigma=sigma_arg,
            label_scale=label_scale_arg,
        )
    else:
        supcon = torch.zeros((), device=x.device)

    if nll_active:
        y_block = model.block_layout.reorder_human_to_block(y)
        finite = torch.isfinite(y_block)

        # Mask out the [alpha/M] channel of the NLL for training stars whose
        # TRUTH alpha falls in the ambiguous mid-band [0.10, 0.20] dex (the
        # disc-bimodality dip). These stars pull the regression head toward
        # the conditional mean and produce a fake mid-alpha overdensity at
        # metal-poor [M/H] (the "M/H=-1, alpha/M=+0.1" hallucinated cluster).
        # By withholding their alpha-NLL gradient, the head only learns from
        # clean disc-component members. Other label channels (Teff, log g,
        # [M/H], [Mg/H]) still contribute for these stars.
        try:
            alpha_idx_human = model.block_layout.label_order_human.index(
                "alpha_m_apogee"
            )
        except ValueError:
            alpha_idx_human = None
        if alpha_idx_human is not None and hasattr(model, "label_scale_human"):
            alpha_idx_block = (
                model.block_layout
                .label_order_block.index("alpha_m_apogee")
            )
            s_alpha = model.label_scale_human[alpha_idx_human]
            m_alpha = model.label_mean_human[alpha_idx_human]
            # Truth alpha in standardised label space (block-ordered y).
            alpha_true_block = y_block[:, alpha_idx_block]
            # Convert thresholds [0.10, 0.20] dex into standardised space.
            lo = (0.10 - m_alpha) / s_alpha
            hi = (0.20 - m_alpha) / s_alpha
            ambig = (alpha_true_block >= lo) & (alpha_true_block <= hi)
            ambig &= torch.isfinite(alpha_true_block)
            # Drop the alpha channel for ambiguous stars only.
            new_mask = finite.clone()
            new_mask[:, alpha_idx_block] = (
                new_mask[:, alpha_idx_block] & ~ambig
            )
            finite = new_mask
        y_clean = torch.where(finite, y_block, mu.detach())
        nll = beta_nll_block_cholesky(
            mu,
            L,
            y_clean,
            beta=lw.beta,
            mask=finite.float(),
            sample_weights=weights,
        )
    else:
        nll = torch.zeros((), device=x.device)

    if barlow_active:
        bt = barlow_twins_loss(h, lam=lw.barlow_lam)
    else:
        bt = torch.zeros((), device=x.device)

    # Soft-ARI chemistry-cluster contamination penalty. y and mu are both in
    # LabelScaler-normalised space during training, so the physical [α/M]
    # threshold (lw.ari_alpha_threshold dex, ~0.15) and kernel sigma must be
    # converted into the same space. We do that via the per-label buffers
    # registered on the model (``label_mean_human`` / ``label_scale_human``,
    # set in ``train_model`` after fitting the scaler). When the buffers are
    # absent (e.g. unit tests / smoke checks running on a freshly constructed
    # model with no scaler attached) we fall back to interpreting the
    # threshold directly in scaled space.
    ari_active = lw.ari != 0.0
    if ari_active:
        try:
            alpha_idx_human = model.block_layout.label_order_human.index("alpha_m_apogee")
        except ValueError:
            ari_active = False
            ari = torch.zeros((), device=x.device)
        else:
            mu_human = model.block_layout.reorder_block_to_human(mu)
            alpha_pred = mu_human[:, alpha_idx_human]
            alpha_true = y[:, alpha_idx_human]
            finite_alpha = torch.isfinite(alpha_true)
            if int(finite_alpha.sum()) >= 8:
                a_pred = alpha_pred[finite_alpha]
                a_true = alpha_true[finite_alpha]
                if hasattr(model, "label_scale_human") and hasattr(model, "label_mean_human"):
                    s_alpha = model.label_scale_human[alpha_idx_human]
                    m_alpha = model.label_mean_human[alpha_idx_human]
                    threshold = (lw.ari_alpha_threshold - m_alpha) / s_alpha
                    kernel = lw.ari_kernel_sigma / s_alpha
                else:
                    threshold = lw.ari_alpha_threshold
                    kernel = lw.ari_kernel_sigma
                p_high_pred = torch.sigmoid((a_pred - threshold) / kernel)
                p_high_true = torch.sigmoid((a_true - threshold) / kernel)
                pred_K2 = torch.stack([1.0 - p_high_pred, p_high_pred], dim=1)
                true_K2 = torch.stack([1.0 - p_high_true, p_high_true], dim=1)
                ari = soft_ari_loss(pred_K2, true_K2)
            else:
                ari = torch.zeros((), device=x.device)
    else:
        ari = torch.zeros((), device=x.device)

    total = (
        lw.supcon * supcon
        + lw.beta_nll * nll
        + lw.barlow * bt
        + lw.ari * ari
    )
    parts = {
        "loss": float(total.detach()),
        "supcon": float(supcon.detach()),
        "nll": float(nll.detach()),
        "barlow": float(bt.detach()),
        "ari": float(ari.detach()),
        "tau": float(tau.detach()),
    }
    return total, parts, z.detach(), y.detach()


def train_one_epoch(  # noqa: PLR0913 — one-epoch dispatch has many collaborators
    model: XpAbundanceModel,
    log_temp: nn.Parameter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg: TrainingConfig,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    adapter: XpFeatureAdapter | None = None,
    queue: ContrastiveQueue | None = None,
) -> dict[str, float]:
    """One training epoch; returns mean per-batch metrics."""
    if adapter is None:
        raise ValueError("adapter is required; build via _build_model_and_temperature")
    model.train()
    amp_dtype = _AMP_DTYPES[cfg.amp_dtype]
    use_amp = amp_dtype is not None and device.type == "cuda"

    sums: dict[str, float] = {
        "loss": 0.0,
        "supcon": 0.0,
        "nll": 0.0,
        "barlow": 0.0,
        "ari": 0.0,
        "tau": 0.0,
    }
    n = 0
    grad_norm_max = 0.0
    grad_norm_sum = 0.0

    for batch in loader:
        x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        w = (
            batch[2].to(device, non_blocking=True)
            if cfg.inverse_freq_weighting and len(batch) > 2
            else None
        )
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            total, parts, z_det, y_det = _compute_losses(
                model,
                log_temp,
                x,
                y,
                cfg,
                adapter,
                weights=w,
                queue=queue,
            )

        if scaler is not None and scaler.is_enabled():
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            g = torch.nn.utils.clip_grad_norm_(
                [*model.parameters(), log_temp],
                cfg.grad_clip_norm,
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            g = torch.nn.utils.clip_grad_norm_(
                [*model.parameters(), log_temp],
                cfg.grad_clip_norm,
            )
            optimizer.step()
        scheduler.step()

        if queue is not None:
            queue.enqueue(z_det, y_det)

        g_val = float(g) if torch.isfinite(g) else float("inf")
        grad_norm_max = max(grad_norm_max, g_val)
        grad_norm_sum += g_val

        if g_val > cfg.grad_norm_abort_threshold:
            raise RuntimeError(
                f"grad_norm={g_val:.2f} exceeded abort threshold "
                f"{cfg.grad_norm_abort_threshold:.2f} "
                f"(batch {n}; loss={float(total):.4f}, "
                f"nll={parts.get('nll', float('nan')):.4f}, "
                f"supcon={parts.get('supcon', float('nan')):.4f}). "
                f"β=0 canary: pure Gaussian NLL likely exploded on a high-σ "
                f"sample; inspect the offending batch before lowering β further.",
            )

        for k, v in parts.items():
            sums[k] += v
        n += 1

    out = {k: v / max(n, 1) for k, v in sums.items()}
    out["grad_norm_max"] = grad_norm_max
    out["grad_norm_mean"] = grad_norm_sum / max(n, 1)
    return out


def validate(
    model: XpAbundanceModel,
    log_temp: nn.Parameter,
    loader: DataLoader,
    cfg: TrainingConfig,
    device: torch.device,
    adapter: XpFeatureAdapter | None = None,
) -> dict[str, float]:
    """One validation pass; same metrics as training, no gradients."""
    if adapter is None:
        raise ValueError("adapter is required; build via _build_model_and_temperature")
    model.eval()
    sums: dict[str, float] = {
        "loss": 0.0,
        "supcon": 0.0,
        "nll": 0.0,
        "barlow": 0.0,
        "ari": 0.0,
        "tau": 0.0,
    }
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1].to(device, non_blocking=True)
            _, parts, _, _ = _compute_losses(model, log_temp, x, y, cfg, adapter)
            for k, v in parts.items():
                sums[k] += v
            n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}


def train_model(  # noqa: PLR0913 — explicit collaborators beat a mega-config object
    cfg: TrainingConfig,
    layout: FeatureLayout,
    tiers: LabelTiers,
    *,
    seed: int,
    train_loader: DataLoader | None = None,
    val_loader: DataLoader | None = None,
    label_scaler: LabelScaler | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train a single-seed model end-to-end; return a result dict with history.

    If ``train_loader`` / ``val_loader`` are omitted, they are built from
    ``cfg.train_parquet`` and a fitted :class:`LabelScaler` is produced as a
    side effect. When loaders *are* supplied (unit tests), ``label_scaler``
    must be supplied too — it is required to save checkpoints and to invert
    the standardisation at inference.
    """
    seed_everything(seed, deterministic=cfg.deterministic)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_scaler: FeatureScaler | None = None
    if train_loader is None or val_loader is None:
        (
            train_loader, val_loader, _split_ids,
            label_scaler, feature_scaler,
        ) = build_dataloaders(cfg, layout, tiers, seed=seed)
    elif label_scaler is None:
        raise ValueError(
            "label_scaler is required when train_loader/val_loader are supplied"
            " (the scaler is a side-effect of build_dataloaders; pass whichever"
            " scaler was fit on the matching train partition).",
        )

    model, log_temp, adapter = _build_model_and_temperature(cfg, layout, tiers, device)
    # Expose the per-label LabelScaler stats to losses that need physical-unit
    # thresholds (e.g. soft-ARI on the [α/M] disc-thick boundary). Buffers
    # follow ``tiers.all_labels`` order, which equals
    # ``model.block_layout.label_order_human``.
    model.register_buffer(
        "label_mean_human",
        torch.tensor(label_scaler.mean, dtype=torch.float32, device=device),
        persistent=False,
    )
    model.register_buffer(
        "label_scale_human",
        torch.tensor(label_scaler.scale, dtype=torch.float32, device=device),
        persistent=False,
    )
    optimizer = _build_optimizer(model, log_temp, cfg)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[g["lr"] for g in optimizer.param_groups],
        steps_per_epoch=max(len(train_loader), 1),
        epochs=cfg.epochs,
        pct_start=cfg.pct_start,
    )
    use_fp16 = cfg.amp_dtype == "float16" and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda") if use_fp16 else None

    queue: ContrastiveQueue | None = None
    if cfg.queue_size > 0:
        queue = ContrastiveQueue(
            latent_dim=cfg.latent_dim,
            n_labels=tiers.n_labels,
            size=cfg.queue_size,
            device=device,
            warm_start=cfg.queue_warm_start,
        )
        _LOG.info(
            "momentum queue enabled: size=%d, D=%d, n_labels=%d, warm_start=%s",
            cfg.queue_size,
            cfg.latent_dim,
            tiers.n_labels,
            cfg.queue_warm_start,
        )

    history: list[dict[str, float]] = []
    best_vl = math.inf
    best_epoch = -1
    patience = 0
    best_model_state: dict[str, torch.Tensor] = {}
    best_temp: torch.Tensor = log_temp.detach().clone()
    cadence_paths: list[Path] = []

    for epoch in range(cfg.epochs):
        tr = train_one_epoch(
            model,
            log_temp,
            train_loader,
            optimizer,
            scheduler,
            cfg,
            device,
            scaler,
            adapter=adapter,
            queue=queue,
        )
        va = validate(model, log_temp, val_loader, cfg, device, adapter=adapter)
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}": v for k, v in va.items()},
        }
        history.append(row)
        _LOG.info(
            "epoch %d: train_loss=%.4f val_loss=%.4f tau=%.3f grad_max=%.2f grad_mean=%.2f",
            epoch,
            tr["loss"],
            va["loss"],
            va["tau"],
            tr.get("grad_norm_max", float("nan")),
            tr.get("grad_norm_mean", float("nan")),
        )

        if (
            epoch == 0
            and cfg.loss_weights.beta_nll != 0.0
            and math.isfinite(cfg.first_epoch_sanity_k)
        ):
            _first_epoch_sanity_check(
                model,
                val_loader,
                adapter,
                label_scaler,
                tiers=tiers,
                device=device,
                k=cfg.first_epoch_sanity_k,
            )

        min_delta = (
            cfg.early_stop_min_delta * abs(best_vl)
            if cfg.relative_min_delta and math.isfinite(best_vl)
            else cfg.early_stop_min_delta
        )
        if va["loss"] < best_vl - min_delta:
            best_vl, best_epoch, patience = va["loss"], epoch, 0
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_temp = log_temp.detach().cpu().clone()
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                _LOG.info("early stopping at epoch %d (best %d)", epoch, best_epoch)
                break

        if cfg.checkpoint_every_n_epochs > 0 and ((epoch + 1) % cfg.checkpoint_every_n_epochs == 0):
            cadence_paths.append(
                _save_cadence_checkpoint(
                    cfg,
                    model,
                    log_temp,
                    layout,
                    tiers,
                    label_scaler,
                    seed=seed,
                    epoch=epoch,
                    history=history,
                ),
            )

    if best_model_state:
        model.load_state_dict(best_model_state)
        with torch.no_grad():
            log_temp.copy_(best_temp.to(device))

    return {
        "model": model,
        "log_temp": log_temp,
        "adapter": adapter,
        "label_scaler": label_scaler,
        "feature_scaler": feature_scaler,
        "history": history,
        "best_val_loss": best_vl,
        "best_epoch": best_epoch,
        "seed": seed,
        "device": str(device),
        "cadence_checkpoints": cadence_paths,
    }


def _first_epoch_sanity_check(  # noqa: PLR0913 — all collaborators are load-bearing
    model: XpAbundanceModel,
    val_loader: DataLoader,
    adapter: XpFeatureAdapter,
    label_scaler: LabelScaler,
    *,
    tiers: LabelTiers,
    device: torch.device,
    k: float = 2.0,
) -> None:
    """Un-scale epoch-0 predictions and truth; halt on gross per-label bias.

    The Run A fine-tune shipped with a placeholder label scaler, leaving the
    single-layer head unable to learn Tier-1 offsets (Teff ~4600 K, log g
    ~2.4 dex). The symptom only surfaced at #135 calibration. This check
    catches the same class of failure after one epoch by comparing each
    label's mean prediction against its mean truth in raw units: a deviation
    exceeding ``2 × std(truth)`` means the head is not tracking the bulk of
    the label distribution and further training is wasted.

    Missing-label rows are handled per column — NaN truth is ignored so a
    mostly-Tier-3 column (lots of NaN) doesn't falsely trigger the halt.
    """
    model.eval()
    mu_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device, non_blocking=True)
            y = batch[1]  # human-order, scaled, CPU-movable
            mu_block, _L, _h, _z = model(adapter(x))
            mu_human = model.block_layout.reorder_block_to_human(mu_block)
            mu_chunks.append(mu_human.detach().cpu().float().numpy())
            y_chunks.append(y.detach().cpu().float().numpy())
    mu_scaled = np.concatenate(mu_chunks, axis=0)
    y_scaled = np.concatenate(y_chunks, axis=0)

    mu_raw = label_scaler.inverse_mean(mu_scaled)
    y_raw = label_scaler.inverse_mean(y_scaled)

    offenders: list[str] = []
    for j, name in enumerate(tiers.all_labels):
        col_y = y_raw[:, j]
        mask = np.isfinite(col_y)
        if mask.sum() < 2:  # noqa: PLR2004 — one-sample std is meaningless
            continue
        y_mean = float(col_y[mask].mean())
        y_std = float(col_y[mask].std(ddof=0))
        mu_mean = float(mu_raw[mask, j].mean())
        dev = abs(mu_mean - y_mean)
        _LOG.info(
            "first-epoch sanity %s: mean_pred=%.4g mean_truth=%.4g std_truth=%.4g dev=%.4g",
            name,
            mu_mean,
            y_mean,
            y_std,
            dev,
        )
        if y_std > 0 and dev > k * y_std:
            offenders.append(
                f"{name}: |{mu_mean:.4g} - {y_mean:.4g}| = {dev:.4g} > {k:g}×{y_std:.4g}",
            )
    if offenders:
        raise RuntimeError(
            f"first-epoch sanity check failed — mean prediction > {k:g}·std(truth) "
            "from mean truth on:\n  "
            + "\n  ".join(offenders)
            + "\nCheck the label scaler was fit on the train partition and "
            "that save_checkpoint stores it (not the zeros/ones placeholder).",
        )
    model.train()


def _build_optimizer(
    model: XpAbundanceModel,
    log_temp: nn.Parameter,
    cfg: TrainingConfig,
) -> torch.optim.Optimizer:
    """Single or two-group AdamW depending on ``cfg.encoder_lr_ratio``.

    Ratio 1.0 gives the historical single-group AdamW. Anything else splits
    encoder (trunk + projection) from head + log_temp so fine-tuning can
    train the head at full LR while nudging the pretrained encoder at a
    smaller rate.
    """
    head_params = [*model.head.parameters(), log_temp]
    encoder_params = list(model.encoder.parameters())
    # Fused AdamW fuses per-parameter updates into one CUDA kernel — for a
    # ~100k-param model on a 6 GB 3060 this drops the optimizer step from
    # ~8 ms to <0.5 ms (profiled 2026-04-22). Only supported on CUDA.
    fused = torch.cuda.is_available()
    if cfg.encoder_lr_ratio == 1.0:
        return torch.optim.AdamW(
            [*encoder_params, *head_params],
            lr=cfg.max_lr,
            weight_decay=cfg.weight_decay,
            fused=fused,
        )
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": cfg.max_lr * cfg.encoder_lr_ratio, "name": "encoder"},
            {"params": head_params, "lr": cfg.max_lr, "name": "head"},
        ],
        weight_decay=cfg.weight_decay,
        fused=fused,
    )


def _save_cadence_checkpoint(  # noqa: PLR0913 — each field is an independent save dependency
    cfg: TrainingConfig,
    model: XpAbundanceModel,
    log_temp: torch.Tensor,
    layout: FeatureLayout,
    tiers: LabelTiers,
    label_scaler: LabelScaler,
    *,
    seed: int,
    epoch: int,
    history: list[dict[str, float]],
) -> Path:
    """Write a cadence checkpoint named with the epoch number.

    Cadence checkpoints are rollback points, not the best-val checkpoint the
    run ultimately returns. Caller decides what to keep after training.
    """
    fname = f"{cfg.output_prefix}_seed{seed}_epoch{epoch:04d}.pt"
    path = cfg.output_dir / "cadence" / fname
    return save_checkpoint(
        path,
        model=model,
        log_temp=log_temp,
        cfg=cfg,
        layout=layout,
        tiers=tiers,
        label_scaler=label_scaler,
        seed=seed,
        training_metrics={"epoch": epoch, "history": history},
    )


def save_checkpoint(  # noqa: PLR0913 — each field is an independent reload dependency
    path: Path | str,
    *,
    model: XpAbundanceModel,
    log_temp: torch.Tensor,
    cfg: TrainingConfig,
    layout: FeatureLayout,
    tiers: LabelTiers,
    label_scaler: LabelScaler,
    seed: int,
    training_metrics: dict[str, Any] | None = None,
    git_sha: str = "",
    feature_scaler: FeatureScaler | None = None,
) -> Path:
    """Persist a v2 checkpoint matching DESIGN §Checkpoint schema.

    ``label_scaler`` is required: the regressor head was trained on
    standardised labels, so inference cannot recover raw-unit predictions
    without the fitted mean/scale. The scaler's ``label_names`` must equal
    ``tiers.all_labels`` — we enforce it here so a reorder bug cannot write
    a silently-wrong checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tier_map = {lab: 1 for lab in tiers.tier1}
    tier_map.update({lab: 2 for lab in tiers.tier2})
    tier_map.update({lab: 3 for lab in tiers.tier3})

    if tuple(label_scaler.label_names) != tuple(tiers.all_labels):
        raise ValueError(
            "label_scaler.label_names does not match tiers.all_labels — "
            f"{label_scaler.label_names} vs {tiers.all_labels}. Refusing to "
            "save a checkpoint with a misaligned scaler.",
        )
    if label_scaler.is_default():
        raise ValueError(
            "label_scaler is the zeros/ones default — fit it on the train "
            "partition before saving (see build_dataloaders).",
        )

    blob: dict[str, Any] = {
        "version": CHECKPOINT_VERSION,
        "input_dim": layout.input_dim,
        "latent_dim": cfg.latent_dim,
        "n_labels": tiers.n_labels,
        "label_names": list(tiers.all_labels),
        "tier_map": tier_map,
        "encoder": model.encoder.state_dict(),
        "regressor": model.head.state_dict(),
        "evol_stage_head": {},
        "label_scaler_mean": label_scaler.mean.astype(np.float32, copy=True),
        "label_scaler_scale": label_scaler.scale.astype(np.float32, copy=True),
        "calibration": {
            "temperature_per_cell": {},
            "isotonic_per_label": {},
            "conformal_scores": np.zeros(0, dtype=np.float32),
        },
        "config_yaml": json.dumps(_config_to_jsonable(cfg)),
        "random_seed": seed,
        "git_sha": git_sha,
        "training_metrics": training_metrics or {},
        "log_temperature": log_temp.detach().cpu(),
        "block_layout": model.block_layout.to_dict(),
    }
    if feature_scaler is not None:
        # Persist the feature scaler so inference can apply the SAME
        # standardisation the encoder was trained on (NaN-aware z-score on
        # aux + log10 + z-score on residual RMS).
        blob["feature_scaler"] = {
            "mean": feature_scaler.mean.astype(np.float32, copy=True),
            "scale": feature_scaler.scale.astype(np.float32, copy=True),
            "feature_names": list(feature_scaler.feature_names),
            "log10_mask": feature_scaler.log10_mask.astype(bool, copy=True),
            "apply_mask": feature_scaler.apply_mask.astype(bool, copy=True),
        }
    torch.save(blob, path)
    return path


def load_checkpoint(
    path: Path | str,
    map_location: torch.device | str | None = None,
) -> dict[str, Any]:
    """Load a v2 checkpoint; caller is responsible for constructing modules."""
    blob = cast(dict[str, Any], torch.load(path, map_location=map_location, weights_only=False))
    if blob.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"checkpoint {path}: version {blob.get('version')} != {CHECKPOINT_VERSION}"
        )
    return blob


def _config_to_jsonable(cfg: TrainingConfig) -> dict[str, Any]:
    """Flatten a :class:`TrainingConfig` into JSON-safe primitives."""
    d = asdict(cfg)
    # Path instances aren't JSON-serialisable by default.
    for k, v in list(d.items()):
        if isinstance(v, Path):
            d[k] = str(v)
    return d


def train_ensemble(
    cfg: TrainingConfig,
    layout: FeatureLayout,
    tiers: LabelTiers,
    *,
    date_tag: str,
    git_sha: str = "",
) -> list[Path]:
    """Sequentially train ``len(cfg.ensemble_seeds)`` models and persist each.

    Filenames follow DESIGN convention:
    ``<cfg.output_prefix>_<date_tag>_<git-sha7>_seed<N>.pt``. Stage drivers
    override ``output_prefix`` to ``"xp_abundances_main_contrastive"`` /
    ``"xp_abundances_main_finetune"`` etc. so stages don't clobber each other.
    """
    sha7 = (git_sha[:7] or "nogit").ljust(7, "0")
    out_paths: list[Path] = []
    for seed in cfg.ensemble_seeds:
        _LOG.info("training ensemble member seed=%d", seed)
        result = train_model(cfg, layout, tiers, seed=seed)
        fname = f"{cfg.output_prefix}_{date_tag}_{sha7}_seed{seed}.pt"
        path = save_checkpoint(
            cfg.output_dir / fname,
            model=result["model"],
            log_temp=result["log_temp"],
            cfg=cfg,
            layout=layout,
            tiers=tiers,
            label_scaler=result["label_scaler"],
            seed=seed,
            training_metrics={
                "best_val_loss": result["best_val_loss"],
                "best_epoch": result["best_epoch"],
                "history": result["history"],
            },
            git_sha=git_sha,
        )
        out_paths.append(path)
    return out_paths


__all__ = [
    "CHECKPOINT_VERSION",
    "build_dataloaders",
    "load_checkpoint",
    "save_checkpoint",
    "seed_everything",
    "train_ensemble",
    "train_model",
    "train_one_epoch",
    "validate",
]
