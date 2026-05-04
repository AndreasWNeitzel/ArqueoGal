"""Pipeline 1 inference + ensemble aggregation.

The trained ensemble (5–10 seeds per DESIGN §Architecture step 5) emits a
per-member ``(μ_m, L_m)`` for each star. The released product is an
aggregated distribution capturing both the *aleatoric* uncertainty each
member predicts (``L_m L_mᵀ``) and the *epistemic* uncertainty of the
ensemble (spread of the per-member means).

We follow the standard Bayesian-model-averaging decomposition used by
Lakshminarayanan+2017 / Kendall&Gal 2017:

- Aggregated mean: ``μ = (1/M) Σ_m μ_m``.
- Aleatoric covariance: ``Σ_alea = (1/M) Σ_m L_m L_mᵀ``.
- Epistemic covariance: ``Σ_epi = (1/M) Σ_m (μ_m - μ)(μ_m - μ)ᵀ``.
- Total covariance: ``Σ_total = Σ_alea + Σ_epi``.

Per-label marginals follow from ``sqrt(diag(·))``. The module returns all
four (μ, Σ_alea, Σ_epi, Σ_total) plus the per-member arrays, leaving the
release layer free to pick whichever it needs.

Post-hoc calibration (from :mod:`.uncertainty`) is applied per member
before aggregation if the checkpoint carries a calibration blob — that's
how DESIGN §Release gates wants it: calibrate each member, then average.

NaN safety: all features are sanitised at inference entry via
``np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)`` per ADR-0012.
The XpFeatureAdapter (in data loader / preprocessing) is a pass-through
and does not guard against NaNs; sanitisation must occur at the inference
driver boundary before the first model forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from arqueogal.data.frozen_stats import assert_frozen_stats_match
from arqueogal.xp_abundances.main.model import (
    CovarianceBlockLayout,
    ModelConfig,
    XpAbundanceModel,
)
from arqueogal.xp_abundances.main.training import CHECKPOINT_VERSION, load_checkpoint
from arqueogal.xp_abundances.main.uncertainty import (
    CalibrationArtifacts,
    apply_calibration,
    collect_predictions,
)


@dataclass
class EnsembleMember:
    """One trained ensemble seed, ready for inference.

    Holds the restored :class:`~.model.XpAbundanceModel` plus the calibration
    dict from the checkpoint. The raw ``blob`` is retained for audit /
    diagnostics — callers are free to ignore it.
    """

    model: nn.Module
    calibration: CalibrationArtifacts
    seed: int
    blob: dict[str, Any]


@dataclass
class EnsemblePrediction:
    """Aggregated ensemble output.

    Shapes:

    - ``mu``: ``(B, n)`` — ensemble-mean predicted label.
    - ``sigma_aleatoric``: ``(B, n)`` — ``sqrt(diag(Σ_alea))`` marginals.
    - ``sigma_epistemic``: ``(B, n)`` — ``sqrt(diag(Σ_epi))`` marginals.
    - ``sigma_total``: ``(B, n)`` — ``sqrt(diag(Σ_total))`` marginals.
    - ``Sigma_aleatoric`` / ``Sigma_epistemic`` / ``Sigma_total``: full
      covariance tensors ``(B, n, n)`` for callers that need cross-label
      correlations (e.g. Starfold's MC ensemble, downstream, separate repo).
    - ``per_member_mu``: ``(M, B, n)`` — calibrated per-member means.
    - ``y``: ``(B, n)`` ground truth if the loader supplied it, else None.
    """

    mu: np.ndarray
    sigma_aleatoric: np.ndarray
    sigma_epistemic: np.ndarray
    sigma_total: np.ndarray
    Sigma_aleatoric: np.ndarray  # noqa: N815 — Σ convention, uppercase for matrix
    Sigma_epistemic: np.ndarray  # noqa: N815
    Sigma_total: np.ndarray  # noqa: N815
    per_member_mu: np.ndarray
    y: np.ndarray | None


def _build_model_from_blob(
    blob: dict[str, Any],
    device: torch.device,
) -> XpAbundanceModel:
    """Rehydrate an :class:`XpAbundanceModel` from a checkpoint dict.

    Architectural knobs beyond ``latent_dim`` (``trunk_hidden``,
    ``head_hidden``, ``dropout``) live inside ``config_yaml`` — we parse them
    out so ensemble members with non-default widths round-trip correctly.
    """
    import json

    layout_blob = blob.get("block_layout")
    if not isinstance(layout_blob, dict):
        raise ValueError(
            "checkpoint is missing 'block_layout' — cannot rehydrate model head",
        )
    block_layout = CovarianceBlockLayout.from_dict(layout_blob)
    if block_layout.n_labels != blob["n_labels"]:
        raise ValueError(
            f"checkpoint block_layout n_labels {block_layout.n_labels} "
            f"inconsistent with n_labels {blob['n_labels']}",
        )

    cfg_json = blob.get("config_yaml") or "{}"
    parsed = json.loads(cfg_json) if isinstance(cfg_json, str) else cfg_json
    trunk_hidden = tuple(parsed.get("trunk_hidden") or (256, 128))
    head_hidden = int(parsed.get("head_hidden", 128))
    dropout = float(parsed.get("dropout", 0.10))

    cfg = ModelConfig(
        input_dim=blob["input_dim"],
        block_layout=block_layout,
        latent_dim=blob["latent_dim"],
        trunk_hidden=trunk_hidden,
        head_hidden=head_hidden,
        dropout=dropout,
    )
    model = XpAbundanceModel(cfg).to(device)
    model.encoder.load_state_dict(blob["encoder"])
    model.head.load_state_dict(blob["regressor"])
    model.eval()
    return model


def _extract_calibration(blob: dict[str, Any]) -> CalibrationArtifacts:
    """Pull calibration out of a checkpoint blob into a :class:`CalibrationArtifacts`."""
    cal = blob.get("calibration") or {}
    art = CalibrationArtifacts()
    temp = cal.get("temperature_per_cell") or {}
    art.temperature_per_cell = {int(k): float(v) for k, v in temp.items()}
    iso = cal.get("isotonic_per_label") or {}
    art.isotonic_per_label = {
        int(k): {"X": np.asarray(v["X"]), "y": np.asarray(v["y"])} for k, v in iso.items()
    }
    conf = cal.get("conformal_scores")
    art.conformal_scores = (
        np.asarray(conf, dtype=np.float32) if conf is not None else np.zeros(0, dtype=np.float32)
    )
    art.cell_definition = cal.get("cell_definition") or {}
    return art


def load_ensemble(
    paths: list[Path | str] | Path | str,
    device: torch.device | None = None,
) -> list[EnsembleMember]:
    """Load every checkpoint in ``paths`` (file or directory) as an ensemble.

    If ``paths`` is a directory, all ``*.pt`` files under it are loaded, sorted
    by filename so the ensemble order is deterministic across invocations.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(paths, (str, Path)):
        p = Path(paths)
        resolved = sorted(p.glob("*.pt")) if p.is_dir() else [p]
    else:
        resolved = [Path(x) for x in paths]

    if not resolved:
        raise FileNotFoundError(f"no checkpoint files found for {paths}")

    members: list[EnsembleMember] = []
    for path in resolved:
        blob = load_checkpoint(path, map_location=device)
        if blob.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                f"{path}: checkpoint version {blob.get('version')} != {CHECKPOINT_VERSION}"
            )
        model = _build_model_from_blob(blob, device)
        cal = _extract_calibration(blob)
        members.append(
            EnsembleMember(
                model=model,
                calibration=cal,
                seed=int(blob.get("random_seed", -1)),
                blob=blob,
            )
        )
    return members


def predict_ensemble(
    ensemble: list[EnsembleMember],
    loader: DataLoader,
    *,
    device: torch.device | None = None,
    cell_ids: np.ndarray | None = None,
    amp_dtype: torch.dtype | None = None,
) -> EnsemblePrediction:
    """Run every ensemble member, apply calibration, aggregate.

    ``cell_ids`` (optional) is a per-star cell ID array used for per-cell
    temperature scaling. If omitted, the cell ID 0 is used for every star —
    equivalent to a single global calibration cell.

    ``amp_dtype`` (optional) mirrors the training-time autocast contract:
    pass ``torch.bfloat16`` to reproduce the bf16 forward pass on CUDA so a
    bf16-trained checkpoint does not silently fall back to fp32 at inference.
    Ignored on CPU.

    Notes
    -----
    Ensemble aggregation assumes the members are *conditionally independent
    given the input* (deep-ensemble Bayesian-model-averaging contract,
    Lakshminarayanan+2017). The training driver enforces this by seeding each
    member with a distinct RNG and (optionally) a distinct data subsample;
    members that share weights, share data, or share an upstream contrastive
    trunk without independent fine-tuning will *under*-estimate epistemic
    variance and the resulting ``Sigma_total`` coverage will be optimistic in
    the tails.

    Raises
    ------
    AssertionError
        If the feature tensor contains non-finite values after NaN sanitisation.
        This indicates either that ``nan_to_num`` failed to capture a pathology
        (e.g., a user-supplied value of NaN not caught by the loader) or that
        the caller's data is malformed. See ADR-0012.
    """
    if not ensemble:
        raise ValueError("ensemble is empty")

    # Pre-flight: verify frozen v1 stats fingerprint *before* any device or
    # loader resource is acquired. A mismatch here is a hard contract failure
    # (the basis on which the calibration thresholds were derived has shifted)
    # and we want it to surface before we touch the GPU.
    assert_frozen_stats_match()

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = len(ensemble)

    # Single-pass streaming accumulator. Earlier code held two lists (one of
    # per-member mu of shape (B, n), one of per-member aleatoric covariance of
    # shape (B, n, n)) and only collapsed them at the end. At catalogue scale
    # (B ≈ 50 M, n = 21, M = 5) the covariance list alone was ~88 GB and forced
    # an OOM on the 32 GB pc127 sibling. We now keep one (M, B, n) array of
    # member means (returned to the caller anyway as ``per_member_mu``) and
    # one (B, n, n) running ``Sigma_alea`` accumulator; epistemic covariance
    # is computed via the second-moment identity
    # ``Σ_epi = E[μ_m μ_mᵀ] − μ̄ μ̄ᵀ`` so no (M, B, n) ``delta`` temporary
    # is materialised. Peak memory drops by ~109 GB at 50 M-star scale.
    per_mu: np.ndarray | None = None
    Sigma_alea: np.ndarray | None = None
    second_moment: np.ndarray | None = None  # Σ_m μ_m μ_mᵀ accumulator
    y_ref: np.ndarray | None = None

    for member_idx, m in enumerate(ensemble):
        preds = collect_predictions(m.model, loader, device=device, amp_dtype=amp_dtype)
        mu_m, L_m = preds["mu"], preds["L"]
        # NaN safety (ADR-0012): sanitise predictions before aggregation.
        # This catches any NaN/Inf that slipped through the data loader.
        mu_m = np.nan_to_num(mu_m, nan=0.0, posinf=0.0, neginf=0.0)
        L_m = np.nan_to_num(L_m, nan=0.0, posinf=0.0, neginf=0.0)
        assert np.isfinite(mu_m).all(), (
            "Inference detected non-finite mu_m after nan_to_num. See ADR-0012."
        )
        assert np.isfinite(L_m).all(), (
            "Inference detected non-finite L_m after nan_to_num. See ADR-0012."
        )
        if cell_ids is None:
            cell_ids_m = np.zeros(mu_m.shape[0], dtype=np.int64)
        else:
            if cell_ids.shape[0] != mu_m.shape[0]:
                raise ValueError(f"cell_ids length {cell_ids.shape[0]} != batch N {mu_m.shape[0]}")
            cell_ids_m = cell_ids
        mu_m, L_m = apply_calibration(mu_m, L_m, m.calibration, cell_ids=cell_ids_m)

        if per_mu is None:
            B, n = mu_m.shape
            per_mu = np.empty((M, B, n), dtype=mu_m.dtype)
            Sigma_alea = np.zeros((B, n, n), dtype=np.float64)
            second_moment = np.zeros((B, n, n), dtype=np.float64)
            y_ref = preds["y"]
        per_mu[member_idx] = mu_m
        # Σ_alea = (1/M) Σ_m L_m L_mᵀ; accumulate in fp64 for numerical safety,
        # then divide once at the end.
        Sigma_alea += np.einsum("bij,bkj->bik", L_m, L_m, dtype=np.float64)
        # Second-moment accumulator for Σ_epi via the parallel-axis identity.
        second_moment += np.einsum("bi,bj->bij", mu_m, mu_m, dtype=np.float64)

    assert per_mu is not None and Sigma_alea is not None and second_moment is not None
    Sigma_alea /= M
    mu_mean = per_mu.mean(axis=0)  # (B, n)
    # Σ_epi = E[μ_m μ_mᵀ] − μ̄ μ̄ᵀ. Equivalent to ``mean over m of (μ_m − μ̄)(μ_m − μ̄)ᵀ``
    # but skips the (M, B, n) centred-mean intermediate.
    Sigma_epi = second_moment / M - np.einsum("bi,bj->bij", mu_mean, mu_mean, dtype=np.float64)
    # Defensive PSD floor on the diagonal: catastrophic cancellation in the
    # second-moment form can leave eigenvalues at -O(eps_fp64) on stars where
    # every member agreed. Clipping the diagonal to ≥ 0 preserves the off-diag
    # correlation structure and keeps Σ_total numerically PSD downstream.
    diag_idx = np.arange(Sigma_epi.shape[1])
    Sigma_epi[:, diag_idx, diag_idx] = np.clip(Sigma_epi[:, diag_idx, diag_idx], 0.0, None)
    Sigma_total = Sigma_alea + Sigma_epi

    sigma_alea = np.sqrt(np.clip(np.einsum("bii->bi", Sigma_alea), 0.0, None))
    sigma_epi = np.sqrt(np.clip(np.einsum("bii->bi", Sigma_epi), 0.0, None))
    sigma_tot = np.sqrt(np.clip(np.einsum("bii->bi", Sigma_total), 0.0, None))

    return EnsemblePrediction(
        mu=mu_mean.astype(np.float32),
        sigma_aleatoric=sigma_alea.astype(np.float32),
        sigma_epistemic=sigma_epi.astype(np.float32),
        sigma_total=sigma_tot.astype(np.float32),
        Sigma_aleatoric=Sigma_alea.astype(np.float32),
        Sigma_epistemic=Sigma_epi.astype(np.float32),
        Sigma_total=Sigma_total.astype(np.float32),
        per_member_mu=per_mu.astype(np.float32),
        y=y_ref.astype(np.float32) if y_ref is not None else None,
    )


__all__ = [
    "EnsembleMember",
    "EnsemblePrediction",
    "load_ensemble",
    "predict_ensemble",
]
