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
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

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
      correlations (e.g. Pipeline 2's MC ensemble).
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
    blob: dict[str, Any], device: torch.device,
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
        int(k): {"X": np.asarray(v["X"]), "y": np.asarray(v["y"])}
        for k, v in iso.items()
    }
    conf = cal.get("conformal_scores")
    art.conformal_scores = (
        np.asarray(conf, dtype=np.float32) if conf is not None
        else np.zeros(0, dtype=np.float32)
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
        members.append(EnsembleMember(
            model=model, calibration=cal,
            seed=int(blob.get("random_seed", -1)), blob=blob,
        ))
    return members


def predict_ensemble(
    ensemble: list[EnsembleMember],
    loader: DataLoader,
    *,
    device: torch.device | None = None,
    cell_ids: np.ndarray | None = None,
) -> EnsemblePrediction:
    """Run every ensemble member, apply calibration, aggregate.

    ``cell_ids`` (optional) is a per-star cell ID array used for per-cell
    temperature scaling. If omitted, the cell ID 0 is used for every star —
    equivalent to a single global calibration cell.
    """
    if not ensemble:
        raise ValueError("ensemble is empty")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mus: list[np.ndarray] = []
    Sigmas_alea: list[np.ndarray] = []
    y_ref: np.ndarray | None = None

    for m in ensemble:
        preds = collect_predictions(m.model, loader, device=device)
        mu_m, L_m = preds["mu"], preds["L"]
        if cell_ids is None:
            cell_ids_m = np.zeros(mu_m.shape[0], dtype=np.int64)
        else:
            if cell_ids.shape[0] != mu_m.shape[0]:
                raise ValueError(
                    f"cell_ids length {cell_ids.shape[0]} != batch N {mu_m.shape[0]}"
                )
            cell_ids_m = cell_ids
        mu_m, L_m = apply_calibration(mu_m, L_m, m.calibration, cell_ids=cell_ids_m)
        mus.append(mu_m)
        Sigmas_alea.append(np.einsum("bij,bkj->bik", L_m, L_m))
        if y_ref is None:
            y_ref = preds["y"]

    per_mu = np.stack(mus, axis=0)  # (M, B, n)
    per_Sig = np.stack(Sigmas_alea, axis=0)  # (M, B, n, n)
    mu_mean = per_mu.mean(axis=0)
    Sigma_alea = per_Sig.mean(axis=0)
    delta = per_mu - mu_mean[None]
    Sigma_epi = np.einsum("mbi,mbj->bij", delta, delta) / per_mu.shape[0]
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
