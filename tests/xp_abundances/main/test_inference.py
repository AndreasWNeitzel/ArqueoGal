"""Tests for xp_abundances.main.inference — ensemble load + aggregation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from arqueogal.xp_abundances.main.config import TrainingConfig
from arqueogal.xp_abundances.main.data import (
    FeatureLayout,
    LabelScaler,
    LabelTiers,
    XpAbundanceDataset,
)
from arqueogal.xp_abundances.main.inference import (
    EnsembleMember,
    load_ensemble,
    predict_ensemble,
)
from arqueogal.xp_abundances.main.model import (
    CovarianceBlockLayout,
    ModelConfig,
    XpAbundanceModel,
)
from arqueogal.xp_abundances.main.training import save_checkpoint
from arqueogal.xp_abundances.main.uncertainty import CalibrationArtifacts


def _small_layout_tiers() -> tuple[FeatureLayout, LabelTiers]:
    layout = FeatureLayout(
        xp_bp_indices=tuple(range(1, 5)),
        xp_rp_indices=tuple(range(1, 5)),
        xp_scalar_cols=(),
        residual_cols=(),
        aux_cols=(),
    )
    tiers = LabelTiers(
        tier1=("teff_apogee", "logg_apogee"),
        tier2=("alpha_m_apogee",),
        tier3=(),
    )
    return layout, tiers


def _tiny_loader(layout: FeatureLayout, tiers: LabelTiers, n: int = 12) -> DataLoader:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, layout.input_dim)).astype(np.float32)
    Y = rng.standard_normal((n, tiers.n_labels)).astype(np.float32)
    ds = XpAbundanceDataset(X=X, Y=Y)
    return DataLoader(ds, batch_size=4)


def _block_layout_for_tiny_tiers(tiers: LabelTiers) -> CovarianceBlockLayout:
    """2-label dense block + 1 diagonal-only label, matching the tiny test tiers."""
    return CovarianceBlockLayout(
        block_sizes=(2,),
        n_diagonal_only=1,
        label_order_block=tiers.all_labels,
        label_order_human=tiers.all_labels,
    )


def _save_member(
    tmp_path: Path,
    cfg: TrainingConfig,
    layout: FeatureLayout,
    tiers: LabelTiers,
    *,
    seed: int,
) -> Path:
    torch.manual_seed(seed)
    model = XpAbundanceModel(
        ModelConfig(
            input_dim=layout.input_dim,
            block_layout=_block_layout_for_tiny_tiers(tiers),
            latent_dim=cfg.latent_dim,
            trunk_hidden=cfg.trunk_hidden,
            head_hidden=cfg.head_hidden,
            dropout=cfg.dropout,
        )
    )
    log_temp = torch.tensor(0.0)
    rng = np.random.default_rng(seed)
    Y_fit = rng.standard_normal((32, tiers.n_labels)).astype(np.float32)
    scaler = LabelScaler.fit(Y_fit, tiers.all_labels)
    path = tmp_path / f"xp_abundances_main_20260418_abcdef1_seed{seed}.pt"
    save_checkpoint(
        path,
        model=model,
        log_temp=log_temp,
        cfg=cfg,
        layout=layout,
        tiers=tiers,
        label_scaler=scaler,
        seed=seed,
    )
    return path


def test_load_ensemble_from_directory(tmp_path: Path) -> None:
    layout, tiers = _small_layout_tiers()
    cfg = TrainingConfig(latent_dim=16, trunk_hidden=(32, 16), head_hidden=16)
    for s in (0, 1, 2):
        _save_member(tmp_path, cfg, layout, tiers, seed=s)

    members = load_ensemble(tmp_path, device=torch.device("cpu"))
    assert len(members) == 3
    assert {m.seed for m in members} == {0, 1, 2}
    assert all(isinstance(m, EnsembleMember) for m in members)


def test_load_ensemble_from_list(tmp_path: Path) -> None:
    layout, tiers = _small_layout_tiers()
    cfg = TrainingConfig(latent_dim=16, trunk_hidden=(32, 16), head_hidden=16)
    p0 = _save_member(tmp_path, cfg, layout, tiers, seed=0)
    p1 = _save_member(tmp_path, cfg, layout, tiers, seed=1)

    members = load_ensemble([p0, p1], device=torch.device("cpu"))
    assert len(members) == 2


def test_load_ensemble_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        load_ensemble(tmp_path, device=torch.device("cpu"))


def test_predict_ensemble_shapes(tmp_path: Path) -> None:
    layout, tiers = _small_layout_tiers()
    cfg = TrainingConfig(latent_dim=16, trunk_hidden=(32, 16), head_hidden=16)
    for s in (0, 1):
        _save_member(tmp_path, cfg, layout, tiers, seed=s)
    members = load_ensemble(tmp_path, device=torch.device("cpu"))
    loader = _tiny_loader(layout, tiers, n=10)

    out = predict_ensemble(members, loader, device=torch.device("cpu"))
    n = tiers.n_labels
    assert out.mu.shape == (10, n)
    assert out.sigma_aleatoric.shape == (10, n)
    assert out.sigma_epistemic.shape == (10, n)
    assert out.sigma_total.shape == (10, n)
    assert out.Sigma_total.shape == (10, n, n)
    assert out.per_member_mu.shape == (2, 10, n)
    assert out.y is not None and out.y.shape == (10, n)


def test_total_sigma_dominates_aleatoric(tmp_path: Path) -> None:
    """σ_total² = σ_alea² + σ_epi² ≥ σ_alea² elementwise."""
    layout, tiers = _small_layout_tiers()
    cfg = TrainingConfig(latent_dim=16, trunk_hidden=(32, 16), head_hidden=16)
    for s in (0, 1, 2):
        _save_member(tmp_path, cfg, layout, tiers, seed=s)
    members = load_ensemble(tmp_path, device=torch.device("cpu"))
    loader = _tiny_loader(layout, tiers, n=8)
    out = predict_ensemble(members, loader, device=torch.device("cpu"))
    assert np.all(out.sigma_total >= out.sigma_aleatoric - 1e-6)


def test_epistemic_is_zero_for_single_member(tmp_path: Path) -> None:
    """With one seed, ensemble spread is exactly zero."""
    layout, tiers = _small_layout_tiers()
    cfg = TrainingConfig(latent_dim=16, trunk_hidden=(32, 16), head_hidden=16)
    _save_member(tmp_path, cfg, layout, tiers, seed=0)
    members = load_ensemble(tmp_path, device=torch.device("cpu"))
    loader = _tiny_loader(layout, tiers, n=6)
    out = predict_ensemble(members, loader, device=torch.device("cpu"))
    assert np.allclose(out.sigma_epistemic, 0.0, atol=1e-6)
    assert np.allclose(out.sigma_total, out.sigma_aleatoric, atol=1e-6)


def test_predict_ensemble_applies_calibration(tmp_path: Path) -> None:
    """If a member's calibration has a scale factor, aleatoric σ should shift."""
    layout, tiers = _small_layout_tiers()
    cfg = TrainingConfig(latent_dim=16, trunk_hidden=(32, 16), head_hidden=16)
    _save_member(tmp_path, cfg, layout, tiers, seed=0)
    members = load_ensemble(tmp_path, device=torch.device("cpu"))
    loader = _tiny_loader(layout, tiers, n=6)

    baseline = predict_ensemble(members, loader, device=torch.device("cpu"))
    # Mutate calibration on the loaded in-memory member — scale by 2×.
    members[0].calibration = CalibrationArtifacts(temperature_per_cell={0: 2.0})
    scaled = predict_ensemble(members, loader, device=torch.device("cpu"))
    assert np.allclose(scaled.sigma_aleatoric, 2.0 * baseline.sigma_aleatoric, atol=1e-4)


def test_predict_ensemble_empty_raises(tmp_path: Path) -> None:
    layout, tiers = _small_layout_tiers()
    loader = _tiny_loader(layout, tiers, n=4)
    with pytest.raises(ValueError, match="ensemble is empty"):
        predict_ensemble([], loader)


def test_predict_ensemble_cell_ids_length_validation(tmp_path: Path) -> None:
    layout, tiers = _small_layout_tiers()
    cfg = TrainingConfig(latent_dim=16, trunk_hidden=(32, 16), head_hidden=16)
    _save_member(tmp_path, cfg, layout, tiers, seed=0)
    members = load_ensemble(tmp_path, device=torch.device("cpu"))
    loader = _tiny_loader(layout, tiers, n=6)
    with pytest.raises(ValueError, match="cell_ids length"):
        predict_ensemble(
            members,
            loader,
            device=torch.device("cpu"),
            cell_ids=np.zeros(999, dtype=np.int64),
        )
