"""Tests for v1.1 architectural changes.

Coverage:
1. 21-label head emits 21 outputs.
2. Evolutionary-stage head emits 4 softmax probabilities summing to 1.
3. ARI contamination loss returns a finite scalar that increases with worse ARI.
4. Feature-noise injection adds noise of the right scale to the right feature columns.
5. Feature-noise marginalisation increases σ_total monotonically with σ_feature.
"""

from __future__ import annotations

import pytest
import torch

from arqueogal.xp_abundances.main.losses import (
    add_feature_noise,
    propagate_feature_noise_uncertainty,
    soft_ari_loss,
)
from arqueogal.xp_abundances.main.model import (
    EvolutionaryStageHead,
    ModelConfig,
    XpAbundanceModel,
    default_pipeline1_layout,
)


class TestTwentyOneLabel:
    """Verify 21-label head emits 21 outputs with proper block structure."""

    def test_default_layout_has_21_labels(self) -> None:
        layout = default_pipeline1_layout()
        assert layout.n_labels == 21
        assert sum(layout.block_sizes) + layout.n_diagonal_only == 21

    def test_model_forward_emits_21_labels(self) -> None:
        cfg = ModelConfig(input_dim=108, block_layout=default_pipeline1_layout())
        model = XpAbundanceModel(cfg)
        x = torch.randn(8, 108)

        result = model(x)

        assert len(result) == 4
        mu, L, h, z = result

        assert mu.shape == (8, 21)
        assert L.shape == (8, 21, 21)
        assert h.shape == (8, 32)
        assert z.shape == (8, 32)

    def test_cholesky_factor_block_structure(self) -> None:
        cfg = ModelConfig(input_dim=108, block_layout=default_pipeline1_layout())
        model = XpAbundanceModel(cfg)
        x = torch.randn(8, 108)

        result = model(x)
        mu, L, _, _ = result

        layout = default_pipeline1_layout()
        block_sizes = layout.block_sizes
        n_diagonal_only = layout.n_diagonal_only

        offset = 0
        for block_size in block_sizes:
            block_end = offset + block_size
            within_block = L[:, offset:block_end, offset:block_end]
            assert within_block.shape[-2:] == (block_size, block_size)
            assert torch.isfinite(within_block).all()
            offset = block_end

        diagonal_tail_start = offset
        for i in range(n_diagonal_only):
            idx = diagonal_tail_start + i
            assert torch.isfinite(L[:, idx, idx]).all()
            assert torch.allclose(L[:, idx, :idx], torch.zeros_like(L[:, idx, :idx]))


class TestEvolutionaryStageHead:
    """Verify 4-way diagnostic head emits valid soft probabilities."""

    def test_evol_stage_head_shape(self) -> None:
        head = EvolutionaryStageHead(latent_dim=32)
        h = torch.randn(16, 32)
        logits = head(h)
        assert logits.shape == (16, 4)

    def test_model_with_evol_stage_enabled(self) -> None:
        cfg = ModelConfig(input_dim=108, block_layout=default_pipeline1_layout())
        model = XpAbundanceModel(cfg, include_evol_stage_head=True)
        x = torch.randn(8, 108)

        result = model(x)

        assert len(result) == 5
        mu, L, h, z, evol_logits = result

        assert evol_logits is not None
        assert evol_logits.shape == (8, 4)
        assert torch.isfinite(evol_logits).all()

    def test_evol_stage_softmax_probabilities(self) -> None:
        head = EvolutionaryStageHead(latent_dim=32)
        h = torch.randn(16, 32)
        logits = head(h)
        probs = torch.softmax(logits, dim=-1)

        assert probs.shape == (16, 4)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(16), atol=1e-6)
        assert (probs >= 0.0).all()
        assert (probs <= 1.0).all()


class TestSoftARILoss:
    """Verify ARI contamination loss behaves correctly."""

    def test_soft_ari_loss_perfect_agreement(self) -> None:
        B = 16
        y_true = torch.zeros(B, 2)
        y_true[:8, 0] = 1.0
        y_true[8:, 1] = 1.0

        y_pred = y_true.clone()

        loss = soft_ari_loss(y_pred, y_true)
        assert torch.isfinite(loss)
        assert loss < 0.01

    def test_soft_ari_loss_random_assignment(self) -> None:
        B = 16
        torch.manual_seed(42)
        y_true = torch.randn(B, 2)
        y_true = torch.softmax(y_true, dim=-1)

        y_pred = torch.rand(B, 2)
        y_pred = torch.softmax(y_pred, dim=-1)

        loss = soft_ari_loss(y_pred, y_true)
        assert torch.isfinite(loss)
        assert 0.0 <= loss.item() <= 2.0

    def test_soft_ari_loss_varies_with_prediction_quality(self) -> None:
        B = 16
        y_true = torch.zeros(B, 2)
        y_true[:8, 0] = 1.0
        y_true[8:, 1] = 1.0

        y_pred_perfect = y_true.clone()

        y_pred_partial = y_true.clone()
        y_pred_partial[:4, :] = 0.5

        loss_perfect = soft_ari_loss(y_pred_perfect, y_true)
        loss_partial = soft_ari_loss(y_pred_partial, y_true)

        assert torch.isfinite(loss_perfect)
        assert torch.isfinite(loss_partial)
        assert loss_partial >= loss_perfect

    def test_soft_ari_loss_invalid_shapes(self) -> None:
        y_true = torch.ones(16, 2)
        y_pred_wrong = torch.ones(16, 3)

        with pytest.raises(ValueError):
            soft_ari_loss(y_pred_wrong, y_true)

    def test_soft_ari_loss_finite_scalar(self) -> None:
        B = 16
        y_true = torch.softmax(torch.randn(B, 2), dim=-1)
        y_pred = torch.softmax(torch.randn(B, 2), dim=-1)

        loss = soft_ari_loss(y_pred, y_true)
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss)


class TestFeatureNoiseInjection:
    """Verify feature-noise injection adds noise at the right scale."""

    def test_noise_injection_disabled_when_not_training(self) -> None:
        x = torch.ones(8, 10)
        sigma = torch.ones(10) * 0.1

        x_noisy = add_feature_noise(x, sigma, training=False)
        assert torch.allclose(x, x_noisy)

    def test_noise_injection_adds_noise_training(self) -> None:
        torch.manual_seed(42)
        x = torch.zeros(1000, 10)
        sigma = torch.ones(10) * 0.5

        x_noisy = add_feature_noise(x, sigma, training=True)

        noise = x_noisy - x
        empirical_std = noise.std(dim=0)

        expected_std = 0.5 * torch.ones(10)
        assert torch.allclose(empirical_std, expected_std, atol=0.05)

    def test_noise_injection_respects_zero_sigma(self) -> None:
        torch.manual_seed(42)
        x = torch.randn(100, 10)
        sigma = torch.zeros(10)
        sigma[0] = 0.5

        x_noisy = add_feature_noise(x, sigma, training=True)

        assert torch.allclose(x_noisy[:, 1:], x[:, 1:])
        assert not torch.allclose(x_noisy[:, 0], x[:, 0])

    def test_noise_injection_selective_columns(self) -> None:
        torch.manual_seed(42)
        x = torch.zeros(100, 10)
        sigma = torch.ones(10) * 0.5
        feature_cols = [0, 1, 2]

        x_noisy = add_feature_noise(x, sigma, feature_cols=feature_cols, training=True)

        assert not torch.allclose(x_noisy[:, feature_cols], x[:, feature_cols])
        assert torch.allclose(x_noisy[:, 3:], x[:, 3:])

    def test_noise_injection_shape_validation(self) -> None:
        x = torch.randn(8, 10)
        sigma = torch.randn(9)

        with pytest.raises(ValueError):
            add_feature_noise(x, sigma, training=True)


class TestFeatureNoisePropagation:
    """Verify analytical feature-noise marginalisation increases uncertainty."""

    def test_noise_marginalisation_shape(self) -> None:
        B, n_labels = 8, 21
        x = torch.randn(B, 108, requires_grad=True)
        mu = torch.randn(B, n_labels, requires_grad=True)
        L = torch.eye(n_labels).unsqueeze(0).expand(B, -1, -1)
        sigma_features = torch.ones(108) * 0.1

        mu_out, L_out = propagate_feature_noise_uncertainty(mu, L, x, sigma_features)

        assert mu_out.shape == mu.shape
        assert L_out.shape == L.shape

    def test_noise_marginalisation_increases_diagonal(self) -> None:
        B, n_labels = 8, 5
        x = torch.randn(B, 20, requires_grad=True)
        mu = torch.randn(B, n_labels, requires_grad=True)
        L_base = torch.eye(n_labels).unsqueeze(0).expand(B, -1, -1) * 0.1
        sigma_features = torch.ones(20) * 0.1

        mu_out, L_out = propagate_feature_noise_uncertainty(mu, L_base, x, sigma_features)

        L_diag_base = L_base.diagonal(dim1=-2, dim2=-1)
        L_diag_out = L_out.diagonal(dim1=-2, dim2=-1)

        assert (L_diag_out >= L_diag_base).all()

    def test_noise_marginalisation_zero_sigma(self) -> None:
        B, n_labels = 8, 5
        x = torch.randn(B, 20, requires_grad=True)
        mu = torch.randn(B, n_labels, requires_grad=True)
        L_base = torch.eye(n_labels).unsqueeze(0).expand(B, -1, -1) * 0.5
        sigma_features = torch.zeros(20)

        mu_out, L_out = propagate_feature_noise_uncertainty(mu, L_base, x, sigma_features)

        assert torch.allclose(L_out, L_base, atol=1e-5)

    def test_noise_marginalisation_shape_mismatch(self) -> None:
        B, n_labels = 8, 5
        x = torch.randn(B, 20, requires_grad=True)
        mu = torch.randn(B, n_labels, requires_grad=True)
        L = torch.eye(n_labels).unsqueeze(0).expand(B, -1, -1)
        sigma_features = torch.ones(19) * 0.1

        with pytest.raises(ValueError):
            propagate_feature_noise_uncertainty(mu, L, x, sigma_features)


class TestV11ModelIntegration:
    """Integration tests for v1.1 model with all features."""

    def test_v11_model_forward_complete(self) -> None:
        cfg = ModelConfig(input_dim=108, block_layout=default_pipeline1_layout())
        model = XpAbundanceModel(cfg, include_evol_stage_head=True)
        x = torch.randn(16, 108)

        result = model(x)
        assert len(result) == 5
        mu, L, h, z, evol_logits = result

        assert mu.shape == (16, 21)
        assert L.shape == (16, 21, 21)
        assert h.shape == (16, 32)
        assert z.shape == (16, 32)
        assert evol_logits.shape == (16, 4)

        assert torch.isfinite(mu).all()
        assert torch.isfinite(L).all()
        assert torch.isfinite(evol_logits).all()

    def test_v11_backward_pass_works(self) -> None:
        cfg = ModelConfig(input_dim=108, block_layout=default_pipeline1_layout())
        model = XpAbundanceModel(cfg, include_evol_stage_head=True)
        x = torch.randn(8, 108, requires_grad=True)

        mu, L, h, z, evol_logits = model(x)

        loss_abund = mu.sum() + L.sum()
        loss_evol = evol_logits.sum()
        loss_total = loss_abund + 0.05 * loss_evol

        loss_total.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert model.encoder.trunk[0].weight.grad is not None

    def test_device_consistency(self) -> None:
        cfg = ModelConfig(input_dim=108, block_layout=default_pipeline1_layout())
        model = XpAbundanceModel(cfg, include_evol_stage_head=True)

        device = torch.device("cpu")
        model = model.to(device)

        x = torch.randn(8, 108, device=device)
        mu, L, h, z, evol_logits = model(x)

        assert mu.device == device
        assert L.device == device
        assert evol_logits.device == device
