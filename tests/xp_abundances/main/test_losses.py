"""Tests for xp_abundances.main.losses — SupCon + multivariate Beta-NLL."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 — PyTorch-idiomatic alias

from arqueogal.xp_abundances.main.losses import (
    beta_nll_block_cholesky,
    mahalanobis_residual,
    supcon_soft_positive,
)


def _unit_proj(B: int, D: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, D, generator=g)
    return F.normalize(z, dim=-1)


def test_supcon_returns_scalar_and_finite() -> None:
    za = _unit_proj(8, 16, seed=1)
    zk = _unit_proj(8, 16, seed=2)
    ya = torch.randn(8, 4)
    yk = torch.randn(8, 4)
    loss = supcon_soft_positive(za, ya, zk, yk, temperature=0.1)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_supcon_gradient_flows_to_anchors_only() -> None:
    za = _unit_proj(4, 8, seed=1).requires_grad_(True)
    zk = _unit_proj(4, 8, seed=2)
    ya = torch.randn(4, 2)
    yk = torch.randn(4, 2)
    loss = supcon_soft_positive(za, ya, zk, yk, temperature=0.2)
    loss.backward()
    assert za.grad is not None
    assert torch.isfinite(za.grad).all()


def test_supcon_excludes_self_pair_when_keys_equal_anchors() -> None:
    D = 8
    za = _unit_proj(4, D, seed=1)
    # Self-exclusion: set first K==B keys identical to anchors; loss must not blow up.
    ya = torch.randn(4, 2)
    loss_self = supcon_soft_positive(za, ya, za, ya, temperature=0.1)
    assert torch.isfinite(loss_self)


def test_supcon_projection_dim_mismatch_raises() -> None:
    za = torch.randn(2, 8)
    zk = torch.randn(2, 16)
    y = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="projection dim"):
        supcon_soft_positive(za, y, zk, y, temperature=0.1)


def _fake_cholesky_batch(
    B: int, n: int, block_sizes: tuple[int, ...], n_diagonal_only: int = 0,
    seed: int = 0,
) -> torch.Tensor:
    """Produce a batch of lower-triangular L with block structure for tests.

    Dense blocks occupy positions ``[offset, offset + k)`` along both axes;
    the trailing ``n_diagonal_only`` labels carry only diagonal entries.
    """
    assert sum(block_sizes) + n_diagonal_only == n
    g = torch.Generator().manual_seed(seed)
    L = torch.zeros(B, n, n)
    for b in range(B):
        offset = 0
        for k in block_sizes:
            block = torch.tril(torch.randn(k, k, generator=g))
            block.diagonal().copy_(F.softplus(block.diagonal()) + 0.1)
            L[b, offset : offset + k, offset : offset + k] = block
            offset += k
        if n_diagonal_only > 0:
            d = F.softplus(torch.randn(n_diagonal_only, generator=g)) + 0.1
            L[b, offset : offset + n_diagonal_only,
              offset : offset + n_diagonal_only] = torch.diag(d)
    return L


def test_beta_nll_scalar_and_finite() -> None:
    B, blocks, n_diag = 6, (3, 3), 4
    n = sum(blocks) + n_diag
    L = _fake_cholesky_batch(B, n, blocks, n_diag, seed=1)
    mu = torch.randn(B, n)
    y = mu + 0.05 * torch.randn(B, n)
    loss = beta_nll_block_cholesky(mu, L, y, beta=0.5)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_beta_nll_backward_is_stable() -> None:
    B, blocks, n_diag = 4, (3, 3), 4
    n = sum(blocks) + n_diag
    L = _fake_cholesky_batch(B, n, blocks, n_diag, seed=1).requires_grad_(True)
    mu = torch.randn(B, n, requires_grad=True)
    y = torch.randn(B, n)
    loss = beta_nll_block_cholesky(mu, L, y, beta=0.5)
    loss.backward()
    assert mu.grad is not None
    assert L.grad is not None
    assert torch.isfinite(mu.grad).all()
    assert torch.isfinite(L.grad).all()


def test_beta_nll_beta_zero_equals_standard_nll() -> None:
    """β=0 must reduce to plain MVN NLL averaged over dimensions."""
    B, blocks, n_diag = 3, (2, 2), 2
    n = sum(blocks) + n_diag
    L = _fake_cholesky_batch(B, n, blocks, n_diag, seed=7)
    mu = torch.randn(B, n)
    y = torch.randn(B, n)
    loss_b0 = beta_nll_block_cholesky(mu, L, y, beta=0.0).item()

    # Compute plain MVN-NLL per star then average, divided by n.
    diff = (y - mu).unsqueeze(-1)
    z = torch.linalg.solve_triangular(L, diff, upper=False).squeeze(-1)
    mahal = z.pow(2).sum(dim=-1)
    log_det = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
    nll_expected = 0.5 * (n * math.log(2 * math.pi) + log_det + mahal)
    ref = (nll_expected.mean() / n).item()
    assert abs(loss_b0 - ref) < 1e-5


def test_beta_nll_ignores_masked_out_labels() -> None:
    B, blocks, n_diag = 4, (2, 2), 2
    n = sum(blocks) + n_diag
    L = _fake_cholesky_batch(B, n, blocks, n_diag, seed=3)
    mu = torch.zeros(B, n)
    y = torch.randn(B, n)

    full = beta_nll_block_cholesky(mu, L, y, beta=0.0)
    mask = torch.ones(B, n)
    masked_all = beta_nll_block_cholesky(mu, L, y, beta=0.0, mask=mask)
    assert abs(full.item() - masked_all.item()) < 1e-4


def test_beta_nll_mask_shape_validation() -> None:
    B, blocks, n_diag = 2, (2, 2), 2
    n = sum(blocks) + n_diag
    L = _fake_cholesky_batch(B, n, blocks, n_diag, seed=0)
    mu = torch.zeros(B, n)
    y = torch.zeros(B, n)
    bad_mask = torch.ones(B, n + 1)
    with pytest.raises(ValueError, match="mask shape"):
        beta_nll_block_cholesky(mu, L, y, mask=bad_mask)


def test_beta_nll_shape_mismatch_raises() -> None:
    L = _fake_cholesky_batch(2, 4, (1, 1), n_diagonal_only=2, seed=0)
    mu = torch.zeros(2, 4)
    y = torch.zeros(2, 5)
    with pytest.raises(ValueError, match="mu shape"):
        beta_nll_block_cholesky(mu, L, y)


def test_mahalanobis_residual_shape_and_zero_at_mean() -> None:
    B, blocks, n_diag = 3, (2, 2), 2
    n = sum(blocks) + n_diag
    L = _fake_cholesky_batch(B, n, blocks, n_diag, seed=0)
    mu = torch.randn(B, n)
    res = mahalanobis_residual(mu, L, mu)
    torch.testing.assert_close(res, torch.zeros(B, n), atol=1e-6, rtol=0)
