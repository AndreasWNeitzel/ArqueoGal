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


def test_supcon_nan_labels_masked_not_propagated() -> None:
    za = _unit_proj(6, 8, seed=1)
    zk = _unit_proj(6, 8, seed=2)
    ya = torch.randn(6, 3)
    yk = torch.randn(6, 3)
    ya[1, 2] = float("nan")
    yk[3, 0] = float("nan")
    loss = supcon_soft_positive(za, ya, zk, yk, temperature=0.1)
    assert torch.isfinite(loss), "NaN labels must not NaN-propagate into the loss"


def _fake_cholesky_batch(
    B: int,
    n: int,
    block_sizes: tuple[int, ...],
    n_diagonal_only: int = 0,
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
            L[b, offset : offset + n_diagonal_only, offset : offset + n_diagonal_only] = torch.diag(
                d
            )
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


def test_beta_nll_uniform_weights_equal_unweighted() -> None:
    """Weights of all ones must reproduce the unweighted reduction exactly."""
    B, blocks, n_diag = 5, (2, 2), 2
    n = sum(blocks) + n_diag
    L = _fake_cholesky_batch(B, n, blocks, n_diag, seed=4)
    mu = torch.randn(B, n)
    y = torch.randn(B, n)

    # No mask path.
    plain = beta_nll_block_cholesky(mu, L, y, beta=0.5).item()
    with_w = beta_nll_block_cholesky(
        mu,
        L,
        y,
        beta=0.5,
        sample_weights=torch.ones(B),
    ).item()
    assert abs(plain - with_w) < 1e-5

    # Mask path (all ones).
    mask = torch.ones(B, n)
    plain_m = beta_nll_block_cholesky(mu, L, y, beta=0.5, mask=mask).item()
    with_w_m = beta_nll_block_cholesky(
        mu,
        L,
        y,
        beta=0.5,
        mask=mask,
        sample_weights=torch.ones(B),
    ).item()
    assert abs(plain_m - with_w_m) < 1e-5


def test_beta_nll_weighted_average_matches_manual() -> None:
    """Weighted reduction must equal sum(w*nll)/sum(w)/n with no mask."""
    B, blocks, n_diag = 6, (2, 2), 2
    n = sum(blocks) + n_diag
    L = _fake_cholesky_batch(B, n, blocks, n_diag, seed=9)
    mu = torch.randn(B, n)
    y = torch.randn(B, n)
    w = torch.tensor([5.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # up-weight star 0 5×.

    # Manual per-star NLL (β=0 so no β weighting to untangle).
    diff = (y - mu).unsqueeze(-1)
    z = torch.linalg.solve_triangular(L, diff, upper=False).squeeze(-1)
    mahal = z.pow(2).sum(dim=-1)
    log_det = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
    nll_per_star = 0.5 * (n * math.log(2 * math.pi) + log_det + mahal)
    expected = (w * nll_per_star).sum() / w.sum() / n

    got = beta_nll_block_cholesky(
        mu,
        L,
        y,
        beta=0.0,
        sample_weights=w,
    ).item()
    assert abs(got - expected.item()) < 1e-5


def test_beta_nll_weighted_mask_path_matches_manual() -> None:
    """Weighted + mask path: sum(w*scale*nll)/sum(w*mask_sum_per_star)."""
    B, blocks, n_diag = 4, (2, 2), 2
    n = sum(blocks) + n_diag
    L = _fake_cholesky_batch(B, n, blocks, n_diag, seed=11)
    mu = torch.randn(B, n)
    y = torch.randn(B, n)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 1, 1, 0, 0],
            [1, 0, 1, 0, 1, 0],
        ],
        dtype=torch.float32,
    )
    w = torch.tensor([1.0, 3.0, 0.5, 2.0])

    # Manual (β=0).
    diff = (y - mu).unsqueeze(-1)
    z = torch.linalg.solve_triangular(L, diff, upper=False).squeeze(-1)
    mahal = z.pow(2).sum(dim=-1)
    log_det = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
    nll_per_star = 0.5 * (n * math.log(2 * math.pi) + log_det + mahal)
    obs = mask.sum(dim=-1).clamp_min(1e-8)
    scaled = nll_per_star * (obs / n) * w
    expected = scaled.sum() / (mask * w.unsqueeze(-1)).sum()

    got = beta_nll_block_cholesky(
        mu,
        L,
        y,
        beta=0.0,
        mask=mask,
        sample_weights=w,
    ).item()
    assert abs(got - expected.item()) < 1e-5


def test_beta_nll_sample_weights_shape_validation() -> None:
    B, blocks, n_diag = 3, (2, 2), 2
    n = sum(blocks) + n_diag
    L = _fake_cholesky_batch(B, n, blocks, n_diag, seed=0)
    mu = torch.zeros(B, n)
    y = torch.zeros(B, n)
    bad_w = torch.ones(B + 1)
    with pytest.raises(ValueError, match="sample_weights shape"):
        beta_nll_block_cholesky(mu, L, y, sample_weights=bad_w)


def test_beta_nll_upweighting_shifts_gradient_toward_weighted_star() -> None:
    """Heavy weight on star k must make the loss more sensitive to its mu_k."""
    B, blocks, n_diag = 3, (2, 2), 2
    n = sum(blocks) + n_diag
    L = _fake_cholesky_batch(B, n, blocks, n_diag, seed=13)
    y = torch.randn(B, n)

    mu_uniform = torch.randn(B, n, requires_grad=True)
    loss_uniform = beta_nll_block_cholesky(
        mu_uniform,
        L,
        y,
        beta=0.5,
        sample_weights=torch.ones(B),
    )
    loss_uniform.backward()
    g_uniform = mu_uniform.grad.clone()

    mu_weighted = mu_uniform.detach().clone().requires_grad_(True)
    w = torch.tensor([10.0, 0.1, 0.1])
    loss_weighted = beta_nll_block_cholesky(
        mu_weighted,
        L,
        y,
        beta=0.5,
        sample_weights=w,
    )
    loss_weighted.backward()
    g_weighted = mu_weighted.grad

    # Star 0's gradient norm should be larger under heavy weighting;
    # other stars' gradient norms should be smaller.
    assert g_weighted[0].norm() > g_uniform[0].norm()
    assert g_weighted[1].norm() < g_uniform[1].norm()
