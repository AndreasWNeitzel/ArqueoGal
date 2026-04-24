"""Tests for xp_abundances.main.adapter — c0 masking + label reorder."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from arqueogal.xp_abundances.main.adapter import (
    XpFeatureAdapter,
    reorder_labels_human_to_block,
)
from arqueogal.xp_abundances.main.data import (
    DEFAULT_XP_SCALAR_COLS,
    FeatureLayout,
    LabelTiers,
)
from arqueogal.xp_abundances.main.model import (
    CovarianceBlockLayout,
    Encoder,
    default_pipeline1_layout,
)


# --- XpFeatureAdapter: shape + c0 masking ------------------------------------


def test_adapter_full_layout_input_dim() -> None:
    """Default layout: 54 BP + 54 RP + 2 c0 + 3 residuals + 26 aux = 139."""
    layout = FeatureLayout()
    adapter = XpFeatureAdapter(layout)
    # 54 + 54 + 2 + 3 + 26 = 139
    assert adapter.input_dim == 139
    assert adapter.output_dim == 139


def test_adapter_truncated_43d_input_dim() -> None:
    """Truncated: 19 BP + 22 RP + 2 c0 + 3 residuals + 26 aux = 72.

    The "43-D XP block" is 19 + 22 + 2 = 43; total flat feature dim is
    43 + residuals + aux.
    """
    layout = FeatureLayout.truncated_43d()
    adapter = XpFeatureAdapter(layout)
    assert adapter.input_dim == 19 + 22 + 2 + 3 + 26  # 72


def test_adapter_identity_when_use_c0_scalars_true() -> None:
    layout = FeatureLayout()
    adapter = XpFeatureAdapter(layout, use_c0_scalars=True)
    x = torch.randn(8, layout.input_dim)
    out = adapter(x)
    torch.testing.assert_close(out, x)


def test_adapter_zeroes_c0_channels_when_flag_false() -> None:
    layout = FeatureLayout()
    adapter = XpFeatureAdapter(layout, use_c0_scalars=False)
    # Fill with non-zero so we can distinguish zeroed positions from input.
    x = torch.ones(4, layout.input_dim)
    out = adapter(x)

    cols = layout.all_required_columns
    c0_positions = [i for i, name in enumerate(cols) if name in DEFAULT_XP_SCALAR_COLS]
    assert len(c0_positions) == 2  # bp_c0_z, rp_c0_z

    for p in c0_positions:
        assert torch.all(out[:, p] == 0.0)
    # Everything else must still be ones.
    non_c0 = [i for i in range(layout.input_dim) if i not in c0_positions]
    for p in non_c0:
        assert torch.all(out[:, p] == 1.0)


def test_adapter_does_not_mutate_input() -> None:
    layout = FeatureLayout()
    adapter = XpFeatureAdapter(layout, use_c0_scalars=False)
    x = torch.ones(2, layout.input_dim)
    x_before = x.clone()
    _ = adapter(x)
    torch.testing.assert_close(x, x_before)


def test_adapter_default_flag_is_true() -> None:
    """Default kwarg: supervised fine-tune is the common case, so c0 is passed through."""
    layout = FeatureLayout()
    adapter = XpFeatureAdapter(layout)
    assert adapter.use_c0_scalars is True


def test_adapter_no_c0_in_layout_is_identity() -> None:
    """A layout with xp_scalar_cols=() has no c0 positions to mask — forward is identity."""
    layout = FeatureLayout(xp_scalar_cols=(), residual_cols=(), aux_cols=())
    adapter = XpFeatureAdapter(layout, use_c0_scalars=False)
    x = torch.randn(3, layout.input_dim)
    torch.testing.assert_close(adapter(x), x)


def test_adapter_zeroing_preserves_dtype_and_device() -> None:
    layout = FeatureLayout()
    adapter = XpFeatureAdapter(layout, use_c0_scalars=False)
    x = torch.ones(2, layout.input_dim, dtype=torch.float64)
    out = adapter(x)
    assert out.dtype == torch.float64
    assert out.device == x.device


def test_adapter_batched_and_unbatched() -> None:
    layout = FeatureLayout()
    adapter = XpFeatureAdapter(layout, use_c0_scalars=False)
    x1d = torch.ones(layout.input_dim)
    x2d = torch.ones(5, layout.input_dim)
    out1 = adapter(x1d)
    out2 = adapter(x2d)
    assert out1.shape == x1d.shape
    assert out2.shape == x2d.shape


# --- Integration: adapter → Encoder trunk → projection -----------------------


def test_adapter_feeds_encoder_produces_32d_projection() -> None:
    """Post-#130 sanity: adapter output → trunk → (h_32, z_32) for both layouts.

    This is the one-time cross-check that the 110-D adapter output slots into
    the ported TESS_ML trunk without a shape mismatch, and that the L2-normalised
    projection lands at latent_dim=32.
    """
    for layout in (FeatureLayout(), FeatureLayout.truncated_43d()):
        adapter = XpFeatureAdapter(layout, use_c0_scalars=False)
        encoder = Encoder(input_dim=adapter.output_dim, latent_dim=32)
        x = torch.randn(6, layout.input_dim)
        h, z = encoder(adapter(x))
        assert h.shape == (6, 32)
        assert z.shape == (6, 32)
        # Projection must be unit-norm (sanity on the ported trunk wiring).
        norms = z.pow(2).sum(dim=-1).sqrt()
        torch.testing.assert_close(norms, torch.ones(6), atol=1e-5, rtol=1e-5)


def test_adapter_toggle_produces_consistent_trunk_shapes() -> None:
    """Run A contract: same 110-D trunk, two adapter states, same output shape."""
    layout = FeatureLayout()
    encoder = Encoder(input_dim=layout.input_dim, latent_dim=32)
    x = torch.randn(4, layout.input_dim)

    adapter_off = XpFeatureAdapter(layout, use_c0_scalars=False)
    adapter_on = XpFeatureAdapter(layout, use_c0_scalars=True)
    h_off, z_off = encoder(adapter_off(x))
    h_on, z_on = encoder(adapter_on(x))
    assert h_off.shape == h_on.shape == (4, 32)
    assert z_off.shape == z_on.shape == (4, 32)
    # Zeroing c0 should actually change the trunk output (otherwise the mask is a no-op).
    assert not torch.allclose(h_off, h_on)


# --- reorder_labels_human_to_block -------------------------------------------


def test_reorder_default_layout_matches_tiers() -> None:
    """The default layout's human order IS LabelTiers order, so this must not raise."""
    tiers = LabelTiers()
    block_layout = default_pipeline1_layout()
    Y = np.arange(tiers.n_labels, dtype=np.float32)[None, :]  # (1, 21)
    Y_block = reorder_labels_human_to_block(Y, tiers=tiers, block_layout=block_layout)
    assert Y_block.shape == Y.shape


def test_reorder_permutation_is_correct() -> None:
    """Value at block-position i should equal human-position of label_order_block[i]."""
    tiers = LabelTiers()
    block_layout = default_pipeline1_layout()
    human_index = {name: i for i, name in enumerate(tiers.all_labels)}

    # Y where value at human-position i is simply i, for easy lookup.
    Y = np.arange(tiers.n_labels, dtype=np.float32)
    Y_block = reorder_labels_human_to_block(Y, tiers=tiers, block_layout=block_layout)
    expected = np.array(
        [human_index[name] for name in block_layout.label_order_block],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(Y_block, expected)


def test_reorder_roundtrip_numpy() -> None:
    tiers = LabelTiers()
    block_layout = default_pipeline1_layout()
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((50, tiers.n_labels)).astype(np.float32)
    Y_block = reorder_labels_human_to_block(Y, tiers=tiers, block_layout=block_layout)
    Y_back = block_layout.reorder_block_to_human(torch.from_numpy(Y_block), dim=-1).numpy()
    np.testing.assert_array_equal(Y_back, Y)


def test_reorder_roundtrip_torch() -> None:
    tiers = LabelTiers()
    block_layout = default_pipeline1_layout()
    Y = torch.randn(8, tiers.n_labels)
    Y_block = reorder_labels_human_to_block(Y, tiers=tiers, block_layout=block_layout)
    Y_back = block_layout.reorder_block_to_human(Y_block, dim=-1)
    torch.testing.assert_close(Y_back, Y)


def test_reorder_preserves_input_type() -> None:
    tiers = LabelTiers()
    block_layout = default_pipeline1_layout()
    Y_np = np.zeros((2, tiers.n_labels), dtype=np.float32)
    Y_t = torch.zeros(2, tiers.n_labels)
    assert isinstance(
        reorder_labels_human_to_block(Y_np, tiers=tiers, block_layout=block_layout),
        np.ndarray,
    )
    assert isinstance(
        reorder_labels_human_to_block(Y_t, tiers=tiers, block_layout=block_layout),
        torch.Tensor,
    )


def test_reorder_rejects_mismatched_orderings() -> None:
    """If tiers.all_labels ≠ block_layout.label_order_human, raise — ambiguous."""
    tiers = LabelTiers()
    # Build an anonymous layout whose label_order_human ≠ tiers.all_labels.
    bogus_layout = CovarianceBlockLayout.anonymous(
        block_sizes=(3, 4, 4, 6),
        n_diagonal_only=4,
    )
    Y = np.zeros((1, tiers.n_labels), dtype=np.float32)
    with pytest.raises(ValueError, match="does not match"):
        reorder_labels_human_to_block(Y, tiers=tiers, block_layout=bogus_layout)


def test_reorder_batched_shape_preserved() -> None:
    tiers = LabelTiers()
    block_layout = default_pipeline1_layout()
    Y = torch.randn(3, 5, tiers.n_labels)  # two leading batch axes
    Y_block = reorder_labels_human_to_block(Y, tiers=tiers, block_layout=block_layout)
    assert Y_block.shape == Y.shape
