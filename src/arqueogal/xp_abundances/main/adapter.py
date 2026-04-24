"""Data-to-model adapter: FeatureLayout flat vector → encoder input.

Two jobs, kept in one module because they live at the same boundary:

1. :class:`XpFeatureAdapter` — optionally zeroes out the ``(bp_c0_z, rp_c0_z)``
   absolute-scale channels of the XP flat feature vector. Contrastive pretraining
   runs *shape-only* so that cell-based positive pairs do not align on c0_z
   (which correlates with distance and line-of-sight extinction, confounding the
   encoder's cell geometry); supervised fine-tuning runs with c0_z active so the
   head gains one real degree of freedom on luminosity/extinction.

2. :func:`reorder_labels_human_to_block` — permutes the last axis of a label
   matrix ``Y`` from ``LabelTiers`` (human/release) order into
   :class:`CovarianceBlockLayout` (model/block) order. Intended to run **once at
   load time** so that everything downstream — β-NLL, per-label diagnostics,
   checkpoint ``y`` buffers — lives in model order. Inverting back to human
   order happens on the inference/reporting boundary.

Counting conventions — annotated at every dimension declaration:

- Full FeatureLayout (``DEFAULT_XP_COEF_INDICES`` = ``range(1, 55)``):
  54 BP shape + 54 RP shape + 2 c0 scalars = **110-D XP block**, then
  residual + aux columns on top.
- Truncated FeatureLayout (``FeatureLayout.truncated_43d``):
  19 BP shape (``range(1, 20)``) + 22 RP shape (``range(1, 23)``) + 2 c0 scalars
  = **43-D XP block**. The c0 scalars are *retained* in the 43-D variant — the
  "43" counts stored shape coefs (41) plus the two c0 scalars. Don't conflate
  with the 43 Hermite modes Ye+2024 singles out as information-bearing; those
  coincide in count but arise from different arithmetic.

The adapter keeps its forward pass a single conditional index write so the
masking is traceable in a debugger without stepping through a branch tree.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import torch
from torch import nn

from arqueogal.xp_abundances.main.data import (
    DEFAULT_XP_SCALAR_COLS,
    FeatureLayout,
    LabelTiers,
)
from arqueogal.xp_abundances.main.model import CovarianceBlockLayout

_ArrayLike = TypeVar("_ArrayLike", np.ndarray, torch.Tensor)


class XpFeatureAdapter(nn.Module):
    """Thin bridge between :class:`FeatureLayout` flat vectors and the encoder.

    Parameters
    ----------
    layout
        The :class:`FeatureLayout` used to build the feature matrix. The adapter
        reads the flat column order from :attr:`FeatureLayout.all_required_columns`
        and looks up c0 scalar positions by name.
    use_c0_scalars
        When ``True`` (default), the forward pass is an identity. When ``False``,
        the output tensor has the ``bp_c0_z`` / ``rp_c0_z`` channels **replaced
        with zero** — not dropped. Keeping the dimensionality fixed means the
        encoder trunk is the same 110-D (or 43-D) input space whether contrastive
        pretraining zeroes c0 or supervised fine-tuning passes it through, which
        is what lets Run A reuse a contrastive trunk on a supervised head
        without re-wiring dimensions.

    Notes
    -----
    The c0 positions are resolved once at construction and cached as a buffer,
    so the forward pass costs one boolean branch + (if active) one scatter write.
    No parameter state — the adapter is a shape-and-mask operation.
    """

    use_c0_scalars: bool
    _c0_positions: torch.Tensor  # registered buffer

    def __init__(
        self,
        layout: FeatureLayout,
        *,
        use_c0_scalars: bool = True,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.use_c0_scalars = use_c0_scalars

        # Flat column order: BP shape → RP shape → XP scalars → residuals → aux.
        # See FeatureLayout.all_required_columns.
        cols = layout.all_required_columns
        c0_names = set(DEFAULT_XP_SCALAR_COLS)  # {"bp_c0_z", "rp_c0_z"}
        positions = [i for i, name in enumerate(cols) if name in c0_names]
        self.register_buffer(
            "_c0_positions",
            torch.tensor(positions, dtype=torch.long),
            persistent=False,
        )

    @property
    def input_dim(self) -> int:
        """The flat feature dimension consumed by the encoder trunk.

        For the default layout: 110 XP (54 BP + 54 RP + 2 c0) + 3 residuals +
        26 aux = 139-D. For the 43-D-truncated layout: 43 XP (19 BP + 22 RP +
        2 c0) + residuals + aux.
        """
        return self.layout.input_dim

    @property
    def output_dim(self) -> int:
        """Output dim matches input dim — the adapter never resizes the vector."""
        return self.layout.input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` unchanged, or with c0_z channels zeroed.

        The non-identity path clones (so we don't mutate the caller's tensor)
        and scatters zero into the c0 positions. The identity path returns the
        input tensor as-is — the encoder trunk is free to view-slice it.
        """
        if self.use_c0_scalars or self._c0_positions.numel() == 0:
            return x
        x_out = x.clone()
        x_out[..., self._c0_positions] = 0.0
        return x_out

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, use_c0_scalars={self.use_c0_scalars}, "
            f"n_c0_positions={int(self._c0_positions.numel())}"
        )


# --- Label reordering utility ------------------------------------------------


def reorder_labels_human_to_block(
    Y: _ArrayLike,
    *,
    tiers: LabelTiers,
    block_layout: CovarianceBlockLayout,
) -> _ArrayLike:
    """Reorder the last axis of ``Y`` from ``LabelTiers`` order to block order.

    Validates that ``tiers.all_labels`` equals ``block_layout.label_order_human``
    so that the permutation is well-defined. Works on both :class:`numpy.ndarray`
    and :class:`torch.Tensor` — returned array is the same type as input.

    Parameters
    ----------
    Y
        Label matrix with the last axis in human (``LabelTiers``) order.
        Any leading batch shape is preserved.
    tiers
        The :class:`LabelTiers` the ``Y`` columns were produced in (so we know
        what "human order" means for this run).
    block_layout
        The :class:`CovarianceBlockLayout` owned by the model; its
        ``label_order_human`` is the reference against which ``tiers`` is checked,
        and its ``human_to_block_perm`` drives the reorder.

    Raises
    ------
    ValueError
        If ``tiers.all_labels`` and ``block_layout.label_order_human`` disagree.
    """
    if tiers.all_labels != block_layout.label_order_human:
        raise ValueError(
            "LabelTiers.all_labels does not match block_layout.label_order_human; "
            "refusing to reorder ambiguously.\n"
            f"  tiers.all_labels    = {tiers.all_labels}\n"
            f"  block.label_order_human = {block_layout.label_order_human}"
        )
    if isinstance(Y, torch.Tensor):
        return block_layout.reorder_human_to_block(Y, dim=-1)
    perm = block_layout.human_to_block_perm.numpy()
    return np.take(Y, perm, axis=-1)


__all__ = [
    "XpFeatureAdapter",
    "reorder_labels_human_to_block",
]
