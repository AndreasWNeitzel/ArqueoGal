"""Pipeline 1 model: encoder + projection head + block-structured Cholesky.

Encoder trunk is ported verbatim from the TESS_ML prototype
(``/projects/TESS_ML/src/contrastive/model.py``):
``Linear → LayerNorm → GELU → Dropout`` MLP, ``in_dim → 256 → 128 → D=32``,
plus a two-layer L2-normalised projection head for SupCon contrastive
pretraining. This keeps the pretrained-weight load-path trivial once we get to
#119.3.

The supervised head is the architectural deviation. TESS_ML emits a 2-D
``(mu, log_var)`` scalar pair; we emit a 21-D mean vector plus a
block-structured lower-triangular Cholesky factor ``L``. The block layout
is **physics-motivated, not tier-based** (DESIGN.md §Block-Cholesky):

- Atmospheric block (3): Teff, log g, [M/H]
- α-process block (4): [Mg/H], [Si/H], [Ca/H], [Ti/H]
- Fe-peak block (4): [Fe/H], [Mn/H], [Ni/H], [Cr/H]
- Light / CNO block (6): [C/H], [N/H], [O/H], [Na/H], [Al/H], [K/H]
- Diagonal-only tail (4): [α/M], [S/H], [V/H], [Ce/H]

Cross-block entries of ``L`` are exactly zero by construction (see
:class:`BlockCholeskyHead`), so ``Σ = L Lᵀ`` is strictly block-diagonal.
The atmospheric / α / Fe-peak / light block-ordering is the **model's**
internal label order. The feature matrix delivers labels in ``LabelTiers``
(Tier 1 / 2 / 3) order, which is the documentation/release order; the
adapter in #119.2 reorders between the two via
:class:`CovarianceBlockLayout`. Keep the conversion centralised — mis-ordering
between computation and reporting is the kind of bug that silently falsifies
reliability diagrams.

Beta-NLL (Seitzer+2022 β=0.5) lives in :mod:`.losses`. This file is pure
architecture — loss-agnostic.

Statistical interpretation
--------------------------
The head outputs ``(μ, L)`` and the codebase carries them as a per-star
"posterior" for shorthand convenience. Mathematically they are an MLE point
estimate of the conditional mean plus a learned heteroscedastic Gaussian
likelihood factor, *not* a Bayesian posterior in the generative sense. The
empirical-Bayes shrinkage in :mod:`.uncertainty` calibrates the marginal
``σ_pred`` against APOGEE residuals so the pair ``(μ, L)`` covers truth at
the nominal frequencies (68 / 95 / 99 %), but no actual prior over labels
is imposed — the "prior" referred to in the σ-inflation gate (see
``release._PER_ELEMENT_SIGMA_INFLATED_THRESHOLD``) is the marginal label
distribution under training conditional on auxiliary features, i.e. the
regression head's empirical inductive bias when conditional mutual
information CMI(spectrum; label | aux) → 0. Methods-paper text must reflect
this distinction explicitly: we deliver calibrated frequentist confidence
ellipsoids dressed as posteriors-of-convenience, not generative-Bayesian
posteriors. See B1/B7 of the bayesian-rigor review (2026-04-28).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F  # noqa: N812 — PyTorch-idiomatic alias
from torch import nn

_MIN_CHOLESKY_DIAG: float = 1e-4
"""Floor on Cholesky diagonal entries.

Prevents ``log det Σ`` from diverging on a freshly-initialized head where
softplus output can drift arbitrarily close to zero. Small relative to any
physically-plausible label σ (e.g. σ(Teff) ≳ 50 K in z-scored units is ~0.05,
100× the floor).
"""


@dataclass(frozen=True, slots=True)
class CovarianceBlockLayout:
    """Block-Cholesky structure + label ordering for the 21-D label vector.

    Two orderings are carried as first-class citizens:

    - ``label_order_block`` — the **model's** internal order. Labels within a
      physics block are contiguous; blocks appear in the order
      ``(atmospheric, α, Fe-peak, light, diagonal-only)``. This is the order
      of the ``mu`` and ``L`` tensors emitted by :class:`BlockCholeskyHead`.
    - ``label_order_human`` — the **documentation/release** order, typically
      the ``LabelTiers`` ordering (Tier 1, Tier 2, Tier 3). This is what
      users, plots, and output parquets should see.

    Conversions are one-liner tensor index_selects via
    :meth:`reorder_block_to_human` and :meth:`reorder_human_to_block`. Use
    named conversion methods at the boundary of every component that touches
    labels — never assume an implicit order.

    Parameters
    ----------
    block_sizes
        Dense Cholesky block sizes in model order, e.g. ``(3, 4, 4, 6)``.
    n_diagonal_only
        Number of labels appended at the end with diagonal-only covariance
        (no cross-correlation within or between the tail and other blocks).
    label_order_block
        Label names in model order. Length must equal
        ``sum(block_sizes) + n_diagonal_only``.
    label_order_human
        Label names in human/documentation order. Must be a permutation of
        ``label_order_block``.
    block_names
        Optional display names for the dense blocks (atmospheric, α, etc.).
        Length must equal ``len(block_sizes)`` when provided. Cosmetic only —
        not used in forward passes.
    """

    block_sizes: tuple[int, ...]
    n_diagonal_only: int
    label_order_block: tuple[str, ...]
    label_order_human: tuple[str, ...]
    block_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.block_sizes:
            raise ValueError("block_sizes must be non-empty")
        if any(k <= 0 for k in self.block_sizes):
            raise ValueError(f"block_sizes must all be positive, got {self.block_sizes}")
        if self.n_diagonal_only < 0:
            raise ValueError(f"n_diagonal_only must be >= 0, got {self.n_diagonal_only}")
        n = sum(self.block_sizes) + self.n_diagonal_only
        if len(self.label_order_block) != n:
            raise ValueError(
                f"label_order_block length {len(self.label_order_block)} != total labels {n}"
            )
        if len(self.label_order_human) != n:
            raise ValueError(
                f"label_order_human length {len(self.label_order_human)} != total labels {n}"
            )
        if set(self.label_order_block) != set(self.label_order_human):
            raise ValueError(
                "label_order_block and label_order_human must be permutations of each other"
            )
        if len(set(self.label_order_block)) != n:
            raise ValueError("label_order_block contains duplicates")
        if self.block_names is not None and len(self.block_names) != len(self.block_sizes):
            raise ValueError(
                f"block_names length {len(self.block_names)} != "
                f"number of blocks {len(self.block_sizes)}"
            )

    @property
    def n_labels(self) -> int:
        return sum(self.block_sizes) + self.n_diagonal_only

    @property
    def n_blocks(self) -> int:
        return len(self.block_sizes)

    @property
    def block_to_human_perm(self) -> torch.Tensor:
        """Index tensor ``p`` such that ``human[i] = block[p[i]]``.

        Use via ``tensor.index_select(dim, perm)``: the output tensor is in
        human order, with ``output[..., i] = tensor[..., p[i]]``.
        """
        lookup = {name: i for i, name in enumerate(self.label_order_block)}
        return torch.tensor(
            [lookup[name] for name in self.label_order_human],
            dtype=torch.long,
        )

    @property
    def human_to_block_perm(self) -> torch.Tensor:
        """Index tensor ``p`` such that ``block[i] = human[p[i]]``.

        The inverse of :attr:`block_to_human_perm`.
        """
        lookup = {name: i for i, name in enumerate(self.label_order_human)}
        return torch.tensor(
            [lookup[name] for name in self.label_order_block],
            dtype=torch.long,
        )

    def reorder_block_to_human(
        self,
        tensor: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        """Reorder a block-ordered tensor along ``dim`` to human order."""
        perm = self.block_to_human_perm.to(tensor.device)
        return tensor.index_select(dim, perm)

    def reorder_human_to_block(
        self,
        tensor: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        """Reorder a human-ordered tensor along ``dim`` to model/block order."""
        perm = self.human_to_block_perm.to(tensor.device)
        return tensor.index_select(dim, perm)

    def to_dict(self) -> dict[str, object]:
        """Serialise for checkpoint storage. Round-trips via :meth:`from_dict`."""
        return {
            "block_sizes": list(self.block_sizes),
            "n_diagonal_only": self.n_diagonal_only,
            "label_order_block": list(self.label_order_block),
            "label_order_human": list(self.label_order_human),
            "block_names": list(self.block_names) if self.block_names else None,
        }

    @classmethod
    def from_dict(cls, blob: dict[str, object]) -> "CovarianceBlockLayout":
        return cls(
            block_sizes=tuple(blob["block_sizes"]),  # type: ignore[arg-type]
            n_diagonal_only=int(blob["n_diagonal_only"]),  # type: ignore[arg-type]
            label_order_block=tuple(blob["label_order_block"]),  # type: ignore[arg-type]
            label_order_human=tuple(blob["label_order_human"]),  # type: ignore[arg-type]
            block_names=(
                tuple(blob["block_names"])  # type: ignore[arg-type]
                if blob.get("block_names")
                else None
            ),
        )

    @classmethod
    def anonymous(
        cls,
        block_sizes: Sequence[int],
        n_diagonal_only: int = 0,
    ) -> "CovarianceBlockLayout":
        """Build a layout with placeholder label names — for tests / sizing only.

        Label names are ``("label_0", "label_1", ...)`` and the human order
        equals the block order, so ``reorder_*`` methods become identities.
        """
        n = sum(block_sizes) + n_diagonal_only
        names = tuple(f"label_{i}" for i in range(n))
        return cls(
            block_sizes=tuple(block_sizes),
            n_diagonal_only=n_diagonal_only,
            label_order_block=names,
            label_order_human=names,
            block_names=None,
        )


# --- Default layout for Pipeline 1 ------------------------------------------

_ATMOSPHERIC: tuple[str, ...] = ("teff_apogee", "logg_apogee", "mh_apogee")
_ALPHA: tuple[str, ...] = (
    "mg_h_apogee",
    "si_h_apogee",
    "ca_h_apogee",
    "ti_h_apogee",
)
_FE_PEAK: tuple[str, ...] = (
    "fe_h_apogee",
    "mn_h_apogee",
    "ni_h_apogee",
    "cr_h_apogee",
)
_LIGHT: tuple[str, ...] = (
    "c_h_apogee",
    "n_h_apogee",
    "o_h_apogee",
    "na_h_apogee",
    "al_h_apogee",
    "k_h_apogee",
)
_DIAGONAL_ONLY: tuple[str, ...] = (
    "alpha_m_apogee",
    "s_h_apogee",
    "v_h_apogee",
    "ce_h_apogee",
)

# Human / documentation order mirrors ``data.LabelTiers`` (Tier 1 + 2 + 3).
# Kept here as a literal so the model module does not import from data —
# validated against LabelTiers at test time.
_TIER1_HUMAN: tuple[str, ...] = ("teff_apogee", "logg_apogee", "mh_apogee")
_TIER2_HUMAN: tuple[str, ...] = (
    "fe_h_apogee",
    "alpha_m_apogee",
    "mg_h_apogee",
    "c_h_apogee",
    "n_h_apogee",
)
_TIER3_HUMAN: tuple[str, ...] = (
    "o_h_apogee",
    "na_h_apogee",
    "al_h_apogee",
    "si_h_apogee",
    "s_h_apogee",
    "k_h_apogee",
    "ca_h_apogee",
    "ti_h_apogee",
    "v_h_apogee",
    "cr_h_apogee",
    "mn_h_apogee",
    "ni_h_apogee",
    "ce_h_apogee",
)


def default_pipeline1_layout() -> CovarianceBlockLayout:
    """The canonical 4-block + diagonal-only layout for the main pipeline.

    Blocks (3, 4, 4, 6) are physics-motivated; the 4-label diagonal-only tail
    carries [α/M] (redundant with the α-block individual abundances) plus three
    Tier-3 audit-only abundances ([S/H], [V/H], [Ce/H]) whose APOGEE SNR is
    insufficient for robust off-diagonal covariance estimation.

    Human order matches :class:`LabelTiers` (Tier 1 + Tier 2 + Tier 3).
    """
    return CovarianceBlockLayout(
        block_sizes=(3, 4, 4, 6),
        n_diagonal_only=4,
        label_order_block=(_ATMOSPHERIC + _ALPHA + _FE_PEAK + _LIGHT + _DIAGONAL_ONLY),
        label_order_human=(_TIER1_HUMAN + _TIER2_HUMAN + _TIER3_HUMAN),
        block_names=("atmospheric", "alpha", "fe_peak", "light"),
    )


# --- 5-label variant: {Teff, log g, [M/H], [α/M], [Mg/H]} -------------------
# Rationale: 21-label training splits encoder capacity across weakly-supported
# Tier-3 abundances that corrupt the representation used for Tier 1. The
# 5-label variant restricts supervision to labels with strong XP-resolution
# signal: atmospherics (Tier 1) + [α/M] (global α enhancement) + [Mg/H]
# (strongest individual-element signal via the Mg b triplet + MgH band).
# [Mg/H] rather than [Mg/Fe] because the APOGEE native output is [Mg/H];
# [Fe/H] is still implicit via [M/H].
_FIVE_LABEL: tuple[str, ...] = (
    "teff_apogee",
    "logg_apogee",
    "mh_apogee",
    "alpha_m_apogee",
    "mg_h_apogee",
)


def five_label_block_layout() -> CovarianceBlockLayout:
    """Single full 5x5 Cholesky block for the {Teff, logg, [M/H], [α/M], [Mg/H]} variant.

    Block and human orders match :class:`LabelTiers.five_label`. With only 5
    labels, a single dense block (15 Cholesky parameters) is trivially trainable
    and captures every cross-label correlation. This is the "v5 working"
    layout that produced visible disc α-bimodality with the SupCon raw-units
    kernel + ARI=0.1 + identity-projection encoder.
    """
    return CovarianceBlockLayout(
        block_sizes=(5,),
        n_diagonal_only=0,
        label_order_block=_FIVE_LABEL,
        label_order_human=_FIVE_LABEL,
        block_names=("all_5",),
    )


_TWO_LABEL: tuple[str, ...] = ("mh_apogee", "alpha_m_apogee")


def two_label_block_layout() -> CovarianceBlockLayout:
    """Single full 2×2 Cholesky block for the TESS_ML-matched {[M/H], [α/M]} variant.

    Block and human orders match :class:`LabelTiers.two_label`. Three Cholesky
    parameters — captures the chemistry-plane correlation that drives bimodal
    disc segregation.
    """
    return CovarianceBlockLayout(
        block_sizes=(2,),
        n_diagonal_only=0,
        label_order_block=_TWO_LABEL,
        label_order_human=_TWO_LABEL,
        block_names=("all_2",),
    )


# --- ModelConfig / Encoder / Head / Wrapper ---------------------------------


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architectural knobs.

    ``input_dim`` comes from
    :attr:`~arqueogal.xp_abundances.main.data.FeatureLayout.input_dim`;
    ``block_layout`` is the physics-motivated 4-block Cholesky structure
    (see :func:`default_pipeline1_layout`). Keep ``latent_dim`` ≥ the largest
    block size so the Cholesky head isn't under-parametrised.
    """

    input_dim: int
    block_layout: CovarianceBlockLayout = field(default_factory=default_pipeline1_layout)
    latent_dim: int = 32
    trunk_hidden: tuple[int, ...] = (256, 128)
    head_hidden: int = 128
    dropout: float = 0.10
    head_dropout: float = 0.05

    @property
    def n_labels(self) -> int:
        return self.block_layout.n_labels


class Encoder(nn.Module):
    """XP coefficient trunk + L2-normalised contrastive projection head.

    Returns ``(h, z)`` where ``h`` is the trunk embedding consumed by the
    supervised head and ``z`` is the projected unit-norm vector used for
    contrastive loss. Ported verbatim from the TESS_ML prototype — LayerNorm
    throughout (robust to small-batch regimes on 6 GB VRAM), GELU activations,
    dropout only on the first trunk layer.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden: tuple[int, ...] = (256, 128),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden, latent_dim]
        layers: list[nn.Module] = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=False)):
            layers.append(nn.Linear(a, b))
            is_last = i == len(dims) - 2
            if not is_last:
                layers.append(nn.LayerNorm(b))
                layers.append(nn.GELU())
                if dropout > 0 and i == 0:
                    layers.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*layers)
        # 2-layer SimCLR/SupCon-style projection. The L2-normalised trunk
        # output (z = F.normalize(h)) collapses magnitude information onto
        # the unit sphere; high-variance label directions (Teff, log g)
        # dominate the angular geometry and squeeze low-variance label
        # directions (e.g. [α/M]) onto a small angular subspace where the
        # contrastive gradient is too weak to discriminate. The projection
        # MLP gives SupCon a learnable transform from h into a separate
        # contrastive space z, while the supervised regression head keeps
        # reading the magnitude-preserving h. The 2-layer (Linear → GELU →
        # Linear) form is the canonical SupCon projection used by Khosla
        # et al. 2020 and the TESS_ML reference.
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        z = F.normalize(self.proj(h), dim=-1)
        return h, z


class BlockCholeskyHead(nn.Module):
    """Predicts ``(mu, L)`` with block-structured lower-triangular ``L``.

    The Cholesky factor is strictly **block-diagonal** by construction — no
    cross-block entries in the lower-triangular region. Within each of the
    ``len(block_sizes)`` dense blocks, ``L`` is lower-triangular with softplus
    + floor on the diagonal and unconstrained off-diagonals. The trailing
    ``n_diagonal_only`` labels have softplus-floored diagonals and zero
    off-diagonals.

    Cross-block zero is enforced via advanced-indexing writes into a
    zero-initialised ``L``: rows/cols for each block are generated by
    ``tril_indices(k, k)`` and offset into the global label space, so only
    within-block positions are ever assigned — cross-block positions remain
    at their initial zero value.

    Head capacity: widened from the TESS_ML prototype's ``D → 128 → 64 → 32``
    to ``D → hidden → hidden`` (two LN+GELU layers at the same width). For
    21 labels with the default 4-block + 4-diagonal-only layout, the split
    heads emit 21 means + 51 Cholesky-factor parameters, so the bottleneck
    of ``hidden=128 → 72`` outputs is comfortable.
    """

    def __init__(
        self,
        latent_dim: int,
        block_sizes: Sequence[int],
        n_diagonal_only: int = 0,
        hidden: int = 128,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        block_sizes = tuple(block_sizes)
        if not block_sizes:
            raise ValueError("block_sizes must be non-empty")
        if any(k <= 0 for k in block_sizes):
            raise ValueError(f"block_sizes must all be positive, got {block_sizes}")
        if n_diagonal_only < 0:
            raise ValueError(f"n_diagonal_only must be >= 0, got {n_diagonal_only}")

        self.block_sizes = block_sizes
        self.n_diagonal_only = n_diagonal_only
        self.n_labels = sum(block_sizes) + n_diagonal_only

        # Head trunk: widened two-layer MLP at ``hidden`` width.
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(hidden, self.n_labels)

        per_block_tril = [k * (k + 1) // 2 for k in block_sizes]
        n_cov_params = sum(per_block_tril) + n_diagonal_only
        self.cov_head = nn.Linear(hidden, n_cov_params)

        # Concatenated per-block tril indices — offset into the global label
        # space. Writing via ``L[:, all_rows, all_cols] = vals`` touches only
        # within-block lower-triangular positions, so cross-block entries in
        # L remain at their zero-initialised value (==> Σ = LLᵀ is block-
        # diagonal by construction, not by optimisation).
        all_rows: list[torch.Tensor] = []
        all_cols: list[torch.Tensor] = []
        all_diag_mask: list[torch.Tensor] = []
        offset = 0
        for k in block_sizes:
            rows, cols = torch.tril_indices(k, k)
            all_rows.append(rows + offset)
            all_cols.append(cols + offset)
            all_diag_mask.append(rows == cols)
            offset += k
        self.register_buffer("_all_rows", torch.cat(all_rows))
        self.register_buffer("_all_cols", torch.cat(all_cols))
        self.register_buffer("_all_diag_mask", torch.cat(all_diag_mask))

        if n_diagonal_only > 0:
            diag_only_idx = torch.arange(n_diagonal_only, dtype=torch.long) + offset
            self.register_buffer("_diag_only_indices", diag_only_idx)
        else:
            self.register_buffer("_diag_only_indices", torch.empty(0, dtype=torch.long))

        self._n_block_params = sum(per_block_tril)
        self._n_cov_params = n_cov_params

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.trunk(h)
        mu = self.mean_head(f)
        raw = self.cov_head(f)

        batch = h.shape[0]
        n = self.n_labels
        # Under autocast (bfloat16/float16), ``softplus`` is one of the ops
        # PyTorch promotes to float32 for numerical stability, so ``vals`` can
        # end up fp32 even when ``raw`` is bf16. Allocate ``L`` with the
        # computed dtype, not ``h.dtype``, so the scatter assignment matches.
        if self._n_block_params > 0:
            block_raw = raw[:, : self._n_block_params]
            pos_diag = F.softplus(block_raw) + _MIN_CHOLESKY_DIAG
            vals = torch.where(
                self._all_diag_mask,
                pos_diag,
                block_raw.to(pos_diag.dtype),
            )
            out_dtype = vals.dtype
        else:
            vals = None
            out_dtype = raw.dtype
        L = torch.zeros(batch, n, n, device=h.device, dtype=out_dtype)

        if vals is not None:
            L[:, self._all_rows, self._all_cols] = vals

        # Diagonal-only tail: softplus+floor, no off-diagonals.
        if self.n_diagonal_only > 0:
            diag_raw = raw[:, self._n_block_params :]
            pos_diag = F.softplus(diag_raw) + _MIN_CHOLESKY_DIAG
            L[:, self._diag_only_indices, self._diag_only_indices] = pos_diag.to(out_dtype)

        return mu, L


class EvolutionaryStageHead(nn.Module):
    """4-way soft classifier for evolutionary stage: RGB, HeCB, OOD_evolved, OOD_unevolved.

    This is a **diagnostic head**, not a gating mechanism. It emits soft probabilities
    that are included in the release parquet for downstream analysis. The head shares the
    encoder latent space ``h`` and is trained jointly with the main abundance regression
    head via a cross-entropy loss with weight 0.05.

    Architecture: ``latent_dim → 64 → 32 → 4`` with LayerNorm + GELU, matching the
    lightweight diagnostic pattern.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden: int = 64,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 4),  # 4 classes: RGB, HeCB, OOD_evolved, OOD_unevolved
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Return logits for 4-way classification.

        Parameters
        ----------
        h
            Trunk hidden state, shape ``(B, latent_dim)``.

        Returns
        -------
        Logits of shape ``(B, 4)``. Apply softmax to get soft probabilities.
        """
        return self.mlp(h)


class XpAbundanceModel(nn.Module):
    """Convenience wrapper: encoder + block-Cholesky head + optional evol-stage head.

    Not an AutoEncoder — just a composition returning everything the training
    loop needs from one forward pass. ``z`` (the L2-normalised projection) is
    computed unconditionally; callers that only need supervised outputs can
    ignore it at negligible cost.

    When ``include_evol_stage_head=True`` (v1.1+), the model also includes a
    4-way evolutionary-stage diagnostic head that emits soft probabilities.
    """

    def __init__(self, config: ModelConfig, include_evol_stage_head: bool = False) -> None:
        super().__init__()
        self.config = config
        self.include_evol_stage_head = include_evol_stage_head
        self.encoder = Encoder(
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            hidden=config.trunk_hidden,
            dropout=config.dropout,
        )
        self.head = BlockCholeskyHead(
            latent_dim=config.latent_dim,
            block_sizes=config.block_layout.block_sizes,
            n_diagonal_only=config.block_layout.n_diagonal_only,
            hidden=config.head_hidden,
            dropout=config.head_dropout,
        )
        if include_evol_stage_head:
            self.evol_stage_head = EvolutionaryStageHead(
                latent_dim=config.latent_dim,
                hidden=64,
                dropout=config.head_dropout,
            )
        else:
            self.evol_stage_head = None

    @property
    def block_layout(self) -> CovarianceBlockLayout:
        return self.config.block_layout

    def forward(
        self,
        x: torch.Tensor,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Forward pass returning (mu, L, h, z) or (mu, L, h, z, evol_logits).

        Returns tuple of 4 elements when evol_stage_head is disabled (v1.0 compat),
        or tuple of 5 elements when enabled (v1.1+). Callers should check the
        include_evol_stage_head flag or use conditional unpacking.
        """
        h, z = self.encoder(x)
        mu, L = self.head(h)
        if self.include_evol_stage_head:
            evol_logits = self.evol_stage_head(h)
            return mu, L, h, z, evol_logits
        return mu, L, h, z


__all__ = [
    "BlockCholeskyHead",
    "CovarianceBlockLayout",
    "Encoder",
    "EvolutionaryStageHead",
    "ModelConfig",
    "XpAbundanceModel",
    "default_pipeline1_layout",
    "five_label_block_layout",
]
