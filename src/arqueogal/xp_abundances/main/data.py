"""Data contract + loader for Pipeline 1 training — ``xp_abundances.main``.

Three things live here, chosen so inspection code, training code, and tests all
share one vocabulary:

- :class:`FeatureLayout` — exactly which columns of
  ``pipeline1_features_stream1.parquet`` feed the encoder, and in what order.
- :class:`LabelTiers` — the APOGEE DR19 labels the network predicts, split into
  the §3.2 tiers that drive block-structured Cholesky covariance and
  release-time gating. Per the DESIGN, tiers are a **release** decision, not a
  training-time exclusion — the network learns every label jointly.
- :class:`XpAbundanceDataset` — map-style torch Dataset wrapping preloaded
  arrays; and :func:`load_arrays` — selective column reader used to build it.

The feature matrix ships flat scalar columns for the XP Hermite coefficients:
``bp_coef_norm_1..54`` and ``rp_coef_norm_1..54`` as shape coefficients,
``bp_c0_z``/``rp_c0_z`` as the log10+z-scored absolute-scale scalars. The
trivial ``bp_coef_norm_0 ≡ 1`` is not stored. Raw ``bp_coef_0..54`` /
``rp_coef_0..54`` remain on disk as diagnostic-only (audit path opt-in) and are
**not** part of the default ML input.

Stratified splits (:func:`stratified_split_ids`) use quantile bins jointly on
``([Fe/H]_apogee, Teff_apogee, |b_deg|)`` per the DESIGN and return
``source_id`` arrays, not integer row indices — carrying the IDs makes
train/val/test membership survive parquet re-writes and shard changes.

See ``src/arqueogal/xp_abundances/main/DESIGN.md`` for the frozen 2026-04-18
feature contract and the rationale for each departure from the earlier
list-typed-array layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# --- Feature layout ----------------------------------------------------------

DEFAULT_XP_COEF_INDICES: Final[tuple[int, ...]] = tuple(range(1, 55))
"""Which Hermite-coefficient indices feed the encoder (default: 1..54).

Index 0 is the absolute-scale coefficient — after c0-normalization the
``bp_coef_norm_0`` column is trivially 1.0 and is not stored in the parquet.
The absolute-scale information survives via :data:`DEFAULT_XP_SCALAR_COLS`
(``bp_c0_z``/``rp_c0_z``).

For the 43-D noise-floor truncation studied in research_brief §3.1, override
this with ``tuple(range(1, 20))`` on BP and ``tuple(range(1, 23))`` on RP —
see ``FeatureLayout.truncated_43d()``.
"""

DEFAULT_XP_SCALAR_COLS: Final[tuple[str, ...]] = ("bp_c0_z", "rp_c0_z")
"""Log10 + z-scored absolute c0 scalars, one per band."""

DEFAULT_RESIDUAL_COLS: Final[tuple[str, ...]] = (
    "reprojection_residual_rms",
    "reprojection_residual_rms_bp",
    "reprojection_residual_rms_rp",
)
"""Per-band + combined Hermite-reprojection RMS, fed as ML features.

Per DESIGN.md: residuals are encoder **input features** (not sample weights).
The network learns to inflate predicted uncertainty on high-residual stars;
attribution is auditable via the heteroscedastic head output.
"""

DEFAULT_AUX_COLS: Final[tuple[str, ...]] = (
    # Gaia photometry (Riello+2021-corrected)
    "g_mag", "bp_mag", "rp_mag",
    "bp_rp", "bp_g", "g_rp",
    # Gaia astrometry (Lindegren+2021 zpt-corrected variant)
    "parallax", "parallax_error", "parallax_corr", "ruwe",
    # Bailer-Jones+2021 photogeometric distance triple
    "r_med_photogeo", "r_lo_photogeo", "r_hi_photogeo",
    # IR photometry
    "j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag",
    # Extinction priors (multi-column — model picks which prior to trust per star)
    "av_edenhofer", "av_sfd", "av_lallement",
    "av_nbhd_median", "av_nbhd_std",
    "ag_gspphot", "ag_gspphot_lower", "ag_gspphot_upper",
)
"""Scalar auxiliary columns appended after XP features.

Column order is deterministic — encoder weight meaning depends on it. Any
change goes through the same-commit DESIGN-co-commit discipline.
"""


@dataclass(frozen=True, slots=True)
class FeatureLayout:
    """Declarative contract for the encoder's input vector.

    ``input_dim`` is derived — do not hand-edit. The flattening order is:
    BP shape coefs (indices ``xp_bp_indices``) → RP shape coefs
    (``xp_rp_indices``) → XP scalars (``xp_scalar_cols``) → residuals
    (``residual_cols``) → auxiliaries (``aux_cols``). Keep this order stable
    across training and inference or the encoder weights become meaningless.
    """

    xp_bp_indices: tuple[int, ...] = DEFAULT_XP_COEF_INDICES
    xp_rp_indices: tuple[int, ...] = DEFAULT_XP_COEF_INDICES
    xp_scalar_cols: tuple[str, ...] = DEFAULT_XP_SCALAR_COLS
    residual_cols: tuple[str, ...] = DEFAULT_RESIDUAL_COLS
    aux_cols: tuple[str, ...] = DEFAULT_AUX_COLS

    @classmethod
    def truncated_43d(cls, **overrides) -> "FeatureLayout":
        """Noise-floor-truncated layout: BP[1:20] + RP[1:23] (research_brief §3.1)."""
        kwargs = {
            "xp_bp_indices": tuple(range(1, 20)),
            "xp_rp_indices": tuple(range(1, 23)),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    @property
    def bp_coef_cols(self) -> tuple[str, ...]:
        return tuple(f"bp_coef_norm_{i}" for i in self.xp_bp_indices)

    @property
    def rp_coef_cols(self) -> tuple[str, ...]:
        return tuple(f"rp_coef_norm_{i}" for i in self.xp_rp_indices)

    @property
    def input_dim(self) -> int:
        return (
            len(self.bp_coef_cols)
            + len(self.rp_coef_cols)
            + len(self.xp_scalar_cols)
            + len(self.residual_cols)
            + len(self.aux_cols)
        )

    @property
    def all_required_columns(self) -> tuple[str, ...]:
        return (
            *self.bp_coef_cols,
            *self.rp_coef_cols,
            *self.xp_scalar_cols,
            *self.residual_cols,
            *self.aux_cols,
        )


# --- Label tiers -------------------------------------------------------------

_TIER1_APOGEE: Final[tuple[str, ...]] = (
    "teff_apogee", "logg_apogee", "mh_apogee",
)
_TIER2_APOGEE: Final[tuple[str, ...]] = (
    "fe_h_apogee", "alpha_m_apogee", "mg_h_apogee", "c_h_apogee", "n_h_apogee",
)
_TIER3_APOGEE: Final[tuple[str, ...]] = (
    "o_h_apogee", "na_h_apogee", "al_h_apogee", "si_h_apogee",
    "s_h_apogee", "k_h_apogee", "ca_h_apogee", "ti_h_apogee",
    "v_h_apogee", "cr_h_apogee", "mn_h_apogee", "ni_h_apogee",
    "ce_h_apogee",
)
"""Per-element [X/H]_APOGEE labels classified by research_brief §3.2.

Tier-1 is the ASPCAP atmospheric trio {Teff, log g, [M/H]} — reliable per-star
with meaningful uncertainties, always finite on ``flag_bad == 0`` rows.
Tier-2 holds the per-element chemistry (starting with ``fe_h_apogee``, which
can legitimately be NaN in DR19 even on otherwise-sound stars) released with
uncertainty inflation + RGB gate; Tier-3 is population-level only. The network
learns all of them — these tuples wire the block-structured Cholesky head and
the release gates, nothing else.
"""


@dataclass(frozen=True, slots=True)
class LabelTiers:
    """Which APOGEE labels belong to which release tier.

    ``tier1``/``tier2``/``tier3`` are tuples of column names. Order inside each
    tuple matters: it fixes the block layout of the predicted covariance and
    the column ordering of the label matrix ``Y``.
    """

    tier1: tuple[str, ...] = _TIER1_APOGEE
    tier2: tuple[str, ...] = _TIER2_APOGEE
    tier3: tuple[str, ...] = _TIER3_APOGEE

    @classmethod
    def five_label(cls) -> "LabelTiers":
        """Reduced tier set — {Teff, log g, [M/H], [α/M], [Mg/H]}.

        Use with :func:`~.model.five_label_block_layout` (single full 5×5 block).
        [Mg/H] replaces the 21-label variant's element specifics because Mg has
        the strongest individual line signal in XP (Mg b triplet + MgH band);
        [α/M] is retained as a global α enhancement. [Fe/H] is implicit via [M/H].
        """
        return cls(
            tier1=("teff_apogee", "logg_apogee", "mh_apogee"),
            tier2=("alpha_m_apogee", "mg_h_apogee"),
            tier3=(),
        )

    @property
    def all_labels(self) -> tuple[str, ...]:
        return (*self.tier1, *self.tier2, *self.tier3)

    @property
    def n_labels(self) -> int:
        return len(self.all_labels)

    @property
    def tier_sizes(self) -> tuple[int, int, int]:
        return (len(self.tier1), len(self.tier2), len(self.tier3))

    def label_error_columns(self) -> tuple[str, ...]:
        """APOGEE label-uncertainty columns — ``"teff_apogee"`` → ``"e_teff_apogee"``."""
        return tuple(f"e_{name}" for name in self.all_labels)


# --- Selective column loading ------------------------------------------------

def load_arrays(
    parquet_path: Path | str,
    layout: FeatureLayout,
    tiers: LabelTiers,
    *,
    include_label_errors: bool = True,
    include_source_id: bool = True,
    dtype: np.dtype = np.float32,
) -> dict[str, np.ndarray]:
    """Read only the columns the model needs, flatten, return a dict of arrays.

    Returns
    -------
    dict with keys:
        ``X`` : (N, input_dim) float array — encoder input (column order per
                :meth:`FeatureLayout.all_required_columns`).
        ``Y`` : (N, n_labels) float array — label targets in tier order.
        ``sigma_Y`` : (N, n_labels) float array, only if ``include_label_errors``.
        ``source_id`` : (N,) int64 array, only if ``include_source_id``.
    """
    feature_cols = list(layout.all_required_columns)
    label_cols = list(tiers.all_labels)
    cols = [*feature_cols, *label_cols]
    if include_label_errors:
        cols.extend(tiers.label_error_columns())
    if include_source_id:
        cols.append("source_id")

    df = pd.read_parquet(parquet_path, columns=cols)

    X = np.column_stack(
        [df[c].to_numpy(dtype=dtype) for c in feature_cols]
    ) if feature_cols else np.empty((len(df), 0), dtype=dtype)
    Y = np.column_stack([df[c].to_numpy(dtype=dtype) for c in label_cols])

    out: dict[str, np.ndarray] = {"X": X, "Y": Y}
    if include_label_errors:
        err_cols = list(tiers.label_error_columns())
        out["sigma_Y"] = np.column_stack([df[c].to_numpy(dtype=dtype) for c in err_cols])
    if include_source_id:
        out["source_id"] = df["source_id"].to_numpy(dtype=np.int64)
    return out


# --- Stratified split --------------------------------------------------------

DEFAULT_STRAT_COLS: Final[tuple[str, str, str]] = ("fe_h_apogee", "teff_apogee", "b_deg")
"""Quantile-stratify on [Fe/H], Teff, and |b|.

``b_deg`` is Galactic latitude; if absent, :func:`stratified_split_ids`
falls back to |dec| — a good-enough proxy for the training split's
geographic balance.
"""


def stratified_split_ids(
    df: pd.DataFrame,
    *,
    fracs: tuple[float, float, float] = (0.70, 0.15, 0.15),
    strat_cols: tuple[str, str, str] = DEFAULT_STRAT_COLS,
    n_quantile_bins: int = 4,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Quantile-stratified 3-way split returning ``source_id`` arrays.

    Joint-quantile-binning on ``strat_cols``: each star is assigned a cell
    label in the Cartesian product of ``n_quantile_bins`` quantiles per
    column. Within each cell, rows are randomly partitioned into
    ``train / val / test`` with the given fractions.

    Returns
    -------
    dict with keys ``"train"``, ``"val"``, ``"test"``, each an int64 array of
    ``source_id`` values.
    """
    if not np.isclose(sum(fracs), 1.0):
        raise ValueError(f"fracs must sum to 1.0, got {fracs}")
    if "source_id" not in df.columns:
        raise KeyError("stratified_split_ids requires a 'source_id' column")

    rng = np.random.default_rng(seed)
    n = len(df)

    cell_codes = np.zeros(n, dtype=np.int64)
    for col in strat_cols:
        values = _lookup_strat_column(df, col)
        binned = _quantile_bin(values, n_quantile_bins, rng)
        cell_codes = cell_codes * n_quantile_bins + binned

    assignments = np.empty(n, dtype=object)
    f_train, f_val, _ = fracs
    for cell in np.unique(cell_codes):
        idx = np.flatnonzero(cell_codes == cell)
        rng.shuffle(idx)
        n_cell = len(idx)
        n_train = int(round(n_cell * f_train))
        n_val = int(round(n_cell * (f_train + f_val))) - n_train
        assignments[idx[:n_train]] = "train"
        assignments[idx[n_train:n_train + n_val]] = "val"
        assignments[idx[n_train + n_val:]] = "test"

    source_ids = df["source_id"].to_numpy(dtype=np.int64)
    return {
        split: source_ids[assignments == split].copy()
        for split in ("train", "val", "test")
    }


def _lookup_strat_column(df: pd.DataFrame, col: str) -> np.ndarray:
    """Resolve a stratification column, falling back to |dec| if ``b_deg`` missing."""
    if col == "b_deg" and col not in df.columns:
        for fallback in ("dec_deg", "dec"):
            if fallback in df.columns:
                return np.abs(df[fallback].to_numpy(dtype=np.float64))
    if col not in df.columns:
        raise KeyError(f"stratification column {col!r} not in frame")
    values = df[col].to_numpy(dtype=np.float64)
    return np.abs(values) if col.startswith("b_") else values


def _quantile_bin(values: np.ndarray, n_bins: int, rng: np.random.Generator) -> np.ndarray:
    """Assign ``values`` into ``n_bins`` quantile bins; NaNs get a random bin."""
    finite = np.isfinite(values)
    out = np.zeros(len(values), dtype=np.int64)
    if finite.any():
        q = np.quantile(values[finite], np.linspace(0, 1, n_bins + 1)[1:-1])
        out[finite] = np.digitize(values[finite], q)
    if (~finite).any():
        out[~finite] = rng.integers(0, n_bins, size=int((~finite).sum()))
    return np.clip(out, 0, n_bins - 1)


# --- Label scaler ------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LabelScaler:
    """Per-label mean/std standardiser fit on the training partition only.

    The XP abundance model's head emits near-zero outputs at init — fine for
    labels that are natively near-zero ([X/H] abundances, std ~0.3 dex), but
    catastrophic for labels in raw physical units (Teff in Kelvin, log g in
    dex). Without standardisation the regressor cannot learn Tier-1 offsets
    from a standard AdamW init; per-label means are stuck at ≈ 0 while truth
    ≈ 4600 K. Run A fine-tune surfaced this as a 14-σ Teff bias after #134.

    The scaler stores its parameters in ``label_names`` order. Covariance
    un-scaling follows ``Σ_raw[i,j] = scale[i] * scale[j] * Σ_scaled[i,j]``,
    which for the Cholesky factor is ``L_raw = diag(scale) @ L_scaled``.

    NaN labels (missing per-element abundances for some stars) are ignored
    when computing per-label statistics; ``transform`` preserves NaN so the
    β-NLL loss's missing-label mask stays meaningful.
    """

    mean: np.ndarray  # (n_labels,) float32, in ``label_names`` order
    scale: np.ndarray  # (n_labels,) float32, strictly positive, same order
    label_names: tuple[str, ...]

    @classmethod
    def fit(
        cls,
        y: np.ndarray,
        label_names: tuple[str, ...] | list[str],
        *,
        eps: float = 1e-8,
    ) -> "LabelScaler":
        """Fit per-label mean / std on finite entries; label order matches ``y``."""
        if y.ndim != 2:
            raise ValueError(f"expected 2-D label matrix, got shape {y.shape}")
        if y.shape[1] != len(label_names):
            raise ValueError(
                f"y has {y.shape[1]} columns but label_names has {len(label_names)} entries",
            )
        mean = np.zeros(y.shape[1], dtype=np.float32)
        scale = np.ones(y.shape[1], dtype=np.float32)
        for j in range(y.shape[1]):
            col = y[:, j]
            col = col[np.isfinite(col)]
            if col.size == 0:
                continue
            mean[j] = np.float32(col.mean())
            s = float(col.std(ddof=0))
            scale[j] = np.float32(max(s, eps))
        return cls(mean=mean, scale=scale, label_names=tuple(label_names))

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Standardise ``y`` (same column order as ``label_names``); NaN preserved."""
        return (y - self.mean) / self.scale

    def inverse_mean(self, mu: np.ndarray) -> np.ndarray:
        """Un-scale a mean vector/matrix with columns in ``label_names`` order."""
        return mu * self.scale + self.mean

    def inverse_L(self, L: np.ndarray) -> np.ndarray:
        """Un-scale a Cholesky factor with row/col axis in ``label_names`` order.

        ``L`` shape is either ``(n, n)`` or ``(B, n, n)``. Returns
        ``diag(scale) @ L`` so that the resulting Σ = L_raw @ L_raw.T has
        ``Σ_raw[i,j] = scale[i] * scale[j] * Σ_scaled[i,j]`` — the correct
        covariance transformation for y_raw = scale * y_scaled + mean.
        """
        if L.ndim == 2:
            return L * self.scale.reshape(-1, 1)
        if L.ndim == 3:
            return L * self.scale.reshape(1, -1, 1)
        raise ValueError(f"L must be 2-D or 3-D, got shape {L.shape}")

    def reorder_to(self, new_names: tuple[str, ...] | list[str]) -> "LabelScaler":
        """Return a new scaler with labels permuted to ``new_names`` order."""
        lookup = {name: i for i, name in enumerate(self.label_names)}
        missing = [n for n in new_names if n not in lookup]
        if missing:
            raise KeyError(f"labels not in scaler: {missing}")
        perm = np.asarray([lookup[n] for n in new_names], dtype=np.int64)
        return LabelScaler(
            mean=self.mean[perm].astype(np.float32, copy=True),
            scale=self.scale[perm].astype(np.float32, copy=True),
            label_names=tuple(new_names),
        )

    def is_default(self) -> bool:
        """True iff mean==0 and scale==1 everywhere — the placeholder shape."""
        return bool(np.all(self.mean == 0.0) and np.all(self.scale == 1.0))


# --- Map-style Dataset -------------------------------------------------------

@dataclass
class XpAbundanceDataset(Dataset):
    """In-memory Dataset over preloaded arrays from :func:`load_arrays`.

    Yields ``(X, Y)`` or ``(X, Y, sigma_Y)`` tensors per index. The source_id
    array is retained for diagnostics but not returned per-sample — fetch it
    off the dataset attribute instead.
    """

    X: np.ndarray
    Y: np.ndarray
    sigma_Y: np.ndarray | None = None  # noqa: N815 — σ_Y convention (uppercase Y = matrix)
    source_id: np.ndarray | None = None
    dtype: torch.dtype = torch.float32
    _cache: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.X) != len(self.Y):
            raise ValueError(f"X/Y length mismatch: {len(self.X)} vs {len(self.Y)}")
        if self.sigma_Y is not None and len(self.sigma_Y) != len(self.X):
            raise ValueError(
                f"sigma_Y length mismatch: {len(self.sigma_Y)} vs {len(self.X)}"
            )
        self._cache["X"] = torch.as_tensor(self.X, dtype=self.dtype)
        self._cache["Y"] = torch.as_tensor(self.Y, dtype=self.dtype)
        if self.sigma_Y is not None:
            self._cache["sigma_Y"] = torch.as_tensor(self.sigma_Y, dtype=self.dtype)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        x = self._cache["X"][idx]
        y = self._cache["Y"][idx]
        if "sigma_Y" in self._cache:
            return x, y, self._cache["sigma_Y"][idx]
        return x, y


__all__ = [
    "DEFAULT_AUX_COLS",
    "DEFAULT_RESIDUAL_COLS",
    "DEFAULT_STRAT_COLS",
    "DEFAULT_XP_COEF_INDICES",
    "DEFAULT_XP_SCALAR_COLS",
    "FeatureLayout",
    "LabelScaler",
    "LabelTiers",
    "XpAbundanceDataset",
    "load_arrays",
    "stratified_split_ids",
]
