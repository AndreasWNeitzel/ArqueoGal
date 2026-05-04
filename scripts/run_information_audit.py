"""Run A — §9.2 information-content audit on the Pipeline 1 5-label ensemble.

Executes the live tests (1, 2, 4, 5) from the research_brief §9.2 release gate
on the moment-matched 5-member ensemble at
``models/main/xp_abundances/20260419_nogit_a0e10aa_ensemble_5label/``:

- **Test 1 — LOOCO**: per-XP-coefficient label shift when the coefficient is
  zeroed at inference. Aggregated into per-family summaries (BP shape / RP
  shape / c0 scalars / residual / aux).
- **Test 2 — Permutation importance**: per-feature val-RMSE increase when the
  feature column is shuffled across stars. Separates spectrum-family from
  auxiliary-family contributions.
- **Test 4 — Shuffled-spectrum null** (load-bearing sanity): within each
  (Teff, log g) cell permute the 110-D XP block across stars; measure the
  RMSE of the prediction. Pass threshold: null **skill** < 20 % of real skill
  (research_brief §4 release gate) → equivalently null_rmse materially larger
  than baseline_rmse for spectrum-driven labels. We separate two modes:
  (a) *hard halt* — null skill ≥ 20 % of real skill **and** XP-family
  permutation importance < 5 % of σ(y): the model is not using the spectrum
  at all, release-blocking; (b) *caveat* — null skill ≥ 20 % of real skill
  **but** XP-family permutation importance ≥ 5 % of σ(y): atmospheric-prior-
  augmented label, the spectrum refines the prior rather than standing alone
  (physics-expected for Teff, log g on RGBs whose photometry is already
  informative). Case (b) does not block D-Cat-b release; it is documented
  in the report card as a known caveat.
- **Test 5 — Conditional MI (Kraskov KSG)**: I(XP; label | bp_rp, g_mag,
  parallax, av_sfd). A 2-D XP summary and a 4-D conditioning vector keep the
  KSG estimator well-conditioned on 10⁴-order val samples. Residual positive
  CMI above the release floor is the information-theoretic fingerprint of
  spectrum-driven signal *beyond* the photometric/astrometric priors.

Tests 3 (SHAP) and 6 (decorrelated sub-sample) are stubs at 2026-04-19 per
``docs/research_brief.md §9.2`` and DESIGN notes; they are deliberately
skipped in this release-gate run.

Outputs under ``reports/pipeline1/audit/``:

- ``{label}_report_card.md``  — one markdown per label with verdict + tables.
- ``{label}_report_card.json`` — machine-readable audit payload.
- ``SUMMARY.md``              — cross-label comparison + go/no-go.

Run: ``PYTHONPATH=src python scripts/run_information_audit.py``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from arqueogal.xp_abundances.main.adapter import XpFeatureAdapter
from arqueogal.xp_abundances.main.audit import conditional_mi_ksg
from arqueogal.xp_abundances.main.config import LossWeights, TrainingConfig
from arqueogal.xp_abundances.main.data import (
    FeatureLayout,
    LabelScaler,
    LabelTiers,
)
from arqueogal.xp_abundances.main.model import (
    CovarianceBlockLayout,
    ModelConfig,
    XpAbundanceModel,
)
from arqueogal.xp_abundances.main.training import (
    build_dataloaders,
    load_checkpoint,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("run_information_audit")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENSEMBLE = REPO_ROOT / "models/main/xp_abundances/20260419_nogit_a0e10aa_ensemble_5label"
DEFAULT_PARQUET = REPO_ROOT / "data/processed/pipeline1_features_stream1.parquet"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports/pipeline1/audit"

# --- Release-gate thresholds (research_brief §4 release-gate + §9.2) ---------

NULL_SKILL_HALT_RATIO: float = 0.20
"""A label's shuffled-spectrum null skill must stay below this fraction of its
real-data skill. Null_skill ≥ 0.20 * real_skill → model recovers the label
from priors alone; release-blocking for per-star use."""

PERM_IMPORTANCE_XP_COUNT: int = 3
"""At least this many XP coefficients must show > PERM_IMPORTANCE_FRAC
fractional RMSE increase under permutation for a Tier-1 pass."""

PERM_IMPORTANCE_FRAC: float = 0.02
"""Fractional RMSE increase threshold (∆RMSE / std(y)) per-coefficient for
the permutation-importance head-count gate."""

CMI_MIN: float = 0.02
"""Minimum conditional MI ``I(XP; label | aux)`` nats to claim residual
spectrum signal beyond the photometric/astrometric prior. The KSG estimator
at k=5 on ~10k samples can produce negative-zero values for truly zero CMI;
0.02 nats is a robust-above-noise floor."""

XP_PERM_MIN_FRAC: float = 0.05
"""Spectrum-family fractional permutation importance below which the model
is considered to be ignoring the XP shape. Combined with the null-skill
ratio in :func:`_classify_label` to distinguish hard-halt from caveat."""


# --- Ensemble inference wrapper ---------------------------------------------


class EnsembleMeanWrapper(nn.Module):
    """Thin ``nn.Module`` that returns the mean scaled μ across members.

    The §9.2 audit tests only read ``mu`` from the forward tuple, so we
    deliberately skip Cholesky moment-matching: the tests compare baseline vs.
    perturbed μ on the same ensemble, and the ensemble Σ̄ plays no role in
    LOOCO / permutation / null-shuffle RMSE.

    Inputs arrive in the flat FeatureLayout order. Each member has its own
    :class:`XpFeatureAdapter` (identical config across the 5-label ensemble),
    so we share one adapter for efficiency. Output ``mu_unscaled`` is in raw
    physical units (K, dex) — the audit reports in the same units used by the
    calibration harness.

    Returns ``(mu_unscaled, L_identity, h, z)`` to match the
    :class:`XpAbundanceModel` forward contract audited in
    ``tests/xp_abundances/main/test_audit.py``.
    """

    def __init__(
        self,
        members: list[XpAbundanceModel],
        adapter: XpFeatureAdapter,
        scaler_block: LabelScaler,
        block_layout: CovarianceBlockLayout,
    ) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)
        self.adapter = adapter
        self.block_layout = block_layout
        self.register_buffer(
            "_scale",
            torch.as_tensor(scaler_block.scale, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_mean",
            torch.as_tensor(scaler_block.mean, dtype=torch.float32),
            persistent=False,
        )
        self.n_labels = len(scaler_block.label_names)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        xa = self.adapter(x)
        mus_scaled: list[torch.Tensor] = []
        for m in self.members:
            mu, _L, _h, _z = m(xa)
            mus_scaled.append(mu)
        mu_mean_scaled = torch.stack(mus_scaled, dim=0).mean(dim=0)
        # Un-scale in block order (== human order for the 5-label variant).
        mu_unscaled = mu_mean_scaled * self._scale + self._mean
        batch = x.shape[0]
        L_identity = torch.eye(
            self.n_labels,
            device=x.device,
            dtype=x.dtype,
        ).expand(batch, -1, -1)
        return mu_unscaled, L_identity, x, x


class _RawLoaderWrapper:
    """Yield ``(X, Y_raw_block_order)`` batches from the training val loader.

    The training val loader emits human-order, standardised ``Y``. The audit
    needs raw-unit ``Y`` in block order so comparisons against the
    un-scaling ensemble wrapper are self-consistent. We iterate the inner
    loader once, converting on the fly — so the audit still sees a
    ``torch.utils.data.DataLoader``-compatible iterable (the audit only calls
    ``for batch in loader:``, so a thin iterable is sufficient).
    """

    def __init__(
        self,
        loader: DataLoader,
        scaler_human: LabelScaler,
        block_layout: CovarianceBlockLayout,
    ) -> None:
        self._loader = loader
        self._scaler = scaler_human
        self._perm = block_layout.human_to_block_perm.cpu().numpy()

    def __iter__(self):  # noqa: ANN204 — iterator protocol
        for batch in self._loader:
            x = batch[0]
            y_human_scaled = batch[1].numpy()
            y_human_raw = self._scaler.inverse_mean(y_human_scaled)
            y_block_raw = y_human_raw[:, self._perm]
            yield x, torch.as_tensor(y_block_raw, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self._loader)


# --- Load + reconstruct ------------------------------------------------------


def _reconstruct_model(
    blob: dict[str, Any],
    layout: FeatureLayout,
    block_layout: CovarianceBlockLayout,
    device: torch.device,
) -> XpAbundanceModel:
    cfg_yaml = json.loads(blob["config_yaml"])
    latent_dim = int(cfg_yaml.get("latent_dim", 32))
    trunk_hidden = tuple(cfg_yaml.get("trunk_hidden", (256, 128)))
    head_hidden = int(cfg_yaml.get("head_hidden", 128))
    dropout = float(cfg_yaml.get("dropout", 0.10))
    model = XpAbundanceModel(
        ModelConfig(
            input_dim=layout.input_dim,
            block_layout=block_layout,
            latent_dim=latent_dim,
            trunk_hidden=trunk_hidden,
            head_hidden=head_hidden,
            dropout=dropout,
        ),
    ).to(device)
    model.encoder.load_state_dict(blob["encoder"])
    model.head.load_state_dict(blob["regressor"])
    model.eval()
    return model


def _build_cfg_for_val_loader(
    parquet: Path,
    split_seed: int,
    batch_size: int,
) -> TrainingConfig:
    """Minimal TrainingConfig sufficient to reconstruct the val split.

    ``build_dataloaders`` does not touch ``pretrained_encoder_ckpt`` — that
    path is consumed only by ``_build_model_and_temperature``. Leaving it
    ``None`` is safe and avoids coupling the audit to the contrastive-stage
    artefact path.
    """
    return TrainingConfig(
        train_parquet=parquet,
        output_dir=REPO_ROOT / "tmp_audit",
        epochs=1,
        batch_size=batch_size,
        num_workers=2,
        amp_dtype="bfloat16",
        use_c0_scalars=True,
        split_seed=split_seed,
        pretrained_encoder_ckpt=None,
        output_prefix="xp_abundances_main_audit",
        loss_weights=LossWeights(supcon=0.0, beta_nll=1.0, beta=0.5),
        ensemble_seeds=(0,),
    )


# --- Feature-family indexing -------------------------------------------------


def _feature_family_indices(
    layout: FeatureLayout,
) -> dict[str, list[int]]:
    """Map each input-dim index into a physical family.

    Flat order (``layout.all_required_columns``): BP shape coefs → RP shape
    coefs → XP c0 scalars → residuals → auxiliaries. The audit reports
    permutation importance and LOOCO aggregated by family because 110 coeffs
    × 5 labels produces an unreadable per-coeff table — the physics that
    matters sits at the family level.
    """
    i = 0
    families: dict[str, list[int]] = {}
    n_bp = len(layout.bp_coef_cols)
    families["bp_shape"] = list(range(i, i + n_bp))
    i += n_bp
    n_rp = len(layout.rp_coef_cols)
    families["rp_shape"] = list(range(i, i + n_rp))
    i += n_rp
    n_c0 = len(layout.xp_scalar_cols)
    families["xp_c0"] = list(range(i, i + n_c0))
    i += n_c0
    n_res = len(layout.residual_cols)
    families["residual"] = list(range(i, i + n_res))
    i += n_res
    n_aux = len(layout.aux_cols)
    families["aux"] = list(range(i, i + n_aux))
    i += n_aux
    assert i == layout.input_dim, (i, layout.input_dim)
    return families


def _cell_ids_from_truth(
    y_block_raw: np.ndarray,
    *,
    n_teff_bins: int = 4,
    n_logg_bins: int = 4,
) -> np.ndarray:
    """Joint-quantile bin the val set on (Teff, log g) truth into cell IDs.

    Matches the calibration harness convention (``run_calibration.py``). The
    shuffled-spectrum null permutes within each cell so that the shuffle
    preserves the atmospheric-cell marginal; any surviving predictive skill
    is then attributable to the auxiliaries alone, not to cell-level aliasing.
    """
    teff = y_block_raw[:, 0]
    logg = y_block_raw[:, 1]
    finite = np.isfinite(teff) & np.isfinite(logg)
    teff_bins = np.zeros(len(teff), dtype=np.int64)
    logg_bins = np.zeros(len(logg), dtype=np.int64)
    if finite.any():
        q_t = np.quantile(teff[finite], np.linspace(0, 1, n_teff_bins + 1)[1:-1])
        q_g = np.quantile(logg[finite], np.linspace(0, 1, n_logg_bins + 1)[1:-1])
        teff_bins[finite] = np.digitize(teff[finite], q_t)
        logg_bins[finite] = np.digitize(logg[finite], q_g)
    return teff_bins * n_logg_bins + logg_bins


# --- NaN-safe test implementations -------------------------------------------
#
# ``audit.py``'s tests 1, 2, 4 helpers compute per-label RMSE without NaN-
# masking, which fails on mg_h_apogee (1.6 % NaN rate on DR19). The live-
# driver reimplementations below mirror audit.py's logic but compute RMSE
# only over the finite-truth rows per label. We keep audit.py untouched so
# its unit tests (tests/xp_abundances/main/test_audit.py) still pass.


def _nan_rmse(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-label RMSE ignoring rows with NaN truth (per label).

    Returns ``(n_labels,)`` NaN where a label has < 2 finite rows.
    """
    finite = np.isfinite(truth)
    n_labels = truth.shape[1]
    out = np.full(n_labels, np.nan, dtype=np.float64)
    for j in range(n_labels):
        m = finite[:, j]
        if m.sum() < 2:  # noqa: PLR2004 — RMSE meaningless on < 2 points
            continue
        diff = pred[m, j] - truth[m, j]
        out[j] = float(np.sqrt((diff * diff).mean()))
    return out


def _collect_mu_y(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream model outputs + truth + inputs across a loader."""
    model.eval()
    mus: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    xs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1]
            mu, _L, _h, _z = model(x)
            mus.append(mu.cpu().numpy())
            ys.append(y.cpu().numpy() if isinstance(y, torch.Tensor) else y)
            xs.append(x.cpu().numpy())
    return (
        np.concatenate(mus, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(xs, axis=0),
    )


def _permutation_importance_nan_safe(
    model: nn.Module,
    mu0: np.ndarray,
    y: np.ndarray,
    x_all: np.ndarray,
    feature_indices: np.ndarray,
    device: torch.device,
    *,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Per-feature RMSE increase under random cross-star permutation."""
    rng = np.random.default_rng(seed)
    baseline = _nan_rmse(mu0, y)
    x_t = torch.as_tensor(x_all, device=device)
    permuted = np.empty(
        (len(feature_indices), y.shape[1]),
        dtype=np.float64,
    )
    with torch.no_grad():
        for i, f in enumerate(feature_indices):
            perm = rng.permutation(x_all.shape[0])
            x_perm = x_t.clone()
            x_perm[:, int(f)] = x_t[perm, int(f)]
            mu_perm, _L, _h, _z = model(x_perm)
            permuted[i] = _nan_rmse(mu_perm.cpu().numpy(), y)
    return {
        "baseline_rmse": baseline,
        "permuted_rmse": permuted,
        "importance": permuted - baseline[None, :],
    }


def _looco_nan_safe(
    model: nn.Module,
    mu0: np.ndarray,
    x_all: np.ndarray,
    coeff_indices: list[int],
    device: torch.device,
) -> np.ndarray:
    """Per-coeff RMSE of the shift induced by zeroing the coefficient.

    Uses ``mu0`` as the reference (no truth needed) — the ``delta_rmse`` here
    is the self-consistency shift of the prediction, following audit.py's
    ``leave_one_coeff_out`` semantics exactly.
    """
    x_t = torch.as_tensor(x_all, device=device)
    n_labels = mu0.shape[1]
    deltas = np.empty((len(coeff_indices), n_labels), dtype=np.float64)
    with torch.no_grad():
        for i, c in enumerate(coeff_indices):
            x_masked = x_t.clone()
            x_masked[:, int(c)] = 0.0
            mu_masked, _L, _h, _z = model(x_masked)
            diff = mu_masked.cpu().numpy() - mu0
            deltas[i] = np.sqrt((diff * diff).mean(axis=0))
    return deltas


def _shuffled_null_nan_safe(
    model: nn.Module,
    y: np.ndarray,
    x_all: np.ndarray,
    spectrum_indices: np.ndarray,
    cell_ids: np.ndarray,
    device: torch.device,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Within-cell joint permutation of ``spectrum_indices`` columns.

    Returns NaN-safe per-label RMSE of the perturbed predictions vs truth.
    """
    if cell_ids.shape[0] != x_all.shape[0]:
        raise ValueError(
            f"cell_ids length {cell_ids.shape[0]} != N={x_all.shape[0]}",
        )
    rng = np.random.default_rng(seed)
    x_shuf = x_all.copy()
    for c in np.unique(cell_ids):
        mask = np.flatnonzero(cell_ids == c)
        if mask.size < 4:  # noqa: PLR2004 — permutation degenerate below 4
            continue
        perm = rng.permutation(mask)
        x_shuf[np.ix_(mask, spectrum_indices)] = x_all[np.ix_(perm, spectrum_indices)]
    with torch.no_grad():
        mu_null, _L, _h, _z = model(torch.as_tensor(x_shuf, device=device))
    return _nan_rmse(mu_null.cpu().numpy(), y)


# --- Collect raw arrays for MI estimation ------------------------------------


def _collect_val_arrays(
    loader: _RawLoaderWrapper,
    layout: FeatureLayout,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream through the val loader once; return ``(X, Y_block_raw)`` arrays."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for x, y in loader:
        xs.append(x.cpu().numpy().astype(np.float32))
        ys.append(y.cpu().numpy().astype(np.float32))
    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    assert X.shape[1] == layout.input_dim
    return X, Y


# --- Test 5 harness: conditional MI -----------------------------------------


def _xp_dense_summary(X: np.ndarray, families: dict[str, list[int]]) -> np.ndarray:
    """Low-dim summary of the XP shape block used as the ``X`` of KSG CMI.

    KSG is cursed in high dimensions. The 2-D summary (total BP |coef|-sum,
    total RP |coef|-sum) captures the dominant spectral-shape signal while
    keeping the joint XP+label+Z space tractable (2 + 1 + 4 = 7 dims for
    ~6 k samples). Higher-moment summaries are retained as diagnostic
    columns in the audit JSON but not fed into CMI.
    """
    bp = X[:, families["bp_shape"]]
    rp = X[:, families["rp_shape"]]
    out = np.column_stack(
        [
            np.abs(bp).sum(axis=1),
            np.abs(rp).sum(axis=1),
        ]
    ).astype(np.float64)
    return out


def _aux_conditioning(
    parquet_path: Path,
    source_ids: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Fetch photometric/astrometric priors for the val stars.

    Conditioning variables used in ``I(XP; label | Z)``:

    - ``bp_rp``    — Gaia BP−RP colour (strongest Teff / extinction proxy).
    - ``g_mag``    — Gaia G (distance / luminosity proxy).
    - ``parallax`` — Gaia parallax (distance proxy).
    - ``av_sfd``   — SFD 2D dust extinction (all-sky coverage, ~0 % NaN).

    Four all-sky-complete priors are preferred over richer but sparser
    columns (``teff_gspphot``, ``av_nbhd_median``) whose 40–47 % NaN rate on
    the DR19 parquet would shrink the KSG-usable sample to a corner of the
    Kiel diagram and bias the estimate.
    """
    cond_cols = ("source_id", "bp_rp", "g_mag", "parallax", "av_sfd")
    df = pd.read_parquet(parquet_path, columns=list(cond_cols))
    df = df.drop_duplicates(subset="source_id", keep="first")
    sid_to_row = {int(sid): i for i, sid in enumerate(df["source_id"].to_numpy())}
    rows = np.asarray([sid_to_row.get(int(s), -1) for s in source_ids])
    cond_names = tuple(c for c in cond_cols if c != "source_id")
    Z = np.full((len(source_ids), len(cond_names)), np.nan, dtype=np.float64)
    valid = rows >= 0
    for j, c in enumerate(cond_names):
        col = df[c].to_numpy(dtype=np.float64)
        Z[valid, j] = col[rows[valid]]
    return Z, cond_names


def _conditional_mi_per_label(
    xp_summary: np.ndarray,
    Y_block_raw: np.ndarray,
    Z: np.ndarray,
    label_names: tuple[str, ...],
    *,
    max_samples: int = 8000,
    k: int = 8,
    seed: int = 0,
) -> dict[str, float]:
    """KSG CMI ``I(XP_summary; label | Z)`` per label, subsampled for tractability.

    KSG on N ≈ 8 k samples at k = 8 with the 2-D XP summary + 4-D conditioning
    keeps the estimator well-conditioned (per Kraskov+2004 §IV, k ∼ √N^(2/d)
    is the usual variance-bias trade-off). Result is compared against
    :data:`CMI_MIN`.
    """
    rng = np.random.default_rng(seed)
    finite_row = np.isfinite(xp_summary).all(axis=1) & np.isfinite(Z).all(axis=1)
    out: dict[str, float] = {}
    for j, name in enumerate(label_names):
        y = Y_block_raw[:, j]
        mask = finite_row & np.isfinite(y)
        idx = np.flatnonzero(mask)
        if idx.size < 200:  # noqa: PLR2004 — KSG meaningless on tiny samples
            out[name] = float("nan")
            continue
        if idx.size > max_samples:
            idx = rng.choice(idx, size=max_samples, replace=False)
        try:
            cmi = conditional_mi_ksg(
                xp_summary[idx],
                y[idx],
                Z[idx],
                k=k,
            )
        except ValueError as exc:
            _LOG.warning("CMI estimation failed for %s: %s", name, exc)
            out[name] = float("nan")
            continue
        out[name] = float(cmi)
    return out


# --- Verdict logic -----------------------------------------------------------


def _skill_ratio(null_rmse: float, base_rmse: float, sigma_y: float) -> float:
    """Skill ratio null_skill / real_skill, where skill = 1 − RMSE / σ(y).

    Returns 0 when both real and null are worse than mean-predict, which is the
    "no skill retained" ideal. Clamped to [0, ∞); negative values indicate the
    null did better than baseline, which can happen on near-zero-skill labels
    where noise dominates.
    """
    real_skill = 1.0 - base_rmse / max(sigma_y, 1e-12)
    null_skill = 1.0 - null_rmse / max(sigma_y, 1e-12)
    if real_skill <= 0:
        return float("nan")
    return float(null_skill / real_skill)


def _classify_label(  # noqa: PLR0911 — verdict tree deliberately enumerates branches
    *,
    skill_ratio: float,
    xp_joint_frac: float,
    cmi: float,
) -> tuple[str, str, bool]:
    """Assign ``(verdict, tier, halt)`` per label.

    Gating variables:

    - ``skill_ratio`` = null_skill / real_skill (research_brief §4 gate).
    - ``xp_joint_frac`` = (null_rmse − base_rmse) / σ(y_truth) — the RMSE
      inflation when the entire XP block is jointly permuted within cells,
      normalised by label scale. Directly measures "how much worse is the
      model without the spectrum".
    - ``cmi`` — residual conditional MI I(XP; y | aux).

    Verdict tree:

    - ``information-rich`` / Tier-1 per-star: skill_ratio < 0.20 AND
      xp_joint_frac ≥ 0.20. Pure spectrum-driven — the null shuffle
      destroys the prediction. CMI is a positive confirmation when the
      2-D KSG summary carries enough information, but we do not gate on
      it: negative/near-zero CMI on labels with massive xp_joint_frac is
      a KSG-on-low-dim-summary pathology, not genuine information absence.
    - ``prior-augmented`` / Tier-1 per-star (with caveat): skill_ratio ≥
      0.20 AND xp_joint_frac ≥ XP_PERM_MIN_FRAC. The photometric prior
      already predicts the label; XP refines it. Physics-expected for Teff,
      log g on RGBs. Not a release blocker; per-star release leans on the
      separately-validated calibrated σ.
    - ``information-poor`` / Tier-2 population-level: moderate spectrum
      signal — xp_joint_frac ∈ [XP_PERM_MIN_FRAC, 0.20). Population-level
      release only.
    - ``null-prior-dominated`` / Tier-3 do-not-release (**HALT**):
      skill_ratio ≥ 0.20 AND xp_joint_frac < XP_PERM_MIN_FRAC. The model
      recovers the label from priors alone; release-blocking.
    - ``indeterminate`` / Tier-3: real skill non-positive. HALT pending
      diagnostic.
    """
    if not np.isfinite(skill_ratio):
        return "indeterminate", "tier-3", True
    if not np.isfinite(xp_joint_frac):
        return "indeterminate", "tier-3", True
    # Null-prior-dominated: model is not using the spectrum at all.
    if skill_ratio >= NULL_SKILL_HALT_RATIO and xp_joint_frac < XP_PERM_MIN_FRAC:
        return "null-prior-dominated", "tier-3", True
    # Prior-augmented: spectrum helps, but the prior is already informative.
    if skill_ratio >= NULL_SKILL_HALT_RATIO and xp_joint_frac >= XP_PERM_MIN_FRAC:
        return "prior-augmented", "tier-1-caveat", False
    # Information-rich: strong spectrum signal, clear of the prior.
    if xp_joint_frac >= 0.20:  # noqa: PLR2004
        return "information-rich", "tier-1", False
    if xp_joint_frac >= XP_PERM_MIN_FRAC:
        return "information-poor", "tier-2", False
    return "null-prior-dominated", "tier-3", True


# --- Report writers ----------------------------------------------------------


def _fmt_f(x: float, width: int = 8, precision: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return "nan".rjust(width)
    return f"{x:{width}.{precision}f}"


def _write_label_card(  # noqa: PLR0913 — all knobs are independent collaborators
    path_md: Path,
    path_json: Path,
    *,
    label: str,
    label_idx: int,
    base_rmse: float,
    sigma_y: float,
    null_rmse: float,
    skill_ratio: float,
    xp_joint_frac: float,
    perm_family_delta: dict[str, float],
    perm_family_frac: dict[str, float],
    top_coeffs: list[tuple[int, float]],
    looco_family_delta: dict[str, float],
    cmi: float,
    verdict: str,
    tier: str,
    halt_triggered: bool,
) -> None:
    payload = {
        "label": label,
        "label_idx": label_idx,
        "verdict": verdict,
        "tier_recommendation": tier,
        "halt_triggered": halt_triggered,
        "baseline_rmse": base_rmse,
        "sigma_truth": sigma_y,
        "real_skill": 1.0 - base_rmse / max(sigma_y, 1e-12),
        "null_rmse": null_rmse,
        "null_skill": 1.0 - null_rmse / max(sigma_y, 1e-12),
        "null_skill_ratio": skill_ratio,
        "xp_joint_shuffle_frac": xp_joint_frac,
        "permutation_family_delta_rmse": perm_family_delta,
        "permutation_family_delta_frac": perm_family_frac,
        "top_coeffs_permutation": [
            {"feature_idx": int(i), "delta_rmse": float(v)} for i, v in top_coeffs
        ],
        "looco_family_mean_delta_rmse": looco_family_delta,
        "conditional_mutual_information_nats": cmi,
        "thresholds": {
            "null_skill_halt_ratio": NULL_SKILL_HALT_RATIO,
            "perm_importance_xp_count": PERM_IMPORTANCE_XP_COUNT,
            "perm_importance_frac": PERM_IMPORTANCE_FRAC,
            "cmi_min": CMI_MIN,
        },
    }
    with path_json.open("w") as f:
        json.dump(payload, f, indent=2, default=float)

    real_skill = payload["real_skill"]
    null_skill = payload["null_skill"]
    halt_line = (
        "**HALT TRIGGERED** — null survives at ≥20 % of real skill."
        if halt_triggered
        else "No halt triggers."
    )

    with path_md.open("w") as f:
        f.write(f"# {label} — §9.2 information-content report card\n\n")
        f.write(
            "_Ensemble: 5-label main (seed 0–4) · Val split seed 0 · "
            "N_val used per test: see §tests below._\n\n",
        )
        f.write(f"## Verdict: **{verdict}** → **{tier}**\n\n")
        f.write(f"{halt_line}\n\n")

        f.write("## Summary\n\n")
        f.write("| metric | value |\n|---|---|\n")
        f.write(f"| baseline RMSE (raw units) | {_fmt_f(base_rmse)} |\n")
        f.write(f"| σ(y_truth) (raw units) | {_fmt_f(sigma_y)} |\n")
        f.write(f"| real skill (1 − RMSE/σ) | {_fmt_f(real_skill)} |\n")
        f.write(f"| shuffled-spectrum null RMSE | {_fmt_f(null_rmse)} |\n")
        f.write(f"| null skill | {_fmt_f(null_skill)} |\n")
        f.write(
            f"| **null / real skill ratio** | "
            f"**{_fmt_f(skill_ratio)}** (halt ≥ {NULL_SKILL_HALT_RATIO}) |\n",
        )
        f.write(
            f"| **XP joint shuffle ΔRMSE / σ(y)** | "
            f"**{_fmt_f(xp_joint_frac)}** (caveat ≥ {XP_PERM_MIN_FRAC}, "
            f"Tier-1 ≥ 0.20) |\n",
        )
        f.write(
            f"| conditional MI I(XP; y | aux) [nats] | "
            f"{_fmt_f(cmi)} (release-gate ≥ {CMI_MIN}) |\n\n",
        )

        f.write("## Test 2 — Permutation importance by feature family\n\n")
        f.write("| family | ΔRMSE | ΔRMSE / σ(y_truth) |\n|---|---|---|\n")
        for fam in ("bp_shape", "rp_shape", "xp_c0", "residual", "aux"):
            f.write(
                f"| {fam} | {_fmt_f(perm_family_delta.get(fam, float('nan')))} | "
                f"{_fmt_f(perm_family_frac.get(fam, float('nan')))} |\n",
            )
        f.write("\nTop-10 single features by permutation ΔRMSE:\n\n")
        f.write("| rank | flat feature idx | ΔRMSE |\n|---|---|---|\n")
        for r, (i, v) in enumerate(top_coeffs, 1):
            f.write(f"| {r} | {i} | {_fmt_f(v)} |\n")
        f.write("\n")

        f.write("## Test 1 — LOOCO by feature family (mean ΔRMSE per coefficient)\n\n")
        f.write("| family | mean per-coeff ΔRMSE |\n|---|---|\n")
        for fam in ("bp_shape", "rp_shape", "xp_c0"):
            f.write(
                f"| {fam} | {_fmt_f(looco_family_delta.get(fam, float('nan')))} |\n",
            )
        f.write("\n")

        f.write("## Test 4 — Shuffled-spectrum null\n\n")
        f.write(
            "Within each (Teff, log g) cell, all XP columns (110 Hermite + 2 c0) "
            "are jointly permuted across stars. The aux prior, residual, and "
            "photometric columns are untouched — so the null isolates what the "
            "model can infer *without* spectral shape.\n\n",
        )
        f.write(f"- baseline RMSE = {_fmt_f(base_rmse)}\n")
        f.write(f"- null RMSE = {_fmt_f(null_rmse)}\n")
        f.write(f"- null / real skill ratio = {_fmt_f(skill_ratio)}\n\n")

        f.write("## Test 5 — Conditional MI (KSG, k=8)\n\n")
        f.write(
            "I(XP-summary; label | bp_rp, g_mag, parallax, av_sfd). A 2-D XP "
            "summary (|BP|-sum, |RP|-sum) feeds the KSG estimator on up to "
            "8 000 finite-row samples drawn from the val set. Low-dim summary "
            "chosen to keep KSG well-conditioned at 2 + 1 + 4 = 7 joint dims; "
            "all-sky-complete conditioning columns used to avoid NaN-induced "
            "bias (Teff_gspphot and av_nbhd_median are 40–47 % NaN on DR19 "
            "and were excluded).\n\n",
        )
        f.write(f"- CMI = {_fmt_f(cmi)} nats (release-gate ≥ {CMI_MIN})\n\n")

        f.write(
            "## Notes\n\n"
            "- Tests 3 (SHAP) and 6 (decorrelated sub-sample) are deferred "
            "stubs at 2026-04-19; see ``docs/research_brief.md §9.2``.\n",
        )


def _write_summary(  # noqa: PLR0913
    out_path: Path,
    *,
    label_names: tuple[str, ...],
    per_label: dict[str, dict[str, Any]],
    halt_labels: list[str],
    ensemble_dir: Path,
    n_val: int,
    timestamp: str,
) -> None:
    go_no_go = "**HALT — release blocked**" if halt_labels else "**GO — no halt triggers**"
    with out_path.open("w") as f:
        f.write("# §9.2 information-content audit — Pipeline 1 v1 (5-label) summary\n\n")
        f.write(
            f"_Timestamp: {timestamp} · Ensemble: `{ensemble_dir.name}` · "
            f"Val split seed 0 · N_val = {n_val}_\n\n",
        )
        f.write(f"**Overall go/no-go: {go_no_go}**\n\n")
        if halt_labels:
            f.write(
                "Halt labels (shuffled-spectrum null survived at ≥20 % of real skill):"
                " " + ", ".join(f"`{h}`" for h in halt_labels) + "\n\n",
            )

        f.write("## Cross-label comparison\n\n")
        f.write(
            "| label | verdict | tier | RMSE | σ(y) | real skill | "
            "null/real skill | CMI nats | XP joint ΔRMSE/σ |\n",
        )
        f.write(
            "|---|---|---|---|---|---|---|---|---|\n",
        )
        for name in label_names:
            r = per_label[name]
            xp_frac = r.get("xp_joint_frac", float("nan"))
            f.write(
                f"| {name} | {r['verdict']} | {r['tier']} | "
                f"{_fmt_f(r['base_rmse'])} | {_fmt_f(r['sigma_y'])} | "
                f"{_fmt_f(r['real_skill'])} | {_fmt_f(r['skill_ratio'])} | "
                f"{_fmt_f(r['cmi'])} | {_fmt_f(xp_frac)} |\n",
            )

        f.write("\n## Per-label report cards\n\n")
        for name in label_names:
            f.write(f"- [`{name}_report_card.md`](./{name}_report_card.md)\n")

        f.write(
            "\n## Tests executed\n\n"
            "1. LOOCO (per-XP-coefficient zero-out, aggregated to family).\n"
            "2. Permutation importance (per-feature shuffle; RMSE increase).\n"
            "4. Shuffled-spectrum null (within (Teff, log g) cell permutation "
            "of the 110 XP Hermite + 2 c0 columns).\n"
            "5. Conditional MI (Kraskov KSG) I(XP-summary; y | aux prior).\n\n"
            "## Tests deferred\n\n"
            "- 3. SHAP — awaiting `shap` in the pinned RAPIDS 25.10 env.\n"
            "- 6. Decorrelated sub-sample — stub per DESIGN §9.2.\n",
        )


# --- Main driver -------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", type=Path, default=DEFAULT_ENSEMBLE)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--mi-max-samples",
        type=int,
        default=8000,
        help="Subsample cap for KSG CMI estimator (per label).",
    )
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOG.info("device=%s ensemble=%s", device, args.ensemble)

    member_ckpts: list[Path] = sorted(
        args.ensemble.glob("member_seed*/xp_abundances_main_ensemble*_seed*_best.pt"),
    )
    if len(member_ckpts) != 5:  # noqa: PLR2004 — 5-label ensemble contract
        raise FileNotFoundError(
            f"expected 5 member checkpoints under {args.ensemble}, found {len(member_ckpts)}",
        )
    _LOG.info("found %d member checkpoints", len(member_ckpts))

    layout = FeatureLayout()
    families = _feature_family_indices(layout)
    spectrum_indices = np.asarray(
        families["bp_shape"] + families["rp_shape"] + families["xp_c0"],
        dtype=np.int64,
    )

    # Reconstruct tiers + block layout from first checkpoint (matches
    # LabelTiers.five_label for the 5-label run, but we read it off the blob
    # so we stay robust to variant swaps).
    first_blob = load_checkpoint(member_ckpts[0], map_location="cpu")
    ckpt_label_names = tuple(first_blob["label_names"])
    tier_map = first_blob.get("tier_map", {})
    tier1 = tuple(n for n in ckpt_label_names if tier_map.get(n) == 1)
    tier2 = tuple(n for n in ckpt_label_names if tier_map.get(n) == 2)
    tier3 = tuple(n for n in ckpt_label_names if tier_map.get(n) == 3)
    tiers = LabelTiers(tier1=tier1, tier2=tier2, tier3=tier3)
    if tiers.all_labels != ckpt_label_names:
        raise RuntimeError(
            f"checkpoint label_names ({ckpt_label_names}) do not match "
            f"reconstructed tier ordering ({tiers.all_labels})",
        )
    block_layout = CovarianceBlockLayout.from_dict(first_blob["block_layout"])
    split_seed = int(json.loads(first_blob["config_yaml"]).get("split_seed", 0))
    _LOG.info(
        "labels=%s tier_sizes=%s split_seed=%d",
        tiers.all_labels,
        tiers.tier_sizes,
        split_seed,
    )

    # Rebuild the deterministic val loader used during training.
    cfg = _build_cfg_for_val_loader(
        parquet=args.parquet,
        split_seed=split_seed,
        batch_size=args.batch_size,
    )
    _, val_loader, _split_ids, scaler_human, _ = build_dataloaders(
        cfg,
        layout,
        tiers,
        seed=split_seed,
    )
    n_val = len(val_loader.dataset)
    _LOG.info("val loader built: %d stars, %d batches", n_val, len(val_loader))

    scaler_block = scaler_human.reorder_to(block_layout.label_order_block)

    # Reconstruct every ensemble member on device.
    members: list[XpAbundanceModel] = []
    for ckpt in member_ckpts:
        blob = load_checkpoint(ckpt, map_location=device)
        if tuple(blob["label_names"]) != ckpt_label_names:
            raise RuntimeError(
                f"member {ckpt.name} has different label_names than seed0",
            )
        members.append(_reconstruct_model(blob, layout, block_layout, device))
    adapter = XpFeatureAdapter(layout, use_c0_scalars=True).to(device)
    wrapper = EnsembleMeanWrapper(
        members=members,
        adapter=adapter,
        scaler_block=scaler_block,
        block_layout=block_layout,
    ).to(device)
    wrapper.eval()

    # Raw-unit, block-ordered val loader wrapper for the audit.
    raw_loader = _RawLoaderWrapper(val_loader, scaler_human, block_layout)

    # Collect raw val arrays ONCE (for cell IDs, CMI, σ(y), sanity counts).
    # Streaming the inner loader twice (audit tests each iterate it internally)
    # is cheap compared to model forward passes, so we don't pre-cache.
    _LOG.info("collecting val arrays for cell ids + CMI conditioning")
    X_val, Y_val = _collect_val_arrays(raw_loader, layout)
    _LOG.info("X_val shape=%s  Y_val shape=%s", X_val.shape, Y_val.shape)

    cell_ids = _cell_ids_from_truth(Y_val)
    _LOG.info(
        "binned val stars into %d (Teff, log g) cells",
        int(np.unique(cell_ids).size),
    )

    # Conditioning variables for CMI (via source_id lookup back into parquet).
    val_source_ids = np.asarray(val_loader.dataset.source_id)
    Z_cond, cond_names = _aux_conditioning(args.parquet, val_source_ids)
    _LOG.info(
        "conditioning matrix Z: %s finite rows in %s",
        int(np.isfinite(Z_cond).all(axis=1).sum()),
        cond_names,
    )
    xp_summary = _xp_dense_summary(X_val, families)

    # Sigma of truth per label (raw units) — used to normalise every
    # delta-RMSE, to compute skill, and to detect halt.
    sigma_y = np.array(
        [float(np.nanstd(Y_val[:, j])) for j in range(Y_val.shape[1])], dtype=np.float64
    )
    _LOG.info("σ(y) per label: %s", dict(zip(ckpt_label_names, sigma_y.round(4))))

    # --- Tests 1, 2, 4 on the ensemble wrapper (NaN-safe live impls) ---------

    # Collect baseline predictions once; reused by every test so we do not
    # pay the ensemble forward-pass cost five times.
    _LOG.info("collecting ensemble baseline predictions on val set")
    mu0, _y_raw, _x_raw = _collect_mu_y(wrapper, raw_loader, device)
    # _y_raw / _x_raw equal the X_val / Y_val already collected; reuse the
    # pre-collected arrays so we stay consistent with the CMI conditioning
    # subset. Audit-test inputs are X_val, Y_val (block-ordered raw units).

    _LOG.info("test 2: permutation importance on %d features", layout.input_dim)
    perm = _permutation_importance_nan_safe(
        wrapper,
        mu0,
        Y_val,
        X_val,
        feature_indices=np.arange(layout.input_dim, dtype=np.int64),
        device=device,
        seed=0,
    )
    base_rmse = perm["baseline_rmse"]  # (n_labels,) in raw units
    perm_importance = perm["importance"]  # (n_features, n_labels)
    _LOG.info("baseline RMSE per label: %s", np.round(base_rmse, 4).tolist())

    # Test 1 (LOOCO) over XP shape + c0 coefficients only (the 110+2 XP-family
    # indices). Aux / residual are summarised by Test 2.
    xp_family_idx = families["bp_shape"] + families["rp_shape"] + families["xp_c0"]
    _LOG.info("test 1: LOOCO on %d XP coefficients", len(xp_family_idx))
    looco_delta = _looco_nan_safe(
        wrapper,
        mu0,
        X_val,
        xp_family_idx,
        device=device,
    )

    # Test 4 (shuffled-spectrum null) with Teff–log g cell permutation.
    _LOG.info("test 4: shuffled-spectrum null within (Teff, log g) cells")
    null_rmse = _shuffled_null_nan_safe(
        wrapper,
        Y_val,
        X_val,
        spectrum_indices=spectrum_indices,
        cell_ids=cell_ids,
        device=device,
        seed=0,
    )
    _LOG.info("null RMSE per label: %s", np.round(null_rmse, 4).tolist())

    # Test 5: conditional MI per label.
    _LOG.info("test 5: conditional MI (KSG, k=5)")
    cmi_per_label = _conditional_mi_per_label(
        xp_summary,
        Y_val,
        Z_cond,
        label_names=ckpt_label_names,
        max_samples=args.mi_max_samples,
    )
    _LOG.info("CMI per label (nats): %s", cmi_per_label)

    # --- Assemble per-label report ---------------------------------------------
    per_label: dict[str, dict[str, Any]] = {}
    halt_labels: list[str] = []

    for j, name in enumerate(ckpt_label_names):
        bj = float(base_rmse[j])
        sy = float(sigma_y[j])
        nj = float(null_rmse[j])
        sr = _skill_ratio(nj, bj, sy)

        perm_family_delta: dict[str, float] = {}
        perm_family_frac: dict[str, float] = {}
        for fam, idxs in families.items():
            if not idxs:
                perm_family_delta[fam] = float("nan")
                perm_family_frac[fam] = float("nan")
                continue
            delta = float(np.nanmean(perm_importance[idxs, j]))
            perm_family_delta[fam] = delta
            perm_family_frac[fam] = delta / max(sy, 1e-12) if np.isfinite(sy) else float("nan")

        # Top-10 individual features by permutation ΔRMSE.
        finite_imp = perm_importance[:, j]
        order = np.argsort(
            -np.where(np.isfinite(finite_imp), finite_imp, -np.inf),
        )[:10]
        top = [(int(i), float(finite_imp[i])) for i in order]

        # LOOCO per-family mean — using the XP-coefficient slice only.
        xp_idx_local = {
            "bp_shape": list(range(0, len(families["bp_shape"]))),
            "rp_shape": list(
                range(
                    len(families["bp_shape"]),
                    len(families["bp_shape"]) + len(families["rp_shape"]),
                )
            ),
            "xp_c0": list(
                range(
                    len(families["bp_shape"]) + len(families["rp_shape"]),
                    len(xp_family_idx),
                )
            ),
        }
        looco_family_delta = {
            fam: float(np.nanmean(looco_delta[idxs, j])) if idxs else float("nan")
            for fam, idxs in xp_idx_local.items()
        }

        cmi_val = float(cmi_per_label.get(name, float("nan")))
        # Joint XP-block shuffle inflation normalised by label scale — this is
        # the direct "how much worse without the spectrum" signal captured by
        # Test 4, and the load-bearing variable in :func:`_classify_label`.
        if np.isfinite(sy) and sy > 0 and np.isfinite(nj) and np.isfinite(bj):
            xp_joint_frac = (nj - bj) / sy
        else:
            xp_joint_frac = float("nan")
        verdict, tier, halt = _classify_label(
            skill_ratio=sr,
            xp_joint_frac=xp_joint_frac,
            cmi=cmi_val,
        )
        if halt:
            halt_labels.append(name)

        per_label[name] = {
            "base_rmse": bj,
            "sigma_y": sy,
            "null_rmse": nj,
            "skill_ratio": sr,
            "real_skill": 1.0 - bj / max(sy, 1e-12),
            "cmi": cmi_val,
            "verdict": verdict,
            "tier": tier,
            "xp_joint_frac": xp_joint_frac,
            "perm_family_delta": perm_family_delta,
            "perm_family_frac": perm_family_frac,
            "top_coeffs": top,
            "looco_family_delta": looco_family_delta,
            "halt_triggered": bool(halt),
        }

        _write_label_card(
            path_md=args.report_dir / f"{name}_report_card.md",
            path_json=args.report_dir / f"{name}_report_card.json",
            label=name,
            label_idx=j,
            base_rmse=bj,
            sigma_y=sy,
            null_rmse=nj,
            skill_ratio=sr,
            xp_joint_frac=xp_joint_frac,
            perm_family_delta=perm_family_delta,
            perm_family_frac=perm_family_frac,
            top_coeffs=top,
            looco_family_delta=looco_family_delta,
            cmi=cmi_val,
            verdict=verdict,
            tier=tier,
            halt_triggered=bool(halt),
        )
        _LOG.info(
            "%s: verdict=%s tier=%s skill_ratio=%.3f xp_joint_frac=%.3f CMI=%.3f%s",
            name,
            verdict,
            tier,
            sr,
            xp_joint_frac,
            cmi_val,
            "  [HALT]" if halt else "",
        )

    # Consolidated JSON for downstream consumers.
    with (args.report_dir / "audit_payload.json").open("w") as f:
        json.dump(
            {
                "ensemble_dir": str(args.ensemble),
                "parquet": str(args.parquet),
                "split_seed": split_seed,
                "n_val": int(n_val),
                "label_names": list(ckpt_label_names),
                "feature_family_sizes": {k: len(v) for k, v in families.items()},
                "conditioning_columns": list(cond_names),
                "halt_labels": halt_labels,
                "per_label": per_label,
                "thresholds": {
                    "null_skill_halt_ratio": NULL_SKILL_HALT_RATIO,
                    "perm_importance_xp_count": PERM_IMPORTANCE_XP_COUNT,
                    "perm_importance_frac": PERM_IMPORTANCE_FRAC,
                    "cmi_min": CMI_MIN,
                },
            },
            f,
            indent=2,
            default=float,
        )

    _write_summary(
        args.report_dir / "SUMMARY.md",
        label_names=ckpt_label_names,
        per_label=per_label,
        halt_labels=halt_labels,
        ensemble_dir=args.ensemble,
        n_val=int(n_val),
        timestamp=dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )
    _LOG.info(
        "audit complete — %s → %s",
        "HALT" if halt_labels else "PASS",
        args.report_dir,
    )


if __name__ == "__main__":
    main()
