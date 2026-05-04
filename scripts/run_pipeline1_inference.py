"""Pipeline 1 inference driver for Thread 3 — D-Cat-b (Aug 2026).

Runs the tagged 5-label production ensemble (the Pipeline 1 v1 release,
tagged ``pipeline1-v1-2026-04-19``) on a Stream-3 feature matrix and writes a release Parquet
+ provenance sidecar. Thin glue over
:mod:`arqueogal.xp_abundances.main.inference` (``load_ensemble``,
``predict_ensemble``, ``EnsemblePrediction``) — no training, no z-score
refitting, no checkpoint mutation.

What this script does
---------------------

1. Load frozen per-coefficient z-score stats from the Stream-1 provenance
   JSON (source-of-truth). Verify the current Hermite basis fingerprint
   matches — fail fast on drift via :class:`FrozenStatsMismatchError`.
2. Read the input Parquet (Stream-3 features). Detect whether the XP block
   is already z-scored (columns ``bp_c0_z`` / ``rp_c0_z`` present) or raw
   (``bp_c0_log`` / ``rp_c0_log``) and apply :func:`apply_frozen_zscore` in
   the latter case.
3. Assemble the flat feature matrix in :class:`FeatureLayout` order and run
   the full ensemble in mini-batches.
4. Un-scale ensemble outputs via the checkpoint's :class:`LabelScaler`
   back to raw physical units (Teff in K, log g in dex, abundances in dex).
5. Compute OOD flags (Mahalanobis on the 108-D XP block, ensemble
   disagreement, combined status) using :mod:`.ood` — the training-set
   Mahalanobis bundle is fit on the fly from the frozen-stats reference set
   (c.f. ``scripts/run_ood_eval.py``; the script takes a training parquet
   path so we don't silently accept drift).
6. Compute the Regime-B envelope flag on predicted (Teff, log g) + observed
   ``|b|``, using :class:`RegimeBEnvelope`.
7. Score or pass through the Ye+2024 ``NO_SYNTH_PHOT`` selection
   probability (see :mod:`arqueogal.data.selection_function`).
8. Write an atomic Parquet + JSON sidecar. Sidecar carries SHA-256s of
   inputs, ensemble dir + member checkpoints, basis fingerprint, row
   counts, OOD flag counts, Regime-B exclusion count, and the Option-2
   label-tier annotations with the user-ratified release statements.

Output schema
-------------

One row per input star. Columns:

- ``source_id`` (passthrough, int64)
- Per-label mean in physical units:
  ``teff_pred``, ``logg_pred``, ``mh_pred``, ``alpha_m_pred``, ``mg_h_pred``
  (all float32)
- Per-label total σ (diagonal of ``Σ_total`` = aleatoric + epistemic):
  ``teff_sigma``, ``logg_sigma``, ``mh_sigma``, ``alpha_m_sigma``,
  ``mg_h_sigma``
- Upper-triangular covariance (15 columns for 5 labels, physical units):
  ``cov_{i}_{j}`` for ``0 ≤ i ≤ j ≤ 4`` in block-label order
  (``teff, logg, mh, alpha_m, mg_h``). ``cov_i_i`` equals the diagonal of
  ``Σ_total`` and therefore equals ``<label>_sigma ** 2`` — stored for
  downstream MC-ensemble consumers (Starfold) that need the full
  cross-label correlation.
- Ensemble epistemic variance per label (``diag(Σ_epistemic)``):
  ``teff_epistemic_var``, ``logg_epistemic_var``, ``mh_epistemic_var``,
  ``alpha_m_epistemic_var``, ``mg_h_epistemic_var``
- OOD flags (per :mod:`arqueogal.xp_abundances.main.ood`):

  * ``ood_mahalanobis_score`` — continuous Mahalanobis distance on the
    108-D XP block; NaN where any XP feature is non-finite.
  * ``ood_disagreement_flag`` — bool; epistemic/total σ ratio above
    ``--ood-threshold`` (default 0.5).
  * ``ood_joint_flag`` — bool; True if either Mahalanobis or disagreement
    flag fires (the "yellow-or-red" status per
    :func:`combined_ood_status`). ``combined_ood_status`` also yields a
    3-level code; we collapse to a single bool here per the #136 release
    convention ("either alone is caution, both is red; downstream
    consumers join both flags from per-flag cells themselves").

- ``regime_b_flag`` — bool; True if the star is inside the
  :class:`RegimeBEnvelope` (``|b|<5°`` AND Teff>4750 K AND log g<2.10).
  True = excluded from Tier-1 per-star release (population-level only).
- ``mode_ambiguous_flag`` — bool; True if the predicted
  (Teff, log g, [M/H]) cell has a bimodal training [α/M] distribution
  (thin-disc + thick-disc α-sequences), OR the prediction is outside the
  precomputed grid. Per-star α/M is not recoverable from XP alone for
  flagged stars — Gaussian-NLL regression collapses bimodal targets to
  the valley between the modes. See
  :mod:`arqueogal.xp_abundances.main.bimodality` and ADR 0015.
- ``mode_ambiguous_in_grid`` — bool; True if the predicted
  (Teff, log g, [M/H]) fell inside the precomputed grid bounds. A caller
  that wants to distinguish "cell-flagged bimodal" from "outside grid,
  conservatively flagged" can recover the split as
  ``mode_ambiguous_flag & mode_ambiguous_in_grid``.
- ``selection_prob`` — Ye+2024 NO_SYNTH_PHOT retention probability.
  Passthrough from input if a ``selection_prob`` column exists; else
  scored from ``b_deg`` + ``g_mag`` via
  :func:`arqueogal.data.selection_function.score_selection_prob`.
- Aux-missingness flags (DATA-availability channels, independent of
  ``ood_joint_flag``):

  * ``ir_missing_flag`` — bool; True if any of ``j_mag, h_mag, k_mag,
    w1_mag, w2_mag`` was NaN in the INPUT row.
  * ``parallax_missing_flag`` — bool; True if ``parallax`` or
    ``parallax_error`` is NaN, or ``parallax / parallax_error`` is
    below :data:`PARALLAX_OVER_ERROR_MIN` (default 5.0).
  * ``extinction_missing_flag`` — bool; True if ALL of ``av_edenhofer,
    av_sfd, av_lallement`` are NaN. One successful dust-map entry keeps
    the flag False.
  * ``aux_missing_any`` — bool; logical OR of the three above.

Tier annotation is NOT a per-row column — it's metadata about labels. See
the sidecar's ``label_tiers`` block.

CLI
---

::

    PYTHONPATH=src python scripts/run_pipeline1_inference.py \\
        --ensemble-dir models/main/xp_abundances/20260419_nogit_a0e10aa_ensemble_5label \\
        --input-parquet data/processed/pipeline1_features_stream3.parquet \\
        --frozen-stats data/processed/pipeline1_features_stream1.provenance.json \\
        --output-parquet data/processed/pipeline1_predictions_stream3.parquet \\
        [--batch-size 4096] [--device auto] \\
        [--ood-threshold 0.5] [--regime-b-config path/to/regime_b.json] \\
        [--ood-training-parquet data/processed/pipeline1_features_stream1.parquet]
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import pyarrow as pa
import pyarrow.parquet as pq

from arqueogal.data.frozen_stats import (
    FrozenZScoreStats,
    apply_frozen_zscore,
    load_frozen_zscore_stats,
    verify_basis_fingerprint,
)
from arqueogal.data.gaia_xp import _build_hermite_basis
from arqueogal.data.selection_function import score_selection_prob
from arqueogal.xp_abundances.main.bimodality import BimodalityGrid
from arqueogal.xp_abundances.main.data import (
    FeatureLayout,
    LabelScaler,
    XpAbundanceDataset,
)
from arqueogal.xp_abundances.main.inference import (
    EnsembleMember,
    EnsemblePrediction,
    load_ensemble,
    predict_ensemble,
)
from arqueogal.xp_abundances.main.model import CovarianceBlockLayout
from arqueogal.xp_abundances.main.ood import (
    MahalanobisOODBundle,
    combined_ood_status,
    fit_mahalanobis_ood,
    flag_mahalanobis_ood,
    percentile_mahalanobis_ood,
    score_mahalanobis_ood,
)
from arqueogal.xp_abundances.main.uncertainty import RegimeBEnvelope

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("run_pipeline1_inference")

REPO_ROOT = Path(__file__).resolve().parent.parent
# Default ensemble: Kiel-bounded RGB-only single-seed run trained
# 2026-04-29. Stream 1 was masked to logg ∈ [1.0, 3.5], Teff ∈ [4000, 5500] K
# at the parquet boundary (pipeline1_features_stream1_kiel.parquet);
# contrastive pretrain ran for 100 epochs at batch 2048 with patience 10
# (cfg 8870bbf), supervised fine-tune ran 1 seed at batch 2048 with patience
# 10 (cfg 3790caf, val loss 1.5443 @ epoch 44). The legacy strong-
# contrastive-v2 ensemble at ``20260425_6b96c06_cd1cbb9_ensemble_5label``
# is retained for methodology comparison and can be selected explicitly via
# ``--ensemble-dir``.
DEFAULT_ENSEMBLE_DIR = (
    REPO_ROOT / "models/main/xp_abundances/20260430_1d71682_a5534e4_ensemble_5label"
)
LEGACY_V1_ENSEMBLE_DIR = (
    REPO_ROOT / "models/main/xp_abundances/20260419_nogit_a0e10aa_ensemble_5label"
)
DEFAULT_FROZEN_STATS = REPO_ROOT / "data/processed/pipeline1_features_stream1.provenance.json"
DEFAULT_OOD_TRAIN_PARQUET = REPO_ROOT / "data/processed/pipeline1_features_stream1_kiel.parquet"
DEFAULT_MODE_AMBIGUOUS_GRID = REPO_ROOT / "data/processed/mode_ambiguous_grid.npz"

LABEL_SHORT_NAMES_5: tuple[str, ...] = ("teff", "logg", "mh", "alpha_m", "mg_h")
LABEL_SHORT_NAMES_21: tuple[str, ...] = (
    "teff", "logg", "mh",
    "fe_h", "alpha_m", "mg_h", "c_h", "n_h",
    "o_h", "na_h", "al_h", "si_h", "s_h",
    "k_h", "ca_h", "ti_h", "v_h", "cr_h",
    "mn_h", "ni_h", "ce_h",
)
# Default kept as 5-label for backwards-compat with code that imports the
# constant; the 21-label flow rebinds this at runtime in main().
LABEL_SHORT_NAMES: tuple[str, ...] = LABEL_SHORT_NAMES_5
"""Short per-label names used for Parquet column prefixes. Order must match
the checkpoint's ``block_layout.label_order_block`` for the 5-label tagged
ensemble; enforced at runtime.
"""

# --- Aux-missingness flag definitions ----------------------------------------
#
# These column-name tuples drive the missingness-flag logic in
# :func:`_compute_aux_missingness_flags`. They are defined as module constants
# so the provenance sidecar can record the exact set used at run time and
# downstream consumers can audit the convention. The names must be present in
# :attr:`FeatureLayout.aux_cols` — if they drift, the flag computation falls
# back to ``False`` for that channel (i.e. a column the layout doesn't carry
# can't be "missing" in the output) and the provenance block records the
# mismatch.

IR_COLS: tuple[str, ...] = ("j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag")
"""2MASS (JHK) + WISE (W1, W2) IR photometry columns."""

PARALLAX_COL: str = "parallax"
PARALLAX_ERROR_COL: str = "parallax_error"
PARALLAX_OVER_ERROR_MIN: float = 5.0
"""Stars with parallax_over_error below this threshold are treated as
informationally-missing parallax — a high-fractional-error parallax is close
to useless for distance inference even if the entry is not NaN. Matches the
Andrae+2023 / Bailer-Jones+2021 convention for "reliable parallax" in the
bulk of RGB.
"""

EXTINCTION_COLS: tuple[str, ...] = ("av_edenhofer", "av_sfd", "av_lallement")
"""The three 3D-dust-map A_V prior channels. The flag trips only when ALL
three are NaN — i.e. no dust-map produced an A_V for that star. A single
non-NaN value keeps the flag False.
"""

RELEASE_NOTES_TEFF = (
    "Teff predictions use Gaia XP spectra as the primary information source, "
    "augmented by parallax and magnitudes. Aux-only baseline achieves RMSE "
    "164 K; the full model achieves 67 K (2.4x improvement). XP coefficients "
    "account for 6 of the 10 top-ranked features in permutation importance "
    "analysis."
)
RELEASE_NOTES_LOGG = (
    "log g predictions use Gaia XP spectra augmented by auxiliary features "
    "(parallax, magnitudes, extinction). An aux-only baseline MLP achieves "
    "RMSE 0.225 dex on the validation set; adding XP spectral information "
    "improves this to 0.157 dex (30% improvement). The spectral contribution "
    "is secondary to geometric and photometric features. Users requiring the "
    "full marginal contribution of spectra to log g predictions should note "
    "this and consider their use case accordingly."
)

LABEL_TIERS: dict[str, str] = {
    "teff": "T1",
    "logg": "T1-caveat",
    "mh": "T1",
    "alpha_m": "T1",
    "mg_h": "T1",
}


# --- Utilities ----------------------------------------------------------------


def _resolve_ensemble_checkpoints(ensemble_dir: Path) -> list[Path]:
    """Collect ensemble member ``*.pt`` files from either flat or nested layouts.

    Handles both:

    * ``ensemble_dir/*.pt`` — flat (used by some unit tests).
    * ``ensemble_dir/member_seed*/*.pt`` — the production layout emitted by
      :func:`arqueogal.xp_abundances.main.training.save_checkpoint` for the
      tagged 5-label ensemble.

    Only ``*_best.pt`` under ``member_seed*`` is returned when the nested
    layout is present, which matches how the tagged ensemble is shipped —
    ``cadence/`` epoch snapshots are ignored.
    """
    ensemble_dir = Path(ensemble_dir)
    if not ensemble_dir.is_dir():
        raise FileNotFoundError(f"ensemble dir does not exist: {ensemble_dir}")
    nested = sorted(ensemble_dir.glob("member_seed*/*.pt"))
    if nested:
        return nested
    flat = sorted(ensemble_dir.glob("*.pt"))
    if flat:
        return flat
    raise FileNotFoundError(f"no checkpoint files found under {ensemble_dir}")


def _sha256_of_file(path: Path, block: int = 1 << 20) -> str:
    """Stream SHA-256 of a file (block = 1 MiB)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    """Return the HEAD commit SHA, or ``"nogit"`` if the repo / git is unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _atomic_write_bytes(dest: Path, data: bytes) -> None:
    """tmp-file → rename within the same directory (atomic on POSIX)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dest.name + ".", suffix=".tmp", dir=dest.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_name, dest)
    except BaseException:
        if Path(tmp_name).exists():
            Path(tmp_name).unlink()
        raise


def _atomic_write_parquet(dest: Path, df: pd.DataFrame) -> None:
    """Write a DataFrame to Parquet via a tmp file + rename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dest.name + ".", suffix=".parquet.tmp", dir=dest.parent)
    os.close(fd)
    try:
        df.to_parquet(tmp_name, index=False)
        os.replace(tmp_name, dest)
    except BaseException:
        if Path(tmp_name).exists():
            Path(tmp_name).unlink()
        raise


# --- Streaming parquet writer ------------------------------------------------


class _StreamingParquetWriter:
    """Append-only Parquet writer with atomic-rename semantics.

    The driver processes the input parquet in chunks of ~50 000 rows so peak
    memory is bounded by ``O(chunk_size)`` instead of ``O(N)``. Each chunk's
    output is converted to a :class:`pyarrow.Table` and written via
    :class:`pyarrow.parquet.ParquetWriter.write_table`. The schema is fixed
    on the first chunk; subsequent chunks are cast to that schema so column
    dtypes don't drift between chunks.

    On ``close()`` the on-disk tmp file is renamed atomically into ``dest``,
    matching the prior :func:`_atomic_write_parquet` contract — readers
    never observe a partial file.
    """

    def __init__(self, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._dest = dest
        fd, self._tmp_name = tempfile.mkstemp(
            prefix=dest.name + ".", suffix=".parquet.tmp", dir=dest.parent,
        )
        os.close(fd)
        self._writer: pq.ParquetWriter | None = None
        self._schema: pa.Schema | None = None
        self._n_rows: int = 0
        self._n_cols: int = 0

    def write(self, df: pd.DataFrame) -> None:
        """Convert ``df`` to a PyArrow table and append to the open writer."""
        if self._schema is None:
            table = pa.Table.from_pandas(df, preserve_index=False)
            self._schema = table.schema
            self._n_cols = len(self._schema)
            self._writer = pq.ParquetWriter(self._tmp_name, self._schema)
        else:
            table = pa.Table.from_pandas(
                df, schema=self._schema, preserve_index=False,
            )
        self._writer.write_table(table)  # type: ignore[union-attr]
        self._n_rows += table.num_rows

    def close(self) -> None:
        if self._writer is None:
            # Nothing was written — clean up the empty tmp file.
            if Path(self._tmp_name).exists():
                Path(self._tmp_name).unlink()
            raise RuntimeError(
                f"streaming writer closed without any rows written for {self._dest}",
            )
        try:
            self._writer.close()
            os.replace(self._tmp_name, self._dest)
        except BaseException:
            if Path(self._tmp_name).exists():
                Path(self._tmp_name).unlink()
            raise

    def abort(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
        if Path(self._tmp_name).exists():
            Path(self._tmp_name).unlink()

    @property
    def n_rows_written(self) -> int:
        return self._n_rows

    @property
    def n_columns(self) -> int:
        return self._n_cols


# --- Input schema detection ---------------------------------------------------


def _detect_input_schema(df_columns: set[str]) -> str:
    """Return ``"zscored"`` or ``"raw"``.

    - ``"zscored"``: Stream-1-style schema where ``bp_c0_z`` / ``rp_c0_z`` plus
      ``bp_coef_norm_*`` are already the frozen-z-score-transformed features.
      This is the default convention the tagged 5-label ensemble was trained
      on; the emit script for Stream-1 persists the z-scoring step in place.
    - ``"raw"``: Stream-3-style schema where ``bp_c0_log`` / ``rp_c0_log`` and
      ``bp_coef_norm_*`` hold the pre-z-score values (log10(c0) and c_i/c_0
      ratios). We apply :func:`apply_frozen_zscore` to move to the model's
      input distribution.

    The detection is a presence check — no data sniffing. If both (``_z`` and
    ``_log``) are present the script raises; the Stream-3 emit has to pick
    one convention. If neither is present we also raise.
    """
    has_z = "bp_c0_z" in df_columns and "rp_c0_z" in df_columns
    has_log = "bp_c0_log" in df_columns and "rp_c0_log" in df_columns
    if has_z and has_log:
        raise ValueError(
            "input parquet has both z-scored and raw c0 columns "
            "(bp_c0_z/rp_c0_z AND bp_c0_log/rp_c0_log); remove one to "
            "disambiguate the emit convention",
        )
    if has_z:
        return "zscored"
    if has_log:
        return "raw"
    raise ValueError(
        "input parquet is missing both the z-scored c0 columns "
        "(bp_c0_z, rp_c0_z) and the raw c0 columns (bp_c0_log, rp_c0_log); "
        "emit the Stream-3 features with one of the two conventions",
    )


# --- Feature-matrix assembly --------------------------------------------------


def _assemble_feature_matrix(
    df: pd.DataFrame,
    layout: FeatureLayout,
    stats: FrozenZScoreStats,
    schema: str,
) -> np.ndarray:
    """Build the flat ``(N, input_dim)`` feature matrix the encoder expects.

    Column order matches :attr:`FeatureLayout.all_required_columns`. If the
    input is raw-schema, we apply :func:`apply_frozen_zscore` to the BP/RP
    coefficient blocks and the c0 log-scalars; z-scored-schema inputs pass
    through unchanged.
    """
    bp_cols = list(layout.bp_coef_cols)
    rp_cols = list(layout.rp_coef_cols)

    if schema == "raw":
        # apply_frozen_zscore requires the full 1..54 block per arm to match the
        # frozen sigma table, so we hand it everything present in the parquet
        # and only slice down to the layout-selected indices afterwards.
        bp_all_cols = sorted(
            (c for c in df.columns if c.startswith("bp_coef_norm_")),
            key=lambda s: int(s.removeprefix("bp_coef_norm_")),
        )
        rp_all_cols = sorted(
            (c for c in df.columns if c.startswith("rp_coef_norm_")),
            key=lambda s: int(s.removeprefix("rp_coef_norm_")),
        )
        bp_full = df[bp_all_cols].to_numpy(dtype=np.float64)
        rp_full = df[rp_all_cols].to_numpy(dtype=np.float64)
        bp_c0_log = df["bp_c0_log"].to_numpy(dtype=np.float64)
        rp_c0_log = df["rp_c0_log"].to_numpy(dtype=np.float64)
        bp_z, rp_z, bp_c0_z, rp_c0_z = apply_frozen_zscore(
            bp_full,
            rp_full,
            bp_c0_log,
            rp_c0_log,
            stats,
        )
        # Re-index to the layout's requested subset (indices are 1-based).
        bp_index_map = {int(c.removeprefix("bp_coef_norm_")): k for k, c in enumerate(bp_all_cols)}
        rp_index_map = {int(c.removeprefix("rp_coef_norm_")): k for k, c in enumerate(rp_all_cols)}
        bp_coef = bp_z[:, [bp_index_map[i] for i in layout.xp_bp_indices]]
        rp_coef = rp_z[:, [rp_index_map[i] for i in layout.xp_rp_indices]]
        c0_scalars_z = {"bp_c0_z": bp_c0_z, "rp_c0_z": rp_c0_z}
    else:
        bp_coef = df[bp_cols].to_numpy(dtype=np.float64)
        rp_coef = df[rp_cols].to_numpy(dtype=np.float64)
        c0_scalars_z = {name: df[name].to_numpy(dtype=np.float64) for name in layout.xp_scalar_cols}

    parts: list[np.ndarray] = [bp_coef, rp_coef]
    for name in layout.xp_scalar_cols:
        parts.append(c0_scalars_z[name][:, None])
    for col in layout.residual_cols:
        parts.append(df[col].to_numpy(dtype=np.float64)[:, None])
    for col in layout.aux_cols:
        parts.append(df[col].to_numpy(dtype=np.float64)[:, None])

    X = np.concatenate(parts, axis=1).astype(np.float32)
    if X.shape[1] != layout.input_dim:
        raise RuntimeError(
            f"assembled feature matrix width {X.shape[1]} != layout.input_dim {layout.input_dim}",
        )
    return X


def _compute_aux_missingness_flags(
    df: pd.DataFrame,
    layout: FeatureLayout,
) -> dict[str, np.ndarray]:
    """Compute the three aux-missingness flags from the INPUT DataFrame.

    This must run BEFORE :func:`np.nan_to_num` is applied to the assembled
    feature matrix — otherwise NaN aux entries have already been replaced
    with 0.0 and the flags are always False.

    Definitions
    -----------
    ``ir_missing_flag``
        True iff ANY of :data:`IR_COLS` is NaN for that row.
    ``parallax_missing_flag``
        True iff ``parallax`` is NaN OR ``parallax_error`` is NaN OR the
        derived ``parallax / parallax_error`` ratio is below
        :data:`PARALLAX_OVER_ERROR_MIN`. The rationale is that a
        high-fractional-error parallax carries almost no information even
        when the entry is not NaN.
    ``extinction_missing_flag``
        True iff ALL THREE of :data:`EXTINCTION_COLS` are NaN — at least
        one successful dust-map A_V keeps the flag False.
    ``aux_missing_any``
        Logical OR of the three above. A compound "this star had some aux
        data missing" signal for downstream consumers.

    Columns the layout doesn't carry contribute ``False`` — i.e. a driver
    running against a trimmed feature layout (e.g. unit tests with 2 aux
    cols) still produces sensible flags, and the provenance block records
    which of the configured channels were actually available.

    Returns
    -------
    dict with keys ``ir_missing_flag``, ``parallax_missing_flag``,
    ``extinction_missing_flag``, ``aux_missing_any`` — each a ``(N,)``
    boolean array.
    """
    n = len(df)
    layout_aux = set(layout.aux_cols)

    # IR: any-NaN across the configured IR columns present in the layout.
    ir_present = [c for c in IR_COLS if c in layout_aux and c in df.columns]
    if ir_present:
        ir_arr = df[ir_present].to_numpy(dtype=np.float64)
        ir_missing = ~np.isfinite(ir_arr).all(axis=1)
    else:
        ir_missing = np.zeros(n, dtype=bool)

    # Parallax: NaN OR error NaN OR |parallax| / parallax_error < threshold.
    has_parallax = PARALLAX_COL in layout_aux and PARALLAX_COL in df.columns
    has_plx_err = PARALLAX_ERROR_COL in layout_aux and PARALLAX_ERROR_COL in df.columns
    if has_parallax and has_plx_err:
        plx = df[PARALLAX_COL].to_numpy(dtype=np.float64)
        plx_err = df[PARALLAX_ERROR_COL].to_numpy(dtype=np.float64)
        plx_nan = ~np.isfinite(plx)
        plx_err_nan = ~np.isfinite(plx_err)
        # Guard against divide-by-zero; also counts 0-error entries as missing.
        with np.errstate(invalid="ignore", divide="ignore"):
            plx_over_err = np.where(
                (plx_err > 0.0) & np.isfinite(plx) & np.isfinite(plx_err),
                plx / plx_err,
                np.nan,
            )
        low_snr = ~(np.abs(plx_over_err) >= PARALLAX_OVER_ERROR_MIN)
        parallax_missing = plx_nan | plx_err_nan | low_snr
    elif has_parallax:
        plx = df[PARALLAX_COL].to_numpy(dtype=np.float64)
        parallax_missing = ~np.isfinite(plx)
    else:
        parallax_missing = np.zeros(n, dtype=bool)

    # Extinction: ALL three dust-map A_V NaN. Columns missing from the
    # layout count as "present but non-NaN" (i.e. they do NOT push the
    # flag toward True) — a layout that carries only one dust map still
    # trips iff that single map is NaN.
    ext_present = [c for c in EXTINCTION_COLS if c in layout_aux and c in df.columns]
    if ext_present:
        ext_arr = df[ext_present].to_numpy(dtype=np.float64)
        extinction_missing = (~np.isfinite(ext_arr)).all(axis=1)
    else:
        extinction_missing = np.zeros(n, dtype=bool)

    aux_missing_any = ir_missing | parallax_missing | extinction_missing
    return {
        "ir_missing_flag": ir_missing.astype(bool),
        "parallax_missing_flag": parallax_missing.astype(bool),
        "extinction_missing_flag": extinction_missing.astype(bool),
        "aux_missing_any": aux_missing_any.astype(bool),
    }


def _xp_108d_block(
    df: pd.DataFrame,
    layout: FeatureLayout,
    schema: str,
    stats: FrozenZScoreStats,
) -> np.ndarray:
    """Pull the 108-D ``(bp_coef_norm_1..54, rp_coef_norm_1..54)`` block.

    Used only for the Mahalanobis OOD flag, which was *fit* on this block at
    training time (see :mod:`scripts.run_ood_eval`). The bundle's mean /
    precision live in z-scored space because Stream-1 persists the z-scored
    values in place. Stream-3 persists *raw* c_i/c_0 ratios — we must apply
    the frozen Stream-1 z-score before scoring, otherwise every Stream-3
    star lands at huge Mahalanobis distance from the z-scored centroid and
    the OOD flag fires universally. NaN rows (failed reprojection) survive
    as NaN, which :func:`score_mahalanobis_ood` treats as OOD.
    """
    if schema == "raw":
        bp_all_cols = sorted(
            (c for c in df.columns if c.startswith("bp_coef_norm_")),
            key=lambda s: int(s.removeprefix("bp_coef_norm_")),
        )
        rp_all_cols = sorted(
            (c for c in df.columns if c.startswith("rp_coef_norm_")),
            key=lambda s: int(s.removeprefix("rp_coef_norm_")),
        )
        bp_full = df[bp_all_cols].to_numpy(dtype=np.float64)
        rp_full = df[rp_all_cols].to_numpy(dtype=np.float64)
        bp_c0_log = df["bp_c0_log"].to_numpy(dtype=np.float64)
        rp_c0_log = df["rp_c0_log"].to_numpy(dtype=np.float64)
        bp_z, rp_z, _, _ = apply_frozen_zscore(
            bp_full,
            rp_full,
            bp_c0_log,
            rp_c0_log,
            stats,
        )
        bp_index_map = {int(c.removeprefix("bp_coef_norm_")): k for k, c in enumerate(bp_all_cols)}
        rp_index_map = {int(c.removeprefix("rp_coef_norm_")): k for k, c in enumerate(rp_all_cols)}
        bp = bp_z[:, [bp_index_map[i] for i in layout.xp_bp_indices]].astype(np.float32)
        rp = rp_z[:, [rp_index_map[i] for i in layout.xp_rp_indices]].astype(np.float32)
    else:
        bp = df[list(layout.bp_coef_cols)].to_numpy(dtype=np.float32)
        rp = df[list(layout.rp_coef_cols)].to_numpy(dtype=np.float32)
    return np.concatenate([bp, rp], axis=1)


# --- Ensemble batched inference ----------------------------------------------


def _build_loader(X: np.ndarray, n_labels: int, batch_size: int) -> DataLoader:
    """Wrap ``X`` in an :class:`XpAbundanceDataset` + :class:`DataLoader`.

    The inference loader supplies dummy ``Y`` (zeros) so
    :func:`predict_ensemble`'s underlying :func:`collect_predictions` stays
    on its documented ``(x, y)`` API. ``Y`` is discarded downstream.
    """
    Y = np.zeros((X.shape[0], n_labels), dtype=np.float32)
    ds = XpAbundanceDataset(X=X, Y=Y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _unscale_ensemble_output(
    pred: EnsemblePrediction,
    scaler: LabelScaler,
    block_layout: CovarianceBlockLayout,
) -> dict[str, Any]:
    """Un-scale μ, σ_total diag, epistemic diag, and per-(i,j) cov columns back
    to physical units **in human label order**, and aggressively release the
    encoder-space ``Σ_aleatoric`` / ``Σ_epistemic`` / ``Σ_total`` / per-member
    tensors after extraction.

    Block-vs-human label ordering
    ------------------------------
    The 21-label 4-block-Cholesky head produces predictions in
    :attr:`block_layout.label_order_block`, which permutes the human label
    order (e.g. ``alpha_m`` is at human-index 4 but block-index 17 because
    the alpha block is stored before the diagonal-only ``s_h``/``alpha_m``
    block). The output writer indexes columns by the human ``LABEL_SHORT_NAMES``
    tuple, so the unscaler must permute ``pred.mu`` and ``pred.Sigma_*`` from
    block to human order before applying the human-order
    :class:`LabelScaler`. Skipping this permutation produced silently-wrong
    21-label predictions on Stream 1: ``alpha_m_pred`` showed correlation
    ``-0.42`` against APOGEE truth (i.e. inverted) because slot 4 in block
    order is ``si_h``, not ``alpha_m``. Verified post-fix at the call site.

    Memory contract
    ---------------
    Stays in fp32 throughout (the prior fp64 upcast caused a transient ~5 GB
    allocation that OOMed at 617k stars × 21 labels). Folds each
    ``s_i * s_j`` into the per-(i,j) cov column so the full scaled
    ``(B, n, n)`` cov tensor is never materialised — only the 231 upper-
    triangular columns the writer consumes. ``pred.Sigma_*`` and
    ``pred.per_member_mu`` are set to ``None`` to release the encoder-space
    copies once the diagonals and cov columns are extracted.
    """
    block_order = list(block_layout.label_order_block)
    human_order = list(block_layout.label_order_human)
    if len(block_order) != len(human_order) or set(block_order) != set(human_order):
        raise RuntimeError(
            "block_layout label_order_block and label_order_human do not "
            "describe the same set of labels — refusing to permute predictions",
        )
    # ``perm[i_human] = block_index_of(label_human[i_human])`` so that
    # ``pred.mu[:, perm]`` is in human order.
    perm = np.asarray(
        [block_order.index(name) for name in human_order], dtype=np.int64,
    )

    # Use the scaler in HUMAN order directly (it was constructed from the
    # checkpoint's ``label_scaler_mean`` / ``label_scaler_scale`` against
    # human ``label_names``). Applying it to the permuted-to-human pred
    # arrays gives physical-units output column-aligned with the human
    # ``LABEL_SHORT_NAMES`` tuple.
    s32 = scaler.scale.astype(np.float32)
    mean32 = scaler.mean.astype(np.float32)

    n_labels = int(pred.mu.shape[1])
    n_rows = int(pred.mu.shape[0])
    if n_labels != len(human_order):
        raise RuntimeError(
            f"pred.mu has {n_labels} columns but block_layout describes "
            f"{len(human_order)} labels",
        )

    # μ in physical units (B, n) fp32, in human order.
    mu = pred.mu[:, perm] * s32[None, :] + mean32[None, :]

    # Per-(i, j) covariance columns in physical units. ``perm[i]`` /
    # ``perm[j]`` index into the BLOCK-order Sigma_total to recover the
    # entry whose row+column correspond to human-order labels i and j.
    diag_var = np.empty((n_rows, n_labels), dtype=np.float32)
    cov_cols: dict[str, np.ndarray] = {}
    Sigma_total = pred.Sigma_total
    for i in range(n_labels):
        bi = int(perm[i])
        for j in range(i, n_labels):
            bj = int(perm[j])
            scale_ij = float(s32[i]) * float(s32[j])
            col = (Sigma_total[:, bi, bj] * scale_ij).astype(np.float32)
            cov_cols[f"cov_{i}_{j}"] = col
            if i == j:
                diag_var[:, i] = col
    sigma_total_diag = np.sqrt(np.clip(diag_var, 0.0, None))

    # Epistemic-variance diagonal in physical units. Diag of block-order
    # Σ_epistemic, then permute to human order, then multiply by ``s²``.
    epi_diag_block = np.einsum("bii->bi", pred.Sigma_epistemic)
    epi_diag_human = epi_diag_block[:, perm]
    epistemic_var_diag = np.clip(
        epi_diag_human * (s32**2)[None, :], 0.0, None
    ).astype(np.float32)

    # Free the (B, n, n) encoder-space tensors and the per-member μ — the
    # downstream code uses only the (B, n) marginals from ``pred`` and the
    # cov_cols / diag arrays computed above. Saves ~3.6 GB held at 617k × 21.
    pred.Sigma_aleatoric = None  # type: ignore[assignment]
    pred.Sigma_epistemic = None  # type: ignore[assignment]
    pred.Sigma_total = None  # type: ignore[assignment]
    pred.per_member_mu = None  # type: ignore[assignment]

    return {
        "mu": mu,
        "cov_cols": cov_cols,
        "sigma_total": sigma_total_diag,
        "epistemic_var": epistemic_var_diag,
    }


# --- OOD scoring -------------------------------------------------------------


def _fit_training_ood_bundle(
    training_parquet: Path,
    layout: FeatureLayout,
    p_threshold: float,
) -> MahalanobisOODBundle:
    """Fit the Mahalanobis OOD bundle from the training parquet's 108-D block.

    This mirrors :mod:`scripts.run_ood_eval`'s training-fit step. We fit the
    bundle on the *full* training parquet (not the train-split subset only)
    because the split-seed metadata is not available at inference time. The
    bundle is near-identical either way — the Mahalanobis distribution is
    dominated by the bulk population, not the 15% val stars.
    """
    _LOG.info("fitting OOD Mahalanobis bundle from %s", training_parquet)
    cols = [*layout.bp_coef_cols, *layout.rp_coef_cols]
    df = pd.read_parquet(training_parquet, columns=cols)
    X = df.to_numpy(dtype=np.float32)
    return fit_mahalanobis_ood(X, p_threshold=p_threshold, regularization=1e-6)


_LABEL_TRUTH_COLS_5: tuple[str, ...] = (
    "teff_apogee", "logg_apogee", "mh_apogee",
    "alpha_m_apogee", "mg_h_apogee",
)


def _fit_label_mahalanobis_bundle(
    training_parquet: Path,
    p_threshold: float = 0.99,
) -> MahalanobisOODBundle | None:
    """Fit a 5-D Mahalanobis bundle on the APOGEE-truth label distribution.

    Used for the **label-extrapolation** flag (Tier-2 demotion gate
    introduced 2026-05-03 to replace the σ-threshold gates that were
    perceived as cherry-picking high-σ predictions).

    The bundle is fit on the joint distribution of the five released
    labels (Teff, log g, [M/H], [α/M], [Mg/H]) as observed by APOGEE on
    the Stream-1 training cohort. At inference, we score the *predicted*
    label vector against this bundle: a star whose μ_pred lies outside
    the training-label envelope is flagged as label-extrapolation.

    Returns None if the training parquet doesn't carry APOGEE truth
    (e.g. when called against a Stream-3 inference parquet).
    """
    try:
        df = pd.read_parquet(training_parquet, columns=list(_LABEL_TRUTH_COLS_5))
    except (KeyError, ValueError) as e:
        _LOG.warning("label-Mahalanobis bundle: training parquet lacks APOGEE truth (%s); "
                     "label_extrapolation_flag will be False everywhere", e)
        return None
    Y = df.to_numpy(dtype=np.float64)
    Y = Y[np.isfinite(Y).all(axis=1)]
    if Y.shape[0] < 100:
        _LOG.warning("label-Mahalanobis bundle: only %d finite truth rows; skipping", Y.shape[0])
        return None
    bundle = fit_mahalanobis_ood(Y, p_threshold=p_threshold, regularization=1e-8)
    _LOG.info("label-Mahalanobis bundle fit on %d APOGEE-truth rows; "
              "threshold=%.3f at p=%.3f",
              bundle.n_training, bundle.threshold, bundle.p_threshold)
    return bundle


# --- Orchestration ------------------------------------------------------------


def _verify_ensemble_label_set(members: list[EnsembleMember]) -> tuple[str, ...]:
    """Verify all ensemble members agree on label_names; return that tuple.

    Accepts either the 5-label set (production v1 / D-Cat-b bridge) or the
    21-label set (production v2). Returns the canonical short-name tuple
    matching ``label_names`` so the output writer can address columns
    correctly.
    """
    first = members[0]
    blob = first.blob
    label_names = tuple(blob["label_names"])

    expected_5 = (
        "teff_apogee", "logg_apogee", "mh_apogee",
        "alpha_m_apogee", "mg_h_apogee",
    )
    expected_21 = (
        "teff_apogee", "logg_apogee", "mh_apogee",
        "fe_h_apogee", "alpha_m_apogee", "mg_h_apogee", "c_h_apogee", "n_h_apogee",
        "o_h_apogee", "na_h_apogee", "al_h_apogee", "si_h_apogee", "s_h_apogee",
        "k_h_apogee", "ca_h_apogee", "ti_h_apogee", "v_h_apogee", "cr_h_apogee",
        "mn_h_apogee", "ni_h_apogee", "ce_h_apogee",
    )
    if label_names == expected_5:
        short = LABEL_SHORT_NAMES_5
    elif label_names == expected_21:
        short = LABEL_SHORT_NAMES_21
    else:
        raise RuntimeError(
            f"ensemble label_names {label_names} matches neither the v1 5-label "
            f"set {expected_5} nor the v2 21-label set {expected_21}",
        )
    bl = CovarianceBlockLayout.from_dict(blob["block_layout"])
    # Both 5-label and 21-label paths are routed through
    # ``_build_output_dataframe``, which permutes mu / Sigma from
    # ``label_order_block`` back to ``label_order_human`` before writing
    # the parquet. The previous 5-label-specific assertion was an over-
    # cautious guard that blocked the post-2026-04-30 v9 layout, where
    # the 5-label variant uses a 4-label dense block + alpha_m diagonal-
    # only tail (so block and human orders deliberately differ).
    _ = bl  # layout retained for downstream callers via the blob.
    for m in members[1:]:
        if tuple(m.blob["label_names"]) != label_names:
            raise RuntimeError(
                f"ensemble member seed={m.seed} has label_names "
                f"{tuple(m.blob['label_names'])} != first member {label_names}",
            )
    return short


# Back-compat alias in case other code imports the old name.
_verify_5label_ensemble = _verify_ensemble_label_set


def _regime_b_envelope(cfg_path: Path | None) -> RegimeBEnvelope:
    """Load a :class:`RegimeBEnvelope` from JSON config, else the default.

    The default carries the 5-label halt-cell envelope thresholds
    (``|b|<5``, Teff>4750, log g<2.10) baked into
    :class:`RegimeBEnvelope`'s class defaults. An explicit config lets
    downstream users pick a tighter or looser envelope without editing the
    script.
    """
    if cfg_path is None:
        return RegimeBEnvelope()
    blob = json.loads(cfg_path.read_text())
    return RegimeBEnvelope.from_dict(blob)


def _selection_prob(
    df: pd.DataFrame,
    *,
    artifact_path: Path | None = None,
) -> np.ndarray:
    """Pass through ``selection_prob`` if present, else score from (b_deg, g_mag)."""
    if "selection_prob" in df.columns:
        return df["selection_prob"].to_numpy(dtype=np.float64)
    required = {"b_deg", "g_mag"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"selection_prob missing and cannot be computed — input parquet "
            f"is also missing {sorted(missing)} for score_selection_prob",
        )
    return score_selection_prob(
        df["b_deg"].to_numpy(dtype=np.float64),
        df["g_mag"].to_numpy(dtype=np.float64),
        artifact_path=artifact_path,
    )


def _assemble_output_frame(  # noqa: PLR0913 — assembles a wide release frame from many upstream arrays
    source_id: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    cov_cols: dict[str, np.ndarray],
    epi_var: np.ndarray,
    *,
    mahal_scores: np.ndarray,
    ens_flags: np.ndarray,
    joint_flags: np.ndarray,
    regime_b_flag: np.ndarray,
    mode_ambiguous_flag: np.ndarray,
    mode_ambiguous_in_grid: np.ndarray,
    selection_prob: np.ndarray,
    aux_missing_flags: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Build the release DataFrame with every required column in order.

    ``cov_cols`` maps ``"cov_{i}_{j}"`` (upper-triangular, ``0 ≤ i ≤ j``) to
    a 1-D array of length ``N`` in physical units. Pre-flattening the cov
    matrix keeps the writer from holding a ``(B, n, n)`` tensor alongside
    the cov columns it derives.
    """
    n_labels = mu.shape[1]
    if n_labels != len(LABEL_SHORT_NAMES):
        raise RuntimeError(
            f"mu has {n_labels} labels but driver expects "
            f"{len(LABEL_SHORT_NAMES)}: {LABEL_SHORT_NAMES}",
        )
    out: dict[str, np.ndarray] = {"source_id": source_id.astype(np.int64)}
    for j, name in enumerate(LABEL_SHORT_NAMES):
        out[f"{name}_pred"] = mu[:, j].astype(np.float32)
    for j, name in enumerate(LABEL_SHORT_NAMES):
        out[f"{name}_sigma"] = sigma[:, j].astype(np.float32)
    for i in range(n_labels):
        for j in range(i, n_labels):
            key = f"cov_{i}_{j}"
            if key not in cov_cols:
                raise KeyError(
                    f"cov_cols missing required key {key!r}; produced keys are "
                    f"{sorted(cov_cols.keys())[:5]}..."
                )
            out[key] = cov_cols[key].astype(np.float32, copy=False)
    for j, name in enumerate(LABEL_SHORT_NAMES):
        out[f"{name}_epistemic_var"] = epi_var[:, j].astype(np.float32)
    out["ood_mahalanobis_score"] = mahal_scores.astype(np.float32)
    out["ood_disagreement_flag"] = ens_flags.astype(bool)
    out["ood_joint_flag"] = joint_flags.astype(bool)
    out["regime_b_flag"] = regime_b_flag.astype(bool)
    out["mode_ambiguous_flag"] = mode_ambiguous_flag.astype(bool)
    out["mode_ambiguous_in_grid"] = mode_ambiguous_in_grid.astype(bool)
    out["selection_prob"] = selection_prob.astype(np.float32)
    # Aux-missingness flags — separate channel from ood_joint_flag by design.
    # Data-availability signal, not OOD-on-XP-distribution signal; downstream
    # is free to combine per its own policy.
    for key in (
        "ir_missing_flag",
        "parallax_missing_flag",
        "extinction_missing_flag",
        "aux_missing_any",
    ):
        out[key] = aux_missing_flags[key].astype(bool)
    return pd.DataFrame(out)


def _iter_input_chunks(
    input_parquet: Path,
    columns: list[str],
    chunk_size: int,
):
    """Yield ``pd.DataFrame`` chunks of length ≤ ``chunk_size`` from ``input_parquet``.

    Uses :meth:`pyarrow.parquet.ParquetFile.iter_batches` so the parquet
    reader streams row groups directly without ever materialising the full
    table in memory. The returned chunks are pandas DataFrames so the rest
    of the driver (which is pandas-flavoured) can stay unchanged.

    Notes
    -----
    PyArrow's ``iter_batches`` already handles row-group boundaries — a
    requested ``batch_size`` of 50 000 may be split across several row
    groups internally without changing the per-batch row count. The driver
    relies on this for predictable per-chunk peak memory.
    """
    pf = pq.ParquetFile(input_parquet)
    for batch in pf.iter_batches(batch_size=chunk_size, columns=columns):
        yield batch.to_pandas()


# --- Per-chunk inference pass ------------------------------------------------


def _process_chunk(  # noqa: PLR0913, PLR0915 — chunk pipeline reads many setup objects produced once in run_inference
    df: pd.DataFrame,
    *,
    layout: FeatureLayout,
    stats: FrozenZScoreStats,
    schema: str,
    members: list[EnsembleMember],
    scaler: LabelScaler,
    block_layout: CovarianceBlockLayout,
    device: torch.device,
    batch_size: int,
    bundle: MahalanobisOODBundle,
    label_bundle: MahalanobisOODBundle | None,
    ood_threshold: float,
    envelope: RegimeBEnvelope,
    ambiguity_grid: BimodalityGrid,
    selection_artifact_path: Path | None,
    accumulators: dict[str, Any],
    feature_scaler: Any = None,  # FeatureScaler | None — loaded from ckpt by run_inference
) -> pd.DataFrame:
    """Run the full per-star pipeline on one chunk and return its output frame.

    Mutates ``accumulators`` in place: flag counters, total-row counters,
    selection_prob raw values (collected for streaming-safe quantile stats
    computed once at end), and per-chunk progress.
    """
    n_rows = len(df)

    # --- assemble feature matrix + 108-D OOD block -----------------------
    X = _assemble_feature_matrix(df, layout, stats, schema)
    xp_block_width = len(layout.bp_coef_cols) + len(layout.rp_coef_cols)
    xp_block_108d = X[:, :xp_block_width].copy()

    # Aux-missingness flags — must be computed BEFORE NaN imputation so a
    # NaN aux entry survives the lookup.
    aux_missing_flags = _compute_aux_missingness_flags(df, layout)

    # Apply the FeatureScaler the encoder was trained with (z-score on aux,
    # log10 + z-score on residual RMS). Must run BEFORE nan_to_num because
    # log10 needs to see the unimputed value. XP block columns are
    # passthrough — they are already standardised by the frozen Hermite
    # z-score basis and the scaler's apply_mask is False on those columns.
    if feature_scaler is not None:
        if tuple(feature_scaler.feature_names) != tuple(layout.all_required_columns):
            raise RuntimeError(
                "checkpoint feature_scaler.feature_names != layout.all_required_columns; "
                "the encoder input contract has drifted between training and inference",
            )
        X = feature_scaler.transform(X)

    # NaN-impute residuals + aux for the forward pass.
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # --- ensemble forward + un-scale -------------------------------------
    loader = _build_loader(X, n_labels=block_layout.n_labels, batch_size=batch_size)
    del X
    gc.collect()
    pred = predict_ensemble(members, loader, device=device)
    unscaled = _unscale_ensemble_output(pred, scaler, block_layout)
    gc.collect()

    # --- OOD scoring against the pre-fit bundles --------------------------
    mahal_scores = score_mahalanobis_ood(xp_block_108d, bundle)
    mahal_flags = flag_mahalanobis_ood(xp_block_108d, bundle)
    mahal_percentile = percentile_mahalanobis_ood(xp_block_108d, bundle)
    del xp_block_108d
    gc.collect()
    # Label-space Mahalanobis (Tier-2 gate, replaces σ-thresholds 2026-05-03).
    # Score the predicted label vector (in physical units) against the
    # APOGEE-truth-trained bundle. Only meaningful for the 5-label release.
    if label_bundle is not None and unscaled["mu"].shape[1] == label_bundle.feature_dim:
        label_mahal_scores = score_mahalanobis_ood(unscaled["mu"], label_bundle)
        label_extrapolation_flags = flag_mahalanobis_ood(unscaled["mu"], label_bundle)
        label_mahal_percentile = percentile_mahalanobis_ood(unscaled["mu"], label_bundle)
    else:
        label_mahal_scores = np.full(unscaled["mu"].shape[0], np.nan, dtype=np.float64)
        label_extrapolation_flags = np.zeros(unscaled["mu"].shape[0], dtype=bool)
        label_mahal_percentile = np.full(unscaled["mu"].shape[0], np.nan, dtype=np.float64)
    epi_tot_ratio_per_label = pred.sigma_epistemic / np.clip(
        np.sqrt(pred.sigma_epistemic**2 + pred.sigma_aleatoric**2),
        1e-12,
        None,
    )
    ens_ratio = epi_tot_ratio_per_label.mean(axis=1).astype(np.float64)
    ens_flags = (ens_ratio > ood_threshold).astype(bool)
    status = combined_ood_status(mahal_flags, ens_flags)
    joint_flags = (status >= 1).astype(bool)

    # --- Regime B + mode ambiguity ---------------------------------------
    if "b_deg" not in df.columns:
        raise KeyError("input parquet is missing b_deg; required for the Regime B envelope")
    b_deg = df["b_deg"].to_numpy(dtype=np.float64)
    teff_pred = unscaled["mu"][:, 0]
    logg_pred = unscaled["mu"][:, 1]
    mh_pred = unscaled["mu"][:, 2]
    regime_b_flag = envelope.mask(teff_pred, logg_pred, b_deg)
    in_grid_flag, in_grid = ambiguity_grid.query(teff_pred, logg_pred, mh_pred)
    mode_ambiguous_flag = in_grid_flag | (~in_grid)

    # --- selection_prob --------------------------------------------------
    selection_prob = _selection_prob(df, artifact_path=selection_artifact_path)
    selection_source_tag = (
        "input_passthrough" if "selection_prob" in df.columns else "scored_from_b_deg_g_mag"
    )
    if accumulators.get("selection_source_tag") is None:
        accumulators["selection_source_tag"] = selection_source_tag
    elif accumulators["selection_source_tag"] != selection_source_tag:
        raise RuntimeError(
            "selection_prob source tag changed mid-stream — chunks disagree on whether "
            "the input parquet carries a selection_prob column",
        )

    # --- assemble + return chunk frame -----------------------------------
    source_id_arr = df["source_id"].to_numpy(dtype=np.int64)
    out_df = _assemble_output_frame(
        source_id=source_id_arr,
        mu=unscaled["mu"],
        sigma=unscaled["sigma_total"],
        cov_cols=unscaled["cov_cols"],
        epi_var=unscaled["epistemic_var"],
        mahal_scores=mahal_scores,
        ens_flags=ens_flags,
        joint_flags=joint_flags,
        regime_b_flag=regime_b_flag,
        mode_ambiguous_flag=mode_ambiguous_flag,
        mode_ambiguous_in_grid=in_grid,
        selection_prob=selection_prob,
        aux_missing_flags=aux_missing_flags,
    )
    # Inject the new label-Mahalanobis columns directly (assemble_output_frame
    # signature kept stable to avoid touching every other caller).
    out_df["label_mahalanobis_score"] = label_mahal_scores.astype(np.float32)
    out_df["label_extrapolation_flag"] = label_extrapolation_flags.astype(bool)
    # Per-star empirical percentiles against training distance ECDF — let
    # the user pick their own cutoff (e.g. 0.95 for stricter, 0.999 for laxer).
    out_df["ood_mahalanobis_percentile"] = mahal_percentile.astype(np.float32)
    out_df["label_mahalanobis_percentile"] = label_mahal_percentile.astype(np.float32)

    # --- accumulate provenance counters ----------------------------------
    accumulators["n_rows"] += n_rows
    accumulators["mahal_count"] += int(mahal_flags.sum())
    accumulators["ens_count"] += int(ens_flags.sum())
    accumulators["joint_count"] += int(joint_flags.sum())
    accumulators["regime_b_count"] += int(regime_b_flag.sum())
    accumulators["mode_ambiguous_count"] += int(mode_ambiguous_flag.sum())
    accumulators["in_grid_bimodal_count"] += int(in_grid_flag.sum())
    accumulators["in_grid_count"] += int(in_grid.sum())
    accumulators["out_of_grid_count"] += int((~in_grid).sum())
    for key in (
        "ir_missing_flag",
        "parallax_missing_flag",
        "extinction_missing_flag",
        "aux_missing_any",
    ):
        accumulators[f"{key}_count"] += int(aux_missing_flags[key].sum())
    accumulators["selection_prob_chunks"].append(selection_prob.astype(np.float64))

    # Drop everything we don't return.
    unscaled.clear()
    del unscaled, pred, df, mahal_scores, mahal_flags
    del ens_flags, joint_flags, regime_b_flag, mode_ambiguous_flag
    del in_grid_flag, in_grid, selection_prob, aux_missing_flags
    gc.collect()

    return out_df


def run_inference(  # noqa: PLR0913, PLR0915 — CLI driver entrypoint; all knobs are release-contract arguments
    *,
    ensemble_dir: Path,
    input_parquet: Path,
    frozen_stats_path: Path,
    output_parquet: Path,
    batch_size: int,
    device: torch.device,
    ood_threshold: float,
    regime_b_config: Path | None,
    ood_training_parquet: Path,
    mode_ambiguous_grid_path: Path | None = None,
    selection_artifact_path: Path | None = None,
    layout: FeatureLayout | None = None,
) -> dict[str, Any]:
    """End-to-end inference driver. Returns the provenance dict it writes.

    Parameters
    ----------
    layout
        Override the :class:`FeatureLayout` used to assemble the feature
        matrix. Defaults to the production 139-D layout. Intended as a test
        hook so fixtures can exercise the driver without materialising the
        full production feature set; production callers always pass
        ``None`` so the driver checks the checkpoint ``input_dim`` against
        the production layout.
    """
    # --- load ensemble ----------------------------------------------------
    # The production 5-label ensemble is laid out as
    # ``ensemble_dir/member_seed{N}/*.pt`` — ``load_ensemble`` only globs top-
    # level ``*.pt``, so resolve checkpoints recursively and pass the list.
    # Flat layouts (``ensemble_dir/*.pt``) continue to work unchanged.
    ckpt_paths = _resolve_ensemble_checkpoints(ensemble_dir)
    members = load_ensemble(ckpt_paths, device=device)
    _LOG.info("loaded %d ensemble members from %s", len(members), ensemble_dir)
    short_names = _verify_ensemble_label_set(members)
    # Rebind module-global LABEL_SHORT_NAMES so _build_output_dataframe and
    # downstream consumers address the right column count for this run.
    global LABEL_SHORT_NAMES
    LABEL_SHORT_NAMES = short_names
    _LOG.info("label set: %d labels (%s)", len(short_names), ", ".join(short_names))

    first_blob = members[0].blob
    block_layout = CovarianceBlockLayout.from_dict(first_blob["block_layout"])
    label_names_human = tuple(first_blob["label_names"])
    scaler = LabelScaler(
        mean=np.asarray(first_blob["label_scaler_mean"], dtype=np.float32),
        scale=np.asarray(first_blob["label_scaler_scale"], dtype=np.float32),
        label_names=label_names_human,
    )
    if scaler.is_default():
        raise RuntimeError(
            "ensemble member 0 has placeholder label scaler (zeros/ones); "
            "checkpoint was saved before the scaler was fit — refuse to run",
        )

    # Pull the FeatureScaler from the checkpoint (None for older models that
    # predate the feature-scaling change). Applied later, after assembly.
    fs_blob = first_blob.get("feature_scaler")
    if fs_blob is not None:
        from arqueogal.xp_abundances.main.data import FeatureScaler
        feature_scaler = FeatureScaler(
            mean=np.asarray(fs_blob["mean"], dtype=np.float32),
            scale=np.asarray(fs_blob["scale"], dtype=np.float32),
            feature_names=tuple(fs_blob["feature_names"]),
            log10_mask=np.asarray(fs_blob["log10_mask"], dtype=bool),
            apply_mask=np.asarray(fs_blob["apply_mask"], dtype=bool),
        )
    else:
        feature_scaler = None

    # --- frozen stats + basis fingerprint --------------------------------
    stats = load_frozen_zscore_stats(frozen_stats_path)
    current_fp = _build_hermite_basis()["fingerprint_sha256"]
    verify_basis_fingerprint(current_fp, stats)
    _LOG.info(
        "basis fingerprint OK (%s...); n_ref=%d",
        stats.basis_fingerprint[:16],
        stats.n_reference_population,
    )

    # --- read input schema + select needed columns -----------------------
    layout = layout or FeatureLayout()  # default 139-D — matches the 5-label ensemble
    if int(first_blob["input_dim"]) != layout.input_dim:
        raise RuntimeError(
            f"checkpoint input_dim {first_blob['input_dim']} != "
            f"FeatureLayout.input_dim {layout.input_dim} — ensemble was "
            "trained on a non-default layout and this driver only supports "
            "the default 139-D layout",
        )
    pf = pq.ParquetFile(input_parquet)
    schema_cols = {f.name for f in pf.schema_arrow}
    n_rows = int(pf.metadata.num_rows)
    needed: set[str] = {"source_id"}
    needed.update(layout.all_required_columns)
    needed.update(c for c in schema_cols if c.startswith(("bp_coef_norm_", "rp_coef_norm_")))
    needed.update({"bp_c0_log", "rp_c0_log", "bp_c0_z", "rp_c0_z"})
    needed.update({"j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag"})
    needed.update({"parallax", "parallax_error", "ruwe"})
    needed.update({"av_edenhofer", "av_lallement", "av_sfd", "av_nbhd_median"})
    needed.update({"av_los", "av_los_source"})
    needed.update({"ra_deg", "dec_deg", "b_deg", "g_mag"})
    if "selection_prob" in schema_cols:
        needed.add("selection_prob")
    keep = sorted(c for c in needed if c in schema_cols)
    if "source_id" not in keep:
        raise KeyError(f"input parquet {input_parquet} missing source_id")
    if "b_deg" not in keep:
        raise KeyError(f"input parquet {input_parquet} missing b_deg (required for Regime B)")
    schema_inferred = _detect_input_schema(set(keep))
    _LOG.info(
        "input %s: schema=%s, n=%d, reading %d/%d cols",
        input_parquet, schema_inferred, n_rows, len(keep), len(schema_cols),
    )

    # --- one-time setup of OOD bundle, regime-B envelope, ambiguity grid -
    bundle = _fit_training_ood_bundle(
        ood_training_parquet,
        layout,
        p_threshold=0.99,
    )
    # Label-space Mahalanobis bundle — fit on APOGEE truth; powers Tier-2.
    label_bundle = _fit_label_mahalanobis_bundle(
        ood_training_parquet,
        p_threshold=0.99,
    )
    envelope = _regime_b_envelope(regime_b_config)
    if mode_ambiguous_grid_path is None or not mode_ambiguous_grid_path.is_file():
        raise FileNotFoundError(
            f"mode-ambiguous grid not found at {mode_ambiguous_grid_path!r}; "
            "build it first with scripts/build_mode_ambiguous_mask.py",
        )
    ambiguity_grid = BimodalityGrid.load(mode_ambiguous_grid_path)

    # --- streaming chunk loop --------------------------------------------
    # 50 000 rows per chunk keeps the per-chunk peak under ~700 MB at
    # 21 labels × 5 ensemble members; bounded by O(chunk_size), not O(N).
    # The earlier all-at-once flow held ~3.3 GB of (B, n, n) covariance
    # tensors plus a ~5 GB transient during fp64 unscale, which OOMed on
    # the 9.7 GB WSL2 instance at 617k stars.
    chunk_size = 50_000
    accumulators: dict[str, Any] = {
        "n_rows": 0,
        "mahal_count": 0,
        "ens_count": 0,
        "joint_count": 0,
        "regime_b_count": 0,
        "mode_ambiguous_count": 0,
        "in_grid_bimodal_count": 0,
        "in_grid_count": 0,
        "out_of_grid_count": 0,
        "ir_missing_flag_count": 0,
        "parallax_missing_flag_count": 0,
        "extinction_missing_flag_count": 0,
        "aux_missing_any_count": 0,
        "selection_prob_chunks": [],
        "selection_source_tag": None,
    }

    writer = _StreamingParquetWriter(output_parquet)
    chunk_idx = 0
    try:
        for chunk_df in _iter_input_chunks(input_parquet, keep, chunk_size):
            chunk_idx += 1
            chunk_n = len(chunk_df)
            _LOG.info(
                "chunk %d: rows=%d (cumulative %d/%d)",
                chunk_idx, chunk_n, accumulators["n_rows"] + chunk_n, n_rows,
            )
            chunk_out = _process_chunk(
                chunk_df,
                layout=layout,
                stats=stats,
                schema=schema_inferred,
                members=members,
                scaler=scaler,
                block_layout=block_layout,
                device=device,
                batch_size=batch_size,
                bundle=bundle,
                label_bundle=label_bundle,
                ood_threshold=ood_threshold,
                envelope=envelope,
                ambiguity_grid=ambiguity_grid,
                selection_artifact_path=selection_artifact_path,
                accumulators=accumulators,
                feature_scaler=feature_scaler,
            )
            writer.write(chunk_out)
            del chunk_out, chunk_df
            gc.collect()
        writer.close()
    except BaseException:
        writer.abort()
        raise

    # Stitch streaming-state into the existing-shape provenance: the prior
    # version held flag arrays in memory; here we work from accumulator
    # counts plus a single concatenated selection_prob array (forming
    # ``selection_prob`` as a (N,) float64 — only ~5 MB at 617k rows so we
    # can compute exact median/p05 without a streaming-quantile estimator).
    if accumulators["n_rows"] != n_rows:
        raise RuntimeError(
            f"streaming wrote {accumulators['n_rows']} rows but input had {n_rows}",
        )
    selection_prob = np.concatenate(accumulators["selection_prob_chunks"])
    accumulators["selection_prob_chunks"].clear()
    selection_source_tag = accumulators["selection_source_tag"] or "scored_from_b_deg_g_mag"
    n_rows_written = writer.n_rows_written
    n_cols_written = writer.n_columns
    schema = schema_inferred
    _LOG.info(
        "wrote %s (%d rows, %d cols) via %d chunks of size %d",
        output_parquet, n_rows_written, n_cols_written, chunk_idx, chunk_size,
    )
    _LOG.info(
        "OOD: mahalanobis_rate=%.4f ensemble_rate=%.4f joint_rate=%.4f",
        accumulators["mahal_count"] / max(n_rows, 1),
        accumulators["ens_count"] / max(n_rows, 1),
        accumulators["joint_count"] / max(n_rows, 1),
    )
    _LOG.info(
        "Regime B: %d/%d (%.3f%%) inside envelope",
        accumulators["regime_b_count"], n_rows,
        100.0 * accumulators["regime_b_count"] / max(n_rows, 1),
    )
    _LOG.info(
        "mode-ambiguous: %d/%d (%.3f%%) flagged",
        accumulators["mode_ambiguous_count"], n_rows,
        100.0 * accumulators["mode_ambiguous_count"] / max(n_rows, 1),
    )
    _LOG.info(
        "aux-missingness: ir=%.4f parallax=%.4f extinction=%.4f any=%.4f",
        accumulators["ir_missing_flag_count"] / max(n_rows, 1),
        accumulators["parallax_missing_flag_count"] / max(n_rows, 1),
        accumulators["extinction_missing_flag_count"] / max(n_rows, 1),
        accumulators["aux_missing_any_count"] / max(n_rows, 1),
    )

    # --- provenance -------------------------------------------------------
    ensemble_member_shas = {ckpt.name: _sha256_of_file(ckpt) for ckpt in ckpt_paths}
    n_rows_safe = max(n_rows, 1)
    column_names = [field.name for field in (writer._schema or pa.schema([]))]
    provenance: dict[str, Any] = {
        "output_file": str(output_parquet.relative_to(REPO_ROOT))
        if output_parquet.is_relative_to(REPO_ROOT)
        else str(output_parquet),
        "script": "scripts/run_pipeline1_inference.py",
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "git_sha": _git_sha(),
        "device": str(device),
        "n_input_rows": int(n_rows),
        "n_output_rows": int(n_rows_written),
        "streaming": {
            "chunk_size": int(chunk_size),
            "n_chunks": int(chunk_idx),
            "writer": "pyarrow.parquet.ParquetWriter (atomic tmp+rename)",
            "rationale": (
                "Per-chunk inference bounds peak RAM by O(chunk_size) instead "
                "of O(N). At 50 000 rows/chunk × 21 labels × 5 members the "
                "predict_ensemble (B, n, n) covariance tensors and the unscale "
                "transients all stay below ~700 MB, enabling 617k-row "
                "inference inside the 9.7 GB WSL2 envelope that previously "
                "OOMed on the all-at-once flow."
            ),
        },
        "input": {
            "path": str(input_parquet),
            "sha256": _sha256_of_file(input_parquet),
            "schema_detected": schema,
        },
        "ensemble": {
            "dir": str(ensemble_dir),
            "n_members": len(members),
            "member_sha256": ensemble_member_shas,
            "label_names": list(label_names_human),
            "block_sizes": list(block_layout.block_sizes),
        },
        "frozen_stats": {
            "path": str(frozen_stats_path),
            "basis_fingerprint_sha256": stats.basis_fingerprint,
            "n_reference_population": int(stats.n_reference_population),
            "sigma_floor": float(stats.sigma_floor),
            "reference_population": stats.reference_population_description,
        },
        "ood": {
            "mahalanobis": {
                "training_parquet": str(ood_training_parquet),
                "training_parquet_sha256": _sha256_of_file(ood_training_parquet),
                "p_threshold": float(bundle.p_threshold),
                "threshold_distance": float(bundle.threshold),
                "n_training": int(bundle.n_training),
                "regularization": float(bundle.regularization),
                "flag_count": int(accumulators["mahal_count"]),
                "flag_rate": float(accumulators["mahal_count"] / n_rows_safe),
            },
            "disagreement": {
                "ratio_threshold": float(ood_threshold),
                "flag_count": int(accumulators["ens_count"]),
                "flag_rate": float(accumulators["ens_count"] / n_rows_safe),
            },
            "joint": {
                "flag_count": int(accumulators["joint_count"]),
                "flag_rate": float(accumulators["joint_count"] / n_rows_safe),
                "convention": (
                    "joint_flag = mahalanobis_flag OR disagreement_flag "
                    "(yellow-or-red per ood.combined_ood_status codes 1+2)"
                ),
            },
        },
        "regime_b": {
            "envelope": envelope.to_dict(),
            "config_path": str(regime_b_config) if regime_b_config else None,
            "n_excluded": int(accumulators["regime_b_count"]),
            "n_released": int(n_rows - accumulators["regime_b_count"]),
            "frac_excluded": float(accumulators["regime_b_count"] / n_rows_safe),
        },
        "mode_ambiguous": {
            "grid_path": str(mode_ambiguous_grid_path),
            "grid_sha256": _sha256_of_file(mode_ambiguous_grid_path),
            "grid_shape": [
                int(len(ambiguity_grid.teff_edges) - 1),
                int(len(ambiguity_grid.logg_edges) - 1),
                int(len(ambiguity_grid.mh_edges) - 1),
            ],
            "criteria": {
                "min_cell_n": int(ambiguity_grid.min_cell_n),
                "min_minor_weight": float(ambiguity_grid.min_minor_weight),
                "min_mean_sep": float(ambiguity_grid.min_mean_sep),
                "bic_delta_min": float(ambiguity_grid.bic_delta_min),
            },
            "n_in_grid": int(accumulators["in_grid_count"]),
            "n_out_of_grid": int(accumulators["out_of_grid_count"]),
            "n_in_grid_bimodal": int(accumulators["in_grid_bimodal_count"]),
            "n_flagged": int(accumulators["mode_ambiguous_count"]),
            "frac_flagged": float(accumulators["mode_ambiguous_count"] / n_rows_safe),
            "convention": (
                "mode_ambiguous_flag = (cell is bimodal in training α/M) "
                "OR (predicted (Teff, log g, [M/H]) is outside the grid). "
                "Out-of-grid defaults to flagged because the training set "
                "does not certify those cells are unimodal."
            ),
            "rationale": (
                "Gaussian-NLL μ* = E[y|x] collapses bimodal targets onto "
                "the conditional mean (the valley between thin-disc and "
                "thick-disc α-sequences). Per-star α/M is not recoverable "
                "from XP alone at these cells; they are excluded from the "
                "per-star release. See ADR 0015."
            ),
        },
        "selection_prob": {
            "source": selection_source_tag,
            "mean": float(np.nanmean(selection_prob)),
            "median": float(np.nanmedian(selection_prob)),
            "p05": float(np.nanquantile(selection_prob, 0.05)),
        },
        "aux_missingness": {
            "definitions": {
                "ir_missing_flag": {
                    "rule": "any of ir_cols is NaN",
                    "ir_cols": list(IR_COLS),
                },
                "parallax_missing_flag": {
                    "rule": (
                        "parallax is NaN OR parallax_error is NaN OR "
                        "parallax / parallax_error < parallax_over_error_min"
                    ),
                    "parallax_col": PARALLAX_COL,
                    "parallax_error_col": PARALLAX_ERROR_COL,
                    "parallax_over_error_min": float(PARALLAX_OVER_ERROR_MIN),
                },
                "extinction_missing_flag": {
                    "rule": "ALL of extinction_cols are NaN",
                    "extinction_cols": list(EXTINCTION_COLS),
                },
                "aux_missing_any": {
                    "rule": ("ir_missing_flag OR parallax_missing_flag OR extinction_missing_flag"),
                },
            },
            "layout_resolution": {
                "ir_cols_in_layout": [c for c in IR_COLS if c in layout.aux_cols],
                "parallax_in_layout": PARALLAX_COL in layout.aux_cols,
                "parallax_error_in_layout": PARALLAX_ERROR_COL in layout.aux_cols,
                "extinction_cols_in_layout": [c for c in EXTINCTION_COLS if c in layout.aux_cols],
            },
            "flag_rates": {
                "ir_missing_flag": float(accumulators["ir_missing_flag_count"] / n_rows_safe),
                "parallax_missing_flag": float(
                    accumulators["parallax_missing_flag_count"] / n_rows_safe,
                ),
                "extinction_missing_flag": float(
                    accumulators["extinction_missing_flag_count"] / n_rows_safe,
                ),
                "aux_missing_any": float(accumulators["aux_missing_any_count"] / n_rows_safe),
            },
            "flag_counts": {
                "ir_missing_flag": int(accumulators["ir_missing_flag_count"]),
                "parallax_missing_flag": int(accumulators["parallax_missing_flag_count"]),
                "extinction_missing_flag": int(accumulators["extinction_missing_flag_count"]),
                "aux_missing_any": int(accumulators["aux_missing_any_count"]),
            },
            "independence_note": (
                "Aux-missingness flags are DATA-availability signals and are "
                "deliberately NOT folded into ood_joint_flag, which remains "
                "the Mahalanobis + ensemble-disagreement combined flag on the "
                "108-D XP block. Downstream consumers combine channels as "
                "their use case requires."
            ),
            "nan_handling": (
                "Input NaN in residuals/aux features is imputed to 0.0 via "
                "np.nan_to_num before the forward pass, mirroring training.py "
                "line 153. Flags above are computed from the RAW input frame "
                "BEFORE imputation — otherwise they would always read False."
            ),
        },
        "label_tiers": dict(LABEL_TIERS),
        "prior_augmented_release_notes": {
            "teff": RELEASE_NOTES_TEFF,
            "logg": RELEASE_NOTES_LOGG,
        },
        "columns": column_names,
        "notes": (
            "Output μ / Σ / σ / epistemic_var are in raw physical units "
            "(Teff: K, log g: dex, abundances: dex). cov_{i}_{j} matrix is "
            "the upper triangle (i ≤ j) of the ensemble total covariance "
            "(aleatoric + epistemic) after per-member calibration and "
            "moment matching, then un-scaled by the checkpoint's LabelScaler. "
            "cov_i_i equals sigma_i**2. Label block order: " + ", ".join(LABEL_SHORT_NAMES) + ". "
            "Tier assignments follow Option 2: all 5 labels are T1 release "
            "quality; log g carries the T1-caveat flag noting spectra are a "
            "secondary information source (see prior_augmented_release_notes)."
        ),
    }
    sidecar_path = output_parquet.with_suffix(output_parquet.suffix + ".provenance.json")
    _atomic_write_bytes(
        sidecar_path,
        json.dumps(provenance, indent=2, default=str).encode("utf-8"),
    )
    _LOG.info("wrote %s", sidecar_path)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-dir", type=Path, default=DEFAULT_ENSEMBLE_DIR)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--frozen-stats", type=Path, default=DEFAULT_FROZEN_STATS)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--device", type=str, default="auto", help="auto | cpu | cuda | cuda:0 | ..."
    )
    parser.add_argument(
        "--ood-threshold",
        type=float,
        default=0.5,
        help="epistemic/total σ ratio cutoff for the ensemble disagreement flag",
    )
    parser.add_argument(
        "--regime-b-config",
        type=Path,
        default=None,
        help="Optional JSON config for RegimeBEnvelope; if "
        "omitted the 5-label halt-cell defaults apply",
    )
    parser.add_argument(
        "--ood-training-parquet",
        type=Path,
        default=DEFAULT_OOD_TRAIN_PARQUET,
        help="Parquet used to refit the training-set Mahalanobis OOD bundle at inference time",
    )
    parser.add_argument(
        "--mode-ambiguous-grid",
        type=Path,
        default=DEFAULT_MODE_AMBIGUOUS_GRID,
        help="Precomputed (Teff, log g, [M/H]) bimodality "
        "grid .npz — see scripts/build_mode_ambiguous_mask.py",
    )
    parser.add_argument(
        "--selection-artifact",
        type=Path,
        default=None,
        help="Optional override for the selection-function v1 Parquet artefact path",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    _LOG.info("device=%s", device)

    run_inference(
        ensemble_dir=args.ensemble_dir,
        input_parquet=args.input_parquet,
        frozen_stats_path=args.frozen_stats,
        output_parquet=args.output_parquet,
        batch_size=args.batch_size,
        device=device,
        ood_threshold=args.ood_threshold,
        regime_b_config=args.regime_b_config,
        ood_training_parquet=args.ood_training_parquet,
        mode_ambiguous_grid_path=args.mode_ambiguous_grid,
        selection_artifact_path=args.selection_artifact,
    )


if __name__ == "__main__":
    main()
