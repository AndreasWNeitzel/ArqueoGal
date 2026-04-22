"""Per-star release-tier assignment for Pipeline 1 predictions.

The six-test protocol in ``research_brief.md`` §3.3 decides which *elements*
are releasable; :mod:`tier_promotion` implements that. This module is the
orthogonal *per-star* layer: given an element has cleared §3.3, which
individual predictions are trustworthy?

Tier semantics (frozen contract — see ``docs/plan/05_release_packaging.md``):

- **1 — per-star science**: all structural gates pass, no missingness, no
  ensemble disagreement. Safe for single-star claims.
- **2 — statistical / ensemble only**: passes the OOD gates but carries a
  caveat (regime-B, mode-ambiguous, aux-missing, or ensemble disagreement).
  Safe for aggregate studies, not for per-star science.
- **3 — do not release**: OOD-flagged (Mahalanobis or latent-support
  convex-hull surrogate) or contains a NaN prediction.

Tier-3 rows are retained in the parquet so downstream consumers can apply
their own, less stringent filters for methodology work; the release
contract is that published catalogues expose Tier 1 only (or Tier 1 + 2
with an explicit caveat) — see `docs/research_brief.md` §3.3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pandas as pd

_PRED_COLS: Final = ("teff_pred", "logg_pred", "mh_pred", "alpha_m_pred", "mg_h_pred")

_OOD_FLAGS: Final = ("ood_joint_flag", "latent_support_flag")

_CAVEAT_FLAGS: Final = (
    "regime_b_flag",
    "mode_ambiguous_flag",
    "ood_disagreement_flag",
    "aux_missing_any",
)


def assign_release_tier(df: pd.DataFrame) -> pd.Series:
    """Compute the per-row ``release_tier`` ∈ {1, 2, 3}.

    Missing flag columns are treated as ``False`` — the conservative
    direction (Tier-3 demotions must be explicit in the input). Missing
    prediction columns are ignored for the NaN check; the caller is
    expected to have run the full inference pipeline before calling this.

    Parameters
    ----------
    df : pd.DataFrame
        Prediction parquet produced by ``run_pipeline1_inference.py`` and
        annotated with ``latent_support_flag`` via
        ``merge_latent_support_into_predictions.py``.

    Returns
    -------
    pd.Series
        ``int8`` series aligned to ``df.index``.
    """
    idx = df.index

    has_nan_pred = pd.Series(False, index=idx)
    for col in _PRED_COLS:
        if col in df.columns:
            has_nan_pred = has_nan_pred | df[col].isna()

    ood = pd.Series(False, index=idx)
    for col in _OOD_FLAGS:
        if col in df.columns:
            ood = ood | df[col].fillna(False).astype(bool)

    tier3 = has_nan_pred | ood

    caveat = pd.Series(False, index=idx)
    for col in _CAVEAT_FLAGS:
        if col in df.columns:
            caveat = caveat | df[col].fillna(False).astype(bool)

    tier = pd.Series(1, index=idx, dtype="int8")
    tier[caveat] = 2
    tier[tier3] = 3
    return tier


def tier_counts(df: pd.DataFrame, tier_col: str = "release_tier") -> dict[int, int]:
    """Return ``{tier: n_rows}`` for the three tiers."""
    s = df[tier_col]
    return {int(t): int((s == t).sum()) for t in (1, 2, 3)}


def annotate_parquet(path: Path) -> dict[str, int | dict[int, int]]:
    """Add or refresh ``release_tier`` in a prediction parquet in place.

    Emits / refreshes the ``*.release_tier.json`` sidecar next to the
    parquet capturing counts and the flag-column provenance. The parquet's
    main provenance sidecar (``*.provenance.json``) is not touched — that
    records upstream inference, not this annotation step.

    Returns
    -------
    dict
        ``{"n_rows": N, "counts": {1: ..., 2: ..., 3: ...}}``.
    """
    df = pd.read_parquet(path)
    df["release_tier"] = assign_release_tier(df).astype("int8")
    df.to_parquet(path)

    counts = tier_counts(df)
    summary: dict[str, int | dict[int, int]] = {
        "n_rows": int(len(df)),
        "counts": counts,
    }

    sidecar = path.with_name(path.stem + ".release_tier.json")
    sidecar_payload = {
        "parquet": path.name,
        "n_rows": int(len(df)),
        "counts": {str(k): v for k, v in counts.items()},
        "ood_flags_considered": [c for c in _OOD_FLAGS if c in df.columns],
        "caveat_flags_considered": [c for c in _CAVEAT_FLAGS if c in df.columns],
        "nan_pred_columns_checked": [c for c in _PRED_COLS if c in df.columns],
    }
    sidecar.write_text(json.dumps(sidecar_payload, indent=2))

    return summary


__all__ = [
    "annotate_parquet",
    "assign_release_tier",
    "tier_counts",
]
