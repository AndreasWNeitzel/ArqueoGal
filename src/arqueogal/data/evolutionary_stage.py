"""Evolutionary-stage filter — RGB + HeCB inference contract.

The brief restricts Pipeline-1 inference to **RGB and HeCB (= red-clump,
core-helium-burning) giants only**. Including non-RGB / non-HeCB stars at
inference time produces wrong [C/N] / Mg-clock predictions because the
training pool (APOGEE DR19 with the §3.3 quality cuts) is dominated by
RGB+HeCB stars; subgiants, AGB tips, and main-sequence contaminants violate
the implicit input distribution.

This module encodes the filter as a **deterministic, fingerprint-able
contract** so train and inference apply identical eligibility logic. Two
sources of evolutionary-stage information are supported:

1. **Andrae+2023 categorical column** (``evolutionary_stage_andrae``):
   ``{"RGB", "RC", "RGB_candidate"}``. RGB and RC are accepted; the
   conservative "RGB_candidate" is also accepted because the Andrae
   vetting catalogue uses it for stars with high-confidence RGB
   classification under a more permissive cut. Anything else (or NaN) is
   rejected.

2. **Atmospheric (Teff, log g) box**: a fallback for stars without an
   Andrae label, using the Pipeline-1 training-pool envelope:
   Teff ∈ [4000, 5500] K and log g ∈ [1.0, 3.5] dex (research_brief.md §7.1
   training-set selection). This is the same box the Stream-1 quality
   cuts use, just enforced at inference too.

The filter is **opt-in for now**: the production v1 release does not call
it because the Stream-1 quality cuts already enforce the box at training
time and the model has not been characterised on subgiant input. The
methods paper recommends running the filter at Stream-2 inference once
the Andrae column lands on the Stream-2 cross-match.

References
----------
- Andrae, Rix & Chandra 2023 (MNRAS 521, 3527; arXiv:2302.02611).
- research_brief.md §7.1 (training-pool selection).
- docs/plan/00_overview.md "Needs clarification" (RGB+HeCB filter).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Andrae+2023 categorical labels accepted at inference (RGB + HeCB / red clump).
ACCEPTED_ANDRAE_STAGES: Final[frozenset[str]] = frozenset({"RGB", "RC", "RGB_candidate"})
"""Andrae+2023 evolutionary-stage labels eligible for Pipeline-1 inference.

``RGB`` is the unambiguous red-giant branch; ``RC`` is the red clump
(core-helium-burning); ``RGB_candidate`` is the Andrae vetting category for
high-confidence RGB stars under a more permissive cut. AGB, MS, subgiant,
and ambiguous labels are excluded.
"""


@dataclass(frozen=True, slots=True)
class EvolutionaryStageFilter:
    """Frozen contract for the RGB+HeCB inference filter.

    The dataclass is frozen so the fingerprint can be embedded in a
    provenance sidecar with no risk of post-construction mutation.

    Attributes
    ----------
    teff_min, teff_max
        Atmospheric-box bounds in Kelvin. Default (4000, 5500) matches
        the research_brief §7.1 training-set selection.
    logg_min, logg_max
        Atmospheric-box bounds in dex. Default (1.0, 3.5) matches §7.1.
    accepted_andrae
        Set of Andrae+2023 categorical labels accepted. Default
        :data:`ACCEPTED_ANDRAE_STAGES`.
    require_andrae
        When True, stars without an Andrae label are *rejected* even if
        they pass the atmospheric box. When False (default), the
        atmospheric box is used as a fallback for stars without an
        Andrae label.
    """

    teff_min: float = 4000.0
    teff_max: float = 5500.0
    logg_min: float = 1.0
    logg_max: float = 3.5
    accepted_andrae: frozenset[str] = ACCEPTED_ANDRAE_STAGES
    require_andrae: bool = False

    def fingerprint(self) -> dict[str, object]:
        """JSON-able snapshot for provenance sidecars."""
        return {
            "teff_min": float(self.teff_min),
            "teff_max": float(self.teff_max),
            "logg_min": float(self.logg_min),
            "logg_max": float(self.logg_max),
            "accepted_andrae": sorted(self.accepted_andrae),
            "require_andrae": bool(self.require_andrae),
        }


DEFAULT_EVOLUTIONARY_STAGE_FILTER: Final[EvolutionaryStageFilter] = EvolutionaryStageFilter()


def is_rgb_or_hecb(  # noqa: PLR0913 — orthogonal column-name knobs
    df: pd.DataFrame,
    *,
    filt: EvolutionaryStageFilter = DEFAULT_EVOLUTIONARY_STAGE_FILTER,
    teff_col: str = "teff",
    logg_col: str = "logg",
    andrae_col: str = "evolutionary_stage_andrae",
) -> dict[str, np.ndarray]:
    """Per-star RGB+HeCB eligibility flags + the composite gate.

    Returns a dict of three boolean arrays:

    - ``andrae_accepted``: True iff the Andrae+2023 categorical label is
      in ``filt.accepted_andrae``. False (not raised) when the column is
      missing or the value is NaN.
    - ``atmospheric_accepted``: True iff (Teff, log g) lies inside the
      training-pool box. False if either column is missing or NaN.
    - ``rgb_or_hecb``: composite gate. When ``filt.require_andrae`` is
      True, equal to ``andrae_accepted``. Otherwise, equal to
      ``andrae_accepted | (no Andrae label AND atmospheric_accepted)``.

    The default policy (``require_andrae=False``) is permissive: stars
    *with* an Andrae label use it as the authoritative evolutionary
    stage, stars *without* one fall back to the atmospheric box. This
    matches the Pipeline-1-on-Stream-2 use case where most stars do not
    yet have an Andrae cross-match but the atmospheric inputs (from a
    GSP-Phot or XGBoost preview) are reliable.
    """
    n = len(df)

    # Andrae column path.
    if andrae_col in df.columns:
        andrae = df[andrae_col].astype("string")
        andrae_accepted = andrae.isin(filt.accepted_andrae).fillna(False).to_numpy()
        has_andrae = andrae.notna().to_numpy()
    else:
        andrae_accepted = np.zeros(n, dtype=bool)
        has_andrae = np.zeros(n, dtype=bool)

    # Atmospheric-box path.
    if teff_col in df.columns and logg_col in df.columns:
        teff = df[teff_col].to_numpy(dtype=np.float64, copy=False)
        logg = df[logg_col].to_numpy(dtype=np.float64, copy=False)
        with np.errstate(invalid="ignore"):
            atm = (
                (teff >= filt.teff_min)
                & (teff <= filt.teff_max)
                & (logg >= filt.logg_min)
                & (logg <= filt.logg_max)
            )
            atmospheric_accepted = np.where(np.isfinite(teff) & np.isfinite(logg), atm, False)
    else:
        atmospheric_accepted = np.zeros(n, dtype=bool)

    if filt.require_andrae:
        composite = andrae_accepted
    else:
        composite = andrae_accepted | (~has_andrae & atmospheric_accepted)

    return {
        "andrae_accepted": andrae_accepted.astype(bool),
        "atmospheric_accepted": atmospheric_accepted.astype(bool),
        "rgb_or_hecb": composite.astype(bool),
    }


def filter_to_rgb_or_hecb(
    df: pd.DataFrame,
    *,
    filt: EvolutionaryStageFilter = DEFAULT_EVOLUTIONARY_STAGE_FILTER,
    teff_col: str = "teff",
    logg_col: str = "logg",
    andrae_col: str = "evolutionary_stage_andrae",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop non-RGB-non-HeCB rows; return the filtered frame + counts.

    The counts dict carries ``n_in``, ``n_out``, and the firing rate of
    each individual flag (``andrae_accepted``, ``atmospheric_accepted``,
    ``rgb_or_hecb``) for provenance / methods-paper diagnostic plots.
    """
    flags = is_rgb_or_hecb(
        df, filt=filt, teff_col=teff_col, logg_col=logg_col, andrae_col=andrae_col
    )
    composite = flags["rgb_or_hecb"]
    out = df.loc[composite].reset_index(drop=True)
    counts = {
        "n_in": int(len(df)),
        "n_out": int(len(out)),
        "n_andrae_accepted": int(flags["andrae_accepted"].sum()),
        "n_atmospheric_accepted": int(flags["atmospheric_accepted"].sum()),
        "n_rgb_or_hecb": int(composite.sum()),
    }
    logger.info(
        "Evolutionary-stage filter: %d → %d rows (Andrae=%d, atm=%d, composite=%d)",
        counts["n_in"],
        counts["n_out"],
        counts["n_andrae_accepted"],
        counts["n_atmospheric_accepted"],
        counts["n_rgb_or_hecb"],
    )
    return out, counts


__all__ = [
    "ACCEPTED_ANDRAE_STAGES",
    "DEFAULT_EVOLUTIONARY_STAGE_FILTER",
    "EvolutionaryStageFilter",
    "filter_to_rgb_or_hecb",
    "is_rgb_or_hecb",
]
