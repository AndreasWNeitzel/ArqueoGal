"""Detect (Teff, log g, [M/H]) cells where training [α/M] is bimodal.

Motivation
----------
Gaussian NLL regression has μ* = E[y | x] as its unique minimiser. When
p(α/M | Teff, log g, [M/H]) is bimodal — e.g. the thin-disc / thick-disc
α-sequences at intermediate metallicity — no amount of encoder training can
avoid collapsing predictions onto the conditional mean, which sits *between*
the two modes. A single-Gaussian head simply cannot express two peaks.

Per-star α/M is therefore not recoverable for stars whose predicted
(Teff, log g, [M/H]) cell has a bimodal training target distribution, and
those stars must be withheld from the per-star release. This module supplies:

- :func:`fit_bimodality_grid` — precompute a 3-D mask over training data
  using a 2-GMM vs 1-GMM BIC comparison plus well-separated-modes guards.
- :class:`BimodalityGrid` — the serialisable artefact (JSON-canonical).
- :meth:`BimodalityGrid.query` — attach a ``mode_ambiguous_flag`` column at
  inference time from the model's predicted (Teff, log g, [M/H]).

Sidecar provenance (``{artifact}.provenance.json``) records the grid edges,
BIC thresholds, cell counts, and the training-set git SHA so downstream
consumers can audit exactly which cells were flagged and why.

The detector is deliberately a *data-side* property of the training set, not
a prediction-clustering artefact. If the training target was bimodal in a
cell, every model on that cell is vulnerable — this flag does not need to be
recomputed per checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# sklearn GaussianMixture is already in the rapids env (via cuml/sklearn). No new dep.
from sklearn.mixture import GaussianMixture

__all__ = ["BimodalityGrid", "fit_bimodality_grid"]


@dataclass
class BimodalityGrid:
    """Serialisable 3-D mask over (Teff, log g, [M/H]) cells.

    ``is_bimodal`` is True wherever the training [α/M] distribution in that
    cell satisfies all of: BIC(2-GMM) < BIC(1-GMM) − ``bic_delta_min``,
    minor mode weight ≥ ``min_minor_weight``, and
    |μ_1 − μ_2| ≥ ``min_mean_sep``. Cells with N < ``min_cell_n`` are left
    False (insufficient evidence — *not* a statement that they are unimodal).
    """

    teff_edges: np.ndarray
    logg_edges: np.ndarray
    mh_edges: np.ndarray
    is_bimodal: np.ndarray
    n_per_cell: np.ndarray
    min_cell_n: int
    min_minor_weight: float
    min_mean_sep: float
    bic_delta_min: float
    per_cell_gmm: dict[tuple[int, int, int], dict] = field(default_factory=dict)

    def query(
        self,
        teff: np.ndarray,
        logg: np.ndarray,
        mh: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Look up the bimodality flag for each (teff, logg, mh) tuple.

        Returns
        -------
        flag : np.ndarray[bool]
            True where the lookup cell is flagged bimodal.
        in_grid : np.ndarray[bool]
            True where (teff, logg, mh) falls inside the grid bounds. Stars
            outside the grid get flag=False and in_grid=False; callers may
            choose to treat out-of-grid stars conservatively (flag=True).
        """
        teff = np.asarray(teff, dtype=np.float64)
        logg = np.asarray(logg, dtype=np.float64)
        mh = np.asarray(mh, dtype=np.float64)
        if not (teff.shape == logg.shape == mh.shape):
            raise ValueError("teff, logg, mh must share shape")

        nT = len(self.teff_edges) - 1
        nL = len(self.logg_edges) - 1
        nM = len(self.mh_edges) - 1

        t_idx = np.digitize(teff, self.teff_edges) - 1
        l_idx = np.digitize(logg, self.logg_edges) - 1
        m_idx = np.digitize(mh, self.mh_edges) - 1

        in_grid = (
            np.isfinite(teff)
            & np.isfinite(logg)
            & np.isfinite(mh)
            & (t_idx >= 0)
            & (t_idx < nT)
            & (l_idx >= 0)
            & (l_idx < nL)
            & (m_idx >= 0)
            & (m_idx < nM)
        )
        flag = np.zeros_like(in_grid)

        if in_grid.any():
            ti = np.clip(t_idx[in_grid], 0, nT - 1)
            li = np.clip(l_idx[in_grid], 0, nL - 1)
            mi = np.clip(m_idx[in_grid], 0, nM - 1)
            flag[in_grid] = self.is_bimodal[ti, li, mi]
        return flag, in_grid

    def save(self, path: Path, *, provenance: dict | None = None) -> None:
        """Persist to an ``.npz`` with a JSON provenance sidecar."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            teff_edges=self.teff_edges,
            logg_edges=self.logg_edges,
            mh_edges=self.mh_edges,
            is_bimodal=self.is_bimodal,
            n_per_cell=self.n_per_cell,
        )
        sidecar = {
            "artifact": path.name,
            "created_utc": datetime.now(UTC).isoformat(),
            "grid": {
                "n_teff": int(len(self.teff_edges) - 1),
                "n_logg": int(len(self.logg_edges) - 1),
                "n_mh": int(len(self.mh_edges) - 1),
                "teff_range": [float(self.teff_edges[0]), float(self.teff_edges[-1])],
                "logg_range": [float(self.logg_edges[0]), float(self.logg_edges[-1])],
                "mh_range": [float(self.mh_edges[0]), float(self.mh_edges[-1])],
            },
            "criteria": {
                "min_cell_n": int(self.min_cell_n),
                "min_minor_weight": float(self.min_minor_weight),
                "min_mean_sep": float(self.min_mean_sep),
                "bic_delta_min": float(self.bic_delta_min),
            },
            "counts": {
                "cells_total": int(self.is_bimodal.size),
                "cells_evaluated": int((self.n_per_cell >= self.min_cell_n).sum()),
                "cells_bimodal": int(self.is_bimodal.sum()),
                "train_stars": int(self.n_per_cell.sum()),
            },
        }
        if provenance is not None:
            sidecar["source"] = provenance
        path.with_suffix(path.suffix + ".provenance.json").write_text(
            json.dumps(sidecar, indent=2),
        )

    @classmethod
    def load(cls, path: Path) -> "BimodalityGrid":
        path = Path(path)
        z = np.load(path)
        sidecar = json.loads(
            path.with_suffix(path.suffix + ".provenance.json").read_text(),
        )
        crit = sidecar["criteria"]
        return cls(
            teff_edges=z["teff_edges"],
            logg_edges=z["logg_edges"],
            mh_edges=z["mh_edges"],
            is_bimodal=z["is_bimodal"],
            n_per_cell=z["n_per_cell"],
            min_cell_n=int(crit["min_cell_n"]),
            min_minor_weight=float(crit["min_minor_weight"]),
            min_mean_sep=float(crit["min_mean_sep"]),
            bic_delta_min=float(crit["bic_delta_min"]),
        )


def _is_cell_bimodal(
    a: np.ndarray,
    *,
    min_minor_weight: float,
    min_mean_sep: float,
    bic_delta_min: float,
    random_state: int = 0,
) -> tuple[bool, dict]:
    """Test a single cell's α/M distribution for bimodality.

    The cell is flagged bimodal iff all three are satisfied:
      1. BIC(2-GMM) + ``bic_delta_min`` < BIC(1-GMM).
      2. min(weights) ≥ ``min_minor_weight``.
      3. |μ_1 − μ_2| ≥ ``min_mean_sep``.

    Returns ``(flag, stats_dict)`` for provenance.
    """
    a = a.reshape(-1, 1)
    if a.shape[0] < 10:
        return False, {"reason": "too few points"}
    try:
        gm1 = GaussianMixture(n_components=1, random_state=random_state).fit(a)
        gm2 = GaussianMixture(
            n_components=2,
            random_state=random_state,
            n_init=3,
            reg_covar=1e-5,
        ).fit(a)
    except Exception as exc:  # noqa: BLE001 — GMM convergence failures get surfaced in stats
        return False, {"reason": f"gmm_failed:{type(exc).__name__}"}

    bic1 = float(gm1.bic(a))
    bic2 = float(gm2.bic(a))
    mu1, mu2 = gm2.means_.flatten().tolist()
    w1, w2 = gm2.weights_.tolist()
    mean_sep = float(abs(mu1 - mu2))
    minor_w = float(min(w1, w2))

    flag = bic2 + bic_delta_min < bic1 and minor_w >= min_minor_weight and mean_sep >= min_mean_sep
    stats = {
        "bic1": bic1,
        "bic2": bic2,
        "bic_delta": bic1 - bic2,
        "weights": [w1, w2],
        "means": [mu1, mu2],
        "mean_sep": mean_sep,
    }
    return flag, stats


def fit_bimodality_grid(
    teff: np.ndarray,
    logg: np.ndarray,
    mh: np.ndarray,
    alpha_m: np.ndarray,
    *,
    teff_edges: np.ndarray | None = None,
    logg_edges: np.ndarray | None = None,
    mh_edges: np.ndarray | None = None,
    min_cell_n: int = 50,
    min_minor_weight: float = 0.15,
    min_mean_sep: float = 0.08,
    bic_delta_min: float = 4.0,
    random_state: int = 0,
) -> BimodalityGrid:
    """Bin training stars by (Teff, log g, [M/H]) and test each cell for α/M bimodality.

    Parameters
    ----------
    teff, logg, mh, alpha_m
        Training labels, same length. NaNs are dropped.
    teff_edges, logg_edges, mh_edges
        Bin edges. Defaults: Teff every 150 K in [3500, 6000]; log g every
        0.3 dex in [0.0, 4.0]; [M/H] every 0.2 dex in [−3.0, 0.5]. These
        produce ~17 × 13 × 17 ≈ 3 800 cells for ~300 k training stars
        (mean ≈ 80 stars / cell).
    min_cell_n
        Skip cells with fewer than this many stars (not enough evidence).
    min_minor_weight, min_mean_sep, bic_delta_min
        Bimodality thresholds. Defaults tuned to flag the thin/thick-disc
        α-sequence cell at ([M/H]≈−0.5, Teff≈4800 K, log g≈2.5) while
        not flagging unimodal cells with mild skew.
    """
    if teff_edges is None:
        teff_edges = np.arange(3500.0, 6000.01, 150.0)
    if logg_edges is None:
        logg_edges = np.arange(0.0, 4.001, 0.30)
    if mh_edges is None:
        # Edges shifted off-grid by half a bin width on the [M/H] axis to
        # avoid placing a cell boundary at exactly [M/H] = 0 dex. The original
        # `np.arange(-3.0, 0.501, 0.20)` put an edge at 0, which produced a
        # tier-filter cliff in Stream-3 inference (mode_ambiguous_flag fired
        # for 65% of stars at -0.005 dex but 0% at +0.005 dex because the
        # adjacent (Teff, logg) cells on either side of the boundary had
        # very different bimodal-coverage statistics). Putting 0.0 in the
        # middle of a cell removes the boundary effect.
        mh_edges = np.arange(-2.9, 0.401, 0.20)

    teff_edges = np.asarray(teff_edges, dtype=np.float64)
    logg_edges = np.asarray(logg_edges, dtype=np.float64)
    mh_edges = np.asarray(mh_edges, dtype=np.float64)

    finite = np.isfinite(teff) & np.isfinite(logg) & np.isfinite(mh) & np.isfinite(alpha_m)
    teff, logg, mh, alpha_m = teff[finite], logg[finite], mh[finite], alpha_m[finite]

    t_idx = np.digitize(teff, teff_edges) - 1
    l_idx = np.digitize(logg, logg_edges) - 1
    m_idx = np.digitize(mh, mh_edges) - 1

    nT = len(teff_edges) - 1
    nL = len(logg_edges) - 1
    nM = len(mh_edges) - 1
    is_bi = np.zeros((nT, nL, nM), dtype=bool)
    n_cell = np.zeros((nT, nL, nM), dtype=np.int32)
    per_cell_gmm: dict[tuple[int, int, int], dict] = {}

    in_grid = (
        (t_idx >= 0) & (t_idx < nT) & (l_idx >= 0) & (l_idx < nL) & (m_idx >= 0) & (m_idx < nM)
    )
    t_idx, l_idx, m_idx, alpha_m = t_idx[in_grid], l_idx[in_grid], m_idx[in_grid], alpha_m[in_grid]

    flat = t_idx * (nL * nM) + l_idx * nM + m_idx
    order = np.argsort(flat, kind="stable")
    flat_sorted = flat[order]
    alpha_sorted = alpha_m[order]
    breaks = np.concatenate(([0], np.flatnonzero(np.diff(flat_sorted)) + 1, [len(flat_sorted)]))
    for seg_start, seg_end in zip(breaks[:-1], breaks[1:], strict=True):
        cell_flat = int(flat_sorted[seg_start])
        i = cell_flat // (nL * nM)
        j = (cell_flat // nM) % nL
        k = cell_flat % nM
        n = seg_end - seg_start
        n_cell[i, j, k] = n
        if n < min_cell_n:
            continue
        flag, stats = _is_cell_bimodal(
            alpha_sorted[seg_start:seg_end],
            min_minor_weight=min_minor_weight,
            min_mean_sep=min_mean_sep,
            bic_delta_min=bic_delta_min,
            random_state=random_state,
        )
        is_bi[i, j, k] = flag
        if flag:
            per_cell_gmm[(i, j, k)] = stats

    return BimodalityGrid(
        teff_edges=teff_edges,
        logg_edges=logg_edges,
        mh_edges=mh_edges,
        is_bimodal=is_bi,
        n_per_cell=n_cell,
        min_cell_n=min_cell_n,
        min_minor_weight=min_minor_weight,
        min_mean_sep=min_mean_sep,
        bic_delta_min=bic_delta_min,
        per_cell_gmm=per_cell_gmm,
    )
