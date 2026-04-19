"""FIRE-2 Ananke hare-and-hounds validation — Subtask 5.1, research_brief §10.6.

Method-validation only. Given predicted cluster labels ``y_pred`` and
ground-truth FIRE-2 labels ``y_true`` (e.g. ``dform`` for
in-situ / accreted, or a chemical high-α/low-α flag), compute:

- **ARI** (Hubert & Arabie 1985) — adjusted Rand index.
- **AMI** (Vinh+2010) — adjusted mutual information.
- **MCC** (Matthews 1975) — multi-class Matthews correlation coefficient.
- **Youden J / informedness** (Youden 1950; bookmaker's informedness
  for multi-class) — macro-averaged ``TPR + TNR - 1`` per matched class.
- Per-cluster best-match precision/recall via Hungarian optimal
  assignment.

This module is strictly the metric layer. It takes labels in, produces
numbers out — it does NOT touch the real-data diagnostic stack (§10.5)
and it does NOT transfer FIRE-2 performance to real-data claims per
DESIGN §40/§10.6. Subtask 5.2 stands alone on the §10.5 diagnostics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    matthews_corrcoef,
)


@dataclass
class HareHoundsReport:
    """Per-run FIRE-2 hare-and-hounds metrics.

    ``ari`` / ``ami`` / ``mcc``: partition-level scores on the matched
    subset.
    ``youden_j``: macro-averaged bookmaker informedness across matched
    classes (``TPR + TNR - 1``).
    ``contingency``: ``(K_pred, K_true)`` count matrix.
    ``predicted_ids`` / ``true_ids``: row / column order of the
    contingency matrix.
    ``match``: Hungarian best-match ``{predicted_id: true_id}`` using
    the negative contingency as the assignment cost.
    ``per_cluster_precision`` / ``per_cluster_recall``: best-match
    precision / recall per predicted cluster (length ``K_pred``).
    ``n_stars_compared``: size of the matched subset (after dropping
    ignored labels).
    """

    ari: float
    ami: float
    mcc: float
    youden_j: float
    contingency: np.ndarray
    predicted_ids: tuple[int, ...]
    true_ids: tuple[Any, ...]
    match: dict[int, Any]
    per_cluster_precision: np.ndarray
    per_cluster_recall: np.ndarray
    n_stars_compared: int


def _contingency(
    y_pred: np.ndarray, y_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_ids = np.unique(y_pred)
    true_ids = np.unique(y_true)
    K, L = pred_ids.size, true_ids.size
    cont = np.zeros((K, L), dtype=np.int64)
    for ki, pid in enumerate(pred_ids):
        sel = y_pred == pid
        for li, lid in enumerate(true_ids):
            cont[ki, li] = int((sel & (y_true == lid)).sum())
    return cont, pred_ids, true_ids


def _macro_youden_j(
    y_pred_int: np.ndarray, y_true_int: np.ndarray, n_classes: int,
) -> float:
    """Macro-averaged Youden J (bookmaker's informedness) over ``n_classes``.

    For each class c: J_c = TPR_c + TNR_c - 1.
    """
    js: list[float] = []
    n = y_true_int.size
    for c in range(n_classes):
        tp = int(((y_pred_int == c) & (y_true_int == c)).sum())
        fn = int(((y_pred_int != c) & (y_true_int == c)).sum())
        fp = int(((y_pred_int == c) & (y_true_int != c)).sum())
        tn = n - tp - fn - fp
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        js.append(tpr + tnr - 1.0)
    return float(np.mean(js)) if js else float("nan")


def compute_hare_hounds_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    ignore_predicted: Sequence[int] = (-1,),
    ignore_true: Sequence[Any] = (),
) -> HareHoundsReport:
    """ARI + AMI + MCC + Youden-J for predicted-vs-true labelings.

    Stars with predicted label in ``ignore_predicted`` (default: noise)
    or true label in ``ignore_true`` are dropped before metric
    computation — standard FIRE-2 practice is to drop noise predictions
    and drop ground-truth categories that are out-of-scope for the
    hounds problem (e.g. mergers that never should have been classified).
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"y_pred {y_pred.shape} vs y_true {y_true.shape} shape mismatch",
        )
    mask = ~np.isin(y_pred, list(ignore_predicted)) & ~np.isin(
        y_true, list(ignore_true),
    )
    yp = y_pred[mask]
    yt = y_true[mask]
    if yp.size < 2:
        return HareHoundsReport(
            ari=float("nan"), ami=float("nan"),
            mcc=float("nan"), youden_j=float("nan"),
            contingency=np.zeros((0, 0), dtype=np.int64),
            predicted_ids=(), true_ids=(), match={},
            per_cluster_precision=np.zeros(0),
            per_cluster_recall=np.zeros(0),
            n_stars_compared=int(yp.size),
        )

    cont, pred_ids, true_ids = _contingency(yp, yt)
    ari = float(adjusted_rand_score(yt, yp))

    true_key = {v: i for i, v in enumerate(true_ids)}
    pred_key = {int(v): i for i, v in enumerate(pred_ids)}
    yt_int = np.fromiter((true_key[v] for v in yt), dtype=np.int64)
    yp_int = np.fromiter((pred_key[int(v)] for v in yp), dtype=np.int64)
    ami = float(adjusted_mutual_info_score(yt_int, yp_int))

    # Hungarian best-match on negative contingency.
    K, L = cont.shape
    size = max(K, L)
    cost = np.zeros((size, size), dtype=np.float64)
    cost[:K, :L] = -cont
    row_ind, col_ind = linear_sum_assignment(cost)
    match: dict[int, Any] = {}
    for r, c in zip(row_ind, col_ind, strict=True):
        if r < K and c < L:
            match[int(pred_ids[r])] = true_ids[c]

    # Map predicted → matched true index for MCC / Youden.
    n_classes = max(K, L)
    # Unmatched predicted ids are given a unique out-of-range class so they
    # never contribute as positives for any true class in the Youden sum.
    remap_pred = np.full(yp_int.shape, -1, dtype=np.int64)
    for pid, tid in match.items():
        remap_pred[yp_int == pred_key[int(pid)]] = true_key[tid]
    yp_remapped = np.where(remap_pred >= 0, remap_pred, n_classes + 1)

    mcc = float(matthews_corrcoef(yt_int, yp_remapped))
    youden_j = _macro_youden_j(yp_remapped, yt_int, n_classes=L)

    row_sums = cont.sum(axis=1).clip(min=1)
    col_sums = cont.sum(axis=0).clip(min=1)
    best_row = cont.max(axis=1) if K else np.zeros(0, dtype=np.int64)
    best_col = cont.max(axis=0) if L else np.zeros(0, dtype=np.int64)
    precision = best_row / row_sums
    recall = best_col / col_sums

    return HareHoundsReport(
        ari=ari,
        ami=ami,
        mcc=mcc,
        youden_j=youden_j,
        contingency=cont,
        predicted_ids=tuple(int(p) for p in pred_ids.tolist()),
        true_ids=tuple(true_ids.tolist()),
        match=match,
        per_cluster_precision=precision.astype(np.float64),
        per_cluster_recall=recall.astype(np.float64),
        n_stars_compared=int(yp.size),
    )


__all__ = [
    "HareHoundsReport",
    "compute_hare_hounds_metrics",
]
