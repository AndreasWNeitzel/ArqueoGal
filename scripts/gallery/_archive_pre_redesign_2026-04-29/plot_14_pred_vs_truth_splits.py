"""Stage 14: pred-vs-truth on the labelled set, IMMEDIATELY after training.

Holds out 70/15/15 train/val/test from Stream 1, runs LOO kNN on train and
frozen kNN on val/test against the strong-contrastive-v2 encoder. Each panel
annotated with n / RMSE / bias / std.

This is the first post-training sanity check: did the trained model recover
the labels on its own training distribution? If pred-vs-truth on Stream 1
val/test is unbiased and tight, downstream Stream-3 inference is grounded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
from _common import apply_style, save_fig

from arqueogal.xp_abundances.main.data import FeatureLayout, LabelTiers, load_arrays
from arqueogal.xp_abundances.main.inference import load_ensemble
from arqueogal.xp_abundances.main.knn_rescue import compute_latents, gpu_knn_search

ENCODER = (
    REPO / "models/main/xp_abundances" / "20260425_6b96c06_cd1cbb9_ensemble_5label" / "member_seed0"
)
S1 = REPO / "data/processed/pipeline1_features_stream1.parquet"
OUT = REPO / "reports/gallery/14_pred_vs_truth_splits"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = (
    ("teff", "Teff (K)", (3500, 6500)),
    ("logg", r"$\log g$ (dex)", (0.5, 4.0)),
    ("mh", "[M/H] (dex)", (-2.5, 0.6)),
    ("alpha_m", r"[$\alpha$/M] (dex)", (-0.2, 0.5)),
    ("mg_h", "[Mg/H] (dex)", (-1.5, 0.5)),
)


def _load() -> tuple[np.ndarray, np.ndarray]:
    arr = load_arrays(S1, FeatureLayout(), LabelTiers.five_label(), include_label_errors=False)
    X = np.asarray(arr["X"])
    Y = np.asarray(arr["Y"])
    sid = np.asarray(arr["source_id"])
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    keep = np.isfinite(Y).all(axis=1)
    X, Y, sid = X[keep], Y[keep], sid[keep]
    _, fi = np.unique(sid, return_index=True)
    fi = np.sort(fi)
    return X[fi], Y[fi]


def _stats(yt, yp):
    finite = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[finite], yp[finite]
    res = yp - yt
    rmse = float(np.sqrt(np.mean(res**2)))
    bias = float(np.median(res))
    std = float(np.std(res - bias))
    return rmse, bias, std, int(finite.sum())


def main() -> None:
    apply_style()
    members = load_ensemble(ENCODER)
    model = members[0].model.to(DEVICE).eval()
    X, Y = _load()
    rng = np.random.default_rng(20260425)
    perm = rng.permutation(len(X))
    n_tr = int(0.70 * len(X))
    n_val = int(0.15 * len(X))
    tri = np.sort(perm[:n_tr])
    vai = np.sort(perm[n_tr : n_tr + n_val])
    tei = np.sort(perm[n_tr + n_val :])
    z_tr = compute_latents(model, X[tri], device=DEVICE)
    z_va = compute_latents(model, X[vai], device=DEVICE)
    z_te = compute_latents(model, X[tei], device=DEVICE)
    z_tr /= np.linalg.norm(z_tr, axis=1, keepdims=True).clip(min=1e-12)
    z_va /= np.linalg.norm(z_va, axis=1, keepdims=True).clip(min=1e-12)
    z_te /= np.linalg.norm(z_te, axis=1, keepdims=True).clip(min=1e-12)
    _, idx_va = gpu_knn_search(z_tr, z_va, k=50, device=DEVICE)
    _, idx_te = gpu_knn_search(z_tr, z_te, k=50, device=DEVICE)
    _, idx_tr_full = gpu_knn_search(z_tr, z_tr, k=51, device=DEVICE)
    idx_tr = idx_tr_full[:, 1:]
    Y_pool = Y[tri]
    pred_tr = np.median(Y_pool[idx_tr], axis=1)
    pred_va = np.median(Y_pool[idx_va], axis=1)
    pred_te = np.median(Y_pool[idx_te], axis=1)

    splits = (
        ("Train (LOO kNN)", Y[tri], pred_tr),
        ("Validation", Y[vai], pred_va),
        ("Test", Y[tei], pred_te),
    )

    fig, axes = plt.subplots(3, 5, figsize=(16, 9))
    for r, (name, Yt, Yp) in enumerate(splits):
        for c, (lbl, axlbl, lim) in enumerate(LABELS):
            ax = axes[r, c]
            rmse, bias, std, n = _stats(Yt[:, c], Yp[:, c])
            m = np.isfinite(Yt[:, c]) & np.isfinite(Yp[:, c])
            ax.hexbin(
                Yt[m, c],
                Yp[m, c],
                gridsize=50,
                mincnt=5,
                cmap="viridis",
                bins="log",
                extent=[lim[0], lim[1], lim[0], lim[1]],
            )
            ax.plot(lim, lim, color="red", lw=0.6, ls="--")
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect("equal")
            if r == 2:
                ax.set_xlabel(f"truth {axlbl}", fontsize=8)
            ax.set_ylabel(f"{name}\npred {axlbl}" if c == 0 else f"pred {axlbl}", fontsize=8)
            unit = "K" if lbl == "teff" else "dex"
            fmt = ".0f" if lbl == "teff" else ".3f"
            txt = (
                f"n={n:,}\nRMSE={rmse:{fmt}} {unit}\n"
                f"bias={bias:+{fmt}} {unit}\nstd={std:{fmt}} {unit}"
            )
            ax.text(
                0.05,
                0.95,
                txt,
                transform=ax.transAxes,
                fontsize=6.5,
                ha="left",
                va="top",
                bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.92, pad=2),
            )
            ax.tick_params(labelsize=7)
    fig.suptitle("Pred vs truth on 70/15/15 splits (kNN+strong-contrastive-v2 hybrid)", fontsize=11)
    fig.tight_layout()
    save_fig(fig, OUT / "pred_vs_truth.png", tight=False)


if __name__ == "__main__":
    main()
