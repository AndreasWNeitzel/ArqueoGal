"""Stage 15: Kiel + chemistry truth-vs-pred on splits (post-training sanity check 2).

Same 70/15/15 split as stage 14. For each split, plots four panels: Kiel
truth, Kiel pred, chemistry truth, chemistry pred. The reader can see at a
glance whether the trained model preserves the giant branch shape (Kiel) and
the disc bimodality (chemistry).
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

ENCODER = (REPO / "models/main/xp_abundances"
            / "20260425_6b96c06_cd1cbb9_ensemble_5label" / "member_seed0")
S1 = REPO / "data/processed/pipeline1_features_stream1.parquet"
OUT = REPO / "reports/gallery/15_kiel_chem_truth_pred"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KIEL_X = (3500, 6500); KIEL_Y = (0.5, 4.0)
CHEM_X = (-2.5, 0.6); CHEM_Y = (-0.2, 0.5)


def _stats(yt, yp):
    finite = np.isfinite(yt) & np.isfinite(yp)
    res = yp[finite] - yt[finite]
    if len(res) == 0:
        return 0.0, 0.0, 0.0, 0
    rmse = float(np.sqrt(np.mean(res**2)))
    bias = float(np.median(res))
    std = float(np.std(res - bias))
    return rmse, bias, std, int(finite.sum())


def _kiel(ax, x, y, title, anno=None):
    m = np.isfinite(x) & np.isfinite(y)
    ax.hexbin(x[m], y[m], gridsize=70, mincnt=5, cmap="viridis", bins="log",
               extent=[KIEL_X[0], KIEL_X[1], KIEL_Y[0], KIEL_Y[1]])
    ax.set_xlim(KIEL_X[1], KIEL_X[0]); ax.set_ylim(KIEL_Y[1], KIEL_Y[0])
    ax.set_xlabel("Teff (K)", fontsize=8); ax.set_ylabel(r"$\log g$ (dex)", fontsize=8)
    ax.set_title(title, fontsize=9)
    if anno:
        ax.text(0.04, 0.04, anno, transform=ax.transAxes, fontsize=6.5,
                 ha="left", va="bottom",
                 bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.92, pad=2))


def _chem(ax, x, y, title, anno=None):
    m = np.isfinite(x) & np.isfinite(y)
    ax.hexbin(x[m], y[m], gridsize=70, mincnt=5, cmap="viridis", bins="log",
               extent=[CHEM_X[0], CHEM_X[1], CHEM_Y[0], CHEM_Y[1]])
    ax.set_xlim(CHEM_X); ax.set_ylim(CHEM_Y)
    ax.set_xlabel("[M/H] (dex)", fontsize=8)
    ax.set_ylabel(r"[$\alpha$/M] (dex)", fontsize=8)
    ax.set_title(title, fontsize=9)
    if anno:
        ax.text(0.04, 0.96, anno, transform=ax.transAxes, fontsize=6.5,
                 ha="left", va="top",
                 bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.92, pad=2))


def _load():
    arr = load_arrays(S1, FeatureLayout(), LabelTiers.five_label(),
                      include_label_errors=False)
    X = np.asarray(arr["X"]); Y = np.asarray(arr["Y"]); sid = np.asarray(arr["source_id"])
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    keep = np.isfinite(Y).all(axis=1)
    X, Y, sid = X[keep], Y[keep], sid[keep]
    _, fi = np.unique(sid, return_index=True); fi = np.sort(fi)
    return X[fi], Y[fi]


def main() -> None:
    apply_style()
    members = load_ensemble(ENCODER)
    model = members[0].model.to(DEVICE).eval()
    X, Y = _load()
    rng = np.random.default_rng(20260425)
    perm = rng.permutation(len(X))
    n_tr = int(0.70 * len(X)); n_val = int(0.15 * len(X))
    tri = np.sort(perm[:n_tr]); vai = np.sort(perm[n_tr:n_tr + n_val])
    tei = np.sort(perm[n_tr + n_val:])
    z_tr = compute_latents(model, X[tri], device=DEVICE)
    z_va = compute_latents(model, X[vai], device=DEVICE)
    z_te = compute_latents(model, X[tei], device=DEVICE)
    for z in (z_tr, z_va, z_te):
        z /= np.linalg.norm(z, axis=1, keepdims=True).clip(min=1e-12)
    _, idx_va = gpu_knn_search(z_tr, z_va, k=50, device=DEVICE)
    _, idx_te = gpu_knn_search(z_tr, z_te, k=50, device=DEVICE)
    _, idx_tr_full = gpu_knn_search(z_tr, z_tr, k=51, device=DEVICE)
    idx_tr = idx_tr_full[:, 1:]
    Y_pool = Y[tri]
    splits = (("Train (LOO kNN)", Y[tri], np.median(Y_pool[idx_tr], axis=1)),
               ("Validation", Y[vai], np.median(Y_pool[idx_va], axis=1)),
               ("Test", Y[tei], np.median(Y_pool[idx_te], axis=1)))

    fig, axes = plt.subplots(3, 4, figsize=(13, 9))
    for r, (name, Yt, Yp) in enumerate(splits):
        teff_t, logg_t = Yt[:, 0], Yt[:, 1]
        teff_p, logg_p = Yp[:, 0], Yp[:, 1]
        mh_t, am_t = Yt[:, 2], Yt[:, 3]
        mh_p, am_p = Yp[:, 2], Yp[:, 3]

        rmse_t, bias_t, *_ = _stats(teff_t, teff_p)
        rmse_g, bias_g, *_ = _stats(logg_t, logg_p)
        rmse_m, bias_m, *_ = _stats(mh_t, mh_p)
        rmse_a, bias_a, _, n = _stats(am_t, am_p)

        _kiel(axes[r, 0], teff_t, logg_t,
               title=("Kiel: truth" if r == 0 else ""), anno=f"n={n:,}")
        axes[r, 0].set_ylabel(f"{name}\n" + r"$\log g$ (dex)", fontsize=8)
        _kiel(axes[r, 1], teff_p, logg_p,
               title=("Kiel: pred" if r == 0 else ""),
               anno=(f"Teff: RMSE={rmse_t:.0f} K, bias={bias_t:+.0f}\n"
                      r"$\log g$: " + f"RMSE={rmse_g:.3f}, bias={bias_g:+.3f}"))
        _chem(axes[r, 2], mh_t, am_t,
               title=("Chemistry: truth" if r == 0 else ""))
        axes[r, 2].set_ylabel(r"[$\alpha$/M] (dex)", fontsize=8)
        _chem(axes[r, 3], mh_p, am_p,
               title=("Chemistry: pred" if r == 0 else ""),
               anno=(f"[M/H]: RMSE={rmse_m:.3f}, bias={bias_m:+.3f}\n"
                      r"[$\alpha$/M]: " + f"RMSE={rmse_a:.3f}, bias={bias_a:+.3f}"))

    fig.suptitle("Kiel and chemistry truth-vs-pred on 70/15/15 splits "
                  "(immediate post-training)", fontsize=11)
    fig.tight_layout()
    save_fig(fig, OUT / "kiel_chem_truth_pred.png", tight=False)


if __name__ == "__main__":
    main()
