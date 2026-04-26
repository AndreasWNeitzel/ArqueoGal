"""Stage 21: 3-component GMM tracking on chemistry (criterion 2 of methods §3.6).

Fits a 3-component GMM on truth chemistry, follows assignments into pred
plane, Hungarian-aligns an independent re-fit. Reports per-component
centroid drift, ARI, purity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

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
OUT = REPO / "reports/gallery/22_gmm_cluster_tracking"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHEM_X = (-2.5, 0.6); CHEM_Y = (-0.2, 0.5)
COMP_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c")
COMP_LABELS = ("G1 (metal-poor)", "G2 (mid)", "G3 (metal-rich)")


def _load_split():
    arr = load_arrays(S1, FeatureLayout(), LabelTiers.five_label(),
                      include_label_errors=False)
    X = np.asarray(arr["X"]); Y = np.asarray(arr["Y"]); sid = np.asarray(arr["source_id"])
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    keep = np.isfinite(Y).all(axis=1)
    X, Y, sid = X[keep], Y[keep], sid[keep]
    _, fi = np.unique(sid, return_index=True); fi = np.sort(fi)
    X, Y = X[fi], Y[fi]
    rng = np.random.default_rng(20260425)
    perm = rng.permutation(len(X))
    n_tr = int(0.70 * len(X)); n_val = int(0.15 * len(X))
    return X, Y, np.sort(perm[:n_tr]), np.sort(perm[n_tr:n_tr + n_val]), np.sort(perm[n_tr + n_val:])


def _knn_pred(model, X, tri, vai, tei, Y):
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
    return (np.median(Y_pool[idx_tr], axis=1),
             np.median(Y_pool[idx_va], axis=1),
             np.median(Y_pool[idx_te], axis=1))


def _fit_align(chem_t, chem_p):
    rs = np.random.RandomState(20260425)
    g_t = GaussianMixture(n_components=3, covariance_type="full",
                          random_state=rs, n_init=5).fit(chem_t)
    rs2 = np.random.RandomState(20260425)
    g_p = GaussianMixture(n_components=3, covariance_type="full",
                          random_state=rs2, n_init=5).fit(chem_p)
    lt = g_t.predict(chem_t); lp_raw = g_p.predict(chem_p)
    order = np.argsort(g_t.means_[:, 0]); relabel = np.argsort(order)
    lt = relabel[lt]
    means_t = g_t.means_[order]
    cost = np.linalg.norm(g_p.means_[:, None, :] - means_t[None, :, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    pred_to_truth = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
    lp = np.array([pred_to_truth[lbl] for lbl in lp_raw])
    means_p = g_p.means_[row_ind][np.argsort(col_ind)]
    return lt, lp, means_t, means_p


def main() -> None:
    apply_style()
    members = load_ensemble(ENCODER)
    model = members[0].model.to(DEVICE).eval()
    X, Y, tri, vai, tei = _load_split()
    pred_tr, pred_va, pred_te = _knn_pred(model, X, tri, vai, tei, Y)
    splits = (("Train (LOO)", Y[tri], pred_tr),
               ("Val", Y[vai], pred_va),
               ("Test", Y[tei], pred_te))

    metrics = {}
    rng = np.random.default_rng(20260425)
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    for r, (name, Yt, Yp) in enumerate(splits):
        ct = np.column_stack([Yt[:, 2], Yt[:, 3]])
        cp = np.column_stack([Yp[:, 2], Yp[:, 3]])
        lt, lp, mt, mp = _fit_align(ct, cp)
        drift = np.linalg.norm(mp - mt, axis=1)
        ari = float(adjusted_rand_score(lt, lp))
        purity = float((lt == lp).mean())
        metrics[name] = {
            "centroid_drift_per_comp": drift.tolist(),
            "centroid_drift_total_rms": float(np.sqrt(np.mean(drift**2))),
            "adjusted_rand_index": ari,
            "purity": purity,
            "truth_centroids": mt.tolist(),
            "pred_centroids_aligned": mp.tolist(),
        }

        sub_n = min(60_000, len(ct))
        idx = rng.choice(len(ct), sub_n, replace=False) if len(ct) > sub_n else np.arange(len(ct))

        for col, (X_plot, Y_plot, labels, title) in enumerate([
            (ct[idx, 0], ct[idx, 1], lt[idx], "Truth (GMM-3)"),
            (cp[idx, 0], cp[idx, 1], lt[idx], "Pred (truth-color tracking)"),
            (cp[idx, 0], cp[idx, 1], lp[idx], "Pred GMM-refit (Hungarian)"),
        ]):
            ax = axes[r, col]
            for k in range(3):
                m = labels == k
                ax.scatter(X_plot[m], Y_plot[m], s=0.5, c=COMP_COLORS[k], alpha=0.4,
                            rasterized=True, label=COMP_LABELS[k] if (r, col) == (0, 0) else None)
            for k in range(3):
                ax.scatter(mt[k, 0], mt[k, 1], marker="*", s=120, c=COMP_COLORS[k],
                            edgecolor="k", lw=0.8, zorder=5)
                if col >= 1:
                    ax.scatter(mp[k, 0], mp[k, 1], marker="*", s=120, facecolor="none",
                                edgecolor=COMP_COLORS[k], lw=1.4, zorder=6)
            ax.set_xlim(CHEM_X); ax.set_ylim(CHEM_Y)
            ax.set_xlabel("[M/H] (dex)", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{name}\n[α/M] (dex)", fontsize=8)
            else:
                ax.set_ylabel("[α/M] (dex)", fontsize=8)
            if r == 0:
                ax.set_title(title, fontsize=9)
                if col == 0:
                    ax.legend(fontsize=7, loc="lower left")
            ax.tick_params(labelsize=7)
        # ARI / drift annotation in last col
        text = (f"ARI={ari:.3f}\npurity={purity:.3f}\n"
                f"drift RMS={metrics[name]['centroid_drift_total_rms']:.3f} dex")
        axes[r, 2].text(0.05, 0.95, text, transform=axes[r, 2].transAxes,
                          fontsize=7, ha="left", va="top",
                          bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.92, pad=2))

    fig.suptitle(
        "3-component GMM cluster tracking — Stream 1 only (truth + pred + GMM-refit).\n"
        "Streams 2 and 3 have no APOGEE truth labels, so this validation cannot be repeated for them. "
        "S2 / S3 inference-side chemistry distributions are shown in plots 20 (planes) and 26 (S2 summary).",
        fontsize=10,
    )
    fig.tight_layout()
    save_fig(fig, OUT / "gmm_cluster_tracking.png", tight=False)
    (OUT / "gmm_cluster_tracking_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
