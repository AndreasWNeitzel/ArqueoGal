"""Stage 22: per-class contamination (criterion 3 of methods §3.6).

Confusion matrix + precision/recall/F1 + flow + Hellinger/TV per cluster
on the truth-vs-pred GMM. Produces the metrics JSON cited in methods.md §3.6.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.mixture import GaussianMixture

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
OUT = REPO / "reports/gallery/23_contamination"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
XB = np.linspace(-2.5, 0.6, 64)
YB = np.linspace(-0.2, 0.5, 64)


def _load_split():
    arr = load_arrays(S1, FeatureLayout(), LabelTiers.five_label(), include_label_errors=False)
    X = np.asarray(arr["X"])
    Y = np.asarray(arr["Y"])
    sid = np.asarray(arr["source_id"])
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    keep = np.isfinite(Y).all(axis=1)
    X, Y, sid = X[keep], Y[keep], sid[keep]
    _, fi = np.unique(sid, return_index=True)
    fi = np.sort(fi)
    X, Y = X[fi], Y[fi]
    rng = np.random.default_rng(20260425)
    perm = rng.permutation(len(X))
    n_tr = int(0.70 * len(X))
    n_val = int(0.15 * len(X))
    return (
        X,
        Y,
        np.sort(perm[:n_tr]),
        np.sort(perm[n_tr : n_tr + n_val]),
        np.sort(perm[n_tr + n_val :]),
    )


def _fit_align(chem_t, chem_p):
    rs = np.random.RandomState(20260425)
    g_t = GaussianMixture(n_components=3, covariance_type="full", random_state=rs, n_init=5).fit(
        chem_t
    )
    rs2 = np.random.RandomState(20260425)
    g_p = GaussianMixture(n_components=3, covariance_type="full", random_state=rs2, n_init=5).fit(
        chem_p
    )
    lt = g_t.predict(chem_t)
    lp_raw = g_p.predict(chem_p)
    order = np.argsort(g_t.means_[:, 0])
    relabel = np.argsort(order)
    lt = relabel[lt]
    means_t = g_t.means_[order]
    cost = np.linalg.norm(g_p.means_[:, None, :] - means_t[None, :, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    pred_to_truth = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
    lp = np.array([pred_to_truth[lbl] for lbl in lp_raw])
    return lt, lp


def _hell(a, b):
    return float(np.sqrt(0.5 * np.sum((np.sqrt(a) - np.sqrt(b)) ** 2)))


def _tv(a, b):
    return float(0.5 * np.sum(np.abs(a - b)))


def main() -> None:
    apply_style()
    members = load_ensemble(ENCODER)
    model = members[0].model.to(DEVICE).eval()
    X, Y, tri, vai, tei = _load_split()
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
    splits = (
        ("Train (LOO)", Y[tri], np.median(Y_pool[idx_tr], axis=1)),
        ("Val", Y[vai], np.median(Y_pool[idx_va], axis=1)),
        ("Test", Y[tei], np.median(Y_pool[idx_te], axis=1)),
    )

    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    metrics = {}
    for r, (name, Yt, Yp) in enumerate(splits):
        ct = np.column_stack([Yt[:, 2], Yt[:, 3]])
        cp = np.column_stack([Yp[:, 2], Yp[:, 3]])
        lt, lp = _fit_align(ct, cp)
        C = confusion_matrix(lt, lp, labels=[0, 1, 2])
        n_truth = C.sum(axis=1).clip(min=1)
        n_pred = C.sum(axis=0).clip(min=1)
        recall = np.diag(C) / n_truth
        purity = np.diag(C) / n_pred
        f1 = 2 * recall * purity / (recall + purity + 1e-12)
        flow_pct = (C / n_truth[:, None]) * 100
        hell, tv = [], []
        for k in range(3):
            ht, _, _ = np.histogram2d(ct[lt == k, 0], ct[lt == k, 1], bins=[XB, YB])
            hp, _, _ = np.histogram2d(cp[lp == k, 0], cp[lp == k, 1], bins=[XB, YB])
            ht /= max(ht.sum(), 1)
            hp /= max(hp.sum(), 1)
            hell.append(_hell(ht, hp))
            tv.append(_tv(ht, hp))

        macro_f1 = float(np.mean(f1))
        ari = float(adjusted_rand_score(lt, lp))
        metrics[name] = {
            "confusion_matrix": C.tolist(),
            "completeness_recall": recall.tolist(),
            "purity_precision": purity.tolist(),
            "f1_per_class": f1.tolist(),
            "macro_f1": macro_f1,
            "flow_pct_truth_to_pred": flow_pct.tolist(),
            "hellinger_per_class": hell,
            "tv_per_class": tv,
            "ari": ari,
        }

        ax = axes[r, 0]
        cm_norm = C / n_truth[:, None]
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    f"{int(C[i, j]):,}\n({100 * cm_norm[i, j]:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                )
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["G1", "G2", "G3"], fontsize=8)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["G1", "G2", "G3"], fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel(f"{name}\ntruth")
        if r == 0:
            ax.set_title("Confusion matrix")

        ax = axes[r, 1]
        x = np.arange(3)
        w = 0.27
        ax.bar(x - w, recall, w, color="#1f77b4", label="recall")
        ax.bar(x, purity, w, color="#ff7f0e", label="purity")
        ax.bar(x + w, f1, w, color="#2ca02c", label="F1")
        for i, (rc, pu, fv) in enumerate(zip(recall, purity, f1)):
            for off, v in zip((-w, 0, w), (rc, pu, fv)):
                ax.text(i + off, v + 0.02, f"{v:.2f}", ha="center", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(["G1", "G2", "G3"])
        ax.set_ylim(0, 1.10)
        ax.set_ylabel("score")
        if r == 0:
            ax.set_title("Per-class recall / purity / F1")
            ax.legend(fontsize=7, loc="upper right")
        ax.text(
            0.02,
            0.98,
            f"macro-F1={macro_f1:.3f}\nARI={ari:.3f}",
            transform=ax.transAxes,
            fontsize=7,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9, pad=2),
        )

        ax = axes[r, 2]
        bottoms = np.zeros(3)
        pal = ("#1f77b4", "#ff7f0e", "#2ca02c")
        for j in range(3):
            bars = ax.bar(
                ["G1", "G2", "G3"],
                flow_pct[:, j],
                0.6,
                bottom=bottoms,
                color=pal[j],
                label=f"→ pred G{j + 1}",
                edgecolor="white",
                lw=0.4,
            )
            for i, b in enumerate(bars):
                if flow_pct[i, j] > 5:
                    ax.text(
                        b.get_x() + b.get_width() / 2,
                        b.get_y() + b.get_height() / 2,
                        f"{flow_pct[i, j]:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white",
                    )
            bottoms += flow_pct[:, j]
        ax.set_ylabel("% of truth class")
        ax.set_xlabel("truth")
        if r == 0:
            ax.set_title("Contamination flow")
            ax.legend(fontsize=6, loc="upper right")

        ax = axes[r, 3]
        x = np.arange(3)
        w = 0.35
        ax.bar(x - w / 2, hell, w, color="#1f77b4", label="Hellinger")
        ax.bar(x + w / 2, tv, w, color="#d62728", label="TV")
        for i, (h, t) in enumerate(zip(hell, tv)):
            ax.text(i - w / 2, h + 0.02, f"{h:.2f}", ha="center", fontsize=6)
            ax.text(i + w / 2, t + 0.02, f"{t:.2f}", ha="center", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(["G1", "G2", "G3"])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("distance")
        if r == 0:
            ax.set_title("Hellinger / TV (truth-Gi vs pred-Gi)")
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Per-class contamination diagnostics (criterion 3)", fontsize=11)
    fig.tight_layout()
    save_fig(fig, OUT / "contamination.png", tight=False)
    (OUT / "contamination_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
