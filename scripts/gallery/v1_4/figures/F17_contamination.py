"""F17: 3-component GMM contamination matrix on Stream-1 Tier 1 (slide 18)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.mixture import GaussianMixture

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts" / "gallery" / "v1_4"))
sys.path.insert(0, str(REPO / "src"))

from _loaders import load_s1_holdout  # noqa: E402

from arqueogal.style import (  # noqa: E402
    LABELS, OKABE_ITO, apply_style, save,
)

K = 3
COMP_LABELS = (r"low-$\alpha$", r"high-$\alpha$", r"MP halo")
COMP_COLORS = (OKABE_ITO["blue"], OKABE_ITO["vermillion"], OKABE_ITO["red_purple"])
INIT = np.array([[+0.05, 0.00], [-0.40, 0.20], [-1.20, 0.25]])


def _fit_gmm(xy):
    mu = xy.mean(axis=0); sd = xy.std(axis=0); sd = np.where(sd > 0, sd, 1.0)
    z = (xy - mu) / sd
    seed_z = (INIT - mu) / sd
    gm = GaussianMixture(
        n_components=K, covariance_type="full",
        random_state=42, n_init=1, max_iter=400,
        init_params="k-means++", means_init=seed_z,
    )
    gm.fit(z)
    z_lab = gm.predict(z)
    means = gm.means_ * sd + mu
    S = np.diag(sd)
    covs = [S @ c @ S for c in gm.covariances_]
    fit_z = gm.means_
    perm = np.empty(K, dtype=int); used = set()
    for s in range(K):
        d = np.linalg.norm(fit_z - seed_z[s], axis=1)
        for f in np.argsort(d):
            if int(f) not in used:
                perm[s] = int(f); used.add(int(f)); break
    means = means[perm]; covs = [covs[k] for k in perm]
    relabel = np.empty(K, dtype=int)
    for new_k, old_k in enumerate(perm):
        relabel[old_k] = new_k
    return means, covs, relabel[z_lab]


def _assign(xy, means, covs):
    log_p = np.empty((len(xy), K))
    for k in range(K):
        cov_inv = np.linalg.pinv(covs[k])
        sign, logdet = np.linalg.slogdet(covs[k])
        if sign <= 0:
            logdet = float(np.log(np.linalg.det(covs[k]) + 1e-30))
        diff = xy - means[k]
        m = np.einsum("ni,ij,nj->n", diff, cov_inv, diff)
        log_p[:, k] = -0.5 * (m + logdet)
    return np.argmax(log_p, axis=1)


def _ellipses(ax, means, covs):
    for k in range(K):
        eigvals, eigvecs = np.linalg.eigh(covs[k])
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
        ang = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        for n in (1, 2):
            w = 2.0 * n * np.sqrt(max(eigvals[0], 1e-6))
            h = 2.0 * n * np.sqrt(max(eigvals[1], 1e-6))
            ax.add_patch(Ellipse(
                xy=means[k], width=w, height=h, angle=ang,
                facecolor="none", edgecolor=COMP_COLORS[k],
                lw=1.0, ls="--", alpha=0.85,
            ))


def main() -> int:
    apply_style()
    df = load_s1_holdout()
    t1 = df.loc[df["release_tier"] == 1].dropna(
        subset=["mh_apogee", "alpha_m_apogee", "mh_pred", "alpha_m_pred"]
    ).reset_index(drop=True)
    truth = t1[["mh_apogee", "alpha_m_apogee"]].to_numpy()
    pred = t1[["mh_pred", "alpha_m_pred"]].to_numpy()
    means, covs, truth_lab = _fit_gmm(truth)
    pred_lab = _assign(pred, means, covs)

    cm = confusion_matrix(truth_lab, pred_lab, labels=list(range(K)))
    # Row-normalised: each truth-component row sums to 1 (recall per
    # truth class). The brief asks specifically for row-normalised, not
    # cell-normalised.
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_total = cm / row_sums
    per_p = precision_score(truth_lab, pred_lab, average=None,
                              labels=list(range(K)), zero_division=0)
    per_r = recall_score(truth_lab, pred_lab, average=None,
                          labels=list(range(K)), zero_division=0)
    per_f = f1_score(truth_lab, pred_lab, average=None,
                      labels=list(range(K)), zero_division=0)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.0),
                              layout="constrained")

    ax = axes[0, 0]
    for k in range(K):
        m = truth_lab == k
        ax.scatter(truth[m, 0], truth[m, 1],
                   s=2, alpha=0.45, color=COMP_COLORS[k],
                   edgecolor="none", rasterized=True,
                   label=f"{COMP_LABELS[k]} (n={int(m.sum()):,})")
    _ellipses(ax, means, covs)
    ax.set_xlim(-2.0, 0.55); ax.set_ylim(-0.10, 0.45)
    ax.set_xlabel(LABELS["Mh"]); ax.set_ylabel(LABELS["alpha_M"])
    ax.set_title("APOGEE truth, by truth component")
    ax.legend(loc="upper right", fontsize=8, markerscale=4)
    ax.grid(False)

    ax = axes[0, 1]
    for k in range(K):
        m = pred_lab == k
        ax.scatter(pred[m, 0], pred[m, 1],
                   s=2, alpha=0.45, color=COMP_COLORS[k],
                   edgecolor="none", rasterized=True,
                   label=f"{COMP_LABELS[k]} (n={int(m.sum()):,})")
    _ellipses(ax, means, covs)
    ax.set_xlim(-2.0, 0.55); ax.set_ylim(-0.10, 0.45)
    ax.set_xlabel(LABELS["Mh"]); ax.set_ylabel(LABELS["alpha_M"])
    ax.set_title("JANUS pred, by predicted component")
    ax.legend(loc="upper right", fontsize=8, markerscale=4)
    ax.grid(False)

    ax = axes[1, 0]
    im = ax.imshow(cm_total, cmap="Blues", vmin=0, vmax=1.0,
                    aspect="auto")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(COMP_LABELS, rotation=20, ha="right")
    ax.set_yticklabels(COMP_LABELS)
    ax.set_xlabel(r"predicted component")
    ax.set_ylabel(r"truth component")
    ax.set_title(r"confusion (rows sum to 1)")
    ax.grid(False)
    thr = (cm_total.max() + cm_total.min()) / 2.0
    for i in range(K):
        for j in range(K):
            v = cm_total[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="white" if v > thr else "#000000", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)

    ax = axes[1, 1]
    width = 0.27
    x = np.arange(K)
    ax.barh(x - width, per_p, width, color=OKABE_ITO["blue"], label=r"precision")
    ax.barh(x, per_r, width, color=OKABE_ITO["vermillion"], label=r"recall")
    ax.barh(x + width, per_f, width, color=OKABE_ITO["green"], label=r"F1")
    ax.set_yticks(x); ax.set_yticklabels(COMP_LABELS)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel(r"score")
    ax.set_title(r"per-component precision / recall / F1")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
               ncol=3, fontsize=10, frameon=False)
    ax.grid(True, axis="x", alpha=0.30)

    save(fig, "F17_contamination")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
