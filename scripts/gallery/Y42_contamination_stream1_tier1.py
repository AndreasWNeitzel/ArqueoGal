"""Y42: Stream-1 Tier-1 contamination via a 3-component GMM in chemistry.

Methodology:
  1. Restrict to the Stream-1 Tier-1 holdout (val + test, seed=0).
  2. Fit a 3-component full-covariance GMM on the APOGEE DR19 truth
     ([M/H], [alpha/M]) plane, seeded at three physical priors:
       - low-alpha thin disc:  ([M/H], [alpha/M]) ~ (+0.05, 0.00)
       - high-alpha thick disc: ~ (-0.40, 0.20)
       - metal-poor halo:       ~ (-1.20, 0.25)
  3. Each star gets a truth-component label = argmax posterior under the
     truth-fit GMM.
  4. Predicted ([M/H], [alpha/M]) gets a predicted-component label by
     evaluating the same GMM at the prediction (per-star migration).

Compact layout (2 rows x 2 cols, 14:9):
  row 0: APOGEE truth, coloured by truth-component | predicted chemistry,
         coloured by truth-component.
  row 1: total-normalised confusion matrix (cells sum to 1) |
         per-component precision / recall / F1 bars.
Global scalar scores (ARI, MCC, macro-F1) are reported in the headline
subtitle so they don't need a panel of their own.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.mixture import GaussianMixture

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
sys.path.insert(0, str(REPO / "src"))

from _presentation import OKABE_ITO, PALETTE, apply_style, headline, save  # noqa: E402

from arqueogal.xp_abundances.main.data import stratified_split_ids  # noqa: E402
from arqueogal.xp_abundances.main.release import assign_release_tier  # noqa: E402

PRED_S1 = REPO / "data/processed/pipeline1_predictions_stream1.parquet"
FEAT_S1 = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"

K = 3
COMP_LABELS = (
    r"low-$\alpha$ (thin)",
    r"high-$\alpha$ (thick)",
    r"MP halo",
)
COMP_COLORS = (OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[3])

# Initial means in physical ([M/H], [alpha/M]) dex.
INIT_MEANS_PHYS = np.array(
    [
        [+0.05, 0.00],   # 0 = low-alpha thin
        [-0.40, 0.20],   # 1 = high-alpha thick
        [-1.20, 0.25],   # 2 = MP halo
    ],
    dtype=np.float64,
)

MH_LIM = (-2.0, 0.55)
AM_LIM = (-0.10, 0.45)

_TITLE_KW = dict(fontsize=11, fontweight="normal", color=PALETTE["ink"], pad=6)


def _load_holdout_t1() -> pd.DataFrame:
    pcols = [
        "source_id", "teff_pred", "logg_pred", "mh_pred",
        "alpha_m_pred", "mg_h_pred",
        "ood_joint_flag", "label_extrapolation_flag",
    ]
    p = pd.read_parquet(PRED_S1, columns=pcols).drop_duplicates("source_id")
    fcols = ["source_id", "fe_h_apogee", "teff_apogee", "b_deg",
             "mh_apogee", "alpha_m_apogee"]
    f = pd.read_parquet(FEAT_S1, columns=fcols).drop_duplicates("source_id")
    df = f.merge(p, on="source_id", how="inner")
    df["release_tier"] = assign_release_tier(df).astype(np.int8)
    split = stratified_split_ids(df, seed=0)
    holdout_ids = np.concatenate([split["val"], split["test"]])
    df = df.loc[df["source_id"].isin(holdout_ids)].reset_index(drop=True)
    df = df.loc[df["release_tier"] == 1].reset_index(drop=True)
    df = df.dropna(subset=["mh_apogee", "alpha_m_apogee",
                            "mh_pred", "alpha_m_pred"]).reset_index(drop=True)
    return df


def _fit_gmm(truth_xy: np.ndarray) -> tuple[GaussianMixture, np.ndarray, list[np.ndarray], np.ndarray]:
    """Fit a 3-component full-cov GMM seeded at INIT_MEANS_PHYS.

    The fit is run in z-space (truth-mean / truth-std), then means and
    covariances are remapped to physical units. Component identity is
    preserved across EM by matching each fitted component to its closest
    seed in z-space, so component 0 always = thin, 1 = thick, 2 = halo.
    """
    mu = truth_xy.mean(axis=0)
    sd = truth_xy.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    z = (truth_xy - mu) / sd
    seed_z = (INIT_MEANS_PHYS - mu) / sd

    gm = GaussianMixture(
        n_components=K, covariance_type="full", random_state=42,
        n_init=1, max_iter=400, init_params="k-means++", means_init=seed_z,
    )
    gm.fit(z)
    z_labels = gm.predict(z)

    means_data = gm.means_ * sd + mu
    S = np.diag(sd)
    covs_data = [S @ cov @ S for cov in gm.covariances_]

    # Reorder so component k always corresponds to seed k.
    fit_means_z = gm.means_
    perm = np.empty(K, dtype=int)
    used: set[int] = set()
    for k_seed in range(K):
        d = np.linalg.norm(fit_means_z - seed_z[k_seed], axis=1)
        for k_fit in np.argsort(d):
            if int(k_fit) not in used:
                perm[k_seed] = int(k_fit)
                used.add(int(k_fit))
                break
    means_data = means_data[perm]
    covs_data = [covs_data[k] for k in perm]
    relabel = np.empty(K, dtype=int)
    for new_k, old_k in enumerate(perm):
        relabel[old_k] = new_k
    truth_label = relabel[z_labels]
    return gm, means_data, covs_data, truth_label


def _assign_to_components(xy: np.ndarray, means: np.ndarray, covs: list[np.ndarray]) -> np.ndarray:
    """Argmax-posterior assignment under per-component Gaussian likelihoods."""
    n = len(xy)
    log_p = np.empty((n, K))
    for k in range(K):
        cov = covs[k]
        cov_inv = np.linalg.pinv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            logdet = float(np.log(np.linalg.det(cov) + 1e-30))
        diff = xy - means[k]
        m_dist = np.einsum("ni,ij,nj->n", diff, cov_inv, diff)
        log_p[:, k] = -0.5 * (m_dist + logdet)
    return np.argmax(log_p, axis=1)


def _draw_ellipses(ax, means: np.ndarray, covs: list[np.ndarray]) -> None:
    for k in range(K):
        eigvals, eigvecs = np.linalg.eigh(covs[k])
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        for n_sig in (1, 2):
            w = 2.0 * n_sig * np.sqrt(max(eigvals[0], 1e-6))
            h = 2.0 * n_sig * np.sqrt(max(eigvals[1], 1e-6))
            ax.add_patch(Ellipse(
                xy=means[k], width=w, height=h, angle=angle,
                facecolor="none", edgecolor=COMP_COLORS[k],
                linestyle="--", linewidth=1.2, alpha=0.85, zorder=3,
            ))


def _scatter_by_component(ax, xy: np.ndarray, labels: np.ndarray,
                          *, title: str, xlabel: str, ylabel: str) -> None:
    for k in range(K):
        m = labels == k
        ax.scatter(xy[m, 0], xy[m, 1],
                   s=3.0, alpha=0.40, color=COMP_COLORS[k],
                   edgecolors="none", rasterized=True,
                   label=f"{COMP_LABELS[k]} (n={int(m.sum()):,})")
    ax.axhline(0.15, color=PALETTE["ink"], lw=0.8, ls=":", alpha=0.55)
    ax.set_xlim(MH_LIM)
    ax.set_ylim(AM_LIM)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, **_TITLE_KW)
    ax.grid(True, alpha=0.20)
    ax.legend(loc="upper right", fontsize=8, frameon=False, markerscale=3)


def _confusion_panel(ax, mat: np.ndarray, *, title: str, fmt: str = "d",
                     vmin=None, vmax=None) -> None:
    im = ax.imshow(mat, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(COMP_LABELS, rotation=20, ha="right", fontsize=9)
    ax.set_yticklabels(COMP_LABELS, fontsize=9)
    ax.set_xlabel("predicted component")
    ax.set_ylabel("truth component")
    ax.set_title(title, **_TITLE_KW)
    ax.grid(False)
    # Choose text colour per cell based on intensity, so dark cells get
    # white text and pale cells get dark text (legible against the Blues
    # ramp regardless of where the value lands).
    threshold = (mat.max() + mat.min()) / 2.0 if mat.size else 0.0
    for i in range(K):
        for j in range(K):
            v = mat[i, j]
            colour = "white" if v >= threshold else PALETTE["ink"]
            ax.text(j, i, format(v, fmt),
                    ha="center", va="center", color=colour, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)


def main() -> int:
    apply_style()
    df = _load_holdout_t1()
    if df.empty:
        print("[Y42] no Stream-1 Tier-1 holdout rows, aborting")
        return 1

    truth = df[["mh_apogee", "alpha_m_apogee"]].to_numpy()
    pred = df[["mh_pred", "alpha_m_pred"]].to_numpy()

    _gm, means, covs, truth_label = _fit_gmm(truth)
    pred_label = _assign_to_components(pred, means, covs)

    cm_counts = confusion_matrix(truth_label, pred_label, labels=list(range(K)))
    cm_total = cm_counts / max(int(cm_counts.sum()), 1)

    ari = float(adjusted_rand_score(truth_label, pred_label))
    mcc = float(matthews_corrcoef(truth_label, pred_label))
    f1_macro = float(f1_score(truth_label, pred_label, average="macro",
                              labels=list(range(K)), zero_division=0))
    per_recall = recall_score(truth_label, pred_label, average=None,
                              labels=list(range(K)), zero_division=0)
    per_precision = precision_score(truth_label, pred_label, average=None,
                                    labels=list(range(K)), zero_division=0)
    per_f1 = f1_score(truth_label, pred_label, average=None,
                      labels=list(range(K)), zero_division=0)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.32,
                          top=0.80, bottom=0.08, left=0.07, right=0.985)

    # Row 0: chemistry scatters.
    ax = fig.add_subplot(gs[0, 0])
    _scatter_by_component(ax, truth, truth_label,
                          title="APOGEE truth, by truth component",
                          xlabel="[M/H] truth (dex)",
                          ylabel=r"[$\alpha$/M] truth (dex)")
    _draw_ellipses(ax, means, covs)

    ax = fig.add_subplot(gs[0, 1])
    _scatter_by_component(ax, pred, truth_label,
                          title="ArqueoGal pred, by TRUTH component",
                          xlabel="[M/H] pred (dex)",
                          ylabel=r"[$\alpha$/M] pred (dex)")
    _draw_ellipses(ax, means, covs)

    # Row 1: total-normalised confusion + per-component bars.
    _confusion_panel(fig.add_subplot(gs[1, 0]), cm_total,
                     title="confusion (cells sum to 1)", fmt=".3f",
                     vmin=0.0, vmax=float(cm_total.max()))

    ax = fig.add_subplot(gs[1, 1])
    width = 0.27
    x = np.arange(K)
    ax.bar(x - width, per_precision, width, color=OKABE_ITO[0],
           edgecolor="white", linewidth=1.0, label="precision")
    ax.bar(x, per_recall, width, color=OKABE_ITO[1],
           edgecolor="white", linewidth=1.0, label="recall")
    ax.bar(x + width, per_f1, width, color=OKABE_ITO[2],
           edgecolor="white", linewidth=1.0, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(COMP_LABELS, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("per-component precision / recall / F1", **_TITLE_KW)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(True, axis="y", alpha=0.20)

    headline(
        fig,
        "Stream 1 Tier 1 holdout: contamination across thin / thick / halo (3-component GMM)",
        f"n = {len(df):,};  ARI = {ari:.3f},  MCC = {mcc:.3f},  macro-F1 = {f1_macro:.3f}. "
        f"GMM seeded at thin (+0.05, 0.00), thick (-0.40, 0.20), halo (-1.20, 0.25).",
        top=0.80,
    )
    save(fig, "Y42_contamination_stream1_tier1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
