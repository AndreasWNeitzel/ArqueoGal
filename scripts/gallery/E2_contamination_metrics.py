"""E2: 3-component GMM contamination diagnostic with full metric panel.

Mirrors E3's truth-fit + migration methodology but with K=3 components
on the (M/H, alpha/M) plane (typical interpretation: thin disc /
thick disc / metal-poor halo). Adds a metrics row reporting ARI, MCC,
macro-F1, plus per-component precision / recall / migration rate.

Methodology:
  1. Fit a K=3 full-covariance GMM on the TRUTH (M/H, alpha/M) plane,
     z-standardised; remap means + covariances back to physical axes.
  2. Each star gets a truth-component label = argmax posterior under
     the truth-fit GMM.
  3. Each star's PREDICTED (M/H, alpha/M) position is also assigned to
     the truth-fit components (under the same Gaussian likelihoods).
  4. Confusion matrix: rows = truth component, cols = predicted
     component. Two normalisations are reported:
        - row-normalised (recall per truth component)
        - col-normalised (precision per predicted component)
  5. Per-component migration rate = fraction of truth-component-i stars
     whose pred lands in any other component.

Layout (3 rows x 3 cols):
  row 0: truth chemistry colored by truth label | predicted chemistry
         colored by truth label | predicted chemistry colored by
         predicted label
  row 1: confusion matrix (counts) | confusion matrix (row-normalised
         = recall) | confusion matrix (col-normalised = precision)
  row 2: scalar metrics bar chart (ARI, MCC, macro-F1) |
         per-component recall + precision + F1 bar chart |
         per-component migration rate bar chart
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

from _common import load_stream1_holdout

from arqueogal.utils.plotting import save_figure, set_aa_style

N_COMPONENTS = 3

# Physical-prior initial means in ([M/H], [alpha/M]) dex.
# Order: 0 = low-alpha thin, 1 = high-alpha thick, 2 = metal-poor halo-like.
# These seed sklearn's GMM via ``means_init`` (in z-standardised space at
# fit time); the GMM still moves them, but the seeded basin is preserved.
INIT_MEANS_PHYS = np.array(
    [
        [+0.25, 0.00],  # low-alpha thin disc
        [-0.50, 0.20],  # high-alpha thick disc
        [-1.25, 0.20],  # metal-poor halo / accreted
    ],
    dtype=np.float64,
)
COMPONENT_LABELS = (
    r"low-$\alpha$ (thin)",
    r"high-$\alpha$ (thick)",
    r"MP halo-like",
)


def fit_gmm(data: np.ndarray, n_components: int, seed: int = 42) -> dict:
    """Fit full-cov GMM on z-standardised inputs with physical-prior seeding.

    The three components are seeded at the user-supplied physical centres
    (INIT_MEANS_PHYS), then standardised into the same z-space used at fit
    time. Component identity is preserved across the fit by indexing
    against the closest seed in z-space post-fit (so component 0 always
    corresponds to the low-alpha thin disc seed, regardless of how the EM
    iterations reorder the internal components).
    """
    mu_data = data.mean(axis=0)
    sd_data = data.std(axis=0)
    sd_data = np.where(sd_data > 0, sd_data, 1.0)
    z = (data - mu_data) / sd_data

    seed_means_z = (INIT_MEANS_PHYS - mu_data) / sd_data

    gm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=seed,
        n_init=1,
        max_iter=400,
        init_params="k-means++",
        means_init=seed_means_z,
    )
    gm.fit(z)
    z_labels = gm.predict(z)

    means_data = gm.means_ * sd_data + mu_data
    S = np.diag(sd_data)
    covs_data = [S @ cov_z @ S for cov_z in gm.covariances_]

    # Match each fitted component to its closest seed in z-space, so
    # component 0 = low-alpha thin, 1 = high-alpha thick, 2 = MP halo
    # regardless of whatever reorder EM produced internally.
    fit_means_z = gm.means_  # in z-space
    perm = np.empty(n_components, dtype=int)
    used = set()
    for k_seed in range(n_components):
        # Closest unassigned fitted component to seed k_seed.
        d = np.linalg.norm(fit_means_z - seed_means_z[k_seed], axis=1)
        for k_fit in np.argsort(d):
            if int(k_fit) not in used:
                perm[k_seed] = int(k_fit)
                used.add(int(k_fit))
                break
    means_data = means_data[perm]
    covs_data = [covs_data[k] for k in perm]
    # Map old labels to new (label k_new = position of k_old in perm).
    relabel = np.empty(n_components, dtype=int)
    for new_k, old_k in enumerate(perm):
        relabel[old_k] = new_k
    labels = relabel[z_labels]
    return {"means": means_data, "labels": labels, "covs": covs_data, "model": gm}


def assign_to_components(data: np.ndarray, means: np.ndarray, covs: list[np.ndarray]) -> np.ndarray:
    """Argmax-posterior assignment under per-component Gaussian likelihoods."""
    K = len(means)
    n = len(data)
    log_p = np.empty((n, K))
    for k in range(K):
        cov = covs[k]
        cov_inv = np.linalg.pinv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            logdet = float(np.log(np.linalg.det(cov) + 1e-30))
        diff = data - means[k]
        m_dist = np.einsum("ni,ij,nj->n", diff, cov_inv, diff)
        log_p[:, k] = -0.5 * (m_dist + logdet)
    return np.argmax(log_p, axis=1)


def main(n_stars: int | None = None) -> None:
    set_aa_style(font_size=9.0)

    data = load_stream1_holdout()
    needed = ["alpha_m_apogee", "mh_apogee", "alpha_m_pred", "mh_pred"]
    data = data.dropna(subset=needed)
    if n_stars is not None and n_stars < len(data):
        data = data.sample(n=n_stars, random_state=42)

    # Tinsley convention: ([M/H], [alpha/M]).
    truth = data[["mh_apogee", "alpha_m_apogee"]].values
    pred = data[["mh_pred", "alpha_m_pred"]].values

    gmm_truth = fit_gmm(truth, n_components=N_COMPONENTS, seed=42)
    truth_label = gmm_truth["labels"]
    pred_label = assign_to_components(pred, gmm_truth["means"], gmm_truth["covs"])

    K = N_COMPONENTS
    # 0 = low-alpha thin (blue), 1 = high-alpha thick (red), 2 = MP halo (purple)
    comp_color = {0: "#1f77b4", 1: "#d62728", 2: "#9467bd"}
    comp_label = {k: COMPONENT_LABELS[k] for k in range(K)}

    cm_counts = confusion_matrix(truth_label, pred_label, labels=list(range(K)))
    row_sums = cm_counts.sum(axis=1, keepdims=True).clip(min=1)
    col_sums = cm_counts.sum(axis=0, keepdims=True).clip(min=1)
    cm_recall = cm_counts / row_sums  # per truth row, fraction landing in each pred col
    cm_precision = cm_counts / col_sums  # per pred col, fraction sourced from each truth row

    ari = float(adjusted_rand_score(truth_label, pred_label))
    mcc = float(matthews_corrcoef(truth_label, pred_label))
    f1_macro = float(
        f1_score(truth_label, pred_label, average="macro", labels=list(range(K)), zero_division=0)
    )

    per_recall = recall_score(
        truth_label, pred_label, average=None, labels=list(range(K)), zero_division=0
    )
    per_precision = precision_score(
        truth_label, pred_label, average=None, labels=list(range(K)), zero_division=0
    )
    per_f1 = f1_score(truth_label, pred_label, average=None, labels=list(range(K)), zero_division=0)
    # Per-component migration rate = 1 - recall_k.
    per_migration = 1.0 - np.asarray(per_recall)

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.30)

    # --- Row 0: chemistry-plane visuals ---
    def _draw_ellipses(ax, n_sigma: float = 2.0):
        for k in range(K):
            mu = gmm_truth["means"][k]
            cov = gmm_truth["covs"][k]
            ev, evec = np.linalg.eigh(cov)
            ev = np.clip(ev, 1e-12, None)
            major = evec[:, 1]
            angle = float(np.degrees(np.arctan2(major[1], major[0])))
            w = 2.0 * n_sigma * np.sqrt(ev[1])
            h = 2.0 * n_sigma * np.sqrt(ev[0])
            ax.add_patch(
                Ellipse(
                    xy=(float(mu[0]), float(mu[1])),
                    width=w,
                    height=h,
                    angle=angle,
                    facecolor="none",
                    edgecolor=comp_color[k],
                    linestyle="--",
                    linewidth=1.4,
                )
            )
            ax.plot(
                mu[0],
                mu[1],
                marker="*",
                color=comp_color[k],
                markeredgecolor="black",
                markersize=14,
                zorder=10,
            )

    from matplotlib.colors import LinearSegmentedColormap

    def _density(ax, x, y, color, label):
        if len(x) == 0:
            return
        cmap = LinearSegmentedColormap.from_list(f"cm_{color}", ["#ffffff00", color])
        ax.hexbin(x, y, gridsize=80, cmap=cmap, mincnt=1, bins="log", alpha=0.7)
        ax.plot([], [], "s", color=color, label=label)

    # Truth panel.
    ax = fig.add_subplot(gs[0, 0])
    for c in range(K):
        m = truth_label == c
        _density(
            ax, truth[m, 0], truth[m, 1], comp_color[c], f"{comp_label[c]} (n={int(m.sum()):,})"
        )
    _draw_ellipses(ax)
    ax.axhline(0.15, color="black", lw=0.8, ls=":", alpha=0.7, label=r"$[\alpha/M]=0.15$")
    ax.set_xlabel(r"$[\mathrm{M/H}]$ (truth)")
    ax.set_ylabel(r"$[\alpha/\mathrm{M}]$ (truth)")
    ax.set_ylim(-0.2, 0.4)
    ax.set_title("(a) Truth GMM (K=3, defines components)")
    ax.legend(fontsize=7, loc="lower left")
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2)

    # Predicted positions, coloured by TRUTH label.
    ax = fig.add_subplot(gs[0, 1])
    migrated = pred_label != truth_label
    for c in range(K):
        m = (truth_label == c) & ~migrated
        _density(
            ax,
            pred[m, 0],
            pred[m, 1],
            comp_color[c],
            f"{comp_label[c]} stayed (n={int(m.sum()):,})",
        )
    for c in range(K):
        m = (truth_label == c) & migrated
        if not m.any():
            continue
        ax.scatter(
            pred[m, 0],
            pred[m, 1],
            s=10,
            alpha=0.85,
            color=comp_color[c],
            edgecolors="black",
            linewidths=0.5,
            rasterized=True,
            label=f"{comp_label[c]} migrated (n={int(m.sum()):,})",
        )
    _draw_ellipses(ax)
    ax.axhline(0.15, color="black", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel(r"$[\mathrm{M/H}]$ (predicted)")
    ax.set_ylabel(r"$[\alpha/\mathrm{M}]$ (predicted)")
    ax.set_ylim(-0.2, 0.4)
    ax.set_title(
        "(b) Predicted positions, coloured by TRUTH label "
        f"(overall migration {migrated.mean() * 100:.1f}%)"
    )
    ax.legend(fontsize=6, loc="lower left")
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2)

    # Predicted positions, coloured by PRED label.
    ax = fig.add_subplot(gs[0, 2])
    for c in range(K):
        m = pred_label == c
        _density(ax, pred[m, 0], pred[m, 1], comp_color[c], f"{comp_label[c]} (n={int(m.sum()):,})")
    _draw_ellipses(ax)
    ax.axhline(0.15, color="black", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel(r"$[\mathrm{M/H}]$ (predicted)")
    ax.set_ylabel(r"$[\alpha/\mathrm{M}]$ (predicted)")
    ax.set_ylim(-0.2, 0.4)
    ax.set_title("(c) Predicted positions, coloured by PRED label")
    ax.legend(fontsize=7, loc="lower left")
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2)

    # --- Row 1: confusion matrices ---
    def _show_cm(ax, M: np.ndarray, title: str, fmt: str, cbar_label: str):
        im = ax.imshow(
            M, cmap="Blues", aspect="equal", vmin=0, vmax=float(M.max()) if M.size else 1.0
        )
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels(
            [f"pred {comp_label[c]}" for c in range(K)], rotation=15, ha="right", fontsize=8
        )
        ax.set_yticklabels([f"truth {comp_label[c]}" for c in range(K)], fontsize=8)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label=cbar_label)
        for i in range(K):
            for j in range(K):
                ax.text(
                    j,
                    i,
                    format(M[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if M[i, j] > 0.5 * float(M.max()) else "black",
                    fontsize=10,
                )

    ax = fig.add_subplot(gs[1, 0])
    _show_cm(ax, cm_counts, "(d) Confusion matrix (counts)", ",d", "stars per cell")

    ax = fig.add_subplot(gs[1, 1])
    _show_cm(ax, cm_recall, "(e) Row-normalised: per-truth recall fraction", ".3f", "row fraction")

    ax = fig.add_subplot(gs[1, 2])
    _show_cm(
        ax, cm_precision, "(f) Col-normalised: per-pred precision fraction", ".3f", "col fraction"
    )

    # --- Row 2: metrics bar charts ---
    ax = fig.add_subplot(gs[2, 0])
    metrics = [("ARI", ari), ("MCC", mcc), ("macro-F1", f1_macro)]
    xs = np.arange(len(metrics))
    bars = ax.bar(xs, [m[1] for m in metrics], color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=10)
    ax.set_ylim(
        min(-0.05, min(m[1] for m in metrics) - 0.05), max(0.05, max(m[1] for m in metrics) + 0.05)
    )
    for b, (_, val) in zip(bars, metrics):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.02,
            f"{val:+.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.axhline(0, color="k", lw=0.6, ls=":", alpha=0.4)
    ax.set_ylabel("score (higher = better)")
    ax.set_title("(g) Scalar agreement metrics")
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(gs[2, 1])
    width = 0.27
    xs = np.arange(K)
    ax.bar(xs - width, per_precision, width, label="precision", color="#1f77b4")
    ax.bar(xs, per_recall, width, label="recall", color="#ff7f0e")
    ax.bar(xs + width, per_f1, width, label="F1", color="#2ca02c")
    ax.set_xticks(xs)
    ax.set_xticklabels([comp_label[c] for c in range(K)], rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("(h) Per-component precision / recall / F1")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(gs[2, 2])
    bars = ax.bar(xs, per_migration, color=[comp_color[c] for c in range(K)], width=0.55)
    for b, val in zip(bars, per_migration):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.01,
            f"{val * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([comp_label[c] for c in range(K)], rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, max(0.05, float(per_migration.max()) * 1.2))
    ax.set_ylabel("migration rate (1 - recall)")
    ax.set_title("(i) Per-component migration rate (stars predicted into a wrong component)")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"E2 - Stream 1 GMM contamination diagnostic with K=3 components "
        f"(n={len(data):,}). "
        f"ARI = {ari:+.3f},  MCC = {mcc:+.3f},  macro-F1 = {f1_macro:.3f},  "
        f"overall migration = {migrated.mean() * 100:.1f}%.",
        fontsize=11,
        fontweight="semibold",
        y=0.995,
    )

    out_dir = REPO / "reports/gallery/E_validation"
    paths = save_figure(fig, out_dir / "E2_contamination_metrics", formats=("pdf", "png"))
    for p in paths:
        print(f"[E2] wrote {p.relative_to(REPO)}")

    # Console summary.
    print(f"\n=== E2 K=3 GMM contamination summary (n={len(data):,}) ===")
    print(
        f"  Scalar: ARI={ari:+.4f}  MCC={mcc:+.4f}  macro-F1={f1_macro:.4f}  "
        f"overall migration={migrated.mean() * 100:.2f}%"
    )
    print("  Per-component (truth → pred):")
    for c in range(K):
        n_truth = int((truth_label == c).sum())
        print(
            f"    {comp_label[c]:>30s}  n_truth={n_truth:>7,}  "
            f"recall={per_recall[c]:.3f}  prec={per_precision[c]:.3f}  "
            f"F1={per_f1[c]:.3f}  migration={per_migration[c] * 100:.1f}%"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-stars", type=int, default=None, help="Optional: downsample to N stars (default: all)"
    )
    args = parser.parse_args()
    main(n_stars=args.n_stars)
