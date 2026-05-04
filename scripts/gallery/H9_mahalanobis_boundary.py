"""H9: Mahalanobis OOD decision boundary visualised explicitly.

The XP-block Mahalanobis gate (``ood_joint_flag``, ``ood.py``) operates in
108-D space: the (bp_coef_norm_1..54, rp_coef_norm_1..54) block, z-scored
under the frozen-stats basis. Decision rule:

  d²(x) = (x − μ)ᵀ Σ⁻¹ (x − μ)
  flag  = d(x) > τ,   τ = 99-th percentile of training d

This figure shows the rule itself rather than the consequences:

  panel 1  d-histogram, training vs held-out, threshold marked
  panel 2  ECDF of d, training vs held-out, p=0.99 line marked
  panel 3  Q-Q plot d² vs χ²(108) — diagnostic for the Gaussian assumption
  panel 4  PC1-PC2 of the 108-D block, held-out coloured by d, level-set
           ellipse at the threshold drawn (necessary but not sufficient
           condition: stars outside the ellipse in this 2D slice are
           certainly flagged, stars inside may still be flagged by the
           remaining 106 dimensions)
  panel 5  PC3-PC4, same layout — demonstrates that the gate is
           multi-dimensional and a single PC pair undersells it
  panel 6  d versus σ_α (the prior-collapse gate), coloured by tier —
           shows how the two gates correlate and where Tier-1 sits

The PC projection uses the eigendecomposition of the training covariance,
so the in-distribution scatter is whitened in (PC1, PC2). The level-set
in PC-space is therefore a circle of radius τ in the unit-variance basis,
and the matplotlib coordinates are (PC1·√λ₁, PC2·√λ₂), so the contour
is an axis-aligned ellipse with semi-axes (τ√λ₁, τ√λ₂).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
sys.path.insert(0, str(REPO / "src"))

from _common import apply_style, save_fig  # noqa: E402

from arqueogal.xp_abundances.main.data import (  # noqa: E402
    FeatureLayout,
    stratified_split_ids,
)
from arqueogal.xp_abundances.main.ood import (  # noqa: E402
    fit_mahalanobis_ood,
    score_mahalanobis_ood,
)
from arqueogal.xp_abundances.main.release import assign_release_tier  # noqa: E402

OUT = REPO / "reports/gallery/H_hybrid_release"
PREDICTIONS_S1 = REPO / "data/processed/pipeline1_predictions_stream1.parquet"
FEATURES_S1 = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"

P_THRESHOLD = 0.99
SIGMA_ALPHA_THRESHOLD = 0.05


def _load() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    layout = FeatureLayout()
    xp108 = list(layout.bp_coef_cols) + list(layout.rp_coef_cols)
    pred_cols = [
        "source_id",
        "ood_joint_flag",
        "ood_mahalanobis_score",
        "teff_sigma",
        "logg_sigma",
        "mh_sigma",
        "alpha_m_sigma",
        "mg_h_sigma",
        "teff_pred",
        "logg_pred",
        "mh_pred",
        "alpha_m_pred",
        "mg_h_pred",
        "label_extrapolation_flag",  # T2 driver, added 2026-05-03
    ]
    df_p = pd.read_parquet(PREDICTIONS_S1, columns=pred_cols)
    feat_extra = [
        "fe_h_apogee",
        "teff_apogee",
        "b_deg",
        "logg_apogee",
        "mh_apogee",
        "alpha_m_apogee",
        "mg_h_apogee",
    ]
    df_f = pd.read_parquet(FEATURES_S1, columns=["source_id", *xp108, *feat_extra])
    df_f = df_f.drop_duplicates(subset="source_id", keep="first")
    df = df_f.merge(df_p, on="source_id", how="inner")

    split = stratified_split_ids(df, seed=0)
    train_mask = np.isin(df["source_id"].to_numpy(), split["train"])
    holdout_mask = np.isin(
        df["source_id"].to_numpy(),
        np.concatenate([split["val"], split["test"]]),
    )

    X108 = df[xp108].to_numpy(dtype=np.float64)
    print(
        f"[H9] full pool: {len(df):,} stars; "
        f"train: {int(train_mask.sum()):,}; "
        f"held-out: {int(holdout_mask.sum()):,}"
    )
    return df, X108, train_mask, holdout_mask


def _pc_project(X_train_centered: np.ndarray, X_query_centered: np.ndarray):
    """Eigendecomposition of training covariance; project query into top-K
    principal components. Returns (PC_train, PC_query, eigenvalues, eigvecs).
    Drops rows with non-finite entries before fitting (matches the precision
    fit in fit_mahalanobis_ood).
    """
    finite = np.isfinite(X_train_centered).all(axis=1)
    Xc = X_train_centered[finite]
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    PC_train = Xc @ eigvecs
    # Project held-out, replacing non-finite stars with NaN in the projection.
    PC_query = np.full((X_query_centered.shape[0], eigvecs.shape[1]), np.nan)
    finite_q = np.isfinite(X_query_centered).all(axis=1)
    PC_query[finite_q] = X_query_centered[finite_q] @ eigvecs
    return PC_train, PC_query, eigvals, eigvecs


def main() -> int:
    apply_style()
    df, X108, train_mask, holdout_mask = _load()

    bundle = fit_mahalanobis_ood(
        X108[train_mask],
        p_threshold=P_THRESHOLD,
        regularization=1e-6,
    )
    print(
        f"[H9] fitted bundle: dim={bundle.feature_dim}, "
        f"τ = {bundle.threshold:.3f}, n_train_used = {bundle.n_training:,}"
    )

    d_train = score_mahalanobis_ood(X108[train_mask], bundle)
    d_hold = score_mahalanobis_ood(X108[holdout_mask], bundle)
    flag_hold = d_hold > bundle.threshold
    print(
        f"[H9] held-out OOD-flag rate: {flag_hold.mean():.3%} "
        f"(expected ~{1.0 - P_THRESHOLD:.0%} if held-out matches train)"
    )

    df_hold = df.loc[holdout_mask].copy().reset_index(drop=True)
    df_hold["release_tier"] = assign_release_tier(df_hold).astype(np.int8)
    tier = df_hold["release_tier"].to_numpy()

    Xc_train = X108[train_mask] - bundle.feature_mean
    Xc_hold = X108[holdout_mask] - bundle.feature_mean
    PC_train, PC_hold, eigvals, _eigvecs = _pc_project(Xc_train, Xc_hold)
    print(f"[H9] PCA done; top eigvals: {eigvals[:6]}")

    # 2 rows × 3 cols: top = XP-Mahalanobis (T3 gate), bottom = label-Mahalanobis (T2 gate).
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(
        2, 3, hspace=0.40, wspace=0.32, top=0.96, bottom=0.06, left=0.05, right=0.97
    )

    # === Panel 1: d-histogram, train vs held-out, threshold ===
    ax = fig.add_subplot(gs[0, 0])
    x_max = float(bundle.threshold * 3.0)  # 3× τ keeps the gate visible
    bins = np.linspace(0, x_max, 80)
    n_train_beyond = int((d_train > x_max).sum())
    n_hold_beyond = int((d_hold > x_max).sum())
    ax.hist(
        np.clip(d_train, 0, x_max),
        bins=bins,
        density=True,
        alpha=0.55,
        color="#1f77b4",
        label=f"training (n={int(train_mask.sum()):,})",
    )
    ax.hist(
        np.clip(d_hold, 0, x_max),
        bins=bins,
        density=True,
        alpha=0.55,
        color="#ff7f0e",
        label=f"held-out (n={int(holdout_mask.sum()):,})",
    )
    ax.axvline(
        bundle.threshold, color="#d62728", lw=1.4, label=f"τ = {bundle.threshold:.2f} (train p99)"
    )
    ax.axvspan(bundle.threshold, x_max, color="#d62728", alpha=0.07)
    ax.set_yscale("log")
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Mahalanobis distance d")
    ax.set_ylabel("density (log)")
    ax.set_title(
        f"d-distribution: training vs held-out  "
        f"(>{x_max:.0f}: train {n_train_beyond:,}, held-out {n_hold_beyond:,})"
    )
    ax.legend(loc="upper right", fontsize=8)

    # === Panel 2: ECDF ===
    ax = fig.add_subplot(gs[0, 1])
    for name, d, color in (("training", d_train, "#1f77b4"), ("held-out", d_hold, "#ff7f0e")):
        ds = np.sort(d[np.isfinite(d)])
        cdf = np.arange(1, len(ds) + 1) / len(ds)
        ax.plot(ds, cdf, color=color, lw=1.3, label=name)
    ax.axvline(bundle.threshold, color="#d62728", lw=1.2, label=rf"$\tau$ = {bundle.threshold:.2f}")
    ax.axhline(P_THRESHOLD, color="#d62728", lw=0.8, ls=":")
    held_p = float((d_hold <= bundle.threshold).mean())
    ax.text(
        0.55,
        0.45,
        f"held-out fraction inside τ:\n  {held_p:.4f}\nflag rate: {1.0 - held_p:.4f}",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#aaaaaa", alpha=0.85),
    )
    ax.set_xlim(0, bundle.threshold * 3.0)
    ax.set_xlabel("Mahalanobis distance d")
    ax.set_ylabel("CDF")
    ax.set_title("ECDF: training vs held-out")
    ax.legend(loc="lower right", fontsize=8)

    # === Panel 3: Q-Q d² vs χ²(108) ===
    from scipy import stats as _st

    ax = fig.add_subplot(gs[0, 2])
    sq = np.sort(d_train[np.isfinite(d_train)] ** 2)
    n_used = min(len(sq), 50_000)
    if len(sq) > n_used:
        sq = sq[:: max(1, len(sq) // n_used)]
    p_grid = (np.arange(1, len(sq) + 1) - 0.5) / len(sq)
    chi_quant = _st.chi2.ppf(p_grid, df=bundle.feature_dim)
    # Diagonal range: cap the plot at the χ² 99.9-th quantile to keep the
    # bulk visible. Heavy-tail divergence is the actual diagnostic.
    qmax = float(_st.chi2.ppf(0.999, df=bundle.feature_dim))
    ymax = max(qmax, float(np.percentile(sq, 99.9)) * 1.05)
    ax.plot(chi_quant, sq, color="#1f77b4", lw=0.8)
    ax.plot(
        [0, ymax], [0, ymax], color="#d62728", lw=1.0, ls="--", label=r"$y=x$ (perfect $\chi^2$)"
    )
    ax.axhline(
        bundle.threshold**2,
        color="#7f7f7f",
        lw=0.8,
        ls=":",
        label=rf"$\tau^2={bundle.threshold**2:.0f}$",
    )
    ax.set_xlim(0, qmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(rf"theoretical $\chi^2_{{{bundle.feature_dim}}}$ quantile")
    ax.set_ylabel(r"empirical $d^2$ quantile (training)")
    ax.set_title(
        r"Q-Q: training $d^2$ vs $\chi^2_{108}$  "
        r"(empirical heavy tail = non-Gaussian XP block)"
    )
    ax.legend(loc="upper left", fontsize=8)

    # === Panel 4: PC1-PC2 with level-set ellipse ===
    def _pc_panel(ax, ix, iy):
        x_all = PC_hold[:, ix]
        y_all = PC_hold[:, iy]
        ok = np.isfinite(x_all) & np.isfinite(y_all) & np.isfinite(d_hold)
        x = x_all[ok]
        y = y_all[ok]
        d_ok = d_hold[ok]
        flag_ok = flag_hold[ok]
        # Subsample for plotting
        if len(x) > 30_000:
            sel = np.random.default_rng(42).choice(len(x), size=30_000, replace=False)
            xs, ys, ds = x[sel], y[sel], d_ok[sel]
            flag_sel = flag_ok[sel]
        else:
            xs, ys, ds = x, y, d_ok
            flag_sel = flag_ok
        order = np.argsort(ds)  # plot high-d on top
        sc = ax.scatter(
            xs[order],
            ys[order],
            c=ds[order],
            cmap="viridis",
            s=3.0,
            alpha=0.55,
            vmin=0,
            vmax=min(np.nanpercentile(d_hold, 99.5), bundle.threshold * 2.0),
            edgecolors="none",
        )
        # Highlight the actually-flagged stars with a red edge so they
        # are visible even when they fall inside the 2-D ellipse.
        red_idx = np.where(flag_sel)[0]
        ax.scatter(
            xs[red_idx],
            ys[red_idx],
            facecolors="none",
            edgecolors="#d62728",
            linewidths=0.45,
            s=14,
            alpha=0.85,
        )
        # Level-set ellipse at d = τ assuming the other 106 dims sit at the
        # mean. Semi-axes: τ·√λ_i along PC_i.
        e = Ellipse(
            (0.0, 0.0),
            width=2.0 * bundle.threshold * np.sqrt(eigvals[ix]),
            height=2.0 * bundle.threshold * np.sqrt(eigvals[iy]),
            facecolor="none",
            edgecolor="#d62728",
            lw=1.6,
            ls="--",
            label=r"$d = \tau$ (other PCs = $\mu$)",
        )
        ax.add_patch(e)
        cb = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("Mahalanobis d", fontsize=8)
        ax.set_xlabel(f"PC{ix + 1}  (λ = {eigvals[ix]:.2f})")
        ax.set_ylabel(f"PC{iy + 1}  (λ = {eigvals[iy]:.2f})")
        # Clamp axes to ~3·τ in PC units (PC variance = λ_i, so ±3·τ·√λ_i is
        # well past the level-set ellipse). Keeps the gate boundary visible.
        ax.set_xlim(
            -3.0 * bundle.threshold * np.sqrt(eigvals[ix]),
            3.0 * bundle.threshold * np.sqrt(eigvals[ix]),
        )
        ax.set_ylim(
            -3.0 * bundle.threshold * np.sqrt(eigvals[iy]),
            3.0 * bundle.threshold * np.sqrt(eigvals[iy]),
        )
        ax.set_title(
            f"PC{ix + 1}-PC{iy + 1} of 108-D XP block "
            f"(held-out plotted, n={len(xs):,})\n"
            "red-edge = star flagged OOD (any of 108 dims); "
            "outside dashed ellipse → certainly flagged"
        )
        ax.legend(loc="upper right", fontsize=7)

    # === Bottom row: label-space Mahalanobis (T2 gate) ===
    # Fit a 5-D Mahalanobis bundle from APOGEE truth on the same training
    # partition; score predicted labels on the held-out cohort.
    label_truth_cols = ("teff_apogee", "logg_apogee", "mh_apogee", "alpha_m_apogee", "mg_h_apogee")
    Y_truth = df[list(label_truth_cols)].to_numpy(dtype=np.float64)
    Y_train_truth = Y_truth[train_mask]
    Y_train_truth = Y_train_truth[np.isfinite(Y_train_truth).all(axis=1)]
    label_bundle = fit_mahalanobis_ood(
        Y_train_truth,
        p_threshold=P_THRESHOLD,
        regularization=1e-8,
    )
    print(
        f"[H9] label bundle: dim={label_bundle.feature_dim}, τ_label = {label_bundle.threshold:.3f}"
    )

    pred_cols = ("teff_pred", "logg_pred", "mh_pred", "alpha_m_pred", "mg_h_pred")
    Y_hold_pred = df.loc[holdout_mask, list(pred_cols)].to_numpy(dtype=np.float64)
    d_label_hold = score_mahalanobis_ood(Y_hold_pred, label_bundle)
    flag_label_hold = d_label_hold > label_bundle.threshold

    # --- Panel 4: label-Mahal histogram ---
    ax = fig.add_subplot(gs[1, 0])
    x_max_label = float(label_bundle.threshold * 4.0)
    bins_label = np.linspace(0, x_max_label, 80)
    ax.hist(
        np.clip(d_label_hold, 0, x_max_label),
        bins=bins_label,
        density=True,
        alpha=0.65,
        color="#ff7f0e",
        label=f"held-out predictions (n={int(holdout_mask.sum()):,})",
    )
    ax.axvline(
        label_bundle.threshold,
        color="#d62728",
        lw=1.4,
        label=rf"$\tau_{{\rm label}}$ = {label_bundle.threshold:.2f} (truth p99)",
    )
    ax.axvspan(label_bundle.threshold, x_max_label, color="#d62728", alpha=0.07)
    ax.set_yscale("log")
    ax.set_xlim(0, x_max_label)
    ax.set_xlabel("label-space Mahalanobis $d_{\\rm label}$")
    ax.set_ylabel("density (log)")
    ax.set_title(
        f"T2 gate: $d_{{\\rm label}}$ on predicted (T_eff, log g, [M/H], [α/M], [Mg/H])  "
        f"flag rate = {flag_label_hold.mean():.4f}"
    )
    ax.legend(loc="upper right", fontsize=8)

    # --- Panel 5: label-Mahal ECDF ---
    ax = fig.add_subplot(gs[1, 1])
    d_train_pred = score_mahalanobis_ood(
        df.loc[train_mask, list(pred_cols)].to_numpy(dtype=np.float64),
        label_bundle,
    )
    for name, d, color in (
        ("training (pred labels)", d_train_pred, "#1f77b4"),
        ("held-out (pred labels)", d_label_hold, "#ff7f0e"),
    ):
        ds = np.sort(d[np.isfinite(d)])
        cdf = np.arange(1, len(ds) + 1) / len(ds)
        ax.plot(ds, cdf, color=color, lw=1.5, label=name)
    ax.axvline(
        label_bundle.threshold,
        color="#d62728",
        lw=1.2,
        label=rf"$\tau_{{\rm label}}$ = {label_bundle.threshold:.2f}",
    )
    ax.axhline(P_THRESHOLD, color="#d62728", lw=0.8, ls=":")
    ax.set_xlim(0, label_bundle.threshold * 3.0)
    ax.set_xlabel("label-space Mahalanobis $d_{\\rm label}$")
    ax.set_ylabel("CDF")
    ax.set_title("ECDF: label-Mahalanobis on predicted vs trained-truth envelope")
    ax.legend(loc="lower right", fontsize=8)

    # --- Panel 6: dual-Mahalanobis scatter coloured by composite tier ---
    ax = fig.add_subplot(gs[1, 2])
    palette = {1: "#2ca02c", 2: "#ff7f0e", 3: "#d62728"}
    sizes = {1: 3.5, 2: 12.0, 3: 6.0}  # T2 boosted so its rare points stand out
    alphas = {1: 0.30, 2: 0.85, 3: 0.65}
    for t in (1, 2, 3):
        m = tier == t
        if m.any():
            ax.scatter(
                d_hold[m],
                d_label_hold[m],
                s=sizes[t],
                c=palette[t],
                alpha=alphas[t],
                edgecolors="white" if t in (2, 3) else "none",
                linewidths=0.4 if t in (2, 3) else 0.0,
                label=f"Tier {t} (n={int(m.sum()):,})",
            )
    # T3 boundary (right of τ_xp) — vertical red.
    ax.axvline(
        bundle.threshold,
        color="#d62728",
        lw=1.6,
        ls="--",
        label=rf"T3 gate $\tau_{{\rm XP}}$ = {bundle.threshold:.2f}",
    )
    ax.axvspan(bundle.threshold, 1e6, color="#d62728", alpha=0.07, zorder=0)
    # T2 boundary (above τ_label) — horizontal orange.
    ax.axhline(
        label_bundle.threshold,
        color="#ff7f0e",
        lw=1.6,
        ls="--",
        label=rf"T2 gate $\tau_{{\rm label}}$ = {label_bundle.threshold:.2f}",
    )
    ax.axhspan(label_bundle.threshold, 1e6, color="#ff7f0e", alpha=0.10, zorder=0)
    ax.set_xlabel("XP-Mahalanobis $d_{\\rm XP}$ (108-D)")
    ax.set_ylabel("label-Mahalanobis $d_{\\rm label}$ (5-D)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(max(1.0, np.nanmin(d_hold) * 0.9), np.nanpercentile(d_hold, 99.95))
    ax.set_ylim(max(0.05, np.nanmin(d_label_hold) * 0.9), np.nanpercentile(d_label_hold, 99.95))
    ax.set_title("dual-Mahalanobis: input OOD vs output OOD, coloured by tier")
    ax.legend(loc="upper left", fontsize=7)

    OUT.mkdir(parents=True, exist_ok=True)
    save_fig(fig, OUT / "H9_mahalanobis_boundary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
