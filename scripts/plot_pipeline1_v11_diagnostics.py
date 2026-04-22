"""Pipeline-1 v1.1 neural-network validation diagnostics.

Consumes
--------
- ``reports/pipeline1/run_a_v11/val_predictions.parquet`` — 5-member ensemble
  predictions + truth on the Stream-1 val partition (41,851 stars).
- ``reports/pipeline1/run_a_v11/ensemble_history.json`` — per-seed epoch-level
  loss trajectories.
- ``data/processed/pipeline1_predictions_stream3_volume_v11.parquet`` — Stream-3
  deployment predictions (for val-vs-deployment σ_α distribution overlay that
  exposes the metal-poor prior-collapse regime).

Emits
-----
``reports/pipeline1/run_a_v11/pipeline1_v11_diagnostics.png`` — 4-row × 5-col
panel:

  Row 1: pred-vs-truth scatter per label (Teff, logg, [M/H], [α/M], [Mg/H])
  Row 2: residual histograms with Gaussian fit
  Row 3: σ-reliability curves (binned reported σ vs empirical |residual|)
  Row 4: training curves · val α_pred vs truth [M/H] · σ_α val-vs-Stream3 ·
         68%/95% coverage bars · epistemic-σ distributions

All abundance-style labels are in dex; Teff is in K; logg is in dex.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

mpl.rcParams["figure.dpi"] = 110
mpl.rcParams["savefig.dpi"] = 140
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.alpha"] = 0.25
mpl.rcParams["grid.linewidth"] = 0.5
mpl.rcParams["axes.titlesize"] = 10
mpl.rcParams["axes.labelsize"] = 9
mpl.rcParams["xtick.labelsize"] = 8
mpl.rcParams["ytick.labelsize"] = 8
mpl.rcParams["legend.fontsize"] = 8

_REPO = Path(__file__).resolve().parents[1]
VAL_PRED = _REPO / "reports/pipeline1/run_a_v11/val_predictions.parquet"
HIST_JSON = _REPO / "reports/pipeline1/run_a_v11/ensemble_history.json"
STREAM3 = _REPO / "data/processed/pipeline1_predictions_stream3_volume_v11.parquet"
OUT_PNG = _REPO / "reports/pipeline1/run_a_v11/pipeline1_v11_diagnostics.png"

LABELS = ("teff", "logg", "mh", "alpha_m", "mg_h")
LABEL_PRETTY = {
    "teff": r"$T_{\rm eff}$ [K]",
    "logg": r"$\log g$ [dex]",
    "mh": r"$[{\rm M/H}]$ [dex]",
    "alpha_m": r"$[\alpha/{\rm M}]$ [dex]",
    "mg_h": r"$[{\rm Mg/H}]$ [dex]",
}
LABEL_SHORT = {
    "teff": "Teff", "logg": "logg", "mh": "[M/H]",
    "alpha_m": "[α/M]", "mg_h": "[Mg/H]",
}
# Nominal plotting ranges (data-driven fallback when exceeded).
LABEL_RANGE = {
    "teff": (4000, 5500),
    "logg": (0.9, 3.7),
    "mh": (-2.0, 0.5),
    "alpha_m": (-0.2, 0.5),
    "mg_h": (-2.0, 0.6),
}


def _scatter_pred_truth(ax, truth, pred, label_key: str) -> None:
    lo, hi = LABEL_RANGE[label_key]
    ax.hexbin(truth, pred, gridsize=50, cmap="viridis", mincnt=1,
              extent=(lo, hi, lo, hi), linewidths=0)
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=0.8, alpha=0.8)
    residual = pred - truth
    bias = float(np.nanmean(residual))
    rmse = float(np.sqrt(np.nanmean(residual ** 2)))
    ax.text(0.03, 0.97,
            f"bias = {bias:+.3g}\nRMSE = {rmse:.3g}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"truth  {LABEL_PRETTY[label_key]}")
    ax.set_ylabel(f"pred  {LABEL_PRETTY[label_key]}")
    ax.set_title(f"pred vs truth — {LABEL_SHORT[label_key]}")


def _residual_hist(ax, truth, pred, label_key: str) -> None:
    residual = pred - truth
    residual = residual[np.isfinite(residual)]
    p1, p99 = np.nanpercentile(residual, [1, 99])
    rng = max(abs(p1), abs(p99))
    bins = np.linspace(-rng, rng, 61)
    ax.hist(residual, bins=bins, color="steelblue", alpha=0.7, density=True,
            edgecolor="none")
    mu, sig = float(np.nanmean(residual)), float(np.nanstd(residual))
    xs = np.linspace(-rng, rng, 200)
    ax.plot(xs, stats.norm.pdf(xs, mu, sig), "r--", linewidth=1.0,
            label=f"N({mu:+.3g}, {sig:.3g})")
    ax.axvline(0.0, color="k", linewidth=0.6, alpha=0.5)
    ax.set_xlim(-rng, rng)
    ax.set_xlabel(f"residual  pred − truth  [{LABEL_PRETTY[label_key]}]")
    ax.set_ylabel("density")
    ax.set_title(f"residuals — {LABEL_SHORT[label_key]}")
    ax.legend(loc="upper right", fontsize=7)


def _sigma_reliability(ax, truth, pred, sigma, label_key: str) -> None:
    """Reported σ vs empirical |residual| std, binned on reported σ.

    Well-calibrated: mean |residual| ≈ reported σ × sqrt(2/π) = 0.7979 σ
    (for a Gaussian). We plot empirical std(residual) per σ-bin which
    should trace 1:1 with reported σ if calibrated.
    """
    residual = pred - truth
    m = np.isfinite(residual) & np.isfinite(sigma) & (sigma > 0)
    if m.sum() < 50:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        return
    s = sigma[m]
    r = residual[m]
    # 10 quantile-bins on reported sigma
    edges = np.nanquantile(s, np.linspace(0.0, 1.0, 11))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    digit = np.digitize(s, edges) - 1
    digit = np.clip(digit, 0, 9)
    sig_bin, emp_std, emp_bias = [], [], []
    for k in range(10):
        mk = digit == k
        if mk.sum() < 10:
            continue
        sig_bin.append(float(np.median(s[mk])))
        emp_std.append(float(np.std(r[mk])))
        emp_bias.append(float(np.mean(r[mk])))
    sig_bin = np.asarray(sig_bin)
    emp_std = np.asarray(emp_std)
    lo = min(sig_bin.min(), emp_std.min()) * 0.9
    hi = max(sig_bin.max(), emp_std.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=0.8, alpha=0.6,
            label="ideal")
    ax.plot(sig_bin, emp_std, "o-", color="steelblue", markersize=4,
            linewidth=1.0, label="empirical std")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"reported σ  [{LABEL_PRETTY[label_key]}]")
    ax.set_ylabel("empirical std(residual) per bin")
    ax.set_title(f"σ calibration — {LABEL_SHORT[label_key]}")
    ax.legend(loc="upper left", fontsize=7)


def _training_curves(ax, history: dict) -> None:
    cmap = plt.get_cmap("tab10")
    for mem in history["members"]:
        seed = mem["seed"]
        hist = mem["history"]
        ep = [h["epoch"] for h in hist]
        tl = [h["train_loss"] for h in hist]
        vl = [h["val_loss"] for h in hist]
        c = cmap(seed)
        ax.plot(ep, tl, "-", color=c, linewidth=1.1, alpha=0.85,
                label=f"seed {seed} train")
        ax.plot(ep, vl, "--", color=c, linewidth=1.1, alpha=0.85)
        # Mark best epoch
        be = mem["best_epoch"]
        ax.scatter([be], [mem["best_val_loss"]], s=22, c=[c],
                   edgecolor="k", linewidths=0.5, zorder=5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("β-NLL block-Cholesky loss (weighted)")
    ax.set_title("training / val loss per seed (— train, -- val)")
    ax.legend(loc="upper right", ncol=2, fontsize=6, handlelength=1.5)


def _alpha_by_mh_bin(ax, df: pd.DataFrame) -> None:
    edges = np.array([-np.inf, -1.5, -1.0, -0.5, 0.0, np.inf])
    mh = df["mh_truth"].to_numpy()
    a_t = df["alpha_m_truth"].to_numpy()
    a_p = df["alpha_m_pred"].to_numpy()
    labels = ["(-∞, −1.5)", "[−1.5, −1)", "[−1, −0.5)", "[−0.5, 0)", "[0, +∞)"]
    centres = np.arange(5)
    means_t, means_p, stds_p, n_per = [], [], [], []
    for k in range(5):
        m = (mh >= edges[k]) & (mh < edges[k + 1])
        if m.sum() == 0:
            means_t.append(np.nan); means_p.append(np.nan)
            stds_p.append(np.nan); n_per.append(0); continue
        means_t.append(float(np.nanmean(a_t[m])))
        means_p.append(float(np.nanmean(a_p[m])))
        stds_p.append(float(np.nanstd(a_p[m])))
        n_per.append(int(m.sum()))
    means_t = np.asarray(means_t); means_p = np.asarray(means_p)
    stds_p = np.asarray(stds_p)
    ax.errorbar(centres - 0.1, means_t, yerr=0, fmt="o", color="black",
                markersize=7, label="APOGEE truth mean")
    ax.errorbar(centres + 0.1, means_p, yerr=stds_p, fmt="s",
                color="tab:blue", markersize=7, capsize=3,
                label="v1.1 pred mean ± std")
    for k, nk in enumerate(n_per):
        ax.text(k, ax.get_ylim()[0] if False else -0.02, f"n={nk:,}",
                ha="center", va="top", fontsize=6)
    ax.set_xticks(centres)
    ax.set_xticklabels(labels, rotation=0, fontsize=7)
    ax.set_xlabel("truth [M/H] bin")
    ax.set_ylabel(r"mean $[\alpha/\mathrm{M}]$ [dex]")
    ax.set_title("α by truth [M/H] bin — val")
    ax.legend(loc="upper right", fontsize=7)


def _sigma_alpha_val_vs_stream3(ax, val_df: pd.DataFrame,
                                 s3_df: pd.DataFrame | None) -> None:
    """σ_α distribution: val (by truth [M/H]) vs Stream-3 (by pred [M/H]).

    Exposes the metal-poor prior-collapse: val σ_α is small and narrow, but
    Stream-3 σ_α grows ×3 in the halo regime because the ensemble is
    broadcasting its uncertainty there.
    """
    edges = np.array([-np.inf, -1.5, -1.0, -0.5, 0.0, np.inf])
    # val — stratified on truth [M/H]
    v_mh = val_df["mh_truth"].to_numpy()
    v_sa = val_df["alpha_m_sigma"].to_numpy()
    cmap = plt.get_cmap("viridis")
    labels = ["(-∞, −1.5)", "[−1.5, −1)", "[−1, −0.5)", "[−0.5, 0)", "[0, +∞)"]
    for k in range(5):
        m = (v_mh >= edges[k]) & (v_mh < edges[k + 1]) & np.isfinite(v_sa)
        if m.sum() < 30:
            continue
        ax.hist(v_sa[m], bins=60, range=(0.0, 0.25), histtype="step",
                color=cmap(k / 4.0), linewidth=1.4, density=True,
                label=f"val {labels[k]} (n={m.sum():,})")
    # Stream-3 halo overlay — stratified on pred [M/H]
    if s3_df is not None and "mh_pred" in s3_df.columns:
        s_mh = s3_df["mh_pred"].to_numpy()
        s_sa = s3_df["alpha_m_sigma"].to_numpy()
        halo = (s_mh < -1.0) & np.isfinite(s_sa)
        if halo.sum() > 100:
            ax.hist(s_sa[halo], bins=60, range=(0.0, 0.25),
                    histtype="stepfilled", color="tab:red", linewidth=1.0,
                    alpha=0.3, density=True,
                    label=f"Stream-3 pred [M/H]<−1 (n={halo.sum():,})")
    ax.set_xlabel(r"reported $\sigma_\alpha$ [dex]")
    ax.set_ylabel("density")
    ax.set_title(r"σ$_α$ val-by-truth-[M/H] vs Stream-3 halo")
    ax.legend(loc="upper right", fontsize=6)


def _coverage_bars(ax, df: pd.DataFrame) -> None:
    """Per-label 68% / 95% empirical coverage vs nominal.

    Well-calibrated: bars hit 0.68 / 0.95.
    """
    ks, cov68, cov95 = [], [], []
    for lab in LABELS:
        r = df[f"{lab}_pred"].to_numpy() - df[f"{lab}_truth"].to_numpy()
        s = df[f"{lab}_sigma"].to_numpy()
        m = np.isfinite(r) & np.isfinite(s) & (s > 0)
        z = np.abs(r[m]) / s[m]
        ks.append(LABEL_SHORT[lab])
        cov68.append(float((z <= 1.0).mean()))
        cov95.append(float((z <= 1.96).mean()))
    x = np.arange(len(ks))
    ax.bar(x - 0.2, cov68, width=0.4, color="steelblue", label="emp 68%")
    ax.bar(x + 0.2, cov95, width=0.4, color="darkorange", label="emp 95%")
    ax.axhline(0.68, color="steelblue", linewidth=0.8, linestyle="--",
               alpha=0.6)
    ax.axhline(0.95, color="darkorange", linewidth=0.8, linestyle="--",
               alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(ks)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("empirical coverage")
    ax.set_title("coverage @ 1σ / 1.96σ (dashed = nominal)")
    ax.legend(loc="lower right", fontsize=7)


def _epistemic_dist(ax, df: pd.DataFrame) -> None:
    """Per-label epistemic σ (ensemble disagreement) distribution."""
    cmap = plt.get_cmap("tab10")
    for i, lab in enumerate(LABELS):
        e = df[f"{lab}_epi"].to_numpy()
        e = e[np.isfinite(e) & (e > 0)]
        if lab == "teff":
            # Teff on a separate x-scale would squash the others; rescale to dex-equivalent.
            e = e / 100.0  # display units: "100 K"
            legend = f"{LABEL_SHORT[lab]} (×100 K)"
        else:
            legend = LABEL_SHORT[lab]
        if e.size == 0:
            continue
        p99 = np.nanpercentile(e, 99)
        ax.hist(e, bins=60, range=(0.0, p99), histtype="step",
                color=cmap(i), linewidth=1.3, density=True, label=legend)
    ax.set_xlabel("ensemble-epistemic σ  [dex or 100 K]")
    ax.set_ylabel("density")
    ax.set_title("ensemble disagreement per label")
    ax.legend(loc="upper right", fontsize=7)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-pred", type=Path, default=VAL_PRED)
    ap.add_argument("--history", type=Path, default=HIST_JSON)
    ap.add_argument("--stream3", type=Path, default=STREAM3)
    ap.add_argument("--output", type=Path, default=OUT_PNG)
    args = ap.parse_args()

    val = pd.read_parquet(args.val_pred)
    history = json.loads(args.history.read_text())
    s3 = pd.read_parquet(args.stream3) if args.stream3.exists() else None

    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(4, 5, hspace=0.38, wspace=0.32)

    # Row 1: pred vs truth per label
    for i, lab in enumerate(LABELS):
        ax = fig.add_subplot(gs[0, i])
        _scatter_pred_truth(ax,
                            val[f"{lab}_truth"].to_numpy(),
                            val[f"{lab}_pred"].to_numpy(), lab)

    # Row 2: residual hist per label
    for i, lab in enumerate(LABELS):
        ax = fig.add_subplot(gs[1, i])
        _residual_hist(ax,
                       val[f"{lab}_truth"].to_numpy(),
                       val[f"{lab}_pred"].to_numpy(), lab)

    # Row 3: σ calibration per label
    for i, lab in enumerate(LABELS):
        ax = fig.add_subplot(gs[2, i])
        _sigma_reliability(ax,
                           val[f"{lab}_truth"].to_numpy(),
                           val[f"{lab}_pred"].to_numpy(),
                           val[f"{lab}_sigma"].to_numpy(), lab)

    # Row 4: training, α-by-mh, σ_α val-vs-stream3, coverage, epistemic
    ax_tr = fig.add_subplot(gs[3, 0])
    _training_curves(ax_tr, history)

    ax_ab = fig.add_subplot(gs[3, 1])
    _alpha_by_mh_bin(ax_ab, val)

    ax_sa = fig.add_subplot(gs[3, 2])
    _sigma_alpha_val_vs_stream3(ax_sa, val, s3)

    ax_cov = fig.add_subplot(gs[3, 3])
    _coverage_bars(ax_cov, val)

    ax_ep = fig.add_subplot(gs[3, 4])
    _epistemic_dist(ax_ep, val)

    n = len(val)
    val_loss = float(history.get("val_loss_mean", float("nan")))
    fig.suptitle(
        f"Pipeline-1 v1.1 NN validation diagnostics  "
        f"(val n={n:,}, 5-member ensemble, "
        f"mean best val loss = {val_loss:+.4f})",
        fontsize=14, y=0.995,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"wrote {args.output}  ({args.output.stat().st_size/1024:.0f} KiB)")

    # Stdout summary — useful for the delta report.
    print()
    print("=== per-label summary (val) ===")
    for lab in LABELS:
        t = val[f"{lab}_truth"].to_numpy()
        p = val[f"{lab}_pred"].to_numpy()
        s = val[f"{lab}_sigma"].to_numpy()
        r = p - t
        m = np.isfinite(r) & np.isfinite(s) & (s > 0)
        z = r[m] / s[m]
        print(f"  {LABEL_SHORT[lab]:>8s}  "
              f"bias={np.mean(r):+.4g}  rmse={np.sqrt(np.mean(r**2)):.4g}  "
              f"σ̄={np.nanmean(s):.4g}  cov68={(np.abs(z) <= 1).mean():.3f}  "
              f"cov95={(np.abs(z) <= 1.96).mean():.3f}")


if __name__ == "__main__":
    main()
