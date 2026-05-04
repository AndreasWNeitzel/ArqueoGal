"""H7: Tier-1 held-out pred vs truth + uncertainty calibration check.

Mirror of D1 (pred-vs-truth + residual-vs-truth hexbin grids), but
restricted to the held-out (val + test) Stream-1 cohort AND only stars
that landed in Tier 1 of the release. This is the "science-grade"
delivery we are advertising.

Three rows per element:
  row 0 = pred vs truth, hex colour = median predicted σ in cell
  row 1 = residual (pred − truth) vs truth, hex colour = log10 density,
          with the empirical ±1σ_resid white reference lines AND the
          mean-predicted-σ blue lines for direct calibration eyeballing
  row 2 = pull (residual / predicted σ) histogram per element. A
          calibrated model has a unit-variance Gaussian here (pull-σ
          ≈ 1, mean ≈ 0). pull-σ < 1 means the model is overconfident
          on average; > 1 means underconfident.

Held-out partition is reproduced via stratified_split_ids(seed=0,
fracs=(0.70, 0.15, 0.15)) — exactly the trainer's split. Tier 1 is
read from the predictions parquet's release_tier column when present;
falls back to the σ-Teff sigma-driven heuristic used elsewhere in the
gallery.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
sys.path.insert(0, str(REPO / "src"))

from _common import save_fig

from arqueogal.utils.plotting import set_aa_style
from arqueogal.xp_abundances.main.data import stratified_split_ids

OUT = REPO / "reports/gallery/H_hybrid_release"
SPLIT_SEED = 0
FRACS = (0.70, 0.15, 0.15)

# Per-label clip ranges to compute the displayed sigma. Mirrors H5/H6.
SIGMA_CLIP = {
    "teff": (10.0, 500.0),
    "logg": (0.05, 2.0),
    "mh": (0.01, 1.0),
    "alpha_m": (0.01, 1.0),
    "mg_h": (0.01, 1.0),
}

PREDS = [
    ("Teff", "teff_apogee", "teff_pred", "teff_sigma", "K"),
    ("log g", "logg_apogee", "logg_pred", "logg_sigma", "dex"),
    ("[M/H]", "mh_apogee", "mh_pred", "mh_sigma", "dex"),
    ("[alpha/M]", "alpha_m_apogee", "alpha_m_pred", "alpha_m_sigma", "dex"),
    ("[Mg/H]", "mg_h_apogee", "mg_h_pred", "mg_h_sigma", "dex"),
]


def main() -> None:
    set_aa_style()

    pred_path = REPO / "data/processed/pipeline1_predictions_stream1.parquet"
    truth_path = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"
    if not pred_path.exists() or not truth_path.exists():
        print(f"Error: required input missing\n  pred: {pred_path}\n  truth: {truth_path}")
        return

    import pyarrow.parquet as _pq

    avail = {f.name for f in _pq.ParquetFile(pred_path).schema_arrow}
    pred_cols = [
        "source_id",
        "teff_pred",
        "logg_pred",
        "mh_pred",
        "alpha_m_pred",
        "mg_h_pred",
        "teff_sigma",
        "logg_sigma",
        "mh_sigma",
        "alpha_m_sigma",
        "mg_h_sigma",
        "ood_joint_flag",
        "label_extrapolation_flag",  # tier drivers
    ]
    if "release_tier" in avail:
        pred_cols.append("release_tier")
    pred = pd.read_parquet(pred_path, columns=pred_cols)
    pred = pred.drop_duplicates("source_id", keep="first")

    truth = pd.read_parquet(
        truth_path,
        columns=[
            "source_id",
            "fe_h_apogee",
            "teff_apogee",
            "logg_apogee",
            "b_deg",
            "mh_apogee",
            "alpha_m_apogee",
            "mg_h_apogee",
        ],
    )
    truth = truth.drop_duplicates("source_id", keep="first")
    df = pred.merge(truth, on="source_id", how="inner")

    # Tier assignment (reuse the same heuristic-or-truth logic as H5/H6).
    if "release_tier" in df.columns:
        df["tier"] = df["release_tier"].astype(int)
    else:
        teff_sig = df["teff_sigma"].clip(SIGMA_CLIP["teff"][0], SIGMA_CLIP["teff"][1])
        df["tier"] = np.where(teff_sig < 100, 1, np.where(teff_sig < 200, 2, 3)).astype(int)

    # Held-out (val + test) partition only.
    splits = stratified_split_ids(df, fracs=FRACS, seed=SPLIT_SEED)
    train_ids = set(splits["train"].tolist())
    df = df[~df["source_id"].isin(train_ids)].reset_index(drop=True)
    n_held = len(df)
    df_t1 = df[df["tier"] == 1].reset_index(drop=True)
    n_t1 = len(df_t1)
    print(
        f"[H7] held-out (val+test) cohort n={n_held:,}; "
        f"Tier 1 subset n={n_t1:,} ({100 * n_t1 / max(n_held, 1):.1f}%)"
    )

    fig, axes = plt.subplots(2, 5, figsize=(20, 8.5))
    GRID = 80

    summary_rows = []
    for i, (elem_label, t_col, p_col, s_col, unit) in enumerate(PREDS):
        truth_v = df_t1[t_col].to_numpy(np.float64)
        pred_v = df_t1[p_col].to_numpy(np.float64)
        sigma_v = df_t1[s_col].to_numpy(np.float64)
        m = np.isfinite(truth_v) & np.isfinite(pred_v) & np.isfinite(sigma_v) & (sigma_v > 0)
        truth_v, pred_v, sigma_v = truth_v[m], pred_v[m], sigma_v[m]
        n = len(truth_v)

        resid = pred_v - truth_v
        rmse = float(np.sqrt(np.mean(resid**2))) if n > 0 else float("nan")
        bias = float(np.median(resid)) if n > 0 else float("nan")
        sigma_med = float(np.median(sigma_v)) if n > 0 else float("nan")
        sigma_std = float(np.std(resid)) if n > 0 else float("nan")
        pull = resid / sigma_v if n > 0 else np.array([])
        pull_mean = float(np.mean(pull)) if n > 0 else float("nan")
        pull_std = float(np.std(pull)) if n > 0 else float("nan")

        # Row 0: pred vs truth, hex coloured by median σ in cell.
        ax = axes[0, i]
        if n > 0:
            hb = ax.hexbin(
                truth_v,
                pred_v,
                C=sigma_v,
                reduce_C_function=np.median,
                gridsize=GRID,
                cmap="viridis",
                mincnt=1,
            )
            cbar = plt.colorbar(hb, ax=ax, pad=0.02)
            cbar.set_label(rf"median $\sigma$ ({unit})", fontsize=7)
            rng_min = float(min(truth_v.min(), pred_v.min()))
            rng_max = float(max(truth_v.max(), pred_v.max()))
            ax.plot([rng_min, rng_max], [rng_min, rng_max], "k--", lw=0.7, alpha=0.4)
        ax.set_xlabel(f"{elem_label} truth", fontsize=8)
        ax.set_ylabel(f"{elem_label} pred", fontsize=8)
        ax.set_title(f"{elem_label}", fontsize=9, fontweight="semibold")
        ax.text(
            0.05,
            0.95,
            f"n={n:,}\nRMSE={rmse:.2g}\nbias={bias:+.2g}",
            transform=ax.transAxes,
            fontsize=6.5,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.88, pad=2),
        )
        ax.grid(True, alpha=0.25)

        # Row 1: residual vs truth, hex coloured by log10 density.
        ax = axes[1, i]
        if n > 0:
            hb2 = ax.hexbin(
                truth_v,
                resid,
                gridsize=GRID,
                cmap="plasma",
                mincnt=1,
                bins="log",
            )
            cbar2 = plt.colorbar(hb2, ax=ax, pad=0.02)
            cbar2.set_label(r"log$_{10}$ N", fontsize=7)
            ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.4)
            ax.axhline(
                sigma_std,
                color="white",
                lw=0.7,
                ls=":",
                alpha=0.85,
                label=rf"$\pm 1\sigma_\mathrm{{resid}}$ ({sigma_std:.2g})",
            )
            ax.axhline(-sigma_std, color="white", lw=0.7, ls=":", alpha=0.85)
            ax.axhline(
                sigma_med,
                color="cyan",
                lw=0.7,
                ls="-.",
                alpha=0.85,
                label=rf"$\pm$ median pred $\sigma$ ({sigma_med:.2g})",
            )
            ax.axhline(-sigma_med, color="cyan", lw=0.7, ls="-.", alpha=0.85)
            ax.legend(fontsize=6, loc="upper right", framealpha=0.85)
        ax.set_xlabel(f"{elem_label} truth", fontsize=8)
        ax.set_ylabel(f"residual ({unit})", fontsize=8)
        ax.set_title(f"{elem_label} residual", fontsize=9, fontweight="semibold")
        ax.grid(True, alpha=0.25)

        # Pull-histogram row dropped 2026-05-03 — Y16 covers calibration
        # in talk-grade form; H7 keeps pred-vs-truth + residual.
        summary_rows.append(
            {
                "label": elem_label,
                "n": n,
                "rmse": rmse,
                "bias": bias,
                "sigma_med_pred": sigma_med,
                "sigma_resid": sigma_std,
                "pull_mean": pull_mean,
                "pull_std": pull_std,
            }
        )

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    save_fig(fig, OUT / "H7_tier1_holdout_pred_vs_truth")

    # Print a concise text summary that goes into the run log.
    print(f"\n=== H7 Tier-1 held-out summary (n={n_t1:,}) ===")
    print(
        f"{'label':>10s}  {'n':>8s}  {'RMSE':>8s}  {'bias':>8s}  "
        f"{'σ_pred':>8s}  {'σ_resid':>8s}  {'pull_μ':>8s}  {'pull_σ':>8s}"
    )
    for r in summary_rows:
        print(
            f"{r['label']:>10s}  {r['n']:>8d}  {r['rmse']:>8.4g}  "
            f"{r['bias']:>+8.4g}  {r['sigma_med_pred']:>8.4g}  "
            f"{r['sigma_resid']:>8.4g}  {r['pull_mean']:>+8.3f}  "
            f"{r['pull_std']:>8.3f}"
        )


if __name__ == "__main__":
    main()
