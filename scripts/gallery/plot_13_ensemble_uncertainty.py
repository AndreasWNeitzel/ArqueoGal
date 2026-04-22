"""Stage 13: ensemble uncertainty + OOD rejection.

Outputs:
  - reports/gallery/13_ensemble_uncertainty/aleatoric_vs_epistemic_per_label.png
  - reports/gallery/13_ensemble_uncertainty/ood_mahalanobis_distribution.png
  - reports/gallery/13_ensemble_uncertainty/ood_disagreement_distribution.png
  - reports/gallery/13_ensemble_uncertainty/ood_joint_decision_plot.png
  - reports/gallery/13_ensemble_uncertainty/regime_b_envelope_footprint.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).resolve().parent))
from _common import DATA_PROCESSED, GALLERY, apply_style, save_fig  # noqa: E402

OUT = GALLERY / "13_ensemble_uncertainty"

LABELS = ["teff", "logg", "mh", "alpha_m", "mg_h"]
LABEL_TEX = {"teff": r"$T_{\rm eff}$", "logg": r"$\log g$",
             "mh": r"$[{\rm M}/{\rm H}]$",
             "alpha_m": r"$[\alpha/{\rm M}]$",
             "mg_h": r"$[{\rm Mg}/{\rm H}]$"}


def _val_df():
    return pq.read_table("reports/pipeline1/run_a_v11/val_predictions.parquet").to_pandas()


def _stream3_df():
    cols = ["source_id", "ood_mahalanobis_score", "ood_disagreement_flag", "ood_joint_flag",
            "regime_b_flag",
            "teff_sigma", "logg_sigma", "mh_sigma", "alpha_m_sigma", "mg_h_sigma",
            "teff_epistemic_var", "logg_epistemic_var", "mh_epistemic_var",
            "alpha_m_epistemic_var", "mg_h_epistemic_var"]
    schema = pq.read_schema(DATA_PROCESSED / "pipeline1_predictions_stream3_v11.parquet")
    have = [c for c in cols if c in {f.name for f in schema}]
    return pq.read_table(DATA_PROCESSED / "pipeline1_predictions_stream3_v11.parquet",
                         columns=have).to_pandas()


def aleatoric_vs_epistemic_per_label() -> None:
    df = _val_df()
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    for ax, lbl in zip(axes.flat, LABELS):
        al = df[f"{lbl}_sigma"].to_numpy()
        ep = df[f"{lbl}_epi"].to_numpy()
        m = np.isfinite(al) & np.isfinite(ep) & (al > 0) & (ep > 0)
        hb = ax.hexbin(al[m], ep[m], xscale="log", yscale="log", gridsize=60,
                       cmap="viridis", bins="log", mincnt=1)
        plt.colorbar(hb, ax=ax, shrink=0.85, pad=0.02, label="log N")
        lim_lo = min(np.percentile(al[m], 0.5), np.percentile(ep[m], 0.5))
        lim_hi = max(np.percentile(al[m], 99.5), np.percentile(ep[m], 99.5))
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "r--", lw=0.8)
        ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
        ax.set_xlabel(rf"aleatoric $\sigma$  ({LABEL_TEX[lbl]})")
        ax.set_ylabel(r"epistemic $\sigma$")
        ax.set_title(LABEL_TEX[lbl], fontsize=10)
    fig.suptitle("Aleatoric vs epistemic uncertainty per label  —  "
                 "below y=x means aleatoric-dominated (label noise); above means epistemic-dominated (model-limited)",
                 fontsize=11, fontweight="semibold", y=1.02)
    save_fig(fig, OUT / "aleatoric_vs_epistemic_per_label.png")


def ood_mahalanobis_distribution() -> None:
    df = _stream3_df()
    fig, ax = plt.subplots(figsize=(10, 5))
    if "ood_mahalanobis_score" not in df.columns:
        ax.text(0.5, 0.5, "ood_mahalanobis_score column absent",
                ha="center", va="center"); save_fig(fig, OUT / "ood_mahalanobis_distribution.png"); return
    vals = df["ood_mahalanobis_score"].to_numpy()
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [0.5, 99.9])
    ax.hist(np.clip(vals, lo, hi), bins=70, color="#1f77b4",
            edgecolor="#333", alpha=0.85)
    # 99th percentile threshold from training (we don't have that explicitly here,
    # so use the Stream-3 p99 as a proxy marker)
    p99 = np.percentile(vals, 99)
    ax.axvline(p99, color="#d62728", lw=1.4, ls="--",
               label=f"Stream-3 p99 = {p99:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("Mahalanobis score  (108-D XP block)")
    ax.set_ylabel("count  (log)")
    ax.set_title(f"Stream 3  —  Mahalanobis distribution  (n={len(vals):,})",
                 fontsize=11, fontweight="semibold")
    ax.legend()
    save_fig(fig, OUT / "ood_mahalanobis_distribution.png")


def ood_disagreement_distribution() -> None:
    """Per-star ensemble disagreement = mean(epistemic σ / total σ) across labels."""
    df = _stream3_df()
    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        ratios = []
        for lbl in LABELS:
            s = df[f"{lbl}_sigma"].to_numpy()
            ev = df[f"{lbl}_epistemic_var"].to_numpy()
            e = np.sqrt(np.clip(ev, 0, None))
            with np.errstate(invalid="ignore", divide="ignore"):
                r = e / s
            ratios.append(r)
        ratios = np.nanmean(np.stack(ratios, axis=0), axis=0)
        ratios = ratios[np.isfinite(ratios)]
        ax.hist(np.clip(ratios, 0, 2), bins=60, color="#9467bd",
                edgecolor="#333", alpha=0.85)
        ax.axvline(0.5, color="#d62728", lw=1.4, ls="--",
                   label=r"disagreement threshold 0.5")
        ax.set_xlabel(r"$\langle\sigma_{\rm epi} / \sigma_{\rm total}\rangle_{\rm labels}$")
        ax.set_ylabel("count")
        ax.set_yscale("log")
        ax.set_title(f"Stream 3 ensemble disagreement  (n={len(ratios):,})",
                     fontsize=11, fontweight="semibold")
        ax.legend()
    except Exception as exc:
        ax.text(0.5, 0.5, f"cannot compute: {exc}", ha="center", va="center",
                fontsize=9, transform=ax.transAxes)
    save_fig(fig, OUT / "ood_disagreement_distribution.png")


def ood_joint_decision_plot() -> None:
    df = _stream3_df()
    fig, ax = plt.subplots(figsize=(9, 7))
    if "ood_mahalanobis_score" not in df.columns:
        ax.text(0.5, 0.5, "missing column", ha="center", va="center"); save_fig(fig, OUT / "ood_joint_decision_plot.png"); return
    # disagreement proxy
    evars = np.stack([df[f"{l}_epistemic_var"].to_numpy() for l in LABELS], axis=0)
    sigs = np.stack([df[f"{l}_sigma"].to_numpy() for l in LABELS], axis=0)
    disagree = np.nanmean(np.sqrt(np.clip(evars, 0, None)) / sigs, axis=0)
    maha = df["ood_mahalanobis_score"].to_numpy()
    mh_thr = np.nanpercentile(maha, 99)
    da_thr = 0.5

    m = np.isfinite(maha) & np.isfinite(disagree)
    hb = ax.hexbin(maha[m], disagree[m], gridsize=70, cmap="Greys", bins="log",
                   mincnt=1, extent=(0, min(50, np.nanpercentile(maha, 99.5)), 0, 2))
    plt.colorbar(hb, ax=ax, shrink=0.85, pad=0.02, label="log N")
    ax.axvline(mh_thr, color="#1f77b4", lw=1.2, ls="--",
               label=f"maha p99 = {mh_thr:.2f}")
    ax.axhline(da_thr, color="#2ca02c", lw=1.2, ls="--",
               label=f"disagreement = {da_thr}")
    # shade joint-flag region
    ax.fill_between([mh_thr, 60], da_thr, 2.5, color="#d62728", alpha=0.2,
                    label="joint-flag (both fire)")
    ax.set_xlim(0, min(50, np.nanpercentile(maha, 99.5)))
    ax.set_ylim(0, 2)
    ax.set_xlabel("Mahalanobis score (108-D XP block)")
    ax.set_ylabel(r"$\langle\sigma_{\rm epi} / \sigma_{\rm total}\rangle$ (disagreement)")
    ax.set_title("Stream-3 joint OOD decision plane", fontsize=11, fontweight="semibold")
    ax.legend(loc="upper right")
    save_fig(fig, OUT / "ood_joint_decision_plot.png")


def regime_b_envelope_footprint() -> None:
    """Kiel diagram with RegimeB envelope: Teff>4750 K, logg<2.1, |b|<5°."""
    # Use Stream-1 truth values for the training footprint
    df = pq.read_table(DATA_PROCESSED / "pipeline1_features_stream1.parquet",
                       columns=["teff_apogee", "logg_apogee", "b_deg"]).to_pandas()
    m = np.isfinite(df["teff_apogee"]) & np.isfinite(df["logg_apogee"])
    low_b = m & (np.abs(df["b_deg"]) < 5)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.hexbin(df.loc[m, "teff_apogee"], df.loc[m, "logg_apogee"],
              gridsize=70, cmap="Greys", bins="log", mincnt=1)
    ax.hexbin(df.loc[low_b, "teff_apogee"], df.loc[low_b, "logg_apogee"],
              gridsize=70, cmap="Reds", bins="log", mincnt=5, alpha=0.8)

    # Envelope outline: Teff>4750, logg<2.1, |b|<5°
    env = Polygon([(4750, 2.1), (5500, 2.1), (5500, 0.5), (4750, 0.5)],
                   closed=True, edgecolor="#d62728", facecolor="none", lw=2,
                   label=r"Regime B envelope ($T_{\rm eff}>4750$, $\log g<2.1$, $|b|<5°$)")
    ax.add_patch(env)
    ax.set_xlim(5600, 3800); ax.set_ylim(3.8, 0.5)
    ax.set_xlabel(r"$T_{\rm eff}$ [K]"); ax.set_ylabel(r"$\log g$")
    ax.set_title(r"Regime B exclusion envelope on Stream 1 Kiel diagram  "
                 r"(red = $|b|<5°$ subsample)",
                 fontsize=11, fontweight="semibold")
    ax.legend(loc="lower left")
    save_fig(fig, OUT / "regime_b_envelope_footprint.png")


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    aleatoric_vs_epistemic_per_label()
    ood_mahalanobis_distribution()
    ood_disagreement_distribution()
    ood_joint_decision_plot()
    regime_b_envelope_footprint()


if __name__ == "__main__":
    main()
