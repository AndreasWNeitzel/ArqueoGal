"""Stage 17: Pipeline-1 per-regime diagnostics.

For each Stream-3 arm (volume, uniform) produces a 3x3 diagnostic figure
showing the ``release_tier`` composition binned along the axes most
relevant to "where does this model work?":

  row 1 (1-D tier composition vs physical axis)
    G-mag    Av (nbhd-median)    distance_pc

  row 2 (2-D Tier-3 fraction surface on physical planes)
    Galactic sky (Mollweide)    HR (Teff vs log g)    chem ([M/H] vs [α/M])

  row 3 (per-label σ stratified by tier / flag contribution)
    σ by label & tier   Tier-3 flag contribution   tier composition bar

Also writes a small ``tier_summary.json`` with the counts to the stage dir.

Run: ``PYTHONPATH=src python scripts/gallery/plot_17_pipeline1_regime_diagnostics.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DATA_PROCESSED,
    GALLERY,
    apply_style,
    galactic_mollweide,
    radec_to_galactic,
    save_fig,
    style_galactic_mollweide,
)

_ROOT = Path(__file__).resolve().parents[2]
_OUT = GALLERY / "17_pipeline1_regime_diagnostics"

_ARMS = ("volume", "uniform")

_LABELS = ("teff", "logg", "mh", "alpha_m", "mg_h")
_LABEL_TEX = {
    "teff": r"$T_{\rm eff}$",
    "logg": r"$\log g$",
    "mh": r"$[{\rm M}/{\rm H}]$",
    "alpha_m": r"$[\alpha/{\rm M}]$",
    "mg_h": r"$[{\rm Mg}/{\rm H}]$",
}

_TIER_COLORS = {1: "#2ca02c", 2: "#f5a623", 3: "#d62728"}
_TIER_LABEL = {1: "Tier 1 (per-star)", 2: "Tier 2 (statistical)", 3: "Tier 3 (withheld)"}

_FLAG_ORDER = (
    "ood_joint_flag",
    "latent_support_flag",
    "regime_b_flag",
    "mode_ambiguous_flag",
    "ood_disagreement_flag",
    "aux_missing_any",
)
_FLAG_COLORS = {
    "ood_joint_flag": "#d62728",
    "latent_support_flag": "#ff7f0e",
    "regime_b_flag": "#9467bd",
    "mode_ambiguous_flag": "#8c564b",
    "ood_disagreement_flag": "#e377c2",
    "aux_missing_any": "#7f7f7f",
}


def _load(arm: str) -> pd.DataFrame:
    pred = pd.read_parquet(
        DATA_PROCESSED / f"pipeline1_predictions_stream3_joint_{arm}.parquet"
    )
    feat_cols = [
        "source_id", "ra_deg", "dec_deg", "b_deg",
        "g_mag", "av_nbhd_median", "distance_pc", "r_med_photogeo",
    ]
    feat = pd.read_parquet(DATA_PROCESSED / "pipeline1_features_stream3.parquet",
                           columns=feat_cols)
    df = pred.merge(feat, on="source_id", how="left")
    if "distance_pc" not in df.columns or df["distance_pc"].isna().all():
        df["distance_pc"] = df["r_med_photogeo"]
    if "release_tier" not in df.columns:
        raise RuntimeError(f"release_tier column missing in arm '{arm}' parquet — "
                           "run scripts/assign_release_tier.py first")
    return df


def _binned_tier_fractions(x: np.ndarray, tier: np.ndarray, edges: np.ndarray) -> dict[int, np.ndarray]:
    idx = np.digitize(x, edges) - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    n = len(edges) - 1
    frac = {t: np.zeros(n, dtype=float) for t in (1, 2, 3)}
    total = np.zeros(n, dtype=float)
    for b in range(n):
        mask = idx == b
        m = int(mask.sum())
        if m == 0:
            continue
        total[b] = m
        for t in (1, 2, 3):
            frac[t][b] = float(((tier == t) & mask).sum()) / m
    return frac, total


def _panel_tier_vs_axis(ax, df, col, edges, xlabel, log_x=False):
    x = df[col].to_numpy()
    tier = df["release_tier"].to_numpy()
    m = np.isfinite(x) & np.isfinite(tier)
    frac, total = _binned_tier_fractions(x[m], tier[m], edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    width = np.diff(edges)
    bot = np.zeros_like(centres)
    for t in (1, 2, 3):
        ax.bar(centres, frac[t], width=width, bottom=bot,
               color=_TIER_COLORS[t], edgecolor="none", align="center",
               label=_TIER_LABEL[t])
        bot = bot + frac[t]
    ax.set_xlabel(xlabel)
    ax.set_ylabel("tier fraction")
    ax.set_ylim(0.0, 1.0)
    if log_x:
        ax.set_xscale("log")
    # secondary axis showing counts per bin (faint line)
    ax2 = ax.twinx()
    ax2.step(centres, total, where="mid", color="#333333", linewidth=0.8, alpha=0.6)
    ax2.set_yscale("log")
    ax2.set_ylabel("n per bin", fontsize=8, color="#555")
    ax2.tick_params(axis="y", labelsize=8, colors="#555")
    ax2.grid(False)


def _panel_tier3_mollweide(ax, df):
    ra = df["ra_deg"].to_numpy()
    dec = df["dec_deg"].to_numpy()
    m = np.isfinite(ra) & np.isfinite(dec)
    l_deg, b_deg = radec_to_galactic(ra[m], dec[m])
    x, y = galactic_mollweide(l_deg, b_deg)
    is_t3 = (df["release_tier"].to_numpy()[m] == 3).astype(float)
    # hexbin mean of is_t3
    hb = ax.hexbin(x, y, C=is_t3, reduce_C_function=np.mean,
                   gridsize=60, cmap="magma", mincnt=30, vmin=0.0, vmax=1.0)
    plt.colorbar(hb, ax=ax, shrink=0.7, pad=0.02, label="Tier-3 fraction")
    style_galactic_mollweide(ax)
    ax.set_title("sky (Mollweide, Galactic)")


def _panel_tier3_2d(ax, df, xcol, ycol, xlim, ylim, xlabel, ylabel, invert_x=False, invert_y=False):
    x = df[xcol].to_numpy()
    y = df[ycol].to_numpy()
    tier = df["release_tier"].to_numpy()
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(tier)
    is_t3 = (tier[m] == 3).astype(float)
    x_extent = (min(xlim), max(xlim))
    y_extent = (min(ylim), max(ylim))
    hb = ax.hexbin(x[m], y[m], C=is_t3, reduce_C_function=np.mean,
                   gridsize=50, cmap="magma", mincnt=10, vmin=0.0, vmax=1.0,
                   extent=(x_extent[0], x_extent[1], y_extent[0], y_extent[1]))
    plt.colorbar(hb, ax=ax, shrink=0.85, pad=0.02, label="Tier-3 fraction")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _panel_sigma_by_tier(ax, df):
    # One boxplot set per label, three boxes per label (tier 1/2/3)
    positions = []
    boxdata = []
    colors = []
    for i, lbl in enumerate(_LABELS):
        col = f"{lbl}_sigma"
        if col not in df.columns:
            continue
        for j, t in enumerate((1, 2, 3)):
            vals = df.loc[df["release_tier"] == t, col].to_numpy()
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            boxdata.append(vals)
            positions.append(3 * i + j)
            colors.append(_TIER_COLORS[t])
    bp = ax.boxplot(
        boxdata, positions=positions, widths=0.7,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
    )
    for patch, c in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(c)
        patch.set_edgecolor("#333333")
        patch.set_alpha(0.85)
    ax.set_xticks([3 * i + 1 for i in range(len(_LABELS))])
    ax.set_xticklabels([_LABEL_TEX[l] for l in _LABELS], fontsize=9)
    ax.set_ylabel(r"$\sigma$ (calibrated)")
    ax.set_yscale("log")
    ax.set_title(r"$\sigma$ by label × tier")


def _panel_flag_contributions(ax, df):
    tier3 = df["release_tier"] == 3
    n_t3 = int(tier3.sum())
    if n_t3 == 0:
        ax.text(0.5, 0.5, "no Tier-3 rows", transform=ax.transAxes,
                ha="center", va="center", fontsize=10)
        ax.set_axis_off()
        return
    counts: dict[str, int] = {}
    for flag in _FLAG_ORDER:
        if flag not in df.columns:
            continue
        counts[flag] = int((tier3 & df[flag].fillna(False).astype(bool)).sum())
    # Also NaN-pred contributions
    pred_cols = ("teff_pred", "logg_pred", "mh_pred", "alpha_m_pred", "mg_h_pred")
    nan_mask = pd.Series(False, index=df.index)
    for c in pred_cols:
        if c in df.columns:
            nan_mask = nan_mask | df[c].isna()
    counts["nan_pred"] = int((tier3 & nan_mask).sum())

    flags = list(counts.keys())
    vals = [counts[f] / n_t3 for f in flags]
    colors = [_FLAG_COLORS.get(f, "#444") for f in flags]
    y = np.arange(len(flags))
    ax.barh(y, vals, color=colors, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(flags, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("fraction of Tier-3 rows flagged (non-exclusive)")
    ax.set_title(f"Tier-3 flag mix  (n={n_t3:,})")


def _panel_tier_summary_bar(ax, df, arm):
    n = len(df)
    counts = {t: int((df["release_tier"] == t).sum()) for t in (1, 2, 3)}
    fracs = {t: counts[t] / n for t in counts}
    ax.bar(
        [1, 2, 3],
        [fracs[t] for t in (1, 2, 3)],
        color=[_TIER_COLORS[t] for t in (1, 2, 3)],
        edgecolor="#333333",
    )
    for t in (1, 2, 3):
        ax.text(
            t, fracs[t] + 0.01,
            f"{fracs[t] * 100:.1f}%\n(n={counts[t]:,})",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Tier 1", "Tier 2", "Tier 3"])
    ax.set_ylim(0.0, 1.05 * max(fracs.values()) + 0.1)
    ax.set_ylabel("fraction")
    ax.set_title(f"{arm} arm  •  n={n:,}")


def _figure_for_arm(arm: str) -> dict[str, int]:
    df = _load(arm)

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle(
        f"Pipeline 1 — regime diagnostics  •  Stream-3 {arm} arm",
        fontsize=13, y=0.995,
    )

    # Row 1 — tier composition vs 1-D axes
    g_edges = np.linspace(8.0, 17.65, 25)
    av_edges = np.linspace(0.0, 3.0, 25)
    d_edges = np.logspace(np.log10(50.0), np.log10(30000.0), 25)

    _panel_tier_vs_axis(axes[0, 0], df, "g_mag", g_edges, "G-mag")
    _panel_tier_vs_axis(axes[0, 1], df, "av_nbhd_median", av_edges,
                        r"$A_V$ (nbhd-median)")
    _panel_tier_vs_axis(axes[0, 2], df, "distance_pc", d_edges,
                        "distance [pc]", log_x=True)

    # Row 2 — 2-D Tier-3 fraction
    axes[1, 0].remove()
    ax_moll = fig.add_subplot(3, 3, 4, projection="mollweide")
    _panel_tier3_mollweide(ax_moll, df)

    _panel_tier3_2d(axes[1, 1], df, "teff_pred", "logg_pred",
                    xlim=(5800, 3800), ylim=(4.2, 0.5),
                    xlabel=r"$T_{\rm eff}$ [K]", ylabel=r"$\log g$")
    axes[1, 1].set_title("HR")

    _panel_tier3_2d(axes[1, 2], df, "mh_pred", "alpha_m_pred",
                    xlim=(-2.0, 0.7), ylim=(-0.15, 0.55),
                    xlabel=r"$[{\rm M}/{\rm H}]$", ylabel=r"$[\alpha/{\rm M}]$")
    axes[1, 2].set_title("chemistry")

    # Row 3 — σ distributions + flag mix + summary
    _panel_sigma_by_tier(axes[2, 0], df)
    _panel_flag_contributions(axes[2, 1], df)
    _panel_tier_summary_bar(axes[2, 2], df, arm)

    # Legend in row 1
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.975), ncol=3, fontsize=10, frameon=False)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    save_fig(fig, _OUT / f"regime_diagnostics_{arm}.png", tight=False)

    return {int(t): int((df["release_tier"] == t).sum()) for t in (1, 2, 3)}


def main() -> None:
    apply_style()
    _OUT.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, int | dict[int, int]]] = {}
    for arm in _ARMS:
        counts = _figure_for_arm(arm)
        summary[arm] = {
            "counts": counts,
            "n_rows": sum(counts.values()),
        }

    (_OUT / "tier_summary.json").write_text(
        json.dumps(
            {arm: {"counts": {str(k): v for k, v in s["counts"].items()},  # type: ignore[index]
                   "n_rows": s["n_rows"]}
             for arm, s in summary.items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
