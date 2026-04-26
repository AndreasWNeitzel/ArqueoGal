"""Stage 21b: Tier 1 + Tier 2 chemistry plane, overlapping scatter.

Two panels, each a SCATTER plot (not hexbin) so individual flag populations
are visible as distinct colored points overlaid on a Tier-1 grey baseline.

- **Panel 1 — Tier coloring**: Tier 1 (grey baseline), Tier 2 (orange),
  Tier 3 not shown (do-not-use). The reader sees how Tier 2 distributes
  relative to the science-grade Tier 1 cloud.
- **Panel 2 — Flag coloring**: Tier 1 (grey baseline) + Tier 2 stars
  colored by which caveat fired. The flag list contains both the v5
  active caveats (σ-inflated, mode_ambiguous on α/M, kin_ood) and the
  v4 retired-but-still-emitted diagnostics (regime_b, aux_missing,
  dist_prior_dominated, ood_disagreement). Diagnostic-only flags are
  shown so the reader can verify the v5 ablation conclusion that they
  fire trivially or shift no chemistry-plane structure. Where flags
  overlap on the same star, we color by highest-priority flag
  (σ-inflated > mode_ambiguous > kin_ood > diagnostic-only set).

Both panels use the hybrid chemistry surface
(``mh_hybrid_pred`` vs ``alpha_m_hybrid_pred``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
from _common import apply_style, save_fig

OUT = REPO / "reports/gallery/21b_flag_coloured_chemistry"
PARQUET_S3 = REPO / "release/D-Cat-b/hybrid_pipeline_run/predictions_with_features.parquet"
PARQUET_S2 = REPO / "release/D-Cat-b/hybrid_pipeline_run_stream2/predictions_with_features.parquet"
CHEM_X = (-2.5, 0.6); CHEM_Y = (-0.2, 0.5)

# Priority-ordered flag list. v5 active gates first (with " (active)"
# suffix), then v4 diagnostic-only flags. When a star has multiple flags,
# the FIRST one matched here gets the colour. σ-inflated takes priority
# because it is the dominant v4/v5 caveat that reroutes the prediction
# to the kNN-median.
FLAG_PRIORITY = (
    ("prediction_sigma_inflated_any", "σ-inflated (active)", "#d62728"),
    ("mode_ambiguous_flag", "mode_ambiguous on α/M (active)", "#bcbd22"),
    ("kin_ood_flag", "kin_ood (active, aux-assisted only)", "#7f7f7f"),
    ("regime_b_flag", "regime_b (diagnostic, retired)", "#ff7f0e"),
    ("dist_prior_dominated", "dist_prior_dominated (diagnostic, retired)", "#9467bd"),
    ("aux_missing_any", "aux_missing_any (diagnostic, retired)", "#17becf"),
    ("ood_disagreement_flag", "ood_disagreement (diagnostic, retired)", "#e377c2"),
)


def _scatter_panel(ax, t1, t2, *, title, color_t2):
    """Panel 1: Tier 1 grey, Tier 2 orange — show overlap."""
    rng = np.random.default_rng(42)
    # Subsample Tier 1 down to 80k for visibility (we have 200k); plot ALL
    # Tier 2 (290k) underneath, then Tier 1 on top so its grey shows through.
    n_t1 = min(80_000, len(t1))
    n_t2 = min(120_000, len(t2))
    i1 = rng.choice(len(t1), n_t1, replace=False) if len(t1) > n_t1 else np.arange(len(t1))
    i2 = rng.choice(len(t2), n_t2, replace=False) if len(t2) > n_t2 else np.arange(len(t2))

    # Tier 2 first, underneath
    ax.scatter(t2["mh_hybrid_pred"].iloc[i2], t2["alpha_m_hybrid_pred"].iloc[i2],
                s=1.2, color=color_t2, alpha=0.18, rasterized=True, zorder=1,
                label=f"Tier 2 ({len(t2):,})")
    # Tier 1 on top
    ax.scatter(t1["mh_hybrid_pred"].iloc[i1], t1["alpha_m_hybrid_pred"].iloc[i1],
                s=1.0, color="0.35", alpha=0.18, rasterized=True, zorder=2,
                label=f"Tier 1 ({len(t1):,})")
    ax.set_xlim(CHEM_X); ax.set_ylim(CHEM_Y)
    ax.set_xlabel("[M/H] (dex)"); ax.set_ylabel(r"[$\alpha$/M] (dex)")
    ax.set_title(title)
    leg = ax.legend(fontsize=8, loc="upper right", frameon=True, framealpha=0.92,
                     facecolor="white", edgecolor="0.4", markerscale=4)
    for h in leg.legend_handles:
        h.set_alpha(1.0)


def _flag_panel(ax, t1, t2, *, available_flags, n_total):
    """Panel 2: Tier 1 grey baseline + Tier 2 stars colored by flag priority."""
    rng = np.random.default_rng(42)

    # Tier 1 grey baseline (subsample for visibility)
    n_t1 = min(80_000, len(t1))
    i1 = rng.choice(len(t1), n_t1, replace=False) if len(t1) > n_t1 else np.arange(len(t1))
    ax.scatter(t1["mh_hybrid_pred"].iloc[i1], t1["alpha_m_hybrid_pred"].iloc[i1],
                s=0.8, color="0.55", alpha=0.10, rasterized=True, zorder=1,
                label=f"Tier 1 baseline ({len(t1):,})")

    # Assign each Tier 2 star to its highest-priority fired flag.
    assigned = np.full(len(t2), -1, dtype=int)
    for k, (col, _label, _color) in enumerate(FLAG_PRIORITY):
        if col not in available_flags: continue
        unflagged = (assigned == -1)
        m = unflagged & t2[col].astype(bool).to_numpy()
        assigned[m] = k

    # Plot Tier 2 stars per flag, sorted by population descending so smaller
    # populations sit on top.
    flag_counts = []
    for k, (col, label, color) in enumerate(FLAG_PRIORITY):
        if col not in available_flags: continue
        n_k = int((assigned == k).sum())
        if n_k > 0:
            flag_counts.append((n_k, k, col, label, color))
    flag_counts.sort(key=lambda x: -x[0])  # plot largest first

    for n_k, k, col, label, color in flag_counts:
        m = (assigned == k)
        # subsample within flag if too dense
        idx = np.flatnonzero(m)
        n_sub = min(40_000, len(idx))
        if len(idx) > n_sub:
            idx = rng.choice(idx, n_sub, replace=False)
        ax.scatter(t2["mh_hybrid_pred"].iloc[idx],
                    t2["alpha_m_hybrid_pred"].iloc[idx],
                    s=2.0, color=color, alpha=0.25, rasterized=True, zorder=3,
                    label=f"{label} ({n_k:,})")

    n_unflagged = int((assigned == -1).sum())
    if n_unflagged > 0:
        m = (assigned == -1)
        idx = np.flatnonzero(m)
        n_sub = min(20_000, len(idx))
        if len(idx) > n_sub:
            idx = rng.choice(idx, n_sub, replace=False)
        ax.scatter(t2["mh_hybrid_pred"].iloc[idx],
                    t2["alpha_m_hybrid_pred"].iloc[idx],
                    s=1.5, color="0.2", alpha=0.20, rasterized=True, zorder=2,
                    label=f"Tier 2, no flag listed ({n_unflagged:,})")

    ax.set_xlim(CHEM_X); ax.set_ylim(CHEM_Y)
    ax.set_xlabel("[M/H] (dex)"); ax.set_ylabel(r"[$\alpha$/M] (dex)")
    ax.set_title("Tier 2 stars by flag (priority-ordered, v5 active gates first)\n"
                  "σ-inflated > mode_ambiguous(α/M) > kin_ood > diagnostic-only set")
    leg = ax.legend(fontsize=7, loc="upper right", frameon=True, framealpha=0.92,
                     facecolor="white", edgecolor="0.4", markerscale=4,
                     ncol=1)
    for h in leg.legend_handles:
        h.set_alpha(1.0)


def _render(parquet: Path, stream_label: str, out_png: Path) -> None:
    if not parquet.exists():
        return
    full = pd.read_parquet(parquet, columns=None).columns
    flag_cols = [c for c, _, _ in FLAG_PRIORITY if c in full]
    cols = ["mh_hybrid_pred", "alpha_m_hybrid_pred", "release_tier"] + flag_cols
    df = pd.read_parquet(parquet, columns=cols)
    t1 = df[df["release_tier"] == 1]
    t2 = df[df["release_tier"] == 2]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    _scatter_panel(axes[0], t1, t2,
                    title=f"{stream_label}: Tier 1 vs Tier 2 (scatter overlay)",
                    color_t2="#ff7f0e")
    _flag_panel(axes[1], t1, t2, available_flags=set(flag_cols), n_total=len(df))
    fig.suptitle(
        f"{stream_label} chemistry plane on the hybrid surface, scatter overlay.\n"
        "Tier 1 is the per-star science slice. Tier 2 stars are demoted because at least one caveat fired; "
        "panel 2 shows which caveat dominates each chemistry-plane region.",
        fontsize=10,
    )
    fig.tight_layout()
    save_fig(fig, out_png, tight=False)


def main() -> None:
    apply_style()
    _render(PARQUET_S3, "Stream 3", OUT / "flag_coloured_chemistry.png")
    _render(PARQUET_S2, "Stream 2", OUT / "flag_coloured_chemistry_stream2.png")


if __name__ == "__main__":
    main()
