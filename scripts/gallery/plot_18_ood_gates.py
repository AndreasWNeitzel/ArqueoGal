"""Stage 18: OOD gates across the Stream-2 and Stream-3 hybrid releases.

Layout 2 × 3:
- (0,0) Per-gate flag firing rates — overlay S2 vs S3, with the v5 active
  vs diagnostic-only distinction made explicit.
- (0,1) Joint-OOD rate vs G overlay.
- (0,2) Mahalanobis score histogram overlay.
- (1,0) S3 release-tier sky map.
- (1,1) S2 release-tier sky map.
- (1,2) Tier composition bars (S2 + S3 side-by-side).

Schema-v5 (2026-04-26) note: only ``ood_joint_flag`` (T3) and
``mode_ambiguous_flag`` on α/M (T2) actually gate the tier. The other flag
columns are still emitted as diagnostics (and shown here so the reader can
verify the v5 ablation conclusion that they fire trivially or shift no
trustworthy-catalog RMSE).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
from _common import (apply_style, save_fig, radec_to_galactic_mollweide,
                     style_galactic_mollweide, sample_index)

OUT = REPO / "reports/gallery/18_ood_gates"
S2 = REPO / "release/D-Cat-b/hybrid_pipeline_run_stream2/predictions_with_features.parquet"
S3 = REPO / "release/D-Cat-b/hybrid_pipeline_run/predictions_with_features.parquet"

FLAG_COLS = ("ood_joint_flag", "ood_aux_mahalanobis_flag", "latent_support_flag",
             "regime_b_flag", "mode_ambiguous_flag", "ood_disagreement_flag",
             "aux_missing_any", "dist_prior_dominated", "kin_ood_flag")

# v5 (2026-04-26) gating distinction. ood_joint_flag → T3; mode_ambiguous_flag
# → T2 on α/M only; kin_ood_flag → T2 on aux-assisted ([α/M], [Mg/H]) only.
# Everything else is diagnostic-only — the column is still emitted but no
# longer feeds release_tier. See release.py docstrings for justification.
ACTIVE_GATES = {"ood_joint_flag", "mode_ambiguous_flag", "kin_ood_flag"}


def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    full = pd.read_parquet(path).iloc[:0].columns
    cols = list(set(FLAG_COLS) & set(full)) + ["ra_deg", "dec_deg", "g_mag",
                                                 "ood_mahalanobis_score", "release_tier"]
    cols = [c for c in cols if c in full]
    return pd.read_parquet(path, columns=cols)


def main() -> None:
    apply_style()
    s2 = _load(S2); s3 = _load(S3)
    if s3 is None and s2 is None:
        return

    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.30)

    # (0,0) Flag firing rates side-by-side. Bold tick labels mark the v5
    # active gates; the rest are emitted as diagnostics only.
    ax = fig.add_subplot(gs[0, 0])
    flags = [c for c in FLAG_COLS if (s3 is not None and c in s3.columns)
              or (s2 is not None and c in s2.columns)]
    rates3 = [100.0 * float(s3[f].mean()) if s3 is not None and f in s3.columns else 0.0
              for f in flags]
    rates2 = [100.0 * float(s2[f].mean()) if s2 is not None and f in s2.columns else 0.0
              for f in flags]
    y = np.arange(len(flags))
    h = 0.4
    ax.barh(y - h/2, rates3, h, color="#d62728", label="Stream 3")
    ax.barh(y + h/2, rates2, h, color="#9467bd", label="Stream 2")
    ax.set_yticks(y)
    label_strs = [f"{f} (active)" if f in ACTIVE_GATES else f"{f} (diag.)"
                  for f in flags]
    ax.set_yticklabels(label_strs, fontsize=7)
    for tick_label, flag in zip(ax.get_yticklabels(), flags):
        if flag in ACTIVE_GATES:
            tick_label.set_fontweight("bold")
            tick_label.set_color("#1f77b4")
        else:
            tick_label.set_color("0.45")
    ax.set_xlabel("% flagged")
    ax.set_title("Per-gate firing rates (S2 vs S3)\n"
                  "bold blue = active in v5 tier gating; grey = diagnostic only",
                  fontsize=9)
    ax.legend(fontsize=7, loc="lower right", frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="0.4")
    for i, (v3, v2) in enumerate(zip(rates3, rates2)):
        ax.text(v3 + 0.3, i - h/2, f"{v3:.1f}", va="center", fontsize=6)
        ax.text(v2 + 0.3, i + h/2, f"{v2:.1f}", va="center", fontsize=6)

    # (0,1) Joint-OOD rate vs G overlay
    ax = fig.add_subplot(gs[0, 1])
    bins = np.linspace(6, 18, 25)
    centres = 0.5 * (bins[1:] + bins[:-1])
    for df, name, color in [(s3, "Stream 3", "#d62728"),
                              (s2, "Stream 2", "#9467bd")]:
        if df is None or "ood_joint_flag" not in df.columns:
            continue
        g = df["g_mag"].to_numpy()
        flag = df["ood_joint_flag"].to_numpy().astype(bool)
        num, _ = np.histogram(g[flag], bins=bins)
        den, _ = np.histogram(g[np.isfinite(g)], bins=bins)
        rate = 100.0 * num / np.maximum(den, 1)
        ax.plot(centres, rate, "o-", color=color, lw=1.2, ms=4, label=name)
    ax.set_xlabel("G (mag)")
    ax.set_ylabel("% ood_joint")
    ax.set_title("Joint-OOD rate vs G")
    ax.legend(fontsize=7, loc="upper left", frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="0.4")

    # (0,2) Mahalanobis score overlay
    ax = fig.add_subplot(gs[0, 2])
    upper = 20.0
    for df, name, color in [(s3, "Stream 3", "#d62728"),
                              (s2, "Stream 2", "#9467bd")]:
        if df is None or "ood_mahalanobis_score" not in df.columns:
            continue
        score = df["ood_mahalanobis_score"].dropna().to_numpy()
        bins_s = np.linspace(0, upper, 60)
        ax.hist(score, bins=bins_s, density=True, histtype="step", color=color,
                 lw=1.4, label=f"{name} (n={len(score):,})")
    ax.set_xlabel("Mahalanobis distance (108-D XP)")
    ax.set_ylabel("density")
    ax.set_yscale("log")
    ax.set_title("Mahalanobis score overlay")
    ax.legend(fontsize=7, loc="upper right", frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="0.4")

    # (1,0) S3 sky map by tier
    rng = np.random.default_rng(0)
    for col, df, name in [(0, s3, "Stream 3"), (1, s2, "Stream 2")]:
        ax = fig.add_subplot(gs[1, col], projection="mollweide")
        if df is None or "release_tier" not in df.columns:
            ax.set_title(f"{name}\n(no tier column)", fontsize=9)
            continue
        idx = sample_index(len(df), 80_000, rng)
        ra = df.ra_deg.iloc[idx].to_numpy(); dec = df.dec_deg.iloc[idx].to_numpy()
        x, y = radec_to_galactic_mollweide(ra, dec)
        tier = df.release_tier.iloc[idx].to_numpy()
        for t, color, label in [(1, "#2ca02c", "Tier 1"),
                                 (2, "#ff7f0e", "Tier 2"),
                                 (3, "#d62728", "Tier 3")]:
            m = tier == t
            if m.any():
                ax.scatter(x[m], y[m], s=0.4, alpha=0.4, color=color,
                            rasterized=True, label=f"{label} ({int(m.sum()):,})")
        style_galactic_mollweide(ax)
        ax.set_title(f"{name} release tier sky", fontsize=9)
        ax.legend(fontsize=6, loc="lower left", frameon=True, framealpha=0.95,
                  facecolor="white", edgecolor="0.4", markerscale=4)

    # (1,2) Tier bars side-by-side
    ax = fig.add_subplot(gs[1, 2])
    tiers = [1, 2, 3]
    n_streams = sum(1 for d in (s3, s2) if d is not None and "release_tier" in d.columns)
    width = 0.85 / max(n_streams, 1)
    x = np.arange(len(tiers))
    for i, (df, name, color) in enumerate([(s3, "Stream 3", "#d62728"),
                                              (s2, "Stream 2", "#9467bd")]):
        if df is None or "release_tier" not in df.columns:
            continue
        n = len(df)
        cnts = [int((df["release_tier"] == t).sum()) for t in tiers]
        offset = (i - (n_streams - 1) / 2) * width
        bars = ax.bar(x + offset, cnts, width=width * 0.95, color=color,
                       label=f"{name} (n={n:,})")
        for b, v in zip(bars, cnts):
            ax.text(b.get_x() + b.get_width() / 2, v,
                     f"{100 * v / n:.0f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([f"Tier {t}" for t in tiers])
    ax.set_ylabel("count")
    ax.set_title("Composite release_tier composition")
    ax.legend(fontsize=7, loc="upper right", frameon=True, framealpha=0.95,
              facecolor="white", edgecolor="0.4")

    fig.suptitle(
        "Stage 18 — OOD gates and tier distribution (S2 vs S3, schema v5).\n"
        "Active gates (bold blue): ood_joint_flag → Tier 3; mode_ambiguous_flag → Tier 2 "
        "on [α/M] only; kin_ood_flag → Tier 2 on aux-assisted elements. "
        "Other flags shown for diagnostics; they no longer feed release_tier "
        "(see release/test_ablations_2026-04-26/REPORT.md).",
        fontsize=9.5,
    )
    save_fig(fig, OUT / "ood_gates.png", tight=False)


if __name__ == "__main__":
    main()
