"""Visualise per-cell gate ablation results.

Reads ``release/test_ablations_2026-04-26/ablations.json`` and produces:
1. ``per_gate_effect.png`` — for each gate, side-by-side bars of T1 fraction
   change and T1+2 RMSE change vs production baseline.
2. ``rmse_vs_tier1_fraction.png`` — Pareto scatter: each ablation as a point
   plotted by (Tier 1 fraction, Tier 1+2 RMSE), coloured by gate identity.
   Shows whether stricter gates buy you sharper predictions.
3. ``per_element_breakdown.png`` — 5-panel grid (one per element) with
   per-ablation Tier 1 RMSE bars.
4. ``kiel_per_ablation.png`` — Kiel diagrams of the Tier 1 stars under
   different ablations (baseline, no_mahalanobis, no_mode_ambiguous,
   minimal_gates) on the Stream-1 test holdout. Shows visually how the
   Swiss-cheese pattern resolves under each ablation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
sys.path.insert(0, str(REPO / "src"))
from _common import apply_style, save_fig

from arqueogal.xp_abundances.main.data import stratified_split_ids

OUT_DIR = REPO / "reports/test_ablations_2026-04-26"
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSON = REPO / "release/test_ablations_2026-04-26/ablations.json"
PREDS = REPO / "release/test_ablations_2026-04-26/predictions_stream1.parquet"
FEATURES = REPO / "data/processed/pipeline1_features_stream1.parquet"

ELEMENTS = ("teff", "logg", "mh", "alpha_m", "mg_h")
EL_LBL = {"teff": "Teff", "logg": "log g", "mh": "[M/H]", "alpha_m": "[α/M]", "mg_h": "[Mg/H]"}
EL_UNIT = {"teff": "K", "logg": "dex", "mh": "dex", "alpha_m": "dex", "mg_h": "dex"}

# Gates that have a meaningful test in the ablation set; ordering chosen to
# group similar gates.
GATE_ORDER = [
    "baseline_prod",
    "no_mahalanobis",
    "no_aux_missing",
    "no_mode_ambiguous",
    "no_regime_b",
    "no_kin_ood",
    "no_latent_support",
    "no_aux_mahalanobis",
    "no_dist_prior",
    "no_disagreement",
    "sigma_global_0p5x",
    "sigma_global_1x",
    "sigma_global_2x",
    "all_caveats_off",
    "all_ood_off",
    "minimal_gates",
    "prod_alpha_tightened",
    "recommended_no_alpha_tighten",
    "recommended",
]

# Configs that are highlighted (green) rather than blue in bar charts.
HIGHLIGHT = {"recommended", "recommended_no_alpha_tighten", "prod_alpha_tightened"}
HIGHLIGHT_COLOR = "#2ca02c"  # green
BASELINE_COLOR = "#d62728"  # red
DEFAULT_COLOR = "#1f77b4"  # blue


def _color_for(name: str) -> str:
    if name == "baseline_prod":
        return BASELINE_COLOR
    if name in HIGHLIGHT:
        return HIGHLIGHT_COLOR
    return DEFAULT_COLOR


def main() -> None:
    apply_style()
    if not JSON.exists():
        print(f"missing {JSON}; run scripts/test_ablations/run_per_cell_ablations.py first")
        return
    blob = json.loads(JSON.read_text())
    n_test = blob["n_test"]
    abl = {a["name"]: a for a in blob["ablations"]}
    base = abl["baseline_prod"]

    # ---- Figure 1: per-gate effect (Tier 1 fraction + Tier 1+2 RMSE shift) ----
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={"height_ratios": [1, 1]})

    names = [n for n in GATE_ORDER if n in abl and n != "baseline_prod"]
    n_gates = len(names)
    width = 0.16
    x = np.arange(n_gates)
    colors = {
        "teff": "#d62728",
        "logg": "#ff7f0e",
        "mh": "#1f77b4",
        "alpha_m": "#2ca02c",
        "mg_h": "#9467bd",
    }

    # Subplot 0: Tier 1 fraction shift
    ax = axes[0]
    for i, e in enumerate(ELEMENTS):
        offset = (i - 2) * width
        base_f = base["per_element"][e]["tier1"]["frac_of_test"]
        shifts = []
        for n in names:
            f = abl[n]["per_element"][e]["tier1"]["frac_of_test"]
            shifts.append(100.0 * (f - base_f))
        ax.bar(x + offset, shifts, width=width * 0.95, color=colors[e], label=EL_LBL[e])
    ax.axhline(0, color="k", lw=0.6, ls="-")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Tier 1 fraction shift vs baseline (pp)")
    ax.set_title(f"Per-gate Tier 1 fraction effect (test holdout n = {n_test:,})")
    ax.legend(
        fontsize=8,
        ncol=5,
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.4",
    )
    ax.grid(True, axis="y", alpha=0.3)

    # Subplot 1: Tier 1+2 RMSE relative shift (%)
    ax = axes[1]
    for i, e in enumerate(ELEMENTS):
        offset = (i - 2) * width
        base_r = base["per_element"][e]["tier12"]["rmse"]
        shifts = []
        for n in names:
            r = abl[n]["per_element"][e]["tier12"]["rmse"]
            shifts.append(100.0 * (r - base_r) / max(base_r, 1e-9))
        ax.bar(x + offset, shifts, width=width * 0.95, color=colors[e], label=EL_LBL[e])
    ax.axhline(0, color="k", lw=0.6, ls="-")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Tier 1+2 RMSE shift vs baseline (%)")
    ax.set_title("Per-gate Tier 1+2 (trustworthy catalog) RMSE effect")
    ax.legend(
        fontsize=8,
        ncol=5,
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.4",
    )
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Ablation study — effect of disabling each per-cell gate.\n"
        "Top panel: how many stars move out of Tier 1 (negative = fewer stars in T1; positive = more).  "
        "Bottom panel: how the trustworthy-catalog RMSE changes.\n"
        "A gate is empirically justified only if disabling it INFLATES the bottom panel — i.e., the gate "
        "is removing genuinely-bad stars, not just relabeling them.",
        fontsize=10,
    )
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "per_gate_effect.png", tight=False)

    # ---- Figure 2: RMSE-vs-Tier-1-fraction Pareto ----
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.4), sharey=False)
    for i, e in enumerate(ELEMENTS):
        ax = axes[i]
        xs, ys, labs, cs = [], [], [], []
        for n in GATE_ORDER:
            if n not in abl:
                continue
            f = abl[n]["per_element"][e]["tier1"]["frac_of_test"]
            r = abl[n]["per_element"][e]["tier1"]["rmse"]
            xs.append(100.0 * f)
            ys.append(r)
            labs.append(n)
            cs.append(_color_for(n))
        ax.scatter(xs, ys, s=60, c=cs, edgecolor="black", lw=0.4)
        # Annotate baseline + extreme + recommended points
        for xv, yv, name in zip(xs, ys, labs):
            if name in (
                "baseline_prod",
                "no_mahalanobis",
                "no_mode_ambiguous",
                "minimal_gates",
                "sigma_global_0p5x",
                "recommended",
                "recommended_no_alpha_tighten",
                "prod_alpha_tightened",
            ):
                ax.annotate(name, (xv, yv), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Tier 1 fraction (%)")
        ax.set_ylabel(f"Tier 1 RMSE ({EL_UNIT[e]})")
        ax.set_title(f"{EL_LBL[e]}", fontsize=10)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        "Pareto: Tier 1 RMSE vs Tier 1 fraction across all ablations.\n"
        "Down-and-right is best (high coverage + low error). Red = production baseline; green = recommended configs.\n"
        "Recommended pushes Teff/log g/[M/H]/[Mg/H] to the right (high coverage) at modest RMSE inflation; "
        "α/M moves left (lower coverage) at sharper RMSE.",
        fontsize=10,
    )
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "rmse_vs_tier1_fraction.png", tight=False)

    # ---- Figure 3: per-element RMSE bars ----
    fig, axes = plt.subplots(1, 5, figsize=(20, 5.8), sharey=False)
    for i, e in enumerate(ELEMENTS):
        ax = axes[i]
        rmse_vals = []
        labels = []
        col_list = []
        for n in GATE_ORDER:
            if n not in abl:
                continue
            r = abl[n]["per_element"][e]["tier1"]["rmse"]
            rmse_vals.append(r)
            labels.append(n)
            col_list.append(_color_for(n))
        y = np.arange(len(labels))
        ax.barh(y, rmse_vals, color=col_list)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(f"Tier 1 RMSE ({EL_UNIT[e]})")
        ax.set_title(EL_LBL[e], fontsize=10)
        ax.axvline(
            rmse_vals[labels.index("baseline_prod")] if "baseline_prod" in labels else 0,
            color="#d62728",
            lw=0.7,
            ls="--",
        )
        for j, v in enumerate(rmse_vals):
            ax.text(v, j, f" {v:.3g}", va="center", fontsize=6)
        ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle(
        "Tier 1 RMSE per ablation, by element. Red dashed = production baseline; green = recommended configs.",
        fontsize=10,
    )
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "per_element_breakdown.png", tight=False)

    # ---- Figure 5: focused recommendation comparison ----
    # 4×5 grid: rows = (Tier 1 fraction, Tier 1 RMSE, Tier 1+2 RMSE, σ-coverage at 1σ),
    # columns = elements. Each cell shows three bars for prod/recommended/minimal_gates.
    cmp_configs = [
        ("baseline_prod", BASELINE_COLOR),
        ("prod_alpha_tightened", "#ff9896"),  # light red — only the σ-tighten
        (
            "recommended_no_alpha_tighten",
            "#aec7e8",
        ),  # light blue — only the gate-set simplification
        ("recommended", HIGHLIGHT_COLOR),
        ("minimal_gates", "#7f7f7f"),  # grey — for reference, the lower bound
    ]
    cmp_names = [c[0] for c in cmp_configs]
    cmp_colors = {c[0]: c[1] for c in cmp_configs}
    metric_specs = [
        (
            "Tier 1 fraction (%)",
            lambda b, e: 100.0 * b["per_element"][e]["tier1"]["frac_of_test"],
            None,
        ),
        ("Tier 1 RMSE", lambda b, e: b["per_element"][e]["tier1"]["rmse"], EL_UNIT),
        ("Tier 1+2 RMSE", lambda b, e: b["per_element"][e]["tier12"]["rmse"], EL_UNIT),
        (
            "Tier 1 σ-coverage at 1σ (%)",
            lambda b, e: 100.0 * b["per_element"][e]["tier1"]["coverage_1sigma"],
            None,
        ),
    ]
    fig, axes = plt.subplots(
        len(metric_specs), len(ELEMENTS), figsize=(22, 4.6 * len(metric_specs)), sharey=False
    )
    for r, (mlabel, mfunc, unit_dict) in enumerate(metric_specs):
        for c, e in enumerate(ELEMENTS):
            ax = axes[r, c]
            vals = [mfunc(abl[n], e) for n in cmp_names]
            colors_row = [cmp_colors[n] for n in cmp_names]
            xpos = np.arange(len(cmp_names))
            bars = ax.bar(xpos, vals, color=colors_row, edgecolor="black", lw=0.4)
            ax.set_xticks(xpos)
            short_names = ["prod", "prod\n+α-tight", "rec.\n(no α-tight)", "REC.", "min."]
            ax.set_xticklabels(short_names, fontsize=8, rotation=0)
            unit = unit_dict[e] if unit_dict else "%"
            if r == 0:
                ax.set_title(f"{EL_LBL[e]}", fontsize=12, fontweight="bold", pad=8)
            ax.set_ylabel(f"{mlabel}\n[{unit}]" if r in (1, 2) else mlabel, fontsize=9)
            # Headroom so the on-bar text fits.
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax + 0.10 * (ymax - ymin))
            for b, v in zip(bars, vals):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v,
                    f"{v:.3g}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
            ax.grid(True, axis="y", alpha=0.3)
            # Light reference line at production
            ax.axhline(vals[0], color=BASELINE_COLOR, lw=0.5, ls=":", alpha=0.5)

    fig.suptitle(
        "Recommended configuration vs production — five-way comparison.\n"
        "prod = current production (47% T1, full per-cell stack);  "
        "prod+α-tight = isolates σ-tighten only;  rec. (no α-tight) = isolates gate-set simplification;  "
        "REC. = both;  min. = NaN-only Tier 3 (lower bound).\n"
        "Tier 1+2 RMSE row is the load-bearing metric — same trustworthy-catalog quality is the target. "
        "σ-coverage row tests whether predicted σ stays well-calibrated under the simplified gates.",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    save_fig(fig, OUT_DIR / "recommended_vs_production.png", tight=False)

    # ---- Figure 4: Kiel under different ablations ----
    print("loading predictions for Kiel panels...")
    # Only request the flag columns that actually exist in the predictions
    # parquet — release-pipeline-added flags (latent_support, ood_aux_mahalanobis,
    # dist_prior_dominated, kin_ood) are absent from raw inference output.
    import pyarrow.parquet as pq

    avail = set(pq.read_schema(PREDS).names)
    base_cols = ["source_id", "teff_pred", "logg_pred"]
    flag_cols = [
        c
        for c in [
            "ood_joint_flag",
            "latent_support_flag",
            "ood_aux_mahalanobis_flag",
            "regime_b_flag",
            "mode_ambiguous_flag",
            "ood_disagreement_flag",
            "aux_missing_any",
            "dist_prior_dominated",
        ]
        if c in avail
    ]
    df_pred = pd.read_parquet(PREDS, columns=base_cols + flag_cols)
    df_pred = df_pred.drop_duplicates(subset="source_id")

    # Compute test-split mask
    feat_for_split = pd.read_parquet(
        FEATURES, columns=["source_id", "teff_apogee", "fe_h_apogee", "b_deg"]
    )
    split_ids = stratified_split_ids(feat_for_split, seed=0, fracs=(0.70, 0.15, 0.15))
    test_ids = set(split_ids["test"])
    df_pred = df_pred[df_pred["source_id"].isin(test_ids)].reset_index(drop=True)

    def t1_mask(
        *, drop_mahal=False, drop_mode_ambig=False, drop_all_caveats=False, drop_all_ood=False
    ) -> np.ndarray:
        n = len(df_pred)
        ood_set = ["ood_joint_flag", "latent_support_flag", "ood_aux_mahalanobis_flag"]
        cav_set = [
            "regime_b_flag",
            "mode_ambiguous_flag",
            "ood_disagreement_flag",
            "aux_missing_any",
            "dist_prior_dominated",
        ]
        if drop_all_ood:
            ood_set = []
        elif drop_mahal:
            ood_set = [c for c in ood_set if c != "ood_joint_flag"]
        if drop_all_caveats:
            cav_set = []
        elif drop_mode_ambig:
            cav_set = [c for c in cav_set if c != "mode_ambiguous_flag"]
        ood = np.zeros(n, dtype=bool)
        for c in ood_set:
            if c in df_pred.columns:
                ood |= df_pred[c].fillna(False).to_numpy().astype(bool)
        cav = np.zeros(n, dtype=bool)
        for c in cav_set:
            if c in df_pred.columns:
                cav |= df_pred[c].fillna(False).to_numpy().astype(bool)
        return (~ood) & (~cav)

    panels = [
        ("baseline (full per-cell stack)", t1_mask(), BASELINE_COLOR),
        ("no Mahalanobis OOD", t1_mask(drop_mahal=True), DEFAULT_COLOR),
        ("no mode-ambiguous", t1_mask(drop_mode_ambig=True), DEFAULT_COLOR),
        (
            "RECOMMENDED (Mahal. only,\nmode-amb. on α/M only)",
            t1_mask(drop_all_caveats=True),
            HIGHLIGHT_COLOR,
        ),
        (
            "all gates off (NaN-only T3)",
            t1_mask(drop_all_caveats=True, drop_all_ood=True),
            "#7f7f7f",
        ),
    ]
    teff = df_pred.teff_pred.to_numpy()
    logg = df_pred.logg_pred.to_numpy()

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.6))
    for ax, (title, mask, edge) in zip(axes, panels):
        n = int(mask.sum())
        if n < 100:
            ax.set_title(f"{title}\n(empty)", fontsize=9)
            continue
        h = ax.hexbin(
            teff[mask],
            logg[mask],
            gridsize=80,
            mincnt=2,
            cmap="viridis",
            bins="log",
            extent=[3500, 5500, 0.0, 4.0],
        )
        plt.colorbar(h, ax=ax, label="log10 N")
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xlim(5500, 3500)
        ax.set_ylim(4.0, 0.0)
        ax.set_xlabel("Teff (K)")
        ax.set_ylabel(r"$\log g$ (dex)")
        ax.set_title(
            f"{title}\nn = {n:,} ({100 * n / len(df_pred):.1f}% of test)", fontsize=9, color=edge
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(edge)
            spine.set_linewidth(1.4)
    fig.suptitle(
        f"Tier 1 Kiel diagrams under different gate configurations (Stream 1 test holdout, n = {len(df_pred):,}).\n"
        "Baseline (panel 1) shows the Swiss-cheese pattern from the per-cell stack. "
        "RECOMMENDED (panel 4) keeps Mahalanobis OOD but drops the relabeling caveats — "
        "matches the natural Kiel distribution while still excluding bona fide OOD stars.",
        fontsize=10,
    )
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "kiel_per_ablation.png", tight=False)

    print(f"\nwrote {len(list(OUT_DIR.glob('*.png')))} PNGs to {OUT_DIR}")


if __name__ == "__main__":
    main()
