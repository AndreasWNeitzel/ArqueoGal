"""Run the Pipeline-1 pre-training sanity battery and emit the audit report.

Gate order:

1–3. Hard-fail checks (halt training gate if any fail): XP-feature NaN
   invariant, Tier-1 atmospheric label completeness (``{Teff, log g, [M/H]}``),
   parameter bounds (DR19-calibrated ranges, not textbook-generic).
4. Per-element NaN rate diagnostic (SOFT, report-only) — baseline rates for
   Tier-2 + Tier-3 [X/H] labels plus ``fe_h_apogee``. Freezes the DR19
   behaviour so a future recalibration surfaces as a numeric delta.
5. Distribution plots — Kiel diagram + Tinsley-Wallerstein diagram —
   operator eyeballs vs Mészáros+2025. No programmatic pass/fail.
6-7. Soft-fail checks: z-score validity, dedup idempotency.
8. Continuity embedding — UMAP on the 43-D XP subspace coloured by
   ``teff_apogee``. Catches data-plumbing bugs (wrong column order,
   accidental shuffling) that the other checks miss. Visual-only pass/fail.

Outputs:

- ``reports/sanity_battery/pretraining_audit.md`` — consolidated report.
- ``reports/sanity_battery/figures/kiel.png``
- ``reports/sanity_battery/figures/tinsley_wallerstein.png``
- ``reports/sanity_battery/figures/umap_continuity.png``
- ``reports/sanity_battery/battery_results.json`` — raw CheckResult details
  (machine-consumable for follow-up tracking).

Run: ``PYTHONPATH=src python scripts/run_pretraining_sanity.py``
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arqueogal.xp_abundances.main.data import FeatureLayout, LabelTiers
from arqueogal.xp_abundances.main.sanity import (
    BatteryVerdict,
    CheckResult,
    run_battery,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_pretraining_sanity")

# The expected dedup output row count — audited on the 2026-04-18 re-emit.
# Waterfall: 354,890 (Stream 1 post-cut) → 324,054 (XP inner-join) →
# 292,948 (post-dedup on source_id). Update when upstream cuts change.
EXPECTED_DEDUP_ROWS_OUT = 292_948


def _plot_kiel(df: pd.DataFrame, out_path: Path) -> None:
    """Kiel diagram (Teff vs logg) with standard astrophysics axis convention."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    teff = df["teff_apogee"].to_numpy()
    logg = df["logg_apogee"].to_numpy()
    mask = np.isfinite(teff) & np.isfinite(logg)
    h = ax.hexbin(
        teff[mask], logg[mask], gridsize=100, bins="log",
        cmap="viridis", mincnt=1,
    )
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel(r"$T_\mathrm{eff,\,APOGEE}$ / K")
    ax.set_ylabel(r"$\log g_\mathrm{APOGEE}$")
    ax.set_title(f"Kiel diagram — Pipeline-1 training pool ({mask.sum():,} stars)")
    cbar = fig.colorbar(h, ax=ax)
    cbar.set_label("log$_{10}$ N")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_tinsley_wallerstein(df: pd.DataFrame, out_path: Path) -> None:
    """[α/M] vs [Fe/H] Tinsley-Wallerstein diagram (α-bimodality signature)."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    feh = df["fe_h_apogee"].to_numpy()
    am = df["alpha_m_apogee"].to_numpy()
    mask = np.isfinite(feh) & np.isfinite(am)
    h = ax.hexbin(
        feh[mask], am[mask], gridsize=100, bins="log",
        cmap="viridis", mincnt=1,
    )
    ax.set_xlabel(r"$[\mathrm{Fe/H}]_\mathrm{APOGEE}$")
    ax.set_ylabel(r"$[\alpha/\mathrm{M}]_\mathrm{APOGEE}$")
    ax.set_title(f"Tinsley-Wallerstein — Pipeline-1 training pool ({mask.sum():,} stars)")
    cbar = fig.colorbar(h, ax=ax)
    cbar.set_label("log$_{10}$ N")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _umap_embedding_43d(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute a 43-D XP UMAP embedding on the normal population.

    Feature vector per star: ``bp_coef_norm_1..19`` (19) + ``rp_coef_norm_1..22``
    (22) + ``bp_c0_z`` + ``rp_c0_z`` = 43 dims. Subset = rows where all 43
    features are finite (the normal population). cuML UMAP on GPU; fallback
    to umap-learn if cuML fails.
    """
    trunc = FeatureLayout.truncated_43d()
    cols = (
        list(trunc.bp_coef_cols) + list(trunc.rp_coef_cols)
        + list(trunc.xp_scalar_cols)
    )
    X = np.column_stack([df[c].to_numpy(np.float32) for c in cols])
    finite = np.isfinite(X).all(axis=1)
    X = X[finite]
    teff = df.loc[finite, "teff_apogee"].to_numpy(np.float32)
    logger.info("UMAP input: %d rows × %d cols (of %d total)",
                X.shape[0], X.shape[1], len(df))

    try:
        from cuml.manifold import UMAP as cuUMAP
        logger.info("using cuml.UMAP on GPU")
        reducer = cuUMAP(
            n_neighbors=25, min_dist=0.1, n_components=2,
            random_state=0,
        )
        emb = reducer.fit_transform(X)
        emb = np.asarray(emb)
    except Exception as e:  # noqa: BLE001 — fallback path
        logger.warning("cuML UMAP failed (%s); falling back to umap-learn CPU", e)
        import umap
        reducer = umap.UMAP(
            n_neighbors=25, min_dist=0.1, n_components=2,
            random_state=0, n_jobs=-1,
        )
        emb = reducer.fit_transform(X)

    return emb, teff, int(finite.sum())


def _plot_umap_continuity(emb: np.ndarray, teff: np.ndarray, n_used: int, out_path: Path) -> None:
    """Scatter the 43-D UMAP embedding coloured by Teff_apogee."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    finite_teff = np.isfinite(teff)
    sc = ax.scatter(
        emb[finite_teff, 0], emb[finite_teff, 1],
        c=teff[finite_teff], cmap="plasma",
        s=1.0, alpha=0.35, rasterized=True,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(
        f"43-D XP UMAP continuity check "
        f"(n={n_used:,}, colour = Teff_apogee)"
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$T_\mathrm{eff,\,APOGEE}$ / K")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _render_markdown(
    verdict: BatteryVerdict,
    figures: dict[str, Path],
    report_path: Path,
    parquet_path: Path,
    n_rows: int,
    n_cols: int,
) -> None:
    """Write the audit report. Plots referenced by relative path."""
    lines: list[str] = []
    banner = {
        "PASS": "PASS — proceed to Pipeline 1 training",
        "SOFT-FAIL": "SOFT-FAIL — review flagged checks before training",
        "HARD-FAIL": "HARD-FAIL — training gate blocked, investigate upstream",
    }[verdict.overall]
    lines.append(f"# Pipeline-1 Pre-training Sanity Battery")
    lines.append("")
    lines.append(f"**Verdict: {banner}**")
    lines.append("")
    lines.append(f"- Input: `{parquet_path}` ({n_rows:,} rows × {n_cols} cols)")
    n_hard = sum(1 for r in verdict.results if r.level == "HARD")
    n_soft = sum(1 for r in verdict.results if r.level == "SOFT")
    lines.append(
        f"- Checks: {len(verdict.results)} ({n_hard} hard-fail, {n_soft} soft-fail)"
    )
    lines.append(f"- Any hard-fail: **{verdict.any_hard_fail}**")
    lines.append(f"- Any soft-fail: **{verdict.any_soft_fail}**")
    lines.append("")
    lines.append("## Per-check results")
    lines.append("")
    lines.append("| # | Check | Level | Result | Summary |")
    lines.append("|---|---|---|---|---|")
    for i, r in enumerate(verdict.results, start=1):
        mark = "PASS" if r.passed else "FAIL"
        lines.append(
            f"| {i} | `{r.name}` | {r.level} | **{mark}** | {r.summary} |"
        )
    lines.append("")

    # Details subsection per check
    for i, r in enumerate(verdict.results, start=1):
        lines.append(f"### {i}. `{r.name}` — {r.level} — {'PASS' if r.passed else 'FAIL'}")
        lines.append("")
        lines.append(f"*{r.summary}*")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(r.details, indent=2, default=str))
        lines.append("```")
        lines.append("")

    # Distribution plots + continuity embedding
    lines.append("## Check 4 — Distribution plots (operator eyeball)")
    lines.append("")
    lines.append(
        "Compare the shapes below against the published Mészáros+2025 "
        "ASPCAP-giant distributions. Shape match matters more than "
        "statistical p-values; our cut imposes known differences vs the "
        "full ASPCAP pool."
    )
    lines.append("")
    lines.append("### Kiel diagram")
    lines.append("")
    lines.append(f"![Kiel diagram]({figures['kiel'].name})")
    lines.append("")
    lines.append("### [α/M] — [Fe/H]")
    lines.append("")
    lines.append(f"![Tinsley-Wallerstein]({figures['tinsley_wallerstein'].name})")
    lines.append("")

    lines.append("## Continuity embedding — UMAP on 43-D XP subspace")
    lines.append("")
    lines.append(
        "Feature vector: `bp_coef_norm_1..19` + `rp_coef_norm_1..22` + "
        "`bp_c0_z` + `rp_c0_z` (43 dims). Normal-population subset. "
        "Expected: a clean smooth Teff gradient across the embedding, no "
        "isolated clusters, no random scatter. A broken gradient indicates "
        "a data-plumbing bug (column order, accidental shuffling, or "
        "miscomputed z-score) that the other checks missed."
    )
    lines.append("")
    lines.append(f"![UMAP continuity]({figures['umap_continuity'].name})")
    lines.append("")

    lines.append("## Why these specific bounds and checks")
    lines.append("")
    lines.append(
        "The checks below are deliberately calibrated to APOGEE DR19 realism, "
        "not textbook-generic finiteness expectations. Keeping them DR19-aware "
        "is the gating criterion for whether the battery catches a real "
        "future drift versus false-alarming on genuine ASPCAP behaviour. "
        "If any of this rationale changes, update `sanity.py` and this "
        "rationale together."
    )
    lines.append("")
    lines.append(
        "- **Tier-1 completeness gates on `{teff_apogee, logg_apogee, "
        "mh_apogee}` only** — not `fe_h_apogee`. ASPCAP DR19 fits [M/H] "
        "globally over Fe-peak + α lines, then runs per-element [Fe/H] "
        "afterwards; the per-element fit legitimately fails on saturated "
        "Fe lines (metal-rich regime), insufficient SNR per individual line "
        "in the blue, or unresolved Fe blends in cool giants. A ~1.6% NaN "
        "rate on `fe_h_apogee` at flag_bad==0 is the DR19 baseline, not a "
        "pipeline bug. See `sanity.py::TIER1_ATMOSPHERIC` and "
        "research_brief §3.2. [Fe/H] is treated as a Tier-2 per-element "
        "label (NaN-masked in training)."
    )
    lines.append("")
    lines.append(
        "- **Parameter bounds match ASPCAP DR19 dynamic range**: "
        "`fe_h_apogee ∈ [-4, 1.1]` widens the canonical +1.0 upper bound "
        "to cover DR19's metal-rich tail (per-element σ ≈ 0.03 dex); "
        "`alpha_m_apogee ∈ [-0.8, 0.8]` widens the disk-dominated "
        "[-0.5, 0.7] envelope to include genuine halo/CEMP α-poor tails "
        "and the upper α-rich fits. Tighter bounds would false-alarm on "
        "real DR19 stars; looser bounds would not flag a future ASPCAP "
        "drift to an even wider dynamic range (e.g. a DR20 recalibration "
        "pushing α/M to [-1, 1] would correctly trip this check)."
    )
    lines.append("")
    lines.append(
        "- **RGB window [Teff 4000-5500 K, log g 1.0-3.5] is enforced as "
        "an explicit builder-time cut** (in "
        "`scripts/build_pipeline1_features_stream1.py`). At 2026-04-18 the "
        "emergent intersection of (ASPCAP flag_bad==0) ∩ (SNR>70) ∩ (Gaia "
        "XP available) already falls inside this box — the cut drops "
        "zero rows on current data. It is retained as a named cut so a "
        "future APOGEE DR20 rebuild with different SNR statistics or XP "
        "coverage surfaces any drift immediately as a nonzero drop count "
        "in provenance, not weeks into training audit."
    )
    lines.append("")
    lines.append(
        "- **Per-element NaN rates (check `per_element_nan_rates`) are "
        "report-only** — no threshold. The baseline rates captured above "
        "are the 2026-04-18 snapshot; future rebuilds should compare "
        "against these. A large delta on any element (e.g. Mg or Al "
        "rates doubling) is a signal to investigate upstream ASPCAP "
        "changes before training."
    )
    lines.append("")
    lines.append("## Bottom line")
    lines.append("")
    if verdict.overall == "PASS":
        lines.append(
            "All six gate checks pass. Pipeline 1 training is unblocked on "
            "the data-layer side; proceed to the TESS_ML port + first "
            "training run."
        )
    elif verdict.overall == "SOFT-FAIL":
        lines.append(
            "Hard-fail checks pass. One or more soft-fail checks flagged; "
            "file a follow-up task and continue training under observation."
        )
    else:
        lines.append(
            "Hard-fail. Training is BLOCKED. Investigate the failing "
            "check(s) above and re-run the re-emit pipeline before "
            "retrying the battery."
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parquet = repo / "data" / "processed" / "pipeline1_features_stream1.parquet"
    out_dir = repo / "reports" / "sanity_battery"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not parquet.exists():
        raise SystemExit(f"missing {parquet} — run the re-emit pipeline first")

    logger.info("loading %s", parquet)
    df = pd.read_parquet(parquet)
    logger.info("%d rows × %d cols", len(df), len(df.columns))

    # Battery (checks 1, 2, 3, 5, 6)
    tiers = LabelTiers()
    verdict = run_battery(
        df, expected_dedup_rows=EXPECTED_DEDUP_ROWS_OUT, tiers=tiers,
    )
    for r in verdict.results:
        mark = "PASS" if r.passed else "FAIL"
        logger.info("  [%s] %s %s: %s", mark, r.level, r.name, r.summary)
    logger.info("verdict: %s", verdict.overall)

    # Plots (check 4 — visual)
    logger.info("rendering Kiel diagram")
    kiel_path = fig_dir / "kiel.png"
    _plot_kiel(df, kiel_path)
    logger.info("rendering Tinsley-Wallerstein diagram")
    tw_path = fig_dir / "tinsley_wallerstein.png"
    _plot_tinsley_wallerstein(df, tw_path)

    # UMAP continuity (check 7 — visual)
    logger.info("computing 43-D UMAP continuity embedding")
    emb, teff, n_umap = _umap_embedding_43d(df)
    umap_path = fig_dir / "umap_continuity.png"
    _plot_umap_continuity(emb, teff, n_umap, umap_path)
    logger.info("UMAP plot written to %s", umap_path)

    # Markdown report
    report_path = out_dir / "pretraining_audit.md"
    figures = {
        "kiel": kiel_path, "tinsley_wallerstein": tw_path,
        "umap_continuity": umap_path,
    }
    # Make figure paths relative to report directory for the markdown link
    rel_figures = {k: Path("figures") / v.name for k, v in figures.items()}
    _render_markdown(
        verdict=verdict, figures=rel_figures, report_path=report_path,
        parquet_path=parquet.relative_to(repo), n_rows=len(df), n_cols=len(df.columns),
    )
    logger.info("wrote %s", report_path)

    # Machine-consumable results
    results_path = out_dir / "battery_results.json"
    payload = {
        "verdict": verdict.overall,
        "any_hard_fail": verdict.any_hard_fail,
        "any_soft_fail": verdict.any_soft_fail,
        "parquet": str(parquet.relative_to(repo)),
        "n_rows": len(df), "n_cols": len(df.columns),
        "checks": [asdict(r) for r in verdict.results],
    }
    results_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", results_path)

    if verdict.overall == "HARD-FAIL":
        raise SystemExit(f"HARD-FAIL — see {report_path}")


if __name__ == "__main__":
    main()
