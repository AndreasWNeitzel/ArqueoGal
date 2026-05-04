"""Locate per-cell μ-bias in parameter space — #140/#135 escalation step.

Follows the joint-correction rejection: before committing to a β=0 retrain,
we need to know *where* the per-cell bias lives:

- If concentrated in known-pathological regions (Teff > 6000 K, |[M/H]| > 1,
  extremes of logg), the fix is a release-scope narrowing, not a retrain.
- If spread across clean thin-disk (-0.3 < [M/H] < 0.1, 4500 < Teff < 5200,
  2 < logg < 3) cells, the pathology is architectural/loss.

Outputs
-------
- ``reports/pipeline1/run_a/bias_location_diagnostic.json`` — full table.
- ``reports/pipeline1/run_a/bias_vs_ncell.png`` — small-sample-noise check.
- stdout: top-worst cells ranked, bin-class tabulation, and verdict.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from arqueogal.xp_abundances.main.data import FeatureLayout, LabelScaler, LabelTiers
from arqueogal.xp_abundances.main.model import default_pipeline1_layout
from arqueogal.xp_abundances.main.training import build_dataloaders, load_checkpoint
from arqueogal.xp_abundances.main.uncertainty import bin_by_cells

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_calibration import (  # noqa: E402
    _build_cfg_for_val_loader,
    _collect_member_preds,
    _moment_match,
    _reconstruct_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("bias_loc")

REPO = Path(__file__).resolve().parent.parent
ENSEMBLE = REPO / "models/main/xp_abundances/20260419_nogit_5ee6908_ensemble"
PARQUET = REPO / "data/processed/pipeline1_features_stream1.parquet"
OUT_DIR = REPO / "reports/pipeline1/run_a"


# Known-pathological / clean-core definitions. Pathology flags fire when any
# condition holds — the cell is "not clean-core".
# Clean-core: well-sampled thin-disk RGB, the regime XP→abundances
# literature (Andrae+2023, Rix+2022) consistently handles well.
PATH_TEFF_HOT = 6000.0  # Teff > this → XP noise-dominated (research_brief §6.1)
PATH_TEFF_COOL = 4000.0  # Teff < this → cool-giant molecular blanketing issues
PATH_LOGG_UPPER = 3.5  # logg > this → subgiant/dwarf contamination
PATH_LOGG_LOWER = 1.0  # logg < this → extended AGB / tip-RGB
PATH_MH_POOR = -1.0  # [M/H] < this → halo regime, sparse training
PATH_MH_RICH = 0.4  # [M/H] > this → metal-rich tail, sparse


def _cell_center_and_flag(
    code: int,
    n_bins: list[int],
    edges: list[list[float]],
) -> tuple[tuple[float, float, float], list[str]]:
    """Decode cell index to (Teff, logg, [M/H]) bin center + pathology flags."""
    # Encoding in bin_by_cells: codes = codes * nb + idx, iterating j 0..2.
    idx: list[int] = []
    c = code
    for nb in reversed(n_bins):
        idx.append(c % nb)
        c //= nb
    idx.reverse()  # now [i_teff, i_logg, i_mh]

    def bin_center(i: int, edges_j: list[float]) -> float:
        # edges_j has nb-1 inner edges; turn into nb+1 (incl. -inf, +inf) mids.
        # Use the two-sided inner midpoint where possible; for edges, use the
        # single inner edge as a one-sided proxy (outer bins are unbounded).
        if len(edges_j) == 0:
            return float("nan")
        lo = (
            edges_j[i - 1]
            if i > 0
            else edges_j[0] - (edges_j[-1] - edges_j[0]) / (len(edges_j) + 1)
        )
        hi = (
            edges_j[i]
            if i < len(edges_j)
            else edges_j[-1] + (edges_j[-1] - edges_j[0]) / (len(edges_j) + 1)
        )
        return 0.5 * (lo + hi)

    teff_c = bin_center(idx[0], edges[0])
    logg_c = bin_center(idx[1], edges[1])
    mh_c = bin_center(idx[2], edges[2])

    flags: list[str] = []
    if teff_c > PATH_TEFF_HOT:
        flags.append("teff_hot")
    if teff_c < PATH_TEFF_COOL:
        flags.append("teff_cool")
    if logg_c > PATH_LOGG_UPPER:
        flags.append("logg_subgiant")
    if logg_c < PATH_LOGG_LOWER:
        flags.append("logg_tipgiant")
    if mh_c < PATH_MH_POOR:
        flags.append("mh_poor")
    if mh_c > PATH_MH_RICH:
        flags.append("mh_rich")
    return (teff_c, logg_c, mh_c), flags


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    member_ckpts = sorted(ENSEMBLE.glob("member_seed*/xp_abundances_main_ensemble_seed*_best.pt"))
    _LOG.info("found %d members", len(member_ckpts))

    layout, tiers = FeatureLayout(), LabelTiers()
    first = load_checkpoint(member_ckpts[0], map_location="cpu")
    first_cfg = json.loads(first["config_yaml"])
    split_seed = int(first_cfg.get("split_seed", 0))
    cfg = _build_cfg_for_val_loader(
        parquet=PARQUET,
        pretrained_ckpt=Path(first_cfg["pretrained_encoder_ckpt"]),
        batch_size=1024,
        seed=split_seed,
    )
    _, val_loader, _, _, _ = build_dataloaders(cfg, layout, tiers, seed=split_seed)
    block_layout = default_pipeline1_layout()

    per_mu: list[np.ndarray] = []
    per_L: list[np.ndarray] = []
    y_human: np.ndarray | None = None
    scaler_block = scaler_human = None
    for ck in member_ckpts:
        blob = load_checkpoint(ck, map_location=device)
        model, adapter = _reconstruct_model(blob, layout, device)
        mu, L, y_h = _collect_member_preds(model, adapter, val_loader, device)
        per_mu.append(mu)
        per_L.append(L)
        if y_human is None:
            y_human = y_h
            scaler_human = LabelScaler(
                mean=np.asarray(blob["label_scaler_mean"], dtype=np.float32),
                scale=np.asarray(blob["label_scaler_scale"], dtype=np.float32),
                label_names=tuple(blob["label_names"]),
            )
            scaler_block = scaler_human.reorder_to(block_layout.label_order_block)
    mu_bar, L_bar = _moment_match(np.stack(per_mu), np.stack(per_L))
    mu_bar = scaler_block.inverse_mean(mu_bar)
    L_bar = scaler_block.inverse_L(L_bar)
    y_block = scaler_human.inverse_mean(y_human)[:, block_layout.human_to_block_perm.cpu().numpy()]
    y_clean = np.where(np.isfinite(y_block), y_block, mu_bar)

    sigma = np.sqrt(np.einsum("bij,bij->bi", L_bar, L_bar)).clip(1e-8, None)
    z = (y_clean - mu_bar) / sigma

    cell_ids, cell_def = bin_by_cells(y_clean[:, :3], n_bins=(4, 4, 4))
    names = list(block_layout.label_order_block)

    rows: list[dict] = []
    for c in np.unique(cell_ids):
        mask = cell_ids == c
        n_c = int(mask.sum())
        if n_c < 8:
            continue
        z_cell = z[mask]  # (n_c, 21)
        bias = z_cell.mean(axis=0)  # (21,)
        (teff_c, logg_c, mh_c), flags = _cell_center_and_flag(
            int(c),
            cell_def["n_bins"],
            cell_def["edges_per_col"],
        )
        # Also record the worst label name.
        j_worst = int(np.argmax(np.abs(bias)))
        rows.append(
            {
                "cell_id": int(c),
                "n": n_c,
                "teff_center": float(teff_c),
                "logg_center": float(logg_c),
                "mh_center": float(mh_c),
                "max_abs_bias": float(np.abs(bias).max()),
                "mean_abs_bias": float(np.abs(bias).mean()),
                "worst_label": names[j_worst],
                "worst_bias": float(bias[j_worst]),
                "bias_per_label": [float(b) for b in bias],
                "is_pathological": bool(len(flags) > 0),
                "pathology_flags": flags,
            }
        )

    rows.sort(key=lambda r: -r["max_abs_bias"])

    # Top-N table.
    print("\n# Top-15 worst cells by max |mean(z|cell)|")
    hdr = f"{'cell':>4} {'n':>6} {'Teff':>6} {'logg':>5} {'[M/H]':>6} {'max|b|':>7} {'worst':<16} {'bias':>7} {'flags':<30}"
    print(hdr)
    for r in rows[:15]:
        print(
            f"{r['cell_id']:>4} {r['n']:>6} {r['teff_center']:>6.0f} "
            f"{r['logg_center']:>5.2f} {r['mh_center']:>6.2f} "
            f"{r['max_abs_bias']:>7.2f} {r['worst_label']:<16} "
            f"{r['worst_bias']:>+7.2f} {','.join(r['pathology_flags']):<30}",
        )

    # Class tabulation: pathological vs clean-core.
    clean = [r for r in rows if not r["is_pathological"]]
    patho = [r for r in rows if r["is_pathological"]]
    print("\n# Tabulation by cell class")
    print(
        f"{'class':<18} {'n_cells':>8} {'stars':>8} {'median max|b|':>14} {'p90 max|b|':>12} {'max max|b|':>12}"
    )

    def _stats(cs):
        if not cs:
            return (0, 0, 0.0, 0.0, 0.0)
        m = np.array([r["max_abs_bias"] for r in cs])
        return (
            len(cs),
            sum(r["n"] for r in cs),
            float(np.median(m)),
            float(np.quantile(m, 0.90)),
            float(m.max()),
        )

    for label, subset in [("clean-core", clean), ("pathological", patho)]:
        n_c, s, med, p90, mx = _stats(subset)
        print(f"{label:<18} {n_c:>8} {s:>8} {med:>14.2f} {p90:>12.2f} {mx:>12.2f}")

    # Which cells fail the halt threshold (max |bias| > 1 σ)?
    bad = [r for r in rows if r["max_abs_bias"] > 1.0]
    bad_clean = [r for r in bad if not r["is_pathological"]]
    bad_patho = [r for r in bad if r["is_pathological"]]
    print(
        f"\n# Cells with max |bias| > 1σ: {len(bad)} total "
        f"({len(bad_clean)} clean-core, {len(bad_patho)} pathological)"
    )

    # Per-flag breakdown.
    from collections import Counter

    flag_ct: Counter[str] = Counter()
    for r in rows:
        for f in r["pathology_flags"]:
            flag_ct[f] += 1
    flag_bad: Counter[str] = Counter()
    for r in bad:
        for f in r["pathology_flags"]:
            flag_bad[f] += 1
    print("\n# Pathology-flag frequency (all / >1σ-bad)")
    for f in sorted(flag_ct.keys()):
        print(f"  {f:<16} {flag_ct[f]:>3} / {flag_bad[f]:>3}")

    # Small-sample-noise check: bias vs cell n.
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.array([r["n"] for r in rows])
    ys = np.array([r["max_abs_bias"] for r in rows])
    colors = ["tab:red" if r["is_pathological"] else "tab:blue" for r in rows]
    ax.scatter(xs, ys, c=colors, s=30, alpha=0.75, edgecolor="k", linewidth=0.3)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, label="bias = 1σ")
    # Expected noise floor for sample mean of z with var 1: std = 1/√n.
    n_grid = np.linspace(max(xs.min(), 10), xs.max(), 100)
    ax.plot(
        n_grid,
        3.0 / np.sqrt(n_grid),
        color="gray",
        linestyle=":",
        label="3/√n (3σ sampling noise at var(z)=1)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("n stars in cell")
    ax.set_ylabel("max |mean(z | cell)| across labels")
    ax.set_title("Per-cell bias vs cell star count")
    ax.legend(
        handles=[
            plt.Line2D([], [], marker="o", color="tab:blue", linestyle="", label="clean-core cell"),
            plt.Line2D(
                [], [], marker="o", color="tab:red", linestyle="", label="pathological cell"
            ),
            plt.Line2D([], [], color="gray", linestyle=":", label="3/√n sampling-noise line"),
            plt.Line2D([], [], color="k", linestyle="--", label="bias = 1σ"),
        ]
    )
    fig.tight_layout()
    fig_path = OUT_DIR / "bias_vs_ncell.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    _LOG.info("wrote %s", fig_path)

    # Correlation stat for the report.
    if (ys > 0).any() and (xs > 0).any():
        rho = float(np.corrcoef(np.log10(xs), ys)[0, 1])
    else:
        rho = float("nan")
    print(f"\n# Pearson r(log10(n_cell), max|bias|) = {rho:+.3f}")
    print(
        "  (strongly negative ⇒ small cells are the noise source; "
        "near zero ⇒ bias is real systematic behavior)"
    )

    # Summary verdict hint.
    if len(bad_clean) == 0:
        verdict = "all >1σ bias cells are pathological — candidate for release-scope narrow"
    elif len(bad_clean) >= len(bad_patho):
        verdict = "clean-core cells have >1σ bias — architectural/loss issue, retrain β=0"
    else:
        verdict = "mixed — retrain β=0 AND consider scope narrow"
    print(f"\n# VERDICT SIGNAL: {verdict}")

    out_path = OUT_DIR / "bias_location_diagnostic.json"
    with out_path.open("w") as f:
        json.dump(
            {
                "cell_definition": cell_def,
                "rows": rows,
                "pathology_flag_counts_all": dict(flag_ct),
                "pathology_flag_counts_bad_1sigma": dict(flag_bad),
                "n_bad_1sigma_total": len(bad),
                "n_bad_1sigma_clean_core": len(bad_clean),
                "n_bad_1sigma_pathological": len(bad_patho),
                "pearson_log10_n_vs_max_bias": rho,
                "verdict_hint": verdict,
            },
            f,
            indent=2,
            default=str,
        )
    _LOG.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
