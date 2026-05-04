"""Y31: Multi-subplot evaluation of the 5-label cadence run.

Reads the per-epoch prediction parquets emitted by
``scripts/emit_cadence_predictions.py`` for a finetune run and renders
on a single figure:

  Row 1 (5 panels): truth-vs-pred hexbin per label at the BEST epoch
                    (lowest val_loss in the cadence stream), RMSE / bias
                    annotated.
  Row 2 (1 panel) : per-epoch RMSE per label (line plot, normalised to the
                    label's epoch-0 RMSE so all curves share a y-range).
  Row 3 (1 panel) : per-epoch [α/M] vs [M/H] chemistry plane at epoch 0
                    side-by-side with the same plot at the best epoch
                    (visual check that the disc bimodality emerges over
                    training).

Usage:
  uv run python scripts/gallery/Y31_cadence_convergence.py \\
      --run-dir data/processed/cadence_predictions/<RUN_ID>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))

from _presentation import PALETTE, apply_style, headline, save  # noqa: E402

LABEL_SPECS = (
    {"key": "teff", "name": r"$T_{\rm eff}$", "unit": "K",
     "extent": (3800, 5800), "rmse_unit": "K"},
    {"key": "logg", "name": r"$\log g$", "unit": "dex",
     "extent": (0.5, 3.7), "rmse_unit": "dex"},
    {"key": "mh", "name": "[M/H]", "unit": "dex",
     "extent": (-2.2, 0.6), "rmse_unit": "dex"},
    {"key": "alpha_m", "name": r"[$\alpha$/M]", "unit": "dex",
     "extent": (-0.10, 0.45), "rmse_unit": "dex"},
    {"key": "mg_h", "name": "[Mg/H]", "unit": "dex",
     "extent": (-2.0, 0.6), "rmse_unit": "dex"},
)
FEAT_S1 = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"


def _load_truth() -> pd.DataFrame:
    return pd.read_parquet(
        FEAT_S1,
        columns=["source_id", "teff_apogee", "logg_apogee", "mh_apogee",
                 "alpha_m_apogee", "mg_h_apogee"],
    ).drop_duplicates("source_id").set_index("source_id")


def _per_epoch_rmse(run_dir: Path, truth: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in sorted(run_dir.glob("epoch_*.parquet")):
        epoch = int(p.stem.split("_")[1])
        df = pd.read_parquet(p).set_index("source_id").join(truth, how="inner")
        row = {"epoch": epoch}
        for spec in LABEL_SPECS:
            k = spec["key"]
            d = (df[f"{k}_pred"] - df[f"{k}_apogee"]).dropna().to_numpy()
            row[f"rmse_{k}"] = float(np.sqrt(np.mean(d ** 2)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)


def _truth_vs_pred(ax, df, spec):
    truth = df[f"{spec['key']}_apogee"].to_numpy()
    pred = df[f"{spec['key']}_pred"].to_numpy()
    ok = np.isfinite(truth) & np.isfinite(pred)
    truth, pred = truth[ok], pred[ok]
    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    bias = float(np.mean(pred - truth))
    lo, hi = spec["extent"]
    hb = ax.hexbin(truth, pred, gridsize=55, extent=(lo, hi, lo, hi),
                   mincnt=1, bins="log", cmap="viridis")
    ax.plot([lo, hi], [lo, hi], color=PALETTE["accent"], lw=1.8, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"truth {spec['name']} ({spec['unit']})")
    ax.set_ylabel(f"pred {spec['name']} ({spec['unit']})")
    ax.set_title(spec["name"], color=PALETTE["navy"])
    ax.text(
        0.04, 0.96,
        f"RMSE = {rmse:.3g} {spec['rmse_unit']}\nbias = {bias:+.3g}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=10, fontweight="bold", color=PALETTE["ink"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=PALETTE["mist"]),
    )


def _chemistry(ax, df, *, title):
    mh = df["mh_pred"].to_numpy()
    am = df["alpha_m_pred"].to_numpy()
    ok = np.isfinite(mh) & np.isfinite(am)
    ax.hexbin(mh[ok], am[ok], gridsize=70,
              extent=(-1.6, 0.55, -0.05, 0.42),
              mincnt=1, bins="log", cmap="viridis")
    ax.axhline(0.15, color=PALETTE["accent"], ls="--", lw=1.4)
    ax.set_xlim(-1.6, 0.55)
    ax.set_ylim(-0.05, 0.42)
    ax.set_xlabel("[M/H] pred  (dex)")
    ax.set_ylabel(r"[$\alpha$/M] pred  (dex)")
    ax.set_title(title, color=PALETTE["navy"])


def _rmse_curves(ax, rmse_df):
    epochs = rmse_df["epoch"].to_numpy()
    for spec in LABEL_SPECS:
        k = spec["key"]
        rmse = rmse_df[f"rmse_{k}"].to_numpy()
        rmse_norm = rmse / rmse[0]
        ax.plot(epochs, rmse_norm, "o-", lw=2.2, ms=6,
                label=f"{spec['name']} (ep0 = {rmse[0]:.3g} {spec['rmse_unit']})")
    ax.axhline(1.0, color=PALETTE["mist"], lw=0.8, ls=":")
    ax.set_xlabel("epoch")
    ax.set_ylabel("RMSE / RMSE(epoch 0)")
    ax.set_title("RMSE convergence per label  (normalised to epoch 0)",
                 color=PALETTE["navy"])
    ax.legend(loc="upper right", fontsize=9, ncol=1)
    ax.set_xlim(epochs.min() - 0.5, epochs.max() + 0.5)


def _training_best_epoch(run_dir: Path) -> int | None:
    """Read the training-selected best epoch from the run's _best.pt.

    Falls back to None if the .pt is missing or doesn't carry the metric.
    """
    import torch
    cadence_root = run_dir.parent.parent / "models/main/xp_abundances" \
        if "cadence_predictions" in str(run_dir) else None
    # The cadence parquet dir mirrors the run-id; find the corresponding
    # model run dir under models/main/xp_abundances/<run_id>/.
    candidates = sorted(REPO.glob(
        f"models/main/xp_abundances/{run_dir.name}/*_best.pt"
    ))
    if not candidates:
        return None
    blob = torch.load(candidates[0], map_location="cpu", weights_only=False)
    m = blob.get("training_metrics", {})
    be = m.get("best_epoch")
    return int(be) if be is not None and be >= 0 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Cadence-predictions run directory.")
    ap.add_argument("--best-epoch", type=int, default=None,
                    help="Override the best-epoch selector. By default, the "
                         "training-selected best (from the run's _best.pt) is "
                         "used; if unavailable, falls back to the heuristic "
                         "min-mean-normalised-RMSE.")
    args = ap.parse_args()

    apply_style()
    truth = _load_truth()
    rmse_df = _per_epoch_rmse(args.run_dir, truth)
    n_epochs = int(rmse_df["epoch"].max()) + 1

    # Best-epoch selection: training-selected (from _best.pt) wins; else
    # heuristic; else explicit override.
    if args.best_epoch is not None:
        best_ep = int(args.best_epoch)
        best_source = "user override"
    else:
        training_best = _training_best_epoch(args.run_dir)
        if training_best is not None and training_best in rmse_df["epoch"].values:
            best_ep = training_best
            best_source = "training-selected (val-loss min)"
        else:
            norm = rmse_df.copy()
            for spec in LABEL_SPECS:
                col = f"rmse_{spec['key']}"
                norm[col] = norm[col] / norm[col].iloc[0]
            norm["mean_norm"] = norm[[f"rmse_{s['key']}" for s in LABEL_SPECS]].mean(axis=1)
            best_ep = int(norm.loc[norm["mean_norm"].idxmin(), "epoch"])
            best_source = "heuristic min-mean-norm-RMSE"

    df_ep0 = pd.read_parquet(args.run_dir / "epoch_0000.parquet").set_index(
        "source_id").join(truth, how="inner")
    df_best = pd.read_parquet(
        args.run_dir / f"epoch_{best_ep:04d}.parquet"
    ).set_index("source_id").join(truth, how="inner")
    n_eval = int(len(df_best))

    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 5, hspace=0.45, wspace=0.30,
                          height_ratios=[1.0, 0.85, 0.85])

    # Row 1: truth-vs-pred per label at best epoch.
    for j, spec in enumerate(LABEL_SPECS):
        _truth_vs_pred(fig.add_subplot(gs[0, j]), df_best, spec)

    # Row 2: RMSE-convergence line plot (spans all 5 columns).
    _rmse_curves(fig.add_subplot(gs[1, :]), rmse_df)

    # Row 3: chemistry plane epoch 0 vs best (each spans 2.5 columns).
    _chemistry(fig.add_subplot(gs[2, 0:2]), df_ep0,
               title=f"Chemistry plane — epoch 0 (n={len(df_ep0):,})")
    _chemistry(fig.add_subplot(gs[2, 3:5]), df_best,
               title=f"Chemistry plane — epoch {best_ep} (n={n_eval:,})")

    headline(
        fig,
        "5-label finetune — per-epoch convergence + truth-vs-pred at the best epoch",
        f"Stream-1 val + test, n = {n_eval:,}.  "
        f"{n_epochs} epochs.  Best epoch = {best_ep}  ({best_source}).",
        top=0.91,
    )
    save(fig, "Y31_cadence_convergence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
