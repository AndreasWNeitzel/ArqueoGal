"""Y40: animation, Kiel + chemistry plane + per-epoch RMSE.

Renders an MP4 (and a GIF fallback) showing the Stream-1 holdout cohort
evolving across the 200 cadence epochs of the v5 finetune chain:

  - left  panel: Kiel diagram (T_eff vs log g), per-epoch hexbin density.
  - middle panel: chemistry plane ([M/H] vs [alpha/M]), per-epoch hexbin.
  - right panel: per-epoch RMSE (pred minus APOGEE truth) for four
                 labels (T_eff, log g, [M/H], [alpha/M]) with two y axes,
                 left = Teff in K, right = log g / [M/H] / [alpha/M] in dex.
                 The "current" epoch is marked with a vertical line that
                 sweeps from frame 0 to the last frame.

Slide-friendly 18:6 layout.  Cadence parquets:
``data/processed/cadence_predictions/20260503_1d71682_2ae55d3_finetune_5label/``.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
sys.path.insert(0, str(REPO / "src"))

from _presentation import OKABE_ITO, PALETTE, apply_style, headline  # noqa: E402

from arqueogal.xp_abundances.main.data import stratified_split_ids  # noqa: E402

CADENCE_DIR = REPO / "data/processed/cadence_predictions/20260503_1d71682_2ae55d3_finetune_5label"
FEAT_S1 = REPO / "data/processed/pipeline1_features_stream1_kiel.parquet"
OUT_DIR = REPO / "reports/gallery/Y_presentation"

KIEL_TEFF = (3500, 6500)
KIEL_LOGG = (5.0, 0.0)   # inverted on plot
MH_LIM = (-1.6, 0.55)
AM_LIM = (-0.10, 0.45)
HEX_KIEL = 80
HEX_CHEM = 80

LABELS = [
    ("teff",    "teff_pred",    "teff_apogee",     OKABE_ITO[1], "left"),   # K, left axis
    ("logg",    "logg_pred",    "logg_apogee",     OKABE_ITO[2], "right"),  # dex, right axis
    ("mh",      "mh_pred",      "mh_apogee",       OKABE_ITO[3], "right"),
    ("alpha_m", "alpha_m_pred", "alpha_m_apogee",  OKABE_ITO[4], "right"),
]

_TITLE_KW = dict(fontsize=11, fontweight="normal", color=PALETTE["ink"], pad=6)


def _epoch_paths() -> list[Path]:
    paths = sorted(CADENCE_DIR.glob("epoch_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no cadence parquets at {CADENCE_DIR}")
    return paths


def _epoch_id(path: Path) -> int:
    m = re.match(r"epoch_(\d+)\.parquet$", path.name)
    if not m:
        raise ValueError(f"cannot parse epoch from {path.name}")
    return int(m.group(1))


def _load_truth() -> pd.DataFrame:
    cols = ["source_id", "teff_apogee", "logg_apogee", "mh_apogee",
            "alpha_m_apogee", "fe_h_apogee", "b_deg"]
    df = pd.read_parquet(FEAT_S1, columns=cols).drop_duplicates("source_id")
    return df


def _holdout_ids(truth: pd.DataFrame) -> np.ndarray:
    split = stratified_split_ids(truth, seed=0)
    return np.concatenate([split["val"], split["test"]])


def _rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    r = pred - truth
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(r * r)))


def _per_epoch_rmse(paths: list[Path], truth: pd.DataFrame,
                    holdout_ids: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n = len(paths)
    epochs = np.empty(n, dtype=np.int32)
    out = {key: np.full(n, np.nan, dtype=np.float64) for key, *_ in LABELS}
    truth_lookup = truth.set_index("source_id")
    for i, p in enumerate(paths):
        epochs[i] = _epoch_id(p)
        df = pd.read_parquet(p, columns=[
            "source_id", "teff_pred", "logg_pred", "mh_pred", "alpha_m_pred",
        ]).drop_duplicates("source_id")
        df = df.loc[df["source_id"].isin(holdout_ids)]
        if df.empty:
            continue
        joined = df.set_index("source_id").join(
            truth_lookup[["teff_apogee", "logg_apogee", "mh_apogee", "alpha_m_apogee"]],
            how="inner",
        )
        for key, pcol, tcol, *_ in LABELS:
            out[key][i] = _rmse(joined[pcol].to_numpy(), joined[tcol].to_numpy())
    return epochs, out


def _draw_kiel(ax, df: pd.DataFrame) -> None:
    x = df["teff_pred"].to_numpy()
    y = df["logg_pred"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    ax.cla()
    if ok.sum() > 0:
        ax.hexbin(
            x[ok], y[ok],
            gridsize=HEX_KIEL,
            extent=(KIEL_TEFF[0], KIEL_TEFF[1], 0.0, 5.0),
            mincnt=1, bins="log", cmap="viridis",
        )
    ax.set_xlim(KIEL_TEFF[1], KIEL_TEFF[0])
    ax.set_ylim(KIEL_LOGG[0], KIEL_LOGG[1])
    ax.set_xlabel(r"$T_{\rm eff,\,pred}$ (K)")
    ax.set_ylabel(r"$\log g_{\rm pred}$ (dex)")
    ax.set_title("Kiel diagram", **_TITLE_KW)
    ax.grid(True, alpha=0.20)


def _draw_chem(ax, df: pd.DataFrame) -> None:
    x = df["mh_pred"].to_numpy()
    y = df["alpha_m_pred"].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    ax.cla()
    if ok.sum() > 0:
        ax.hexbin(
            x[ok], y[ok],
            gridsize=HEX_CHEM,
            extent=(MH_LIM[0], MH_LIM[1], AM_LIM[0], AM_LIM[1]),
            mincnt=1, bins="log", cmap="viridis",
        )
    ax.set_xlim(MH_LIM)
    ax.set_ylim(AM_LIM)
    ax.set_xlabel("[M/H] pred (dex)")
    ax.set_ylabel(r"[$\alpha$/M] pred (dex)")
    ax.set_title("Chemistry plane", **_TITLE_KW)
    ax.grid(True, alpha=0.20)


def main() -> int:
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = _epoch_paths()
    truth = _load_truth()
    holdout = _holdout_ids(truth)

    print(f"[Y40] computing per-epoch RMSE on {len(paths)} epochs, holdout n={len(holdout):,}")
    epochs, rmse = _per_epoch_rmse(paths, truth, holdout)

    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, wspace=0.35,
                          left=0.05, right=0.97, top=0.78, bottom=0.13)
    ax_k = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_r_left = fig.add_subplot(gs[0, 2])
    ax_r_right = ax_r_left.twinx()

    # Static RMSE curves: drawn once, then we sweep a vertical line.
    rmse_lines: dict[str, plt.Line2D] = {}
    teff_color = LABELS[0][3]
    for key, _pcol, _tcol, color, side in LABELS:
        ax = ax_r_left if side == "left" else ax_r_right
        (ln,) = ax.plot(epochs, rmse[key], color=color, lw=1.8,
                        label=key.replace("_", " ").upper())
        rmse_lines[key] = ln

    ax_r_left.set_xlabel("epoch")
    ax_r_left.set_ylabel(r"RMSE($T_{\rm eff}$) (K)", color=teff_color)
    ax_r_left.tick_params(axis="y", colors=teff_color)
    ax_r_left.spines["left"].set_color(teff_color)
    ax_r_right.set_ylabel("RMSE (dex)", color=PALETTE["ink"])
    ax_r_left.set_xlim(epochs.min(), epochs.max())
    ax_r_left.set_title("Per-epoch RMSE on Stream-1 holdout", **_TITLE_KW)
    ax_r_left.grid(True, alpha=0.20)

    # Combine legends from both y-axes onto the left panel.
    handles_l, labels_l = ax_r_left.get_legend_handles_labels()
    handles_r, labels_r = ax_r_right.get_legend_handles_labels()
    ax_r_left.legend(
        handles_l + handles_r, labels_l + labels_r,
        loc="upper right", fontsize=9, frameon=False,
    )

    # Vertical sweep line on the RMSE panel; updated per frame.
    sweep = ax_r_left.axvline(epochs[0], color=PALETTE["ink"], lw=1.4,
                              alpha=0.85, zorder=10)

    headline(
        fig,
        "Stream-1 holdout: training-cadence Kiel + chemistry + RMSE evolution",
        f"v5 finetune chain, n_epochs = {len(paths)}, holdout n = {len(holdout):,}.",
        top=0.78,
    )

    # Cache holdout truth merged-by-source_id so each frame just re-reads
    # one cadence parquet and joins.
    truth_lookup = truth.set_index("source_id")
    holdout_set = set(holdout.tolist())

    def _frame(i: int):
        ep = epochs[i]
        df = pd.read_parquet(paths[i], columns=[
            "source_id", "teff_pred", "logg_pred", "mh_pred", "alpha_m_pred",
        ]).drop_duplicates("source_id")
        df = df.loc[df["source_id"].isin(holdout_set)]
        joined = df.set_index("source_id").join(truth_lookup, how="inner")
        _draw_kiel(ax_k, joined)
        _draw_chem(ax_c, joined)
        sweep.set_xdata([ep, ep])
        ax_k.set_title(f"Kiel diagram, epoch {ep:03d}", **_TITLE_KW)
        ax_c.set_title(f"Chemistry plane, epoch {ep:03d}", **_TITLE_KW)
        return ax_k, ax_c, sweep

    print(f"[Y40] rendering {len(paths)} frames")
    anim = animation.FuncAnimation(
        fig, _frame, frames=len(paths), interval=120, blit=False,
    )

    out_mp4 = OUT_DIR / "Y40_animation_kiel_chem_rmse.mp4"
    out_gif = OUT_DIR / "Y40_animation_kiel_chem_rmse.gif"

    # Try MP4 (ffmpeg) first; fall back to GIF (pillow) if ffmpeg missing.
    try:
        writer = animation.FFMpegWriter(fps=8, bitrate=2800)
        anim.save(out_mp4, writer=writer, dpi=140)
        print(f"[Y40] wrote {out_mp4.relative_to(REPO)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[Y40] ffmpeg path failed ({exc}); writing GIF instead")
        anim.save(out_gif, writer=animation.PillowWriter(fps=6), dpi=110)
        print(f"[Y40] wrote {out_gif.relative_to(REPO)}")

    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
