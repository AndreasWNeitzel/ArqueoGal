"""Stage 14: Pipeline-1 inference on Stream 3.

Outputs:
  - reports/gallery/14_pipeline1_inference/stream3_pred_hrd.png
  - reports/gallery/14_pipeline1_inference/stream3_pred_chemistry.png
  - reports/gallery/14_pipeline1_inference/stream3_pred_sky.png
  - reports/gallery/14_pipeline1_inference/stream3_ood_rate_sky.png
  - reports/gallery/14_pipeline1_inference/stream3_regime_b_sky.png
  - reports/gallery/14_pipeline1_inference/stream3_aux_missingness.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DATA_PROCESSED,
    GALLERY,
    apply_style,
    radec_to_galactic_mollweide,
    sample_index,
    save_fig,
    style_galactic_mollweide,
)

OUT = GALLERY / "14_pipeline1_inference"

LABELS = ["teff", "logg", "mh", "alpha_m", "mg_h"]
LABEL_TEX = {"teff": r"$T_{\rm eff}$  [K]", "logg": r"$\log g$",
             "mh": r"$[{\rm M}/{\rm H}]$",
             "alpha_m": r"$[\alpha/{\rm M}]$",
             "mg_h": r"$[{\rm Mg}/{\rm H}]$"}


def _load_full(version: str = "v11") -> "pd.DataFrame":
    import pandas as pd  # noqa: F401
    pred_path = DATA_PROCESSED / f"pipeline1_predictions_stream3_{version}.parquet"
    feat = DATA_PROCESSED / "pipeline1_features_stream3.parquet"
    preds = pq.read_table(pred_path).to_pandas()
    meta_cols = ["source_id", "ra_deg", "dec_deg", "g_mag",
                 "ir_missing_flag", "extinction_missing_flag"]
    meta_schema = pq.read_schema(feat)
    meta_have = [c for c in meta_cols if c in {f.name for f in meta_schema}]
    meta = pq.read_table(feat, columns=meta_have).to_pandas()
    return preds.merge(meta, on="source_id", how="left")


def stream3_pred_hrd() -> None:
    df = _load_full()
    t = df["teff_pred"].to_numpy()
    g = df["logg_pred"].to_numpy()
    m = np.isfinite(t) & np.isfinite(g)
    fig, ax = plt.subplots(figsize=(9, 7))
    hb = ax.hexbin(t[m], g[m], gridsize=80, cmap="magma", bins="log", mincnt=1)
    plt.colorbar(hb, ax=ax, shrink=0.85, pad=0.02, label="log N")
    ax.set_xlim(5600, 3800); ax.set_ylim(3.8, 0.5)
    ax.set_xlabel(r"$T_{\rm eff}^{\rm pred}$ [K]"); ax.set_ylabel(r"$\log g^{\rm pred}$")
    ax.set_title(f"Stream 3 v1.1 — predicted Kiel diagram  (n={int(m.sum()):,})",
                 fontsize=11, fontweight="semibold")
    save_fig(fig, OUT / "stream3_pred_hrd.png")


def stream3_pred_chemistry() -> None:
    """Overlay v1 and v1.1 chemistry planes."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, ver in zip(axes.flat, ("", "_v11")):
        p = DATA_PROCESSED / f"pipeline1_predictions_stream3{ver}.parquet"
        if not p.exists():
            ax.set_visible(False); continue
        df = pq.read_table(p, columns=["mh_pred", "alpha_m_pred"]).to_pandas()
        mh = df["mh_pred"].to_numpy()
        am = df["alpha_m_pred"].to_numpy()
        m = np.isfinite(mh) & np.isfinite(am)
        hb = ax.hexbin(mh[m], am[m], gridsize=80, cmap="magma", bins="log",
                       extent=(-2.2, 0.6, -0.3, 0.55), mincnt=1)
        plt.colorbar(hb, ax=ax, shrink=0.85, pad=0.02, label="log N")
        ax.set_xlabel(r"$[{\rm M}/{\rm H}]^{\rm pred}$"); ax.set_ylabel(r"$[\alpha/{\rm M}]^{\rm pred}$")
        label = "v1" if ver == "" else "v1.1"
        ax.set_title(f"Stream 3 chemistry — Pipeline-1 {label}  (n={int(m.sum()):,})",
                     fontsize=11, fontweight="semibold")
        ax.set_xlim(-2.2, 0.6); ax.set_ylim(-0.3, 0.55)
    save_fig(fig, OUT / "stream3_pred_chemistry.png")


def stream3_pred_sky() -> None:
    df = _load_full()
    if "ra_deg" not in df.columns:
        return
    rng = np.random.default_rng(21)
    idx = sample_index(len(df), 120_000, rng)
    sub = df.iloc[idx]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), subplot_kw={"projection": "mollweide"})
    x, y = radec_to_galactic_mollweide(sub["ra_deg"].to_numpy(), sub["dec_deg"].to_numpy())
    for ax, lbl in zip(axes.flat, LABELS):
        vals = sub[f"{lbl}_pred"].to_numpy()
        lo, hi = np.nanpercentile(vals, [2, 98])
        sc = ax.scatter(x, y, c=np.clip(vals, lo, hi), cmap="magma",
                        s=1.0, alpha=0.7, rasterized=True, vmin=lo, vmax=hi)
        plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02, label=LABEL_TEX[lbl])
        ax.set_title(f"{LABEL_TEX[lbl]}  pred", fontsize=10)
        style_galactic_mollweide(ax)
    axes[1, 2].set_visible(False)
    fig.suptitle(f"Stream 3 predicted labels across the sky  —  Galactic coords "
                 f"(n={len(sub):,} / {len(df):,})",
                 fontsize=13, fontweight="bold", y=1.00)
    save_fig(fig, OUT / "stream3_pred_sky.png")


def stream3_ood_rate_sky() -> None:
    df = _load_full()
    if "ra_deg" not in df.columns or "ood_joint_flag" not in df.columns:
        return
    # Bin on Galactic (l, b) grid
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    c = SkyCoord(ra=df["ra_deg"].to_numpy() * u.deg,
                 dec=df["dec_deg"].to_numpy() * u.deg, frame="icrs").galactic
    lg = c.l.degree
    bg = c.b.degree
    lg = np.where(lg > 180, lg - 360, lg)
    flag = df["ood_joint_flag"].to_numpy().astype(bool)
    bins_l = np.linspace(-180, 180, 90)
    bins_b = np.linspace(-90, 90, 45)
    total, _, _ = np.histogram2d(lg, bg, bins=[bins_l, bins_b])
    fl, _, _ = np.histogram2d(lg[flag], bg[flag], bins=[bins_l, bins_b])
    rate = np.where(total >= 5, fl / np.maximum(total, 1), np.nan)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection="mollweide")
    L, B = np.meshgrid(0.5 * (bins_l[:-1] + bins_l[1:]),
                       0.5 * (bins_b[:-1] + bins_b[1:]), indexing="ij")
    sc = ax.pcolormesh(-np.deg2rad(L), np.deg2rad(B), rate,
                       cmap="plasma", vmin=0, vmax=np.nanpercentile(rate, 98),
                       shading="auto")
    plt.colorbar(sc, ax=ax, shrink=0.65, pad=0.02, label="ood-joint rate")
    ax.set_title(f"Stream 3 joint-OOD rate per sky pixel  (n pix $\\geq$ 5)",
                 fontsize=11, fontweight="semibold")
    style_galactic_mollweide(ax)
    save_fig(fig, OUT / "stream3_ood_rate_sky.png")


def stream3_regime_b_sky() -> None:
    df = _load_full()
    if "ra_deg" not in df.columns or "regime_b_flag" not in df.columns:
        return
    flag = df["regime_b_flag"].to_numpy().astype(bool)
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": "mollweide"})
    # plot only flagged stars
    rng = np.random.default_rng(27)
    idx_flag = np.where(flag)[0]
    if len(idx_flag) > 60_000:
        idx_flag = rng.choice(idx_flag, 60_000, replace=False)
    x, y = radec_to_galactic_mollweide(df["ra_deg"].to_numpy()[idx_flag],
                                        df["dec_deg"].to_numpy()[idx_flag])
    # background: all
    idx_all = sample_index(len(df), 80_000, rng)
    xa, ya = radec_to_galactic_mollweide(df["ra_deg"].to_numpy()[idx_all],
                                          df["dec_deg"].to_numpy()[idx_all])
    ax.scatter(xa, ya, s=0.4, alpha=0.2, color="#bbb", rasterized=True)
    ax.scatter(x, y, s=0.8, alpha=0.7, color="#d62728", rasterized=True,
               label=f"regime B  n={int(flag.sum()):,} ({100*flag.mean():.1f}%)")
    ax.set_title("Stream 3 Regime-B flag footprint  (warm upper-RGB ∩ $|b|<5°$)",
                 fontsize=11, fontweight="semibold")
    style_galactic_mollweide(ax)
    ax.legend(loc="lower right")
    save_fig(fig, OUT / "stream3_regime_b_sky.png")


def stream3_aux_missingness() -> None:
    df = _load_full()
    flags = [c for c in ("ir_missing_flag", "parallax_missing_flag",
                         "extinction_missing_flag", "aux_missing_any")
             if c in df.columns]
    if not flags:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    rates = {f: 100.0 * df[f].astype(float).mean() for f in flags}
    names = list(rates.keys())
    vals = [rates[n] for n in names]
    bars = ax.bar(names, vals, color=["#1f77b4", "#ff7f0e", "#9467bd", "#d62728"][:len(names)],
                  edgecolor="#333", alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.2f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("rate  [%]")
    ax.set_title(f"Stream 3 aux-feature missingness rates  (n={len(df):,})",
                 fontsize=11, fontweight="semibold")
    save_fig(fig, OUT / "stream3_aux_missingness.png")


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    stream3_pred_hrd()
    stream3_pred_chemistry()
    stream3_pred_sky()
    stream3_ood_rate_sky()
    stream3_regime_b_sky()
    stream3_aux_missingness()


if __name__ == "__main__":
    main()
