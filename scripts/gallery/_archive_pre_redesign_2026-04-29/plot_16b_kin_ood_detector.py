"""Stage 16b: kinematic-OOD detector diagnostic (Phase B Stage 1).

What the deploy did: ``scripts/build_kin_ood_flag.py`` fit a Gaussian
envelope on the Stream-3 disc-cut kinematic subset (|v_z|<80 km/s,
v_T>100 km/s, n=245,153) in 3-D Galactocentric (v_R, v_T, v_z) space, and
flagged stars whose Mahalanobis distance exceeds the 99th-percentile
threshold. The flag is injected into the release pipeline by
``release_pipeline.attach_kin_ood_flag``; aux-assisted elements ([α/M],
[Mg/H]) demote to Tier 2 when this flag fires.

What we plot, 2×2:
- (0,0) **Toomre with kin_ood overlay**: v_T vs sqrt(v_R²+v_z²) hexbin of
  Tier-1+2 stars in greyscale, flagged stars over-plotted as red dots.
- (0,1) **Mahalanobis score distribution**: log-scaled histogram with the
  threshold (4.00) marked.
- (1,0) **Sky map of flagged stars**: where on the sky kin_ood fires.
- (1,1) **Per-element tier impact**: bars showing how many stars per element
  are demoted to Tier 2 specifically because of kin_ood (i.e., they would
  have been Tier 1 without the flag).
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
from _common import (
    apply_style,
    radec_to_galactic_mollweide,
    sample_index,
    save_fig,
    style_galactic_mollweide,
)

OUT = REPO / "reports/gallery/16b_kin_ood_detector"
KIN = REPO / "data/processed/pipeline2_kinematics_stream3_volume.parquet"
KIN_OOD = REPO / "data/processed/pipeline1_kin_ood_flag.parquet"
BUNDLE_JSON = REPO / "data/processed/pipeline1_kin_ood_bundle.json"
PARQUET = REPO / "release/D-Cat-b/hybrid_pipeline_run/predictions_with_features.parquet"

V_LSR = 230.0


def main() -> None:
    apply_style()
    if not KIN.exists() or not KIN_OOD.exists() or not BUNDLE_JSON.exists():
        return

    bundle = json.loads(BUNDLE_JSON.read_text())
    threshold = float(bundle["threshold"])
    n_train = int(bundle["n_training"])
    p_thr = float(bundle["p_threshold"])

    kin = pd.read_parquet(KIN, columns=["source_id", "v_R_kms", "v_T_kms", "v_z_kms"])
    flags = pd.read_parquet(KIN_OOD)
    df = kin.merge(flags, on="source_id", how="left")
    n_flagged = int(df.kin_ood_flag.sum())
    n_total = len(df)

    # Pull (ra, dec) from the predictions parquet for the sky map.
    rel = pd.read_parquet(PARQUET, columns=["source_id", "ra_deg", "dec_deg"])
    df = df.merge(rel, on="source_id", how="left")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (0,0) Toomre overlay
    ax = axes[0, 0]
    v_T = df["v_T_kms"].to_numpy()
    v_perp = np.sqrt(df["v_R_kms"].to_numpy() ** 2 + df["v_z_kms"].to_numpy() ** 2)
    ok = np.isfinite(v_T) & np.isfinite(v_perp)
    flag = df.kin_ood_flag.fillna(False).to_numpy().astype(bool)
    h = ax.hexbin(
        v_T[ok & ~flag],
        v_perp[ok & ~flag],
        gridsize=80,
        mincnt=10,
        cmap="Greys",
        bins="log",
        extent=[0, 400, 0, 200],
    )
    plt.colorbar(h, ax=ax, label="log10 N (in-distribution)")
    rng = np.random.default_rng(0)
    flagged_idx = np.flatnonzero(ok & flag)
    if len(flagged_idx) > 5000:
        flagged_idx = rng.choice(flagged_idx, 5000, replace=False)
    ax.scatter(
        v_T[flagged_idx],
        v_perp[flagged_idx],
        s=2.5,
        color="#d62728",
        alpha=0.55,
        rasterized=True,
        label=f"kin_ood = True ({n_flagged:,})",
    )
    th = np.linspace(0, 2 * np.pi, 400)
    for r_iso, color, lab in [
        (100, "#9467bd", "100 km/s from LSR"),
        (200, "#ff7f0e", "200 km/s (halo cut)"),
    ]:
        cx = V_LSR + r_iso * np.cos(th)
        cy = r_iso * np.sin(th)
        in_box = (cx >= 0) & (cx <= 400) & (cy >= 0) & (cy <= 200)
        if in_box.any():
            ax.plot(cx[in_box], cy[in_box], color=color, lw=0.9, ls="--", alpha=0.85, label=lab)
    ax.axvline(V_LSR, color="orange", lw=0.8, ls=":", label=f"$V_{{\\rm LSR}}$ = {V_LSR:.0f}")
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 200)
    ax.set_xlabel(r"$v_T$ (km/s)")
    ax.set_ylabel(r"$\sqrt{v_R^2+v_z^2}$ (km/s)")
    ax.set_title(f"Toomre with kin_ood overlay (n={int(ok.sum()):,} kinematic stars)")
    ax.legend(
        fontsize=7,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.4",
        markerscale=3,
    )

    # (0,1) Mahalanobis score histogram
    ax = axes[0, 1]
    score = df.kin_ood_score.dropna().to_numpy()
    bins = np.linspace(0, np.percentile(score, 99.9), 80)
    ax.hist(score, bins=bins, color="#1f77b4", alpha=0.75)
    ax.axvline(
        threshold,
        color="#d62728",
        lw=1.4,
        ls="--",
        label=f"threshold = {threshold:.2f}\n(p={p_thr:.2f}, n_train={n_train:,})",
    )
    ax.set_xlabel("Mahalanobis distance (3D velocity)")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.set_title(f"kin_ood Mahalanobis score (n={len(score):,})")
    ax.legend(
        fontsize=8,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.4",
    )

    # (1,0) Sky map of flagged stars
    ax = axes[1, 0]
    ax.set_axis_off()
    ax = fig.add_subplot(2, 2, 3, projection="mollweide")
    sub = df.dropna(subset=["ra_deg", "dec_deg"])
    f_sub = sub[sub.kin_ood_flag.astype(bool)]
    rng = np.random.default_rng(0)
    bg_idx = sample_index(len(sub), 80_000, rng)
    bg_x, bg_y = radec_to_galactic_mollweide(
        sub.ra_deg.iloc[bg_idx].to_numpy(), sub.dec_deg.iloc[bg_idx].to_numpy()
    )
    ax.scatter(
        bg_x,
        bg_y,
        s=0.4,
        alpha=0.10,
        color="0.5",
        rasterized=True,
        label=f"in-distribution ({len(sub) - len(f_sub):,})",
    )
    if len(f_sub):
        x, y = radec_to_galactic_mollweide(f_sub.ra_deg.to_numpy(), f_sub.dec_deg.to_numpy())
        ax.scatter(
            x,
            y,
            s=1.5,
            alpha=0.65,
            color="#d62728",
            rasterized=True,
            label=f"kin_ood = True ({len(f_sub):,})",
        )
    style_galactic_mollweide(ax)
    ax.set_title("Galactic sky map of kinematically-OOD stars", fontsize=10)
    ax.legend(
        fontsize=7,
        loc="lower left",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.4",
        markerscale=4,
    )

    # (1,1) Per-element tier impact
    ax = axes[1, 1]
    rel_full = pd.read_parquet(
        PARQUET,
        columns=["source_id", "kin_ood_flag"]
        + [f"release_tier__{e}" for e in ("teff", "logg", "mh", "alpha_m", "mg_h")],
    )
    elements = ("teff", "logg", "mh", "alpha_m", "mg_h")
    el_labels = {
        "teff": "Teff",
        "logg": "log g",
        "mh": "[M/H]",
        "alpha_m": "[α/M]",
        "mg_h": "[Mg/H]",
    }
    n_kin = int(rel_full["kin_ood_flag"].astype(bool).sum())
    flagged = rel_full[rel_full["kin_ood_flag"].astype(bool)]
    counts = []
    for e in elements:
        n_t1 = int((flagged[f"release_tier__{e}"] == 1).sum())
        n_t2 = int((flagged[f"release_tier__{e}"] == 2).sum())
        n_t3 = int((flagged[f"release_tier__{e}"] == 3).sum())
        counts.append((n_t1, n_t2, n_t3))
    x = np.arange(len(elements))
    w = 0.25
    t1 = [c[0] for c in counts]
    t2 = [c[1] for c in counts]
    t3 = [c[2] for c in counts]
    ax.bar(x - w, t1, w, color="#2ca02c", label="Tier 1")
    ax.bar(x, t2, w, color="#ff7f0e", label="Tier 2")
    ax.bar(x + w, t3, w, color="#d62728", label="Tier 3")
    ax.set_xticks(x)
    ax.set_xticklabels([el_labels[e] for e in elements])
    ax.set_ylabel("count of kin_ood-flagged stars")
    ax.set_title(f"Per-element tier among kin_ood stars (n={n_kin:,})")
    ax.legend(
        fontsize=8,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.4",
    )
    # Annotate the spectrum-dominant vs aux-assisted asymmetry
    ax.text(
        0.02,
        0.97,
        "Spectrum-dominant (Teff, logg, [M/H]) NOT demoted by kin_ood;\n"
        "aux-assisted ([α/M], [Mg/H]) demoted to Tier 2 (population prior breaks down).",
        transform=ax.transAxes,
        fontsize=7,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.92, pad=2),
    )

    fig.suptitle(
        "kinematic-OOD detector (Phase B Stage 1, 2026-04-25): "
        f"{n_flagged:,} / {n_total:,} stars flagged ({100 * n_flagged / n_total:.2f}%)",
        fontsize=11,
    )
    fig.tight_layout()
    save_fig(fig, OUT / "kin_ood_detector.png", tight=False)


if __name__ == "__main__":
    main()
