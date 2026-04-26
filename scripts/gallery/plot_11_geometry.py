"""Stage 11: Galactocentric and Heliocentric geometry of Stream 3.

Four projections of the SAME stars in a 2×2 grid:

- **Row 1 — Galactocentric** (GC at origin, Sun at (-R_⊙, 0, ≈0)):
    - col 0: edge-on (R_gal vs z_gal)
    - col 1: face-on (X_gal vs Y_gal)
- **Row 2 — Heliocentric** (Sun at origin, GC at (+R_⊙, 0, 0)):
    - col 0: edge-on (r_helio vs z_helio)
    - col 1: face-on (x_helio vs y_helio)

A star at radius R from the relevant origin in the edge-on view sits on the
circle of radius R from that origin in the face-on view. Reference R-rings at
R = 5, 8, 12 kpc are drawn in both rows so the user can verify the
transformation.

All four panels share the same population (n stars), filtered to non-NaN
distance + sky position.
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

OUT = REPO / "reports/gallery/11_geometry"

R_SUN = 8.122  # kpc, astropy default galcen_distance
RINGS = ((5.0, "#9467bd", "$R$ = 5 kpc"),
          (R_SUN, "orange", f"$R_\\odot$"),
          (12.0, "#1f77b4", "$R$ = 12 kpc"))


def _hex_panel(ax, x, y, *, x_lo, x_hi, y_lo, y_hi, xlab, ylab, title, gridsize=110):
    m = np.isfinite(x) & np.isfinite(y)
    h = ax.hexbin(x[m], y[m], gridsize=gridsize, mincnt=1, cmap="viridis",
                   bins="log", extent=[x_lo, x_hi, y_lo, y_hi])
    plt.colorbar(h, ax=ax, label="log10 N")
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_title(f"{title} (n={int(m.sum()):,})")


def _draw_rings_edge(ax, R_max, *, x_axis_label):
    """For an R-vs-z subplot, draw vertical lines at the ring radii so the
    reader sees R = 5 / 8 / 12 dashed in subplot 1, matching the rings in 2."""
    for R_ref, color, label in RINGS:
        if R_ref <= R_max:
            ax.axvline(R_ref, color=color, lw=0.9, ls="--", alpha=0.85, label=label)


def _draw_rings_face(ax, x_lo, x_hi, y_lo, y_hi, *, origin=(0.0, 0.0)):
    """For an X-vs-Y subplot, draw circles of radius R at the ring radii."""
    th = np.linspace(0, 2 * np.pi, 400)
    for R_ref, color, label in RINGS:
        cx = R_ref * np.cos(th) + origin[0]
        cy = R_ref * np.sin(th) + origin[1]
        in_box = (cx >= x_lo) & (cx <= x_hi) & (cy >= y_lo) & (cy <= y_hi)
        if in_box.any():
            ax.plot(cx[in_box], cy[in_box], color=color, lw=1.0, ls="--",
                     alpha=0.85, label=label)


def main() -> None:
    apply_style()
    s3 = REPO / "data/processed/pipeline1_features_stream3.parquet"
    full = pd.read_parquet(s3).iloc[:0].columns

    if "ra_deg" in full and "r_med_photogeo" in full:
        df = pd.read_parquet(s3, columns=["r_med_photogeo", "ra_deg", "dec_deg"])
    else:
        return

    # Use astropy's official Galactocentric transform — gold standard.
    from astropy.coordinates import SkyCoord, Galactocentric
    import astropy.units as u
    sky = SkyCoord(ra=df["ra_deg"].to_numpy() * u.deg,
                    dec=df["dec_deg"].to_numpy() * u.deg,
                    distance=(df["r_med_photogeo"].to_numpy() / 1000.0) * u.kpc,
                    frame="icrs")

    # --- Galactocentric (GC at origin, Sun at -R_sun on x-axis) ---
    gc = sky.transform_to(Galactocentric())
    x_gc = gc.x.to(u.kpc).value
    y_gc = gc.y.to(u.kpc).value
    z_gc = gc.z.to(u.kpc).value
    R_gal = np.sqrt(x_gc ** 2 + y_gc ** 2)

    # --- Heliocentric Cartesian (Sun at origin) ---
    g = sky.galactic
    l_rad = g.l.rad
    b_rad = g.b.rad
    r_kpc = (df["r_med_photogeo"].to_numpy() / 1000.0)
    x_h = r_kpc * np.cos(b_rad) * np.cos(l_rad)
    y_h = r_kpc * np.cos(b_rad) * np.sin(l_rad)
    z_h = r_kpc * np.sin(b_rad)
    r_h = np.sqrt(x_h ** 2 + y_h ** 2)  # cylindrical heliocentric R

    m = (np.isfinite(R_gal) & np.isfinite(z_gc) & np.isfinite(x_gc) & np.isfinite(y_gc)
         & np.isfinite(r_h) & np.isfinite(x_h) & np.isfinite(y_h) & np.isfinite(z_h))

    # Common percentile-based extents — but EXTENDED so the relevant origin
    # marker (GC for Galactocentric, Sun for Heliocentric — but we mark BOTH
    # in BOTH frames) is visible in every face-on panel.
    R_gal_max = float(np.nanpercentile(R_gal[m], 99.5))
    z_abs = float(np.nanpercentile(np.abs(z_gc[m]), 99.5))
    x_gc_lo, x_gc_hi = np.nanpercentile(x_gc[m], [0.5, 99.5])
    y_gc_lo, y_gc_hi = np.nanpercentile(y_gc[m], [0.5, 99.5])
    pad_xgc = 0.05 * (x_gc_hi - x_gc_lo); pad_ygc = 0.05 * (y_gc_hi - y_gc_lo)
    x_gc_lo -= pad_xgc; x_gc_hi += pad_xgc
    y_gc_lo -= pad_ygc; y_gc_hi += pad_ygc
    # Force-extend Galactocentric face-on to include GC at (0,0) and Sun at
    # (-R_sun, 0). With 0.5 kpc breathing room.
    x_gc_lo = min(x_gc_lo, -R_SUN - 0.5)
    x_gc_hi = max(x_gc_hi, 0.5)
    y_gc_lo = min(y_gc_lo, -0.5)
    y_gc_hi = max(y_gc_hi, 0.5)

    r_h_max = float(np.nanpercentile(r_h[m], 99.5))
    z_h_abs = float(np.nanpercentile(np.abs(z_h[m]), 99.5))
    x_h_lo, x_h_hi = np.nanpercentile(x_h[m], [0.5, 99.5])
    y_h_lo, y_h_hi = np.nanpercentile(y_h[m], [0.5, 99.5])
    pad_xh = 0.05 * (x_h_hi - x_h_lo); pad_yh = 0.05 * (y_h_hi - y_h_lo)
    x_h_lo -= pad_xh; x_h_hi += pad_xh
    y_h_lo -= pad_yh; y_h_hi += pad_yh
    # Force-extend Heliocentric face-on to include Sun at (0,0) and GC at
    # (+R_sun, 0).
    x_h_lo = min(x_h_lo, -0.5)
    x_h_hi = max(x_h_hi, R_SUN + 0.5)
    y_h_lo = min(y_h_lo, -0.5)
    y_h_hi = max(y_h_hi, 0.5)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    # ONE master legend on subplot 1; subsequent panels share the same
    # ring-colour conventions so individual legends would be redundant.
    # frameon=True is explicit because apply_style() sets it False globally
    # (which would silently make framealpha + facecolor have no effect).
    LEGEND_KW = dict(fontsize=8.5, loc="upper right", frameon=True,
                      framealpha=0.97, facecolor="white", edgecolor="0.4",
                      borderpad=0.5, labelspacing=0.4)

    # ----- Row 1, col 0: Galactocentric R–z (edge-on) -----
    _hex_panel(axes[0, 0], R_gal[m], z_gc[m],
                x_lo=0, x_hi=R_gal_max, y_lo=-z_abs, y_hi=z_abs,
                xlab=r"$R_{\rm gal}$ (kpc)",
                ylab=r"$z_{\rm gal}$ (kpc)",
                title="Galactocentric edge-on: R–z")
    axes[0, 0].axhline(0, color="white", lw=0.4, ls="--", alpha=0.6)
    _draw_rings_edge(axes[0, 0], R_gal_max, x_axis_label="R_gal")
    # Mark GC as the R=0 origin and Sun at R_sun on this edge-on panel.
    axes[0, 0].scatter([0], [0], marker="x", s=130, color="red", lw=2.6,
                        zorder=10, label="GC (R=0)")
    axes[0, 0].legend(**LEGEND_KW)

    # ----- Row 1, col 1: Galactocentric X–Y (face-on) -----
    _hex_panel(axes[0, 1], x_gc[m], y_gc[m],
                x_lo=x_gc_lo, x_hi=x_gc_hi, y_lo=y_gc_lo, y_hi=y_gc_hi,
                xlab=r"$x_{\rm gc}$ (kpc) — GC at 0, Sun at $-R_\odot$",
                ylab=r"$y_{\rm gc}$ (kpc)",
                title="Galactocentric face-on: X–Y")
    _draw_rings_face(axes[0, 1], x_gc_lo, x_gc_hi, y_gc_lo, y_gc_hi,
                      origin=(0.0, 0.0))
    axes[0, 1].scatter([-R_SUN], [0], marker="*", s=220, color="orange",
                        edgecolor="k", lw=1.2, zorder=10,
                        label=f"Sun ($-R_\\odot$, 0)")
    axes[0, 1].scatter([0], [0], marker="x", s=140, color="red", lw=2.6,
                        zorder=10, label="GC (0, 0)")
    axes[0, 1].set_aspect("equal")
    axes[0, 1].grid(True, alpha=0.25, lw=0.4)
    # No legend on subsequent panels — convention shared with subplot (0,0).

    # ----- Row 2, col 0: Heliocentric r–z (edge-on) -----
    _hex_panel(axes[1, 0], r_h[m], z_h[m],
                x_lo=0, x_hi=max(r_h_max, R_SUN + 0.5),
                y_lo=-z_h_abs, y_hi=z_h_abs,
                xlab=r"$r_{\rm helio}\;=\;\sqrt{x_h^2+y_h^2}$ (kpc)",
                ylab=r"$z_{\rm helio}$ (kpc)",
                title="Heliocentric edge-on: r–z")
    axes[1, 0].axhline(0, color="white", lw=0.4, ls="--", alpha=0.6)
    for r_ref, color, label in [(1.0, "#9467bd", "r=1 kpc"),
                                  (2.0, "orange", "r=2 kpc (typical)"),
                                  (5.0, "#1f77b4", "r=5 kpc")]:
        if r_ref <= max(r_h_max, R_SUN + 0.5):
            axes[1, 0].axvline(r_ref, color=color, lw=0.9, ls="--", alpha=0.85,
                                label=label)
    # GC sits at heliocentric r = R_sun on the +x axis (so b=0, l=0); mark it.
    axes[1, 0].scatter([R_SUN], [0], marker="x", s=140, color="red", lw=2.6,
                        zorder=10, label=f"GC ($r=R_\\odot$, $z\\approx 0$)")
    axes[1, 0].scatter([0], [0], marker="*", s=220, color="orange",
                        edgecolor="k", lw=1.2, zorder=10, label="Sun (origin)")
    # No legend on subsequent panels — convention shared with subplot (0,0).

    # ----- Row 2, col 1: Heliocentric x_h-y_h (face-on) -----
    _hex_panel(axes[1, 1], x_h[m], y_h[m],
                x_lo=x_h_lo, x_hi=x_h_hi, y_lo=y_h_lo, y_hi=y_h_hi,
                xlab=r"$x_{\rm helio}$ (kpc) — toward $l=0°$ (GC)",
                ylab=r"$y_{\rm helio}$ (kpc) — toward $l=90°$",
                title="Heliocentric face-on: $x_h$–$y_h$")
    th = np.linspace(0, 2 * np.pi, 400)
    for r_ref, color, label in [(1.0, "#9467bd", "r=1 kpc"),
                                  (2.0, "orange", "r=2 kpc"),
                                  (5.0, "#1f77b4", "r=5 kpc")]:
        cx = r_ref * np.cos(th); cy = r_ref * np.sin(th)
        in_box = (cx >= x_h_lo) & (cx <= x_h_hi) & (cy >= y_h_lo) & (cy <= y_h_hi)
        if in_box.any():
            axes[1, 1].plot(cx[in_box], cy[in_box], color=color, lw=1.0, ls="--",
                              alpha=0.85, label=label)
    axes[1, 1].scatter([0], [0], marker="*", s=220, color="orange",
                        edgecolor="k", lw=1.2, zorder=10, label="Sun (0, 0)")
    axes[1, 1].scatter([R_SUN], [0], marker="x", s=140, color="red", lw=2.6,
                        zorder=10, label=f"GC ($R_\\odot$, 0)")
    axes[1, 1].set_aspect("equal")
    axes[1, 1].grid(True, alpha=0.25, lw=0.4)
    # No legend on subsequent panels — convention shared with subplot (0,0).

    # Overlay Stream 2 on ALL FOUR projections so the reader sees TESS giants
    # transformed into Galactocentric + Heliocentric frames consistently with
    # Stream 3 (hex). No legend — Stream 2 is purple, Stream 3 is the hex
    # density; the suptitle states the convention.
    s2_path = REPO / "data/processed/pipeline1_features_stream2.parquet"
    if s2_path.exists():
        s2_full = pd.read_parquet(s2_path).iloc[:0].columns
        if "ra_deg" in s2_full and "r_med_photogeo" in s2_full:
            s2 = pd.read_parquet(s2_path,
                                  columns=["r_med_photogeo", "ra_deg", "dec_deg"])
            sky2 = SkyCoord(ra=s2["ra_deg"].to_numpy() * u.deg,
                             dec=s2["dec_deg"].to_numpy() * u.deg,
                             distance=(s2["r_med_photogeo"].to_numpy() / 1000.0) * u.kpc,
                             frame="icrs")
            gc2 = sky2.transform_to(Galactocentric())
            x_gc2 = gc2.x.to(u.kpc).value
            y_gc2 = gc2.y.to(u.kpc).value
            z_gc2 = gc2.z.to(u.kpc).value
            R_gal2 = np.sqrt(x_gc2 ** 2 + y_gc2 ** 2)
            g2 = sky2.galactic
            r_kpc2 = (s2["r_med_photogeo"].to_numpy() / 1000.0)
            x_h2 = r_kpc2 * np.cos(g2.b.rad) * np.cos(g2.l.rad)
            y_h2 = r_kpc2 * np.cos(g2.b.rad) * np.sin(g2.l.rad)
            z_h2 = r_kpc2 * np.sin(g2.b.rad)
            r_h2 = np.sqrt(x_h2 ** 2 + y_h2 ** 2)
            m2 = (np.isfinite(R_gal2) & np.isfinite(z_gc2) & np.isfinite(x_gc2)
                  & np.isfinite(y_gc2) & np.isfinite(r_h2) & np.isfinite(x_h2)
                  & np.isfinite(y_h2) & np.isfinite(z_h2))
            rng = np.random.default_rng(0)
            n_show = min(15000, int(m2.sum()))
            idx2 = rng.choice(np.flatnonzero(m2), n_show, replace=False)
            scatter_kw = dict(s=1.2, color="#9467bd", alpha=0.45,
                               rasterized=True, zorder=8)
            axes[0, 0].scatter(R_gal2[idx2], z_gc2[idx2], **scatter_kw)
            axes[0, 1].scatter(x_gc2[idx2], y_gc2[idx2], **scatter_kw)
            axes[1, 0].scatter(r_h2[idx2], z_h2[idx2], **scatter_kw)
            axes[1, 1].scatter(x_h2[idx2], y_h2[idx2], **scatter_kw)

    fig.suptitle(
        "Geometry of inference cohorts — Stream 3 (hex density) + Stream 2 (purple points).\n"
        "Top row: Galactocentric (GC at origin). Bottom row: Heliocentric (Sun at origin). "
        "Stream 2 (TESS bright giants) clusters near the Sun in helio frames; "
        "in galactocentric it shows up as a ~1-2 kpc shell around the Sun.",
        fontsize=10,
    )
    fig.tight_layout()
    save_fig(fig, OUT / "geometry.png", tight=False)


if __name__ == "__main__":
    main()
