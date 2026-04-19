"""Six-panel diagnostics for the APOGEE DR19 interim extraction.

    Top row:    sky (Mollweide in galactic l, b) | Kiel | CMD (BP-RP vs G)
    Bottom row: [M/H] vs [α/M]  |  r_med_photogeo |  dust-map availability

Points plotted via `hexbin` so 354 k rows don't blow up the PDF. Saved as
PNG + PDF into ``reports/figures/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from arqueogal.utils.plotting import (
    AA_DOUBLE_COLUMN_IN,
    WONG_PALETTE,
    save_figure,
    set_aa_style,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("plot_extraction_diagnostics")


def _load_apogee(path: Path) -> pd.DataFrame:
    cols = [
        "ra_deg", "dec_deg",
        "teff", "logg", "m_h_atm", "alpha_m_atm",
        "g_mag", "bp_mag", "rp_mag",
        "r_med_photogeo",
        "ebv", "ebv_edenhofer_2023", "ebv_bayestar_2019",
        "ebv_zhang_2023", "ebv_sfd",
    ]
    df = pd.read_parquet(path, columns=cols)
    logger.info("APOGEE interim rows: %d", len(df))
    return df


def _galactic(ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (l_rad, b_rad) with l wrapped to [-pi, pi] for Mollweide."""
    sky = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs").galactic
    l_deg = sky.l.wrap_at(180 * u.deg).deg
    b_deg = sky.b.deg
    return np.deg2rad(l_deg), np.deg2rad(b_deg)


def _panel_sky(ax, l_rad, b_rad, *, gridsize: int = 180) -> None:
    hb = ax.hexbin(l_rad, b_rad, gridsize=gridsize, cmap="magma", bins="log", mincnt=1)
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.set_xticklabels([])  # Mollweide tick labels overlap the map edge
    ax.set_title(r"Galactic sky (Mollweide, $\ell$, $b$)")
    cb = ax.figure.colorbar(hb, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
    cb.set_label(r"$\log_{10}\,N$")


def _panel_kiel(ax, teff, logg) -> None:
    hb = ax.hexbin(teff, logg, gridsize=80, cmap="viridis", bins="log", mincnt=1)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel(r"$T_{\mathrm{eff}}$ [K]")
    ax.set_ylabel(r"$\log g$")
    ax.set_title("Kiel diagram")
    ax.figure.colorbar(hb, ax=ax, label=r"$\log_{10}\,N$")


def _panel_cmd(ax, bp_mag, rp_mag, g_mag) -> None:
    color = bp_mag - rp_mag
    hb = ax.hexbin(color, g_mag, gridsize=80, cmap="viridis", bins="log", mincnt=1)
    ax.invert_yaxis()
    ax.set_xlabel(r"$G_{\mathrm{BP}} - G_{\mathrm{RP}}$")
    ax.set_ylabel(r"$G$ [mag]")
    ax.set_title("Observed CMD (pre-extinction correction)")
    ax.figure.colorbar(hb, ax=ax, label=r"$\log_{10}\,N$")


def _panel_alpha_mh(ax, m_h, alpha_m) -> None:
    hb = ax.hexbin(m_h, alpha_m, gridsize=80, cmap="viridis", bins="log", mincnt=1)
    ax.set_xlabel(r"[M/H]")
    ax.set_ylabel(r"[$\alpha$/M]")
    ax.set_title(r"Chemical plane ([M/H] vs [$\alpha$/M])")
    ax.figure.colorbar(hb, ax=ax, label=r"$\log_{10}\,N$")


def _panel_distance(ax, r_photogeo_pc) -> None:
    r_kpc = r_photogeo_pc / 1000.0
    r_kpc = r_kpc[np.isfinite(r_kpc) & (r_kpc > 0)]
    bins = np.logspace(np.log10(0.05), np.log10(r_kpc.max() + 1), 80)
    ax.hist(r_kpc, bins=bins, color=WONG_PALETTE[2], edgecolor="none")
    ax.set_xscale("log")
    ax.set_xlabel(r"$r_{\mathrm{med,photogeo}}$ [kpc]")
    ax.set_ylabel("N stars")
    ax.set_title(f"Bailer-Jones+2021 distance (n={len(r_kpc):,})")
    ax.axvline(1.25, color="0.5", lw=0.6, ls="--")
    ax.text(1.25 * 1.05, ax.get_ylim()[1] * 0.85,
            "Edenhofer\nhorizon", color="0.4", fontsize=7)


def _panel_dust(ax, df: pd.DataFrame) -> None:
    bins = np.linspace(0.0, 2.0, 80)
    for col, color, label in (
        ("ebv_edenhofer_2023", WONG_PALETTE[1], "Edenhofer+2023 (3D, <1.25 kpc)"),
        ("ebv_bayestar_2019",  WONG_PALETTE[2], "Bayestar+2019 (3D all-sky)"),
        ("ebv_zhang_2023",     WONG_PALETTE[3], "Zhang+2023 (XP-based 3D)"),
        ("ebv_sfd",            WONG_PALETTE[5], "SFD (2D upper limit)"),
    ):
        x = df[col].to_numpy()
        x = x[np.isfinite(x)]
        frac = len(x) / len(df)
        ax.hist(x, bins=bins, histtype="step", color=color, lw=1.2,
                label=f"{label}  [{frac:.1%} cov]")
    ax.set_yscale("log")
    ax.set_xlabel(r"$E(B-V)$ [mag]")
    ax.set_ylabel("N stars")
    ax.set_title("Per-star dust (pre-baked in DR19)")
    ax.legend(loc="upper right", fontsize=6.5)


def main() -> None:
    import matplotlib.pyplot as plt  # local import so env without display still runs

    repo = Path(__file__).resolve().parents[1]
    interim = repo / "data" / "interim" / "apogee_dr19_precorrected.parquet"
    out_path = repo / "reports" / "figures" / "apogee_dr19_diagnostics"

    df = _load_apogee(interim)

    set_aa_style(colorblind=True, font_size=9.0)
    fig = plt.figure(figsize=(AA_DOUBLE_COLUMN_IN, AA_DOUBLE_COLUMN_IN * 0.62))

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38)
    ax_sky = fig.add_subplot(gs[0, 0], projection="mollweide")
    ax_kiel = fig.add_subplot(gs[0, 1])
    ax_cmd = fig.add_subplot(gs[0, 2])
    ax_alpha = fig.add_subplot(gs[1, 0])
    ax_dist = fig.add_subplot(gs[1, 1])
    ax_dust = fig.add_subplot(gs[1, 2])

    # Sky
    logger.info("computing galactic coordinates")
    l_rad, b_rad = _galactic(df["ra_deg"].to_numpy(), df["dec_deg"].to_numpy())
    _panel_sky(ax_sky, l_rad, b_rad)

    # Kiel + chemical + CMD + distance + dust
    _panel_kiel(ax_kiel, df["teff"], df["logg"])
    _panel_cmd(ax_cmd, df["bp_mag"], df["rp_mag"], df["g_mag"])
    _panel_alpha_mh(ax_alpha, df["m_h_atm"], df["alpha_m_atm"])
    _panel_distance(ax_dist, df["r_med_photogeo"].to_numpy())
    _panel_dust(ax_dust, df)

    fig.suptitle(
        f"APOGEE DR19 post-cuts interim — {len(df):,} stars  (Stream 1 training pool)",
        fontsize=10,
    )

    paths = save_figure(fig, out_path, formats=("png", "pdf"))
    for p in paths:
        logger.info("wrote %s (%.1f KB)", p, p.stat().st_size / 1024)


if __name__ == "__main__":
    main()
