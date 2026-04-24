"""Six-panel figure showing the Mészáros+2025 Table-3 correction in action.

Reads both ``apogee_dr19_precorrected.parquet`` and
``apogee_dr19_corrected.parquet`` and visualises the per-star Δ applied in
Teff for every corrected element. Six elements chosen to span the largest
corrections (|Δ| ≳ 0.05 dex at 4800 K): Na, Ce, Mg, Al, Ni, O.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from arqueogal.data.apogee_dr19 import MESZAROS2025_COEFFS
from arqueogal.utils.plotting import (
    AA_DOUBLE_COLUMN_IN,
    save_figure,
    set_aa_style,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("plot_meszaros")


PANEL_ELEMENTS: tuple[tuple[str, str], ...] = (
    ("na_h_atm", "[Na/H]"),
    ("ce_h_atm", "[Ce/H]"),
    ("mg_h_atm", "[Mg/H]"),
    ("al_h_atm", "[Al/H]"),
    ("ni_h_atm", "[Ni/H]"),
    ("o_h_atm", "[O/H]"),
)


def main() -> None:
    import matplotlib.pyplot as plt

    repo = Path(__file__).resolve().parents[1]
    pre = repo / "data" / "interim" / "apogee_dr19_precorrected.parquet"
    post = repo / "data" / "interim" / "apogee_dr19_corrected.parquet"
    out_path = repo / "reports" / "figures" / "apogee_dr19_meszaros_deltas"

    cols_pre = ["teff"] + [c for c, _ in PANEL_ELEMENTS]
    cols_post = ["teff"] + [c for c, _ in PANEL_ELEMENTS]
    logger.info("loading pre/post interim parquets")
    df_pre = pd.read_parquet(pre, columns=cols_pre)
    df_post = pd.read_parquet(post, columns=cols_post)

    set_aa_style(colorblind=True, font_size=9.0)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(AA_DOUBLE_COLUMN_IN, AA_DOUBLE_COLUMN_IN * 0.58),
        sharex=True,
    )
    for ax, (col, label) in zip(axes.flat, PANEL_ELEMENTS):
        finite = np.isfinite(df_pre[col].to_numpy()) & np.isfinite(df_post[col].to_numpy())
        teff = df_pre["teff"].to_numpy()[finite]
        delta = df_post[col].to_numpy()[finite] - df_pre[col].to_numpy()[finite]
        hb = ax.hexbin(teff, delta, gridsize=70, cmap="magma", bins="log", mincnt=1)
        a, b, _, _ = MESZAROS2025_COEFFS[col]
        xline = np.linspace(4000, 5500, 80)
        ax.plot(
            xline,
            -(a * xline + b),
            "c-",
            lw=1.2,
            alpha=0.85,
            label=rf"$-(a T_{{\mathrm{{eff}}}}+b)$, $a={a:.2e}$, $b={b:+.3f}$",
        )
        ax.axhline(0.0, color="0.5", lw=0.6, ls="--")
        ax.set_title(rf"{label}  $\Delta$ applied")
        ax.set_ylabel(r"post $-$ pre [dex]")
        ax.legend(loc="upper right", fontsize=6.5, framealpha=0.7)
        ax.figure.colorbar(hb, ax=ax, label=r"$\log_{10}\,N$", pad=0.02)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$T_{\mathrm{eff}}$ [K]")
    axes[0, 0].invert_xaxis()

    fig.suptitle(
        r"M\'esz\'aros+2025 Table 3 $\Delta$[X/M] applied (pre $\to$ post), "
        f"354,890 DR19 giants",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    paths = save_figure(fig, out_path, formats=("png", "pdf"))
    for p in paths:
        logger.info("wrote %s (%.1f KB)", p, p.stat().st_size / 1024)


if __name__ == "__main__":
    main()
