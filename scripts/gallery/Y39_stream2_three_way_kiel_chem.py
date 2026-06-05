"""Y39: Stream-2 (TESS asteroseismic giants) compared three ways.

Three Stream-2 cohorts, side by side:
  - left:   GSP-Spec   (Gaia DR3 RVS-derived chemistry on Stream 2)
  - middle: APOGEE     (Stream 2 cross-matched against APOGEE DR19)
  - right:  Tier 1     (Stream 2 v1.1 Pipeline 1 prediction restricted
                        to release_tier == 1)

Top row: Kiel diagram (T_eff vs log g).
Bottom row: chemistry plane ([M/H] vs [α/M] or [α/Fe]).

Slide-friendly 16:8 layout.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "gallery"))
sys.path.insert(0, str(REPO / "src"))

from _presentation import PALETTE, apply_style, headline, save  # noqa: E402

from arqueogal.xp_abundances.main.release import assign_release_tier  # noqa: E402

PRED_S2 = REPO / "data/processed/pipeline1_predictions_stream2.parquet"
FEAT_S2 = REPO / "data/processed/pipeline1_features_stream2.parquet"
GSPSPEC_RAW = REPO / "data/interim/stream2_gaia_dr3_raw.parquet"
APOGEE_DR19 = REPO / "data/interim/apogee_dr19_corrected.parquet"

KIEL_TEFF = (3500, 6500)
KIEL_LOGG = (5.0, 0.0)
MH_LIM = (-1.6, 0.55)
AM_LIM = (-0.10, 0.45)
HEX_KIEL = 60
HEX_CHEM = 60

_TITLE_KW = dict(fontsize=11, fontweight="normal", color=PALETTE["ink"], pad=6)


def _load_pred() -> pd.DataFrame:
    pcols = [
        "source_id", "teff_pred", "logg_pred", "mh_pred",
        "alpha_m_pred", "mg_h_pred",
        "ood_joint_flag", "label_extrapolation_flag",
    ]
    p = pd.read_parquet(PRED_S2, columns=pcols).drop_duplicates("source_id")
    p["release_tier"] = assign_release_tier(p).astype(np.int8)
    return p


def _load_gspspec() -> pd.DataFrame:
    cols = [
        "source_id", "teff_gspspec", "logg_gspspec",
        "mh_gspspec", "alphafe_gspspec",
    ]
    df = pd.read_parquet(GSPSPEC_RAW, columns=cols).drop_duplicates("source_id")
    return df.dropna(subset=["teff_gspspec", "logg_gspspec",
                             "mh_gspspec", "alphafe_gspspec"]
                     ).reset_index(drop=True)


def _load_apogee_xmatch(s2_source_ids: np.ndarray) -> pd.DataFrame:
    cols = ["source_id", "teff", "logg", "m_h_atm", "alpha_m_atm"]
    df = pd.read_parquet(APOGEE_DR19, columns=cols).drop_duplicates("source_id")
    df = df.loc[df["source_id"].isin(s2_source_ids)].reset_index(drop=True)
    return df.dropna(subset=cols[1:]).reset_index(drop=True)


def _kiel_panel(ax, x, y, *, title: str, n_label: bool = False):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() > 0:
        ax.hexbin(
            x[ok], y[ok],
            gridsize=HEX_KIEL, extent=(KIEL_TEFF[0], KIEL_TEFF[1], 0.0, 5.0),
            mincnt=1, bins="log", cmap="viridis",
        )
    ax.set_xlim(KIEL_TEFF[1], KIEL_TEFF[0])
    ax.set_ylim(KIEL_LOGG[0], KIEL_LOGG[1])
    ax.set_xlabel(r"$T_{\rm eff}$ (K)")
    ax.set_ylabel(r"$\log g$ (dex)")
    suffix = f"  (n = {int(ok.sum()):,})" if n_label else ""
    ax.set_title(f"{title}{suffix}", **_TITLE_KW)
    ax.grid(True, alpha=0.20)


def _chem_panel(ax, x, y, *, title: str, ylabel: str, n_label: bool = False):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() > 0:
        ax.hexbin(
            x[ok], y[ok],
            gridsize=HEX_CHEM, extent=(MH_LIM[0], MH_LIM[1], AM_LIM[0], AM_LIM[1]),
            mincnt=1, bins="log", cmap="viridis",
        )
    ax.axhline(0.15, color="white", lw=1.0, ls=":", alpha=0.85)
    ax.set_xlim(MH_LIM)
    ax.set_ylim(AM_LIM)
    ax.set_xlabel("[M/H] (dex)")
    ax.set_ylabel(ylabel)
    suffix = f"  (n = {int(ok.sum()):,})" if n_label else ""
    ax.set_title(f"{title}{suffix}", **_TITLE_KW)
    ax.grid(True, alpha=0.20)


def main() -> int:
    apply_style()
    pred = _load_pred()
    if pred.empty:
        print("[Y39] no Stream-2 predictions, aborting")
        return 1
    tier1 = pred.loc[pred["release_tier"] == 1].reset_index(drop=True)
    gsp = _load_gspspec()
    apo = _load_apogee_xmatch(pred["source_id"].to_numpy())
    # Restrict GSP-Spec and APOGEE to Stream-2 source ids only.
    s2_ids = set(pred["source_id"].to_numpy().tolist())
    gsp = gsp.loc[gsp["source_id"].isin(s2_ids)].reset_index(drop=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # Top row: Kiel.
    _kiel_panel(axes[0, 0],
                gsp["teff_gspspec"].to_numpy(), gsp["logg_gspspec"].to_numpy(),
                title="GSP-Spec", n_label=True)
    _kiel_panel(axes[0, 1],
                apo["teff"].to_numpy(), apo["logg"].to_numpy(),
                title="APOGEE DR19 cross-match", n_label=True)
    _kiel_panel(axes[0, 2],
                tier1["teff_pred"].to_numpy(), tier1["logg_pred"].to_numpy(),
                title="ArqueoGal Tier 1", n_label=True)

    # Bottom row: chemistry.
    _chem_panel(axes[1, 0],
                gsp["mh_gspspec"].to_numpy(), gsp["alphafe_gspspec"].to_numpy(),
                title="GSP-Spec", ylabel=r"[$\alpha$/Fe] (dex)")
    _chem_panel(axes[1, 1],
                apo["m_h_atm"].to_numpy(), apo["alpha_m_atm"].to_numpy(),
                title="APOGEE DR19 cross-match", ylabel=r"[$\alpha$/M] (dex)")
    _chem_panel(axes[1, 2],
                tier1["mh_pred"].to_numpy(), tier1["alpha_m_pred"].to_numpy(),
                title="ArqueoGal Tier 1", ylabel=r"[$\alpha$/M] (dex)")

    fig.subplots_adjust(left=0.05, right=0.985, top=0.81, bottom=0.08,
                        hspace=0.50, wspace=0.30)
    headline(
        fig,
        "Stream 2: GSP-Spec, APOGEE DR19 cross-match, ArqueoGal Tier 1",
        f"top = Kiel; bottom = chemistry plane.  TESS asteroseismic giants, "
        f"n_total = {len(pred):,}.",
        top=0.81,
    )
    save(fig, "Y39_stream2_three_way_kiel_chem")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
