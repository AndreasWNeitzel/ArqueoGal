"""Diagnostic plot suite for the 2026-04-18 status report.

Reads ``data/processed/pipeline1_features_stream1.parquet`` and produces a
set of figures under ``reports/figures/status_2026-04-18/`` covering:

1. Kiel diagram (Teff vs log g, colored by [Fe/H])
2. Tinsley diagram ([Mg/Fe] vs [Fe/H]) — α-bimodality check
3. Sky distribution + Ye+2024 flag sky (Galactic, Mollweide)
4. Extinction-prior comparison (5 priors cross-correlated)
5. APOGEE element-label grid (18 panels)
6. Sample Ye-corrected XP spectra
7. Data-flow funnel (row counts through the pipeline)
8. Feature availability bar chart

Not intended to be pretty publication plots — these are sanity-check
diagnostics. Publication-quality plots go elsewhere.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from arqueogal.data.gaia_xp import YE2024_SAMPLING_NM

mpl.rcParams["figure.dpi"] = 110
mpl.rcParams["savefig.dpi"] = 140
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.alpha"] = 0.25
mpl.rcParams["font.size"] = 9

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("plot_status")

REPO = Path(__file__).resolve().parents[1]
FEATURES = REPO / "data" / "processed" / "pipeline1_features_stream1.parquet"
OUT_DIR = REPO / "reports" / "figures" / "status_2026-04-18"


def _save(fig, name: str) -> None:
    path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    logger.info("wrote %s", path.relative_to(REPO))
    plt.close(fig)


def plot_kiel(df: pd.DataFrame) -> None:
    mask = df["teff_gspphot"].notna() & df["logg_gspphot"].notna() & df["fe_h_atm"].notna()
    sub = df.loc[mask].sample(min(80_000, int(mask.sum())), random_state=0)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(
        sub["teff_gspphot"],
        sub["logg_gspphot"],
        c=sub["fe_h_atm"],
        cmap="viridis_r",
        s=1.5,
        alpha=0.5,
        vmin=-1.5,
        vmax=0.4,
        rasterized=True,
    )
    ax.set_xlim(6200, 3500)
    ax.set_ylim(5.0, 0.5)
    ax.set_xlabel("Teff (GSP-Phot) [K]")
    ax.set_ylabel("log g (GSP-Phot)")
    ax.set_title(
        f"Kiel diagram — Pipeline-1 training set (n={len(sub):,})\ncolored by APOGEE [Fe/H]"
    )
    plt.colorbar(sc, ax=ax, label="[Fe/H] (APOGEE DR19, Mészáros+2025-corrected)")
    _save(fig, "01_kiel.png")


def plot_tinsley(df: pd.DataFrame) -> None:
    mask = df["fe_h_atm"].notna() & df["mg_h_atm"].notna()
    feh = df.loc[mask, "fe_h_atm"].to_numpy()
    mgfe = (df.loc[mask, "mg_h_atm"] - df.loc[mask, "fe_h_atm"]).to_numpy()
    fig, ax = plt.subplots(figsize=(7, 5))
    hb = ax.hexbin(
        feh, mgfe, gridsize=90, extent=(-2.0, 0.6, -0.3, 0.6), cmap="Blues", mincnt=3, bins="log"
    )
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.5)
    ax.axvline(0.0, color="k", lw=0.5, alpha=0.5)
    ax.set_xlabel("[Fe/H]")
    ax.set_ylabel("[Mg/Fe]  =  [Mg/H] − [Fe/H]")
    ax.set_title(f"Tinsley diagram — α-bimodality check (n={len(feh):,})")
    plt.colorbar(hb, ax=ax, label="log N")
    _save(fig, "02_tinsley_mgfe_feh.png")


def plot_sky(df: pd.DataFrame) -> None:
    c = SkyCoord(
        ra=df["ra_deg"].to_numpy() * u.deg,
        dec=df["dec_deg"].to_numpy() * u.deg,
        frame="icrs",
    ).galactic
    l_rad = c.l.wrap_at(180 * u.deg).radian
    b_rad = c.b.radian

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), subplot_kw={"projection": "mollweide"})

    ax = axes[0]
    ax.hexbin(l_rad, b_rad, gridsize=(90, 45), cmap="magma", mincnt=1, bins="log")
    ax.set_title(f"All Pipeline-1 training stars, Galactic (l, b) (n={len(df):,})")
    ax.grid(alpha=0.3)

    ax = axes[1]
    bad = df["ye2024_flag"] == 1  # NO_SYNTH_PHOT
    ax.scatter(
        l_rad[bad.to_numpy()], b_rad[bad.to_numpy()], s=0.6, c="crimson", alpha=0.4, rasterized=True
    )
    ax.set_title(
        f"Ye+2024 flag=NO_SYNTH_PHOT (n={int(bad.sum()):,})\n"
        "expected: northern sky, outside SkyMapper coverage"
    )
    ax.grid(alpha=0.3)

    _save(fig, "03_sky_distribution.png")


def plot_extinction_priors(df: pd.DataFrame) -> None:
    av_sfd = df["ebv_sfd"].to_numpy() * 2.742
    av_eden = df["ebv_edenhofer_2023"].to_numpy() * 2.742
    av_gsp = df["ag_gspphot"].to_numpy()
    av_nbhd = df["av_nbhd_median"].to_numpy()
    av_lall = df["av_lallement_xcheck"].to_numpy() if "av_lallement_xcheck" in df.columns else None

    priors = [
        ("A_V (SFD×2.742)", av_sfd),
        ("A_V (Edenhofer×2.742)", av_eden),
        ("A_G (GSP-Phot)", av_gsp),
        ("A_V (nbhd-median)", av_nbhd),
    ]
    if av_lall is not None:
        priors.append(("A_V (Lallement)", av_lall))

    n = len(priors)
    fig, axes = plt.subplots(n, n, figsize=(2.3 * n, 2.3 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                vals = priors[i][1]
                ax.hist(vals[np.isfinite(vals)], bins=80, color="steelblue")
                ax.set_yscale("log")
                ax.set_xlabel(priors[i][0], fontsize=8)
            elif j < i:
                y = priors[i][1]
                x = priors[j][1]
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() > 500:
                    ax.hexbin(x[mask], y[mask], gridsize=40, cmap="Greys", mincnt=3, bins="log")
                    lo = min(np.nanpercentile(x[mask], 1), np.nanpercentile(y[mask], 1))
                    hi = max(np.nanpercentile(x[mask], 99), np.nanpercentile(y[mask], 99))
                    ax.plot([lo, hi], [lo, hi], "r-", lw=0.8, alpha=0.7)
                ax.set_xlabel(priors[j][0], fontsize=8)
                ax.set_ylabel(priors[i][0], fontsize=8)
            else:
                ax.set_visible(False)
            ax.tick_params(labelsize=7)
    fig.suptitle("Extinction-prior cross-comparison (log density; red = 1:1)", fontsize=12, y=1.00)
    _save(fig, "04_extinction_priors.png")


def plot_label_grid(df: pd.DataFrame) -> None:
    labels = [
        "m_h_atm",
        "fe_h_atm",
        "c_h_atm",
        "n_h_atm",
        "o_h_atm",
        "na_h_atm",
        "mg_h_atm",
        "al_h_atm",
        "si_h_atm",
        "s_h_atm",
        "k_h_atm",
        "ca_h_atm",
        "ti_h_atm",
        "v_h_atm",
        "cr_h_atm",
        "mn_h_atm",
        "ni_h_atm",
        "ce_h_atm",
    ]
    fig, axes = plt.subplots(3, 6, figsize=(16, 8))
    for ax, col in zip(axes.flat, labels):
        vals = df[col].dropna().to_numpy()
        lo, hi = np.nanpercentile(vals, [0.5, 99.5])
        ax.hist(vals, bins=100, range=(lo, hi), color="darkorange", alpha=0.85)
        ax.set_title(col.replace("_atm", "").replace("_", "/"), fontsize=10)
        ax.tick_params(labelsize=7)
        ax.text(
            0.02,
            0.96,
            f"n={len(vals):,}\nμ={np.nanmean(vals):+.2f}\nσ={np.nanstd(vals):.2f}",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=7,
            family="monospace",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5),
        )
    fig.suptitle(
        "APOGEE DR19 element label distributions (Mészáros+2025-corrected where applicable)",
        fontsize=12,
        y=1.00,
    )
    _save(fig, "05_label_grid.png")


def plot_xp_spectra(df: pd.DataFrame) -> None:
    ok = df[(df["ye2024_flag"] == 0) & df["fe_h_atm"].notna()]
    rng = np.random.default_rng(42)
    # Sample stratified in [Fe/H] so we see the range
    feh_bins = np.array([-2.0, -1.5, -1.0, -0.5, -0.2, 0.0, 0.3])
    samples = []
    for i in range(len(feh_bins) - 1):
        lo, hi = feh_bins[i], feh_bins[i + 1]
        in_bin = ok[(ok["fe_h_atm"] >= lo) & (ok["fe_h_atm"] < hi)]
        if len(in_bin) > 0:
            n_take = min(3, len(in_bin))
            picks = rng.choice(len(in_bin), size=n_take, replace=False)
            samples.append(in_bin.iloc[picks])
    sub = pd.concat(samples).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("coolwarm_r")
    norm = mpl.colors.Normalize(vmin=-2.0, vmax=0.4)
    for _i, row in sub.iterrows():
        flux = np.asarray(row["corrected_flux"])
        # Normalize each spectrum to its median for shape comparison
        flux_n = flux / np.nanmedian(flux)
        ax.plot(YE2024_SAMPLING_NM, flux_n, color=cmap(norm(row["fe_h_atm"])), lw=0.7, alpha=0.85)
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("corrected flux / median")
    ax.set_title(f"Ye+2024-corrected XP spectra — {len(sub)} stars stratified in [Fe/H]")
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="[Fe/H]")
    ax.set_xlim(YE2024_SAMPLING_NM[0], YE2024_SAMPLING_NM[-1])
    _save(fig, "06_xp_spectra.png")


def plot_data_flow(df: pd.DataFrame) -> None:
    steps = [
        ("APOGEE DR19 raw (ASPCAP)", 964_989),
        ("APOGEE DR19 Mészáros-corrected", 354_890),
        ("∩ Gaia DR3 (Stream 1)", 354_231),
        ("Gaia DR3 corrected subset", 320_333),
        ("Raw XP coeffs (Stream 1 ∪ Stream 3)", 457_306),
        ("Ye+2024 OK", 445_255),
        ("Pipeline-1 training (∩ XP)", 324_054),
        ("  of which nbhd-median A_V finite", 170_217),
        ("  of which Lallement A_V finite", 90_504),
    ]
    names = [s[0] for s in steps]
    counts = [s[1] for s in steps]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(names))
    colors = ["#4c72b0"] * 4 + ["#55a868"] * 2 + ["#c44e52"] + ["#8172b3"] * 2
    ax.barh(y, counts, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("row count")
    ax.set_xscale("log")
    for yi, n in zip(y, counts):
        ax.text(n * 1.03, yi, f"{n:,}", va="center", fontsize=8)
    ax.set_title("Pipeline-1 data flow — row counts through the extraction")
    _save(fig, "07_data_flow.png")


def plot_feature_availability(df: pd.DataFrame) -> None:
    nullable = [
        "ebv_edenhofer_2023",
        "ebv_sfd",
        "ag_gspphot",
        "av_nbhd_median",
        "av_lallement_xcheck",
        "r_med_photogeo",
        "teff_gspphot",
        "logg_gspphot",
        "mh_gspphot",
        "g_mag",
        "bp_mag",
        "rp_mag",
        "j_mag",
        "h_mag",
        "k_mag",
        "w1_mag",
        "w2_mag",
        "m_h_atm",
        "fe_h_atm",
        "mg_h_atm",
        "al_h_atm",
        "c_h_atm",
        "n_h_atm",
        "o_h_atm",
        "ni_h_atm",
        "ce_h_atm",
        "v_h_atm",
    ]
    nullable = [c for c in nullable if c in df.columns]
    n = len(df)
    frac = [float(df[c].notna().sum()) / n for c in nullable]
    order = np.argsort(frac)
    nullable = [nullable[i] for i in order]
    frac = [frac[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 7.5))
    y = np.arange(len(nullable))
    colors = ["#c44e52" if f < 0.5 else ("#dd8452" if f < 0.9 else "#55a868") for f in frac]
    ax.barh(y, frac, color=colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(nullable, fontsize=8, family="monospace")
    ax.set_xlim(0, 1.02)
    ax.set_xlabel(f"fraction of stars with finite value (N = {n:,})")
    ax.axvline(0.5, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.axvline(0.9, color="k", lw=0.5, ls="--", alpha=0.5)
    for yi, f in zip(y, frac):
        ax.text(min(f + 0.01, 1.0), yi, f"{f * 100:.1f}%", va="center", fontsize=7)
    ax.set_title(
        "Per-feature availability in pipeline1_features_stream1\n"
        "green ≥ 90%, orange 50–90%, red < 50%"
    )
    _save(fig, "08_feature_availability.png")


def plot_distance_extinction(df: pd.DataFrame) -> None:
    d = df["r_med_photogeo"].to_numpy()
    ag = df["ag_gspphot"].to_numpy()
    mask = np.isfinite(d) & np.isfinite(ag) & (d > 0) & (d < 15000)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    ax.hist(d[mask] / 1000.0, bins=100, color="teal", alpha=0.85)
    ax.set_xlabel("Bailer-Jones photogeometric distance [kpc]")
    ax.set_ylabel("N stars")
    ax.set_title(f"Distance distribution (n={int(mask.sum()):,})")
    ax.set_yscale("log")

    ax = axes[1]
    hb = ax.hexbin(
        d[mask] / 1000.0,
        ag[mask],
        gridsize=80,
        extent=(0, 10, 0, 3.5),
        cmap="viridis",
        mincnt=5,
        bins="log",
    )
    ax.set_xlabel("distance [kpc]")
    ax.set_ylabel("A_G (GSP-Phot) [mag]")
    ax.set_title("Extinction vs distance")
    plt.colorbar(hb, ax=ax, label="log N")
    _save(fig, "09_distance_extinction.png")


def main() -> None:
    if not FEATURES.exists():
        raise SystemExit(f"missing {FEATURES}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("loading %s", FEATURES)
    df = pd.read_parquet(FEATURES)
    logger.info("  %d rows × %d cols", len(df), len(df.columns))

    plot_kiel(df)
    plot_tinsley(df)
    plot_sky(df)
    plot_extinction_priors(df)
    plot_label_grid(df)
    plot_xp_spectra(df)
    plot_data_flow(df)
    plot_feature_availability(df)
    plot_distance_extinction(df)

    logger.info("all plots written to %s", OUT_DIR.relative_to(REPO))


if __name__ == "__main__":
    main()
