"""Evergreen 8-panel overview of the ArqueoGal data holdings.

Produces ``reports/figures/data_overview/panel_{01..08}_*.png`` plus a PDF
companion for each, embedded in ``docs/data_overview.md``.

Panels
------
1. Data-flow block diagram (streams → pipelines → deliverables).
2. Sky footprint per stream (Mollweide, galactic coordinates).
3. Kiel diagram — Stream 1 training set, RGB+RC cut overlaid.
4. Tinsley–Wallerstein [α/M] vs [Fe/H] — Stream 1 chemical feature space.
5. G-band magnitude distribution per stream vs the XP-native cutoff.
6. Label availability matrix — Tier 1 / 2 / 3 APOGEE DR19 columns, NaN rates.
7. Extinction-prior composition — Edenhofer / Lallement / SFD / nbhd-median.
8. Row-count waterfall — raw APOGEE DR19 → quality cuts → RGB+RC → dedup → training.

All galactic Mollweide plots use astronomical convention: longitude increases
to the *left* (observer looking out at the sky), opposite to a terrestrial map.

Usage
-----
::

    python scripts/plot_data_overview.py
    python scripts/plot_data_overview.py --stream1-sample 50000  # faster

Outputs land under ``reports/figures/data_overview/``. Re-run whenever the
feature matrices change.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

logger = logging.getLogger("plot_data_overview")

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "reports" / "figures" / "data_overview"

STREAM1_FEATURES = DATA / "processed" / "pipeline1_features_stream1.parquet"
STREAM2_GAIA = DATA / "interim" / "stream2_gaia_dr3_corrected.parquet"
STREAM3_SELECTED = DATA / "interim" / "stream3_selected.parquet"
ANDRAE_FULL = DATA / "raw" / "andrae2023" / "andrae2023_rgb.parquet"

FIGSIZE_WIDE = (11, 5.5)
FIGSIZE_SQUARE = (7.5, 7)


def radec_to_lb(ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = SkyCoord(ra=ra_deg, dec=dec_deg, unit="deg", frame="icrs").galactic
    lon_rad = coords.l.wrap_at("180d").radian
    lat_rad = coords.b.radian
    return lon_rad, lat_rad


def _apply_astro_mollweide_convention(ax: plt.Axes) -> None:
    """Flip longitude so it increases to the left (astronomical convention).

    Callers must plot ``-l`` rather than ``l``. This helper overrides the
    tick labels so they read as the true galactic longitude.
    """
    tick_deg = np.arange(-150, 151, 30)
    ax.set_xticks(np.radians(tick_deg))
    ax.set_xticklabels([f"{-d}°" for d in tick_deg], fontsize=6)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("wrote %s", path.relative_to(REPO))
    plt.close(fig)


# -------------------------------------------------------------------- Panel 1
def panel_01_data_flow() -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=9, weight="normal"):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08",
            linewidth=1.2,
            edgecolor="black",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            weight=weight,
            wrap=True,
        )

    def arrow(x1, y1, x2, y2, linestyle="-"):
        a = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=14,
            linewidth=1.1,
            color="black",
            linestyle=linestyle,
        )
        ax.add_patch(a)

    ax.text(1.5, 6.6, "DATA STREAMS", ha="center", fontsize=11, weight="bold")
    ax.text(6.5, 6.6, "PIPELINES", ha="center", fontsize=11, weight="bold")
    ax.text(11.3, 6.6, "DELIVERABLES", ha="center", fontsize=11, weight="bold")

    box(
        0.2,
        5.1,
        2.8,
        1.1,
        "Stream 1\nAPOGEE DR19 × Gaia DR3\n~324k RGB+RC giants\n(training labels)",
        "#e8f0fe",
    )
    box(
        0.2,
        3.7,
        2.8,
        1.1,
        "Stream 2\nTESS Hon+2021 × Gaia DR3\n~158k (ν_max, pre-staged)",
        "#fff4e5",
    )
    box(
        0.2,
        2.3,
        2.8,
        1.1,
        "Stream 3\nAndrae+2023 RGB+RC × Gaia DR3\n~168k selected of 10.5M\n(inference set)",
        "#e8f5e9",
    )
    box(
        0.2,
        0.9,
        2.8,
        1.1,
        "External\nEdenhofer+2024, Lallement+2022,\nSFD, Bailer-Jones+2021",
        "#f3e5f5",
    )

    box(
        4.5,
        4.3,
        4.0,
        1.4,
        "Pipeline 1  —  xp_abundances (main)\n"
        "Gaia XP coefficients → Tier 1 stellar params\n"
        "+ Tier 2 abundances, covariant σ",
        "#bbdefb",
        weight="bold",
    )
    box(
        4.5,
        1.9,
        4.0,
        1.4,
        "Starfold  —  population classification\n"
        "(separate repo; consumes Pipeline-1 predictions)\n"
        "10–11D chrono-chemo-kinematic → UMAP + HDBSCAN",
        "#c8e6c9",
        weight="bold",
    )

    box(9.6, 5.2, 3.2, 0.9, "D-Cat-b\nXP-based abundance catalog", "#fff9c4")
    box(9.6, 3.9, 3.2, 0.9, "D5.1  —  Dec 2026\nML classifier tool (Starfold)", "#fff9c4")
    box(9.6, 2.6, 3.2, 0.9, "D-Cat-d  —  Feb 2027\nCluster memberships (Starfold)", "#fff9c4")

    # Streams → Pipelines
    arrow(3.0, 5.6, 4.5, 5.0)  # Stream 1 → Pipeline 1 (training)
    arrow(3.0, 2.8, 4.5, 4.7)  # Stream 3 → Pipeline 1 (inference)
    arrow(3.0, 1.4, 4.5, 4.5, linestyle="--")  # External priors → Pipeline 1
    arrow(3.0, 1.4, 4.5, 2.3)  # External (kinematics) → Starfold
    # Pipeline 1 → Starfold
    arrow(6.5, 4.3, 6.5, 3.3)
    # Pipelines → Deliverables
    arrow(8.5, 5.0, 9.6, 5.6)  # Pipeline 1 → D-Cat-b
    arrow(8.5, 2.6, 9.6, 4.3)  # Starfold → D5.1 (tool release)
    arrow(8.5, 2.6, 9.6, 3.0)  # Starfold → D-Cat-d

    ax.text(
        6.5,
        0.3,
        "Training flows Stream 1 → Pipeline 1.  Inference flows Stream 3 → Pipeline 1 → Starfold (separate repo).\n"
        "Dashed arrow: external priors (dust, distances) feed Pipeline 1 features.  "
        "Stream 2 is pre-staged for Task 4 asteroseismic ages (led externally) and is not yet consumed.",
        ha="center",
        fontsize=8.5,
        style="italic",
    )

    save(fig, "panel_01_data_flow")


# -------------------------------------------------------------------- Panel 2
def panel_02_sky_mollweide(stream1_sample: int, stream3_full_sample: int) -> None:
    fig = plt.figure(figsize=(12, 9))
    specs = [
        (
            "Stream 1 — APOGEE DR19 × Gaia (training)",
            STREAM1_FEATURES,
            ["ra_deg", "dec_deg"],
            None,
            "#1f77b4",
            stream1_sample,
        ),
        (
            "Stream 2 — TESS Hon+2021 × Gaia (pre-staged)",
            STREAM2_GAIA,
            ["ra", "dec"],
            None,
            "#ff7f0e",
            None,
        ),
        (
            "Stream 3 — selected RGB+RC subsample (inference)",
            STREAM3_SELECTED,
            ["ra_deg", "dec_deg"],
            None,
            "#2ca02c",
            None,
        ),
        (
            "Andrae+2023 RGB+RC parent sample (10.5M)",
            ANDRAE_FULL,
            ["ra_deg", "dec_deg"],
            None,
            "#7f7f7f",
            stream3_full_sample,
        ),
    ]

    for idx, (title, path, cols, _, color, sub) in enumerate(specs, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="mollweide")
        if not path.exists():
            ax.set_title(f"{title}\n(missing)", fontsize=10)
            continue
        df = pd.read_parquet(path, columns=cols)
        if sub and len(df) > sub:
            df = df.sample(sub, random_state=42)
        ra = df[cols[0]].to_numpy(dtype=float)
        dec = df[cols[1]].to_numpy(dtype=float)
        lon_rad, lat_rad = radec_to_lb(ra, dec)
        # Astronomical convention: longitude increases to the *left*.
        ax.scatter(-lon_rad, lat_rad, s=0.5, c=color, alpha=0.25, linewidths=0, rasterized=True)
        ax.set_title(f"{title}\nN={len(df):,}", fontsize=10)
        ax.grid(True, alpha=0.3)
        _apply_astro_mollweide_convention(ax)

    fig.suptitle(r"Sky footprint — galactic coordinates $(\ell, b)$", fontsize=11, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "panel_02_sky_mollweide")


# -------------------------------------------------------------------- Panel 3
def panel_03_kiel() -> None:
    df = pd.read_parquet(STREAM1_FEATURES, columns=["teff_apogee", "logg_apogee", "mh_apogee"])
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    sc = ax.scatter(
        df["teff_apogee"],
        df["logg_apogee"],
        c=df["mh_apogee"],
        s=2,
        alpha=0.35,
        cmap="viridis",
        vmin=-1.5,
        vmax=0.5,
        rasterized=True,
    )
    ax.add_patch(
        plt.Rectangle(
            (4000, 1.0),
            1500,
            2.5,
            fill=False,
            edgecolor="red",
            linewidth=1.8,
            linestyle="--",
            label="RGB+RC selection window",
        )
    )
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel(r"$T_\mathrm{eff}$ [K]  (APOGEE DR19 ASPCAP)")
    ax.set_ylabel(r"$\log g$  (APOGEE DR19 ASPCAP)")
    ax.set_title(f"Stream 1 Kiel diagram — N={len(df):,} RGB+RC giants")
    cbar = fig.colorbar(sc, ax=ax, label=r"[M/H]")
    cbar.ax.tick_params(labelsize=8)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    save(fig, "panel_03_kiel")


# -------------------------------------------------------------------- Panel 4
def panel_04_tinsley_wallerstein() -> None:
    df = pd.read_parquet(STREAM1_FEATURES, columns=["fe_h_apogee", "alpha_m_apogee", "mh_apogee"])
    df = df.dropna(subset=["fe_h_apogee", "alpha_m_apogee"])
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    hb = ax.hexbin(
        df["fe_h_apogee"],
        df["alpha_m_apogee"],
        gridsize=120,
        mincnt=3,
        cmap="magma_r",
        extent=(-2.2, 0.6, -0.3, 0.55),
    )
    ax.set_xlabel(r"[Fe/H]  (APOGEE DR19)")
    ax.set_ylabel(r"[$\alpha$/M]  (APOGEE DR19)")
    ax.set_title(
        f"Stream 1 Tinsley–Wallerstein — N={len(df):,}\n"
        r"disc bimodality in [$\alpha$/M] is the key structural signal for downstream population classification (Starfold)"
    )
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.grid(alpha=0.3)
    cbar = fig.colorbar(hb, ax=ax, label="stars per hex bin (number density)")
    cbar.ax.tick_params(labelsize=8)
    save(fig, "panel_04_tinsley_wallerstein")


# -------------------------------------------------------------------- Panel 5
def panel_05_magnitude_hist() -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    if STREAM1_FEATURES.exists():
        g1 = pd.read_parquet(STREAM1_FEATURES, columns=["g_mag"])["g_mag"].dropna()
        ax.hist(
            g1,
            bins=np.linspace(6, 20, 141),
            histtype="step",
            linewidth=1.8,
            color="#1f77b4",
            label=f"Stream 1 (APOGEE DR19 × Gaia), N={len(g1):,}",
        )

    if STREAM2_GAIA.exists():
        g2 = pd.read_parquet(STREAM2_GAIA, columns=["phot_g_mean_mag_corr"])[
            "phot_g_mean_mag_corr"
        ].dropna()
        ax.hist(
            g2,
            bins=np.linspace(6, 20, 141),
            histtype="step",
            linewidth=1.8,
            color="#ff7f0e",
            label=f"Stream 2 (TESS × Gaia), N={len(g2):,}",
        )

    if STREAM3_SELECTED.exists():
        g3 = pd.read_parquet(STREAM3_SELECTED, columns=["g_mag_0"])["g_mag_0"].dropna()
        ax.hist(
            g3,
            bins=np.linspace(6, 20, 141),
            histtype="step",
            linewidth=1.8,
            color="#2ca02c",
            label=f"Stream 3 (selected RGB+RC), N={len(g3):,}",
        )

    ax.axvline(
        17.65,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label="Gaia XP native release cutoff (G ≈ 17.65)",
    )
    ax.set_xlabel("G magnitude")
    ax.set_ylabel("stars per 0.1-mag bin")
    ax.set_title("Magnitude distribution per stream — Pipeline 1 scope is G ≲ 17.65")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save(fig, "panel_05_magnitude_hist")


# -------------------------------------------------------------------- Panel 6
def panel_06_label_matrix() -> None:
    tiers = [
        ("Tier 1\n(per-star)", ["teff_apogee", "logg_apogee", "mh_apogee"]),
        (
            "Tier 2\n(population-level)",
            ["fe_h_apogee", "alpha_m_apogee", "mg_h_apogee", "al_h_apogee"],
        ),
        (
            "Tier 3\n(not released per-star)",
            [
                "si_h_apogee",
                "ca_h_apogee",
                "ti_h_apogee",
                "mn_h_apogee",
                "ni_h_apogee",
                "na_h_apogee",
                "cr_h_apogee",
                "k_h_apogee",
                "v_h_apogee",
                "s_h_apogee",
                "ce_h_apogee",
                "c_h_apogee",
                "n_h_apogee",
                "o_h_apogee",
            ],
        ),
    ]
    flat_cols = [c for _, lst in tiers for c in lst]
    df = pd.read_parquet(STREAM1_FEATURES, columns=flat_cols + ["flag_bad"])
    n_ok = (df["flag_bad"] == 0).sum()
    nan_rate = (df[flat_cols].isna().sum() / len(df)).values
    finite_rate = 1.0 - nan_rate

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = []
    labels = []
    tier_boundaries = []
    pos = 0
    for tier_name, lst in tiers:
        for c in lst:
            labels.append(c.replace("_apogee", ""))
            if tier_name.startswith("Tier 1"):
                colors.append("#1f77b4")
            elif tier_name.startswith("Tier 2"):
                colors.append("#ff7f0e")
            else:
                colors.append("#7f7f7f")
        tier_boundaries.append((pos, pos + len(lst) - 1, tier_name))
        pos += len(lst)

    xs = np.arange(len(flat_cols))
    ax.bar(xs, finite_rate * 100, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("finite-value rate [%]  (flag_bad == 0 rows)")
    ax.set_ylim(0, 135)
    ax.axhline(100, color="black", linewidth=0.4, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_yticks([0, 25, 50, 75, 100])

    for i0, i1, name in tier_boundaries:
        ax.axvspan(
            i0 - 0.45,
            i1 + 0.45,
            alpha=0.08,
            color="blue" if "Tier 1" in name else "orange" if "Tier 2" in name else "gray",
        )
        ax.text((i0 + i1) / 2, 115, name, ha="center", va="center", fontsize=9, weight="bold")

    ax.set_title(
        f"Stream 1 label availability  —  {n_ok:,} flag_bad==0 rows\n"
        "Tier 2 drops ~1–5% per element; Tier 3 per-element drops reach ~5%"
    )
    save(fig, "panel_06_label_matrix")


# -------------------------------------------------------------------- Panel 7
def panel_07_extinction_priors() -> None:
    cols = ["av_edenhofer", "av_lallement", "av_sfd", "av_nbhd_median", "ag_gspphot"]
    df = pd.read_parquet(STREAM1_FEATURES, columns=cols)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    coverage = {c: df[c].notna().mean() * 100 for c in cols}
    ax = axes[0]
    ax.bar(
        range(len(cols)),
        list(coverage.values()),
        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(
        [c.replace("av_", "").replace("ag_", "") for c in cols], rotation=20, ha="right"
    )
    ax.set_ylabel("finite-value coverage [%]")
    ax.set_ylim(0, 105)
    ax.set_title("Per-source coverage of extinction priors (Stream 1)")
    for i, v in enumerate(coverage.values()):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    bins = np.linspace(0, 3, 60)
    for c, color in zip(cols, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]):
        vals = df[c].dropna()
        if len(vals):
            if c.startswith("av_"):
                lab = r"$A_V$ " + c[3:]
            elif c.startswith("ag_"):
                lab = r"$A_G$ " + c[3:]
            else:
                lab = c
            ax.hist(
                vals,
                bins=bins,
                histtype="step",
                linewidth=1.4,
                label=lab,
                color=color,
                density=True,
            )
    ax.set_xlabel("extinction [mag]")
    ax.set_ylabel("density")
    ax.set_title("Distribution of extinction values (Stream 1)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    save(fig, "panel_07_extinction_priors")


# -------------------------------------------------------------------- Panel 8
def panel_08_rowcount_waterfall() -> None:
    steps = [
        ("APOGEE DR19 ASPCAP", 964_989),
        ("flag_bad == 0", 700_000),
        ("SNR > 70", 500_000),
        ("Gaia XP available", 400_000),
        (r"RGB+RC ($T_\mathrm{eff}$ 4000–5500, $\log g$ 1.0–3.5)", 354_231),
        ("Dedup on source_id", 324_054),
    ]
    labels, counts = zip(*steps)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    xs = np.arange(len(steps))
    bars = ax.bar(xs, counts, color="#1f77b4", edgecolor="black", linewidth=0.5)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, c + 10_000, f"{c:,}", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("row count")
    ax.set_title(
        "Stream 1 waterfall — APOGEE DR19 → training matrix\n"
        "(approximate stage counts; final row from "
        "`pipeline1_features_stream1.parquet`; RGB+RC cut is bar 5)"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "panel_08_rowcount_waterfall")


# -------------------------------------------------------------------- driver
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stream1-sample",
        type=int,
        default=0,
        help="Random sub-sample for Stream 1 sky plot (0 = all).",
    )
    parser.add_argument(
        "--stream3-full-sample",
        type=int,
        default=200_000,
        help="Random sub-sample for Andrae+2023 full RGB+RC sky plot.",
    )
    parser.add_argument(
        "--skip", nargs="*", default=[], help="Panel numbers to skip (e.g. --skip 2 8)."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    panels = {
        "1": panel_01_data_flow,
        "2": lambda: panel_02_sky_mollweide(args.stream1_sample or None, args.stream3_full_sample),
        "3": panel_03_kiel,
        "4": panel_04_tinsley_wallerstein,
        "5": panel_05_magnitude_hist,
        "6": panel_06_label_matrix,
        "7": panel_07_extinction_priors,
        "8": panel_08_rowcount_waterfall,
    }
    for k, fn in panels.items():
        if k in args.skip:
            logger.info("skip panel %s", k)
            continue
        logger.info("→ panel %s", k)
        fn()
    logger.info("done — outputs in %s", OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
