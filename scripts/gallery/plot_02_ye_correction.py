"""Stage 02: Ye+2024 NN flux correction — before/after diagnostics.

Outputs:
  - reports/gallery/02_ye_correction/ye_before_after_sed.png
  - reports/gallery/02_ye_correction/ye_delta_distribution.png
  - reports/gallery/02_ye_correction/ye_flag_sky_map.png   (Galactic Mollweide)
  - reports/gallery/02_ye_correction/ye_flag_vs_g.png

"Before" = raw coefficients calibrated via gaiaxpy (no NN correction).
"After"  = xp_sampled_corrected.parquet (post-Ye NN output, SFD-dereddened).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DATA_INTERIM,
    DATA_PROCESSED,
    GALLERY,
    PALETTE,
    apply_style,
    radec_to_galactic_mollweide,
    sample_index,
    save_fig,
    style_galactic_mollweide,
)

OUT = GALLERY / "02_ye_correction"

YE_SAMPLING_NM = np.linspace(360.0, 990.0, 330)
YE_FLAG_NAMES = {0: "OK", 1: "NO_SYNTH_PHOT", 2: "CALIBRATE_FAIL"}


def _pick_before_after_pairs(
    n: int = 120, seed: int = 23
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Run gaiaxpy.calibrate on n raw-coef rows and pair with post-Ye output.

    Oversample n, then drop any row whose 'before' (gaiaxpy) is all-NaN —
    this excludes a known subset of ye_flag=0 rows that gaiaxpy refuses to
    calibrate (typically N_meas-edge cases).
    """
    from gaiaxpy import calibrate

    xp_raw = pq.read_table(DATA_INTERIM / "xp_coeffs_raw_delta.parquet").to_pandas()
    xp_cor = pq.read_table(
        DATA_INTERIM / "xp_sampled_corrected_delta.parquet"
        if (DATA_INTERIM / "xp_sampled_corrected_delta.parquet").exists()
        else DATA_INTERIM / "xp_sampled_corrected.parquet",
        columns=["source_id", "corrected_flux", "ye2024_flag"],
    ).to_pandas()
    merged = xp_raw.merge(xp_cor, on="source_id", how="inner")
    ok = merged[merged["ye2024_flag"] == 0].copy()
    if len(ok) == 0:
        raise RuntimeError("no OK rows after merge")
    idx = sample_index(len(ok), n, np.random.default_rng(seed))
    sub = ok.iloc[idx].reset_index(drop=True)

    cal_in = pd.DataFrame(
        {
            "source_id": sub["source_id"],
            "bp_coefficients": sub["bp_coefficients"].apply(np.asarray).tolist(),
            "rp_coefficients": sub["rp_coefficients"].apply(np.asarray).tolist(),
            "bp_coefficient_errors": sub["bp_coefficient_errors"].apply(np.asarray).tolist(),
            "rp_coefficient_errors": sub["rp_coefficient_errors"].apply(np.asarray).tolist(),
            "bp_n_parameters": [55] * len(sub),
            "rp_n_parameters": [55] * len(sub),
            "bp_coefficient_correlations": [np.zeros(55 * 56 // 2, dtype=np.float32)] * len(sub),
            "rp_coefficient_correlations": [np.zeros(55 * 56 // 2, dtype=np.float32)] * len(sub),
            "bp_standard_deviation": sub["bp_standard_deviation"],
            "rp_standard_deviation": sub["rp_standard_deviation"],
            "bp_n_relevant_bases": sub["bp_n_relevant_bases"],
            "rp_n_relevant_bases": sub["rp_n_relevant_bases"],
            "bp_n_measurements": sub["bp_n_measurements"],
            "rp_n_measurements": sub["rp_n_measurements"],
            "solution_id": [1636148068921376768] * len(sub),
        }
    )

    cal_df, _ = calibrate(
        cal_in,
        sampling=YE_SAMPLING_NM,
        save_file=False,
    )
    cal_df = cal_df.sort_values("source_id").reset_index(drop=True)
    sub = sub.sort_values("source_id").reset_index(drop=True)
    assert (cal_df["source_id"].values == sub["source_id"].values).all()

    flux_before = np.vstack(cal_df["flux"].apply(np.asarray).to_list())
    flux_after = np.vstack(sub["corrected_flux"].apply(np.asarray).to_list())

    # drop rows where gaiaxpy failed to produce any usable flux
    good = np.isfinite(flux_before).all(axis=1) & np.isfinite(flux_after).all(axis=1)
    return flux_before[good], flux_after[good], sub.loc[good].reset_index(drop=True)


def _load_star_info(source_ids: np.ndarray) -> pd.DataFrame:
    """Best-effort metadata join. The delta-XP stars come from Stream 3, so we
    prefer Andrae+2023 labels there; fall back to APOGEE via Stream 1."""
    wanted = {
        "source_id",
        "teff_apogee",
        "logg_apogee",
        "mh_apogee",
        "teff_andrae",
        "logg_andrae",
        "mh_andrae",
        "g_mag",
        "av_sfd",
        "bp_rp",
    }
    frames = []
    for path in (
        DATA_PROCESSED / "pipeline1_features_stream3.parquet",
        DATA_PROCESSED / "pipeline1_features_stream1.parquet",
    ):
        if not path.exists():
            continue
        schema = pq.ParquetFile(path).schema_arrow.names
        cols = [c for c in wanted if c in schema]
        if "source_id" not in cols:
            continue
        t = pq.read_table(path, columns=cols).to_pandas()
        t = t[t["source_id"].isin(source_ids)]
        if len(t):
            frames.append(t)
    if not frames:
        return pd.DataFrame({"source_id": source_ids})
    info = pd.concat(frames, ignore_index=True).drop_duplicates("source_id", keep="first")
    # Collapse APOGEE/Andrae into unified `teff/logg/mh` columns, preferring APOGEE
    for label, suffix in (("teff", "teff"), ("logg", "logg"), ("mh", "mh")):
        ap = f"{suffix}_apogee"
        an = f"{suffix}_andrae"
        if ap in info.columns and an in info.columns:
            info[label] = info[ap].where(info[ap].notna(), info[an])
            info[f"{label}_src"] = np.where(info[ap].notna(), "APOGEE", "Andrae")
        elif ap in info.columns:
            info[label] = info[ap]
            info[f"{label}_src"] = "APOGEE"
        elif an in info.columns:
            info[label] = info[an]
            info[f"{label}_src"] = "Andrae"
    return info


def before_after_sed() -> None:
    print("[stage02] running gaiaxpy.calibrate on ~120 stars for before/after baseline …")
    before, after, sub = _pick_before_after_pairs(n=120, seed=31)
    apply_style()  # gaiaxpy flips text.usetex — restore
    print(f"[stage02]   kept {len(sub)} pairs after NaN filter")

    info = _load_star_info(sub["source_id"].to_numpy())
    sub = sub.merge(info, on="source_id", how="left")

    # normalise each pair to the pre-Ye peak (after can legitimately scale up)
    peak_before = np.nanmax(np.abs(before), axis=1, keepdims=True) + 1e-12
    before_n = before / peak_before
    after_n = after / peak_before

    # Pick 6 stars spanning the metallicity range when possible. Prefer
    # stars with low A_V so the post-Ye SED is not reddened by a large
    # fraction of the dust column.
    rng = np.random.default_rng(7)
    if "av_sfd" in sub.columns:
        low_av = sub["av_sfd"].fillna(99) < 0.5
        if low_av.sum() >= 6:
            sub = sub.loc[low_av].reset_index(drop=True)
            before = before[low_av.to_numpy()]
            after = after[low_av.to_numpy()]
            before_n = before_n[low_av.to_numpy()]
            after_n = after_n[low_av.to_numpy()]
            peak_before = peak_before[low_av.to_numpy()]
    if "mh" in sub.columns and sub["mh"].notna().sum() >= 6:
        order = np.argsort(sub["mh"].fillna(-99).to_numpy())
        picks = np.linspace(0, len(order) - 1, 6).round().astype(int)
        idx_show = order[picks]
    else:
        idx_show = rng.choice(len(sub), size=min(6, len(sub)), replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)

    for ax, i in zip(axes.flat, idx_show):
        ax.plot(
            YE_SAMPLING_NM, before_n[i], color="#888", lw=1.2, label="pre-Ye (gaiaxpy calibrate)"
        )
        ax.plot(
            YE_SAMPLING_NM, after_n[i], color=PALETTE["v11"], lw=1.2, label="post-Ye (+dereddened)"
        )
        ax.set_xlabel(r"$\lambda$ [nm]")
        ax.set_ylabel("flux (pre-Ye peak-normalised)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

        row = sub.iloc[i]
        sid = int(row["source_id"])
        src = row.get("teff_src", "") or ""

        # Line 1: identity + photometry / extinction (non-physical-parameter side)
        line1_bits = [f"source\\_id {sid}"]
        if pd.notna(row.get("g_mag", np.nan)):
            line1_bits.append(rf"$G={row['g_mag']:.2f}$")
        if pd.notna(row.get("av_sfd", np.nan)):
            line1_bits.append(rf"$A_V^{{\rm SFD}}={row['av_sfd']:.2f}$")

        # Line 2: the stellar parameters. \n has to live OUTSIDE r-strings.
        line2_bits = []
        if pd.notna(row.get("teff", np.nan)):
            line2_bits.append(rf"$T_{{\rm eff}}={row['teff']:.0f}\,$K")
        if pd.notna(row.get("logg", np.nan)):
            line2_bits.append(rf"$\log g={row['logg']:.2f}$")
        if pd.notna(row.get("mh", np.nan)):
            line2_bits.append(rf"$[\mathrm{{M/H}}]={row['mh']:+.2f}$")
        line2 = "   ".join(line2_bits)
        if src:
            line2 += f"   (labels: {src})"

        ax.set_title("   ".join(line1_bits) + "\n" + line2, fontsize=9, loc="left")

    fig.suptitle(
        "Ye+2024 NN correction: before / after on 6 Stream-3 delta stars",
        fontsize=12,
        fontweight="bold",
        y=1.00,
    )
    save_fig(fig, OUT / "ye_before_after_sed.png")

    # --- delta-flux distribution -----------------------------------------
    # Use fractional residual Δ/before: the wavelength-dependent correction
    # is a few % at most wavelengths and much larger blueward of 420 nm,
    # so the absolute-flux axis renders the envelopes as "near-zero".
    delta_frac = (after - before) / (np.abs(before) + 1e-12)
    delta_frac = np.clip(delta_frac, -2.0, 2.0)
    p02 = np.nanpercentile(delta_frac, 2, axis=0)
    p16 = np.nanpercentile(delta_frac, 16, axis=0)
    p50 = np.nanpercentile(delta_frac, 50, axis=0)
    p84 = np.nanpercentile(delta_frac, 84, axis=0)
    p98 = np.nanpercentile(delta_frac, 98, axis=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(
        YE_SAMPLING_NM,
        p02,
        p98,
        color="#f0c9a4",
        alpha=0.65,
        edgecolor="#a6560a",
        linewidth=0.7,
        label="2–98%",
    )
    ax.fill_between(
        YE_SAMPLING_NM,
        p16,
        p84,
        color="#d88b4d",
        alpha=0.85,
        edgecolor="#7f2704",
        linewidth=0.9,
        label="16–84%",
    )
    ax.plot(YE_SAMPLING_NM, p50, color="#7f2704", lw=1.8, label="median")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel(r"$\lambda$ [nm]")
    ax.set_ylabel(r"$\Delta\,\mathrm{flux}\,/\,|\mathrm{before}|$  (post $-$ pre, fractional)")
    ax.set_title(
        rf"Ye+2024 correction magnitude vs wavelength "
        rf"(n={len(before)} Stream-3 delta stars)"
    )
    ax.set_ylim(-0.001, 0.001)
    ax.legend(loc="upper right")
    save_fig(fig, OUT / "ye_delta_distribution.png")


def flag_sky_map() -> None:
    xp_cor = pq.read_table(
        DATA_INTERIM / "xp_sampled_corrected.parquet",
        columns=["source_id", "ye2024_flag"],
    ).to_pandas()
    feat = pq.read_table(
        DATA_PROCESSED / "pipeline1_features_stream1.parquet",
        columns=["source_id", "ra_deg", "dec_deg", "g_mag"],
    ).to_pandas()
    merged = xp_cor.merge(feat, on="source_id", how="inner")

    total = len(merged)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": "mollweide"})
    ok = merged[merged["ye2024_flag"] == 0]
    not_ok = merged[merged["ye2024_flag"] != 0]
    for ax, sub, title, col in (
        (
            axes[0],
            ok,
            f"Ye flag = OK  (n={len(ok):,}, {100 * len(ok) / total:.1f}%)",
            PALETTE["ok"],
        ),
        (
            axes[1],
            not_ok,
            f"Ye flag ≠ OK  (n={len(not_ok):,}, {100 * len(not_ok) / total:.1f}%)",
            PALETTE["bad"],
        ),
    ):
        n_plot = min(len(sub), 40_000)
        idx = sample_index(len(sub), n_plot)
        x, y = radec_to_galactic_mollweide(sub["ra_deg"].values[idx], sub["dec_deg"].values[idx])
        ax.scatter(x, y, s=0.4, alpha=0.35, color=col, rasterized=True)
        ax.set_title(title)
        style_galactic_mollweide(ax)

    fig.suptitle(
        "Ye+2024 NN correction outcomes on Stream 1  —  Galactic coordinates",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    save_fig(fig, OUT / "ye_flag_sky_map.png")


def flag_vs_g() -> None:
    xp_cor = pq.read_table(
        DATA_INTERIM / "xp_sampled_corrected.parquet",
        columns=["source_id", "ye2024_flag"],
    ).to_pandas()
    feat = pq.read_table(
        DATA_PROCESSED / "pipeline1_features_stream1.parquet",
        columns=["source_id", "g_mag", "b_deg"],
    ).to_pandas()
    merged = xp_cor.merge(feat, on="source_id", how="inner")
    merged["abs_b"] = np.abs(merged["b_deg"].values)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    g_bins = np.linspace(merged["g_mag"].min(), merged["g_mag"].max(), 40)
    for fv, label, color in [
        (0, "OK", PALETTE["ok"]),
        (1, "NO_SYNTH_PHOT", PALETTE["bad"]),
        (2, "CALIBRATE_FAIL", "#333"),
    ]:
        sub = merged[merged["ye2024_flag"] == fv]
        if len(sub):
            axes[0].hist(
                sub["g_mag"],
                bins=g_bins,
                histtype="step",
                label=f"{label}  (n={len(sub):,})",
                color=color,
                lw=1.4,
            )
    axes[0].set_xlabel(r"$G$ [mag]")
    axes[0].set_ylabel("count")
    axes[0].set_yscale("log")
    axes[0].set_title(r"Ye flag distribution vs $G$-mag")
    axes[0].legend(loc="upper right")

    b_bins = np.linspace(0, 90, 19)
    b_centers = 0.5 * (b_bins[:-1] + b_bins[1:])
    totals, _ = np.histogram(merged["abs_b"], bins=b_bins)
    bad_mask = merged["ye2024_flag"] != 0
    bads, _ = np.histogram(merged.loc[bad_mask, "abs_b"], bins=b_bins)
    rate = np.where(totals > 0, bads / np.maximum(totals, 1), np.nan)
    axes[1].plot(b_centers, 100 * rate, "o-", color=PALETTE["bad"], lw=1.5)
    axes[1].axvline(5, color="#333", lw=0.8, ls="--", label=r"$|b|=5^{\circ}$ (low-$b$)")
    axes[1].set_xlabel(r"$|b|$ [deg]")
    axes[1].set_ylabel(r"Ye flag $\neq$ OK rate  [%]")
    axes[1].set_title("Ye NO\\_SYNTH\\_PHOT rate vs Galactic latitude".replace("\\_", "_"))
    axes[1].legend()

    save_fig(fig, OUT / "ye_flag_vs_g.png")


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    flag_sky_map()
    flag_vs_g()
    before_after_sed()


if __name__ == "__main__":
    main()
