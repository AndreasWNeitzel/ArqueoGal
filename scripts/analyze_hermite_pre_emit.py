"""Three pre-re-emit analyses on the full 315 616 Ye-OK stars.

Motivation (from smoke-test review, 2026-04-18):

1.  The global normal-population residual p99 = 6.5e-13 has a hidden
    Teff dependence. Hot stars (Teff > 6000 K) fit worse by ~14x due to
    sharp Balmer / Paschen features not captured by a 55-mode Hermite
    basis centred at 510 / 825 nm. Applying the global threshold would
    flag hot stars disproportionately, conflating "fit failed" with
    "star is hot". We want a Teff-stratified threshold so the flag has
    unambiguous meaning: "this star's fit is worse than 99% of other
    stars at similar Teff".

2.  The 2.3% catastrophic-Ye rate is close to the 2.6% NO_SYNTH_PHOT
    rate. Are these the same stars, different stars, or partially
    overlapping? Structurally they cannot overlap (flag=1 rows carry
    NaN flux, so the Hermite residual is undefined), but their stellar
    populations may or may not overlap — that's the interesting
    comparison.

3.  The smoke-test 110-D PCA explains only 29.2%/19.2% in PC1/PC2, much
    lower than the 50-70% typical of XP-abundance latent spaces. Two
    hypotheses: (a) Hermite orthonormalisation spreads independent
    physical information across orthogonal modes, (b) the ~65 noise-
    dominated modes (n > 20 BP, n > 22 RP) inflate total variance and
    dilute the physical PCs' fractions. Running PCA on the 43-D noise-
    truncated vector (n=0..19 BP + n=0..22 RP) discriminates.

Outputs under ``reports/figures/hermite_smoke/pre_emit/``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from arqueogal.data.gaia_xp import (
    HERMITE_N_BASIS,
    HERMITE_REPROJECTION_VERSION,
    YE2024_N_OUTPUT,
    reproject_ye_to_hermite,
)
from arqueogal.utils.plotting import (
    AA_DOUBLE_COLUMN_IN,
    WONG_PALETTE,
    save_figure,
    set_aa_style,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("analyze_hermite_pre_emit")

YE_CATASTROPHIC_RESIDUAL = 1e-10
SFD_TO_AV = 2.742

# Teff bin edges (K). NaN / Teff < first edge both fall into the "no_teff" bin.
TEFF_BIN_EDGES: tuple[float, ...] = (4000.0, 4500.0, 5000.0, 5500.0, 6000.0)
TEFF_BIN_LABELS: tuple[str, ...] = (
    "lt_4000", "4000_4500", "4500_5000", "5000_5500", "5500_6000", "ge_6000",
)

# Noise-floor truncation from the smoke test on 1490 stars.
N_BP_INFORMATIVE = 20   # keep modes 0..19
N_RP_INFORMATIVE = 23   # keep modes 0..22


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/pipeline1_features_stream1.parquet"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/figures/hermite_smoke/pre_emit"),
    )
    p.add_argument("--chunk-size", type=int, default=50_000)
    p.add_argument(
        "--pca-sample-size", type=int, default=50_000,
        help="PCA subsample size for speed (full 315k is unneeded for PC directions).",
    )
    p.add_argument("--seed", type=int, default=20260418)
    return p.parse_args()


def _teff_bin(teff: np.ndarray) -> np.ndarray:
    """Return integer bin index; label index into ``TEFF_BIN_LABELS``."""
    idx = np.digitize(teff, TEFF_BIN_EDGES)  # 0..len(edges)
    idx[np.isnan(teff)] = -1  # sentinel for "no Teff" → will be mapped separately
    return idx


def _reproject_full(
    df: pd.DataFrame, chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproject every flag=0 row in chunks.

    Returns
    -------
    residuals : np.ndarray (N,)
    bp_coeffs : np.ndarray (N, 55)
    rp_coeffs : np.ndarray (N, 55)
    """
    n = len(df)
    residuals = np.empty(n, dtype=np.float32)
    bp_c = np.empty((n, HERMITE_N_BASIS), dtype=np.float32)
    rp_c = np.empty((n, HERMITE_N_BASIS), dtype=np.float32)
    for lo in range(0, n, chunk_size):
        hi = min(lo + chunk_size, n)
        logger.info("reprojecting rows [%d, %d) / %d", lo, hi, n)
        flux = np.stack(
            [np.asarray(f, dtype=np.float32) for f in df["corrected_flux"].iloc[lo:hi]]
        )
        assert flux.shape == (hi - lo, YE2024_N_OUTPUT)
        out = reproject_ye_to_hermite(flux)
        residuals[lo:hi] = out["reprojection_residual_rms"]
        bp_c[lo:hi] = out["bp_coeffs"]
        rp_c[lo:hi] = out["rp_coeffs"]
    return residuals, bp_c, rp_c


def _teff_stratified_thresholds(
    residuals: np.ndarray, teff: np.ndarray,
) -> dict:
    """p99 on normal (residual < catastrophic) per Teff bin + global fallback."""
    normal = residuals < YE_CATASTROPHIC_RESIDUAL
    bins = _teff_bin(teff)

    per_bin: dict[str, dict] = {}
    for i, label in enumerate(TEFF_BIN_LABELS):
        mask = (bins == i) & normal
        sub = residuals[mask]
        if sub.size == 0:
            per_bin[label] = {
                "n": 0, "p50": None, "p95": None, "p99": None, "p99_9": None,
            }
            continue
        per_bin[label] = {
            "n": int(sub.size),
            "n_catastrophic_in_bin": int(((bins == i) & ~normal).sum()),
            "p50": float(np.percentile(sub, 50)),
            "p95": float(np.percentile(sub, 95)),
            "p99": float(np.percentile(sub, 99)),
            "p99_9": float(np.percentile(sub, 99.9)),
        }

    no_teff_mask = (bins == -1) & normal
    no_teff = residuals[no_teff_mask]
    if no_teff.size:
        per_bin["no_teff"] = {
            "n": int(no_teff.size),
            "n_catastrophic_in_bin": int(((bins == -1) & ~normal).sum()),
            "p50": float(np.percentile(no_teff, 50)),
            "p95": float(np.percentile(no_teff, 95)),
            "p99": float(np.percentile(no_teff, 99)),
            "p99_9": float(np.percentile(no_teff, 99.9)),
        }
    else:
        per_bin["no_teff"] = {"n": 0}

    global_p99 = float(np.percentile(residuals[normal], 99))
    return {
        "per_teff_bin": per_bin,
        "global_p99_normal": global_p99,
        "teff_bin_edges_K": list(TEFF_BIN_EDGES),
        "teff_bin_labels": list(TEFF_BIN_LABELS) + ["no_teff"],
        "catastrophic_threshold": YE_CATASTROPHIC_RESIDUAL,
    }


def _contingency(
    df_full: pd.DataFrame, residuals_flag0: np.ndarray,
) -> dict:
    """2x2 table of (NO_SYNTH_PHOT) × (Hermite catastrophic) with population overlay.

    Uses the FULL feature frame (all 324 054 rows) by positional index — we
    pass the flag=0 residuals only and leave flag=1 rows with NaN residual
    (no Hermite fit possible).
    """
    flag = df_full["ye2024_flag"].to_numpy()
    resid_full = np.full(len(df_full), np.nan, dtype=np.float32)
    resid_full[flag == 0] = residuals_flag0

    no_synth = flag == 1                             # pre-Hermite Ye failure
    hermite_cat = (flag == 0) & (resid_full >= YE_CATASTROPHIC_RESIDUAL)
    normal = (flag == 0) & (resid_full < YE_CATASTROPHIC_RESIDUAL)

    table = {
        "no_synth_phot": int(no_synth.sum()),
        "hermite_catastrophic": int(hermite_cat.sum()),
        "both_by_flag": 0,  # structurally zero: flag=1 rows have NaN flux
        "normal": int(normal.sum()),
        "total": int(len(df_full)),
        "hermite_catastrophic_rate_of_flag0": float(
            hermite_cat.sum() / max(1, (flag == 0).sum())
        ),
    }

    # Population overlay: are NO_SYNTH_PHOT and HERMITE_CAT similar in (Teff, G, [Fe/H])?
    def _pop_stats(mask: np.ndarray) -> dict:
        sub = df_full.loc[mask]
        return {
            "n": int(mask.sum()),
            "teff_median": (
                float(np.nanmedian(sub["teff_gspphot"])) if mask.sum() else None
            ),
            "g_median": float(np.nanmedian(sub["g_mag"])) if mask.sum() else None,
            "fe_h_median": (
                float(np.nanmedian(sub["fe_h_atm"])) if mask.sum() else None
            ),
            "av_sfd_median": (
                float(np.nanmedian(SFD_TO_AV * sub["ebv_sfd"]))
                if mask.sum() else None
            ),
        }

    return {
        "counts": table,
        "population_stats": {
            "no_synth_phot": _pop_stats(no_synth),
            "hermite_catastrophic": _pop_stats(hermite_cat),
            "normal": _pop_stats(normal),
        },
    }


def _plot_population_overlay(
    df_full: pd.DataFrame, residuals_flag0: np.ndarray, out_path: Path,
) -> None:
    set_aa_style()
    flag = df_full["ye2024_flag"].to_numpy()
    resid_full = np.full(len(df_full), np.nan, dtype=np.float32)
    resid_full[flag == 0] = residuals_flag0
    no_synth = flag == 1
    hermite_cat = (flag == 0) & (resid_full >= YE_CATASTROPHIC_RESIDUAL)
    normal = (flag == 0) & (resid_full < YE_CATASTROPHIC_RESIDUAL)

    fig, axes = plt.subplots(1, 4, figsize=(AA_DOUBLE_COLUMN_IN, 3.0))
    cols = [
        ("teff_gspphot", "Teff (K)", (3500, 8000)),
        ("g_mag", "G (mag)", (6, 17.7)),
        ("fe_h_atm", "[Fe/H]", (-2.2, 0.6)),
        ("ebv_sfd", r"$A_V$ (SFD)", (0, 5)),
    ]
    for ax, (col, label, xlim) in zip(axes, cols):
        data_norm = df_full.loc[normal, col].to_numpy(dtype=np.float64)
        data_ns = df_full.loc[no_synth, col].to_numpy(dtype=np.float64)
        data_hc = df_full.loc[hermite_cat, col].to_numpy(dtype=np.float64)
        if col == "ebv_sfd":
            data_norm = data_norm * SFD_TO_AV
            data_ns = data_ns * SFD_TO_AV
            data_hc = data_hc * SFD_TO_AV
        bins = np.linspace(*xlim, 40)
        ax.hist(
            np.clip(data_norm, *xlim), bins=bins, density=True,
            histtype="step", color="0.4", label=f"normal ({normal.sum()})", lw=1.2,
        )
        ax.hist(
            np.clip(data_ns, *xlim), bins=bins, density=True,
            histtype="step", color=WONG_PALETTE[2],
            label=f"NO_SYNTH_PHOT ({no_synth.sum()})", lw=1.2,
        )
        ax.hist(
            np.clip(data_hc, *xlim), bins=bins, density=True,
            histtype="step", color=WONG_PALETTE[1],
            label=f"Hermite cat ({hermite_cat.sum()})", lw=1.2,
        )
        ax.set_xlabel(label)
        ax.set_xlim(*xlim)
        if ax is axes[0]:
            ax.set_ylabel("normalised density")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(
        "Failure-population overlay: NO_SYNTH_PHOT vs Hermite-catastrophic vs normal"
    )
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def _pca_compare(
    bp_c: np.ndarray, rp_c: np.ndarray, residuals: np.ndarray,
    sample_size: int, seed: int,
) -> dict:
    """Compare PC1/PC2 explained variance for 110-D vs 43-D truncated coefficient vectors."""
    keep = residuals < YE_CATASTROPHIC_RESIDUAL
    bp_k = bp_c[keep]
    rp_k = rp_c[keep]
    rng = np.random.default_rng(seed)
    n_keep = bp_k.shape[0]
    if n_keep > sample_size:
        pick = rng.choice(n_keep, size=sample_size, replace=False)
        bp_k = bp_k[pick]
        rp_k = rp_k[pick]
    full_110 = np.concatenate([bp_k, rp_k], axis=1)
    trunc_43 = np.concatenate(
        [bp_k[:, :N_BP_INFORMATIVE], rp_k[:, :N_RP_INFORMATIVE]], axis=1,
    )

    def _robust_pca(X: np.ndarray) -> np.ndarray:
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0)
        sd = 1.4826 * mad
        sd[sd < 1e-30] = 1.0
        Xz = np.clip((X - med) / sd, -8.0, 8.0)
        return PCA(n_components=5).fit(Xz).explained_variance_ratio_

    ev_110 = _robust_pca(full_110)
    ev_43 = _robust_pca(trunc_43)
    return {
        "n_used": int(bp_k.shape[0]),
        "full_110": {f"pc{i + 1}": float(v) for i, v in enumerate(ev_110)},
        "truncated_43": {
            "n_bp_modes": N_BP_INFORMATIVE,
            "n_rp_modes": N_RP_INFORMATIVE,
            "dim": N_BP_INFORMATIVE + N_RP_INFORMATIVE,
            **{f"pc{i + 1}": float(v) for i, v in enumerate(ev_43)},
        },
    }


def _write_summary(out_dir: Path, thresholds: dict, contingency: dict, pca: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON for programmatic consumption by the re-emit script.
    (out_dir / "pre_emit_decisions.json").write_text(
        json.dumps(
            {
                "basis_version": HERMITE_REPROJECTION_VERSION,
                "thresholds": thresholds,
                "contingency": contingency,
                "pca_comparison": pca,
            },
            indent=2,
        ),
    )

    lines: list[str] = [
        "# Hermite re-projection — pre-re-emit decisions",
        "",
        "Generated by `scripts/analyze_hermite_pre_emit.py` on the FULL 315 616 "
        "Ye-OK rows of `pipeline1_features_stream1.parquet`.",
        "",
        "## 1. Per-Teff-bin RESIDUAL_HIGH thresholds",
        "",
        f"Global normal-pop p99 (fallback for no_teff rows): "
        f"**{thresholds['global_p99_normal']:.3e}**",
        "",
        "| Teff bin (K) | N | catastrophic-in-bin | p50 | p95 | p99 | p99.9 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in list(TEFF_BIN_LABELS) + ["no_teff"]:
        stats = thresholds["per_teff_bin"][label]
        if stats.get("n", 0) == 0:
            lines.append(f"| `{label}` | 0 | — | — | — | — | — |")
            continue
        lines.append(
            f"| `{label}` | {stats['n']} | "
            f"{stats.get('n_catastrophic_in_bin', 0)} | "
            f"{stats['p50']:.3e} | {stats['p95']:.3e} | "
            f"**{stats['p99']:.3e}** | {stats['p99_9']:.3e} |"
        )

    lines += [
        "",
        "The p99 column is the per-bin `XP_FIT_FLAG_RESIDUAL_HIGH_FOR_TEFF_BIN` "
        "threshold. Stars without GSP-Phot Teff fall back to the global p99. ",
        "The global flag `XP_FIT_FLAG_RESIDUAL_HIGH_GLOBAL` "
        f"(threshold {thresholds['global_p99_normal']:.3e}) is retained as an "
        "auxiliary diagnostic column — it is what downstream users get if they "
        "do not stratify.",
        "",
        "## 2. NO_SYNTH_PHOT × Hermite-catastrophic contingency",
        "",
        "| | Hermite normal | Hermite catastrophic | No Hermite fit (NaN flux) |",
        "|---|---:|---:|---:|",
    ]
    c = contingency["counts"]
    lines += [
        f"| Ye flag=0 (OK) | {c['normal']} | {c['hermite_catastrophic']} | 0 |",
        f"| Ye flag=1 (NO_SYNTH_PHOT) | 0 | 0 | {c['no_synth_phot']} |",
        "",
        "**Structural observation:** the overlap cell is zero by construction — "
        "flag=1 rows carry NaN flux, so the Hermite residual is undefined. The "
        "two failure flags address disjoint stages of the pipeline. The relevant "
        "question is whether the two failure modes sample the same stellar "
        "population; the overlay figure below answers that.",
        "",
        "### Population medians (Teff, G, [Fe/H], A_V_SFD)",
        "",
        "| group | N | Teff | G | [Fe/H] | A_V_SFD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group in ("normal", "no_synth_phot", "hermite_catastrophic"):
        p = contingency["population_stats"][group]
        teff = f"{p['teff_median']:.0f}" if p["teff_median"] is not None else "—"
        g = f"{p['g_median']:.2f}" if p["g_median"] is not None else "—"
        feh = f"{p['fe_h_median']:.2f}" if p["fe_h_median"] is not None else "—"
        av = f"{p['av_sfd_median']:.2f}" if p["av_sfd_median"] is not None else "—"
        lines.append(f"| `{group}` | {p['n']} | {teff} | {g} | {feh} | {av} |")

    lines += [
        "",
        "See `failure_population_overlay.png` for the full densities.",
        "",
        "## 3. 110-D vs 43-D PCA (noise-floor truncation test)",
        "",
        f"PCA on {pca['n_used']} normal-population rows "
        f"(catastrophic rows dropped, robust standardisation, ±8σ winsorised).",
        "",
        "| component | full 110-D | truncated 43-D (BP[0:20] + RP[0:23]) |",
        "|---|---:|---:|",
    ]
    for i in range(1, 6):
        a = pca["full_110"][f"pc{i}"]
        b = pca["truncated_43"][f"pc{i}"]
        lines.append(f"| PC{i} | {a:.2%} | {b:.2%} |")

    ev110_12 = pca["full_110"]["pc1"] + pca["full_110"]["pc2"]
    ev43_12 = pca["truncated_43"]["pc1"] + pca["truncated_43"]["pc2"]
    lines += [
        "",
        f"- Sum PC1+PC2: **{ev110_12:.1%}** (110-D) vs **{ev43_12:.1%}** (43-D).",
        "",
        "**Interpretation:** if the 43-D PC1 fraction is substantially higher "
        "(say ≥50%), noise dilution is real and supports truncation at ML input "
        "time. If it stays within a few percent of the 110-D PC1, the Hermite "
        "basis genuinely spreads stellar physics across many orthogonal "
        "directions and the 29% PC1 is intrinsic.",
        "",
        "See `pca_compare.png`.",
        "",
    ]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines))
    logger.info("Wrote %s", out_dir / "SUMMARY.md")


def _plot_pca_compare(
    bp_c: np.ndarray, rp_c: np.ndarray, residuals: np.ndarray,
    df: pd.DataFrame, sample_size: int, seed: int, out_path: Path,
) -> None:
    set_aa_style()
    keep = residuals < YE_CATASTROPHIC_RESIDUAL
    idx_keep = np.where(keep)[0]
    rng = np.random.default_rng(seed)
    if idx_keep.size > sample_size:
        pick = rng.choice(idx_keep, size=sample_size, replace=False)
    else:
        pick = idx_keep
    bp_k = bp_c[pick]
    rp_k = rp_c[pick]
    teff = df["teff_gspphot"].to_numpy()[pick]

    full = np.concatenate([bp_k, rp_k], axis=1)
    trunc = np.concatenate(
        [bp_k[:, :N_BP_INFORMATIVE], rp_k[:, :N_RP_INFORMATIVE]], axis=1,
    )

    def _robust(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0)
        sd = 1.4826 * mad
        sd[sd < 1e-30] = 1.0
        Xz = np.clip((X - med) / sd, -8.0, 8.0)
        pca = PCA(n_components=5).fit(Xz)
        return pca.transform(Xz), pca.explained_variance_ratio_

    pcs_full, ev_full = _robust(full)
    pcs_trunc, ev_trunc = _robust(trunc)

    fig, axes = plt.subplots(1, 2, figsize=(AA_DOUBLE_COLUMN_IN, 4.0))
    for ax, (pcs, ev, title) in zip(
        axes,
        [(pcs_full, ev_full, "110-D full basis"),
         (pcs_trunc, ev_trunc, f"{N_BP_INFORMATIVE + N_RP_INFORMATIVE}-D truncated")],
    ):
        sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=teff, s=3, cmap="inferno", linewidths=0)
        ax.set_xlabel(f"PC1 ({ev[0]:.1%})")
        ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
        ax.set_title(title)
        fig.colorbar(sc, ax=ax, shrink=0.7, label="Teff (K)")
    fig.suptitle(
        "Noise-floor truncation test — does PC1 fraction jump when noise modes are removed?"
    )
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cols = [
        "source_id", "corrected_flux", "ye2024_flag",
        "g_mag", "teff_gspphot", "logg_gspphot",
        "fe_h_atm", "ebv_sfd",
    ]
    df_full = pd.read_parquet(args.features, columns=cols)
    logger.info("Loaded %d total rows", len(df_full))
    df_ok = df_full[df_full["ye2024_flag"] == 0].reset_index(drop=True)
    logger.info("Ye-OK (flag=0) rows: %d", len(df_ok))

    residuals, bp_c, rp_c = _reproject_full(df_ok, args.chunk_size)

    teff_ok = df_ok["teff_gspphot"].to_numpy(dtype=np.float64)
    thresholds = _teff_stratified_thresholds(residuals, teff_ok)
    contingency = _contingency(df_full, residuals)
    pca_stats = _pca_compare(bp_c, rp_c, residuals, args.pca_sample_size, args.seed)

    _plot_population_overlay(
        df_full, residuals, args.out_dir / "failure_population_overlay.png",
    )
    _plot_pca_compare(
        bp_c, rp_c, residuals, df_ok,
        args.pca_sample_size, args.seed, args.out_dir / "pca_compare.png",
    )
    _write_summary(args.out_dir, thresholds, contingency, pca_stats)
    logger.info("Done. Outputs under %s", args.out_dir)


if __name__ == "__main__":
    main()
