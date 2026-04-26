"""Assemble the Pipeline-1 Stream-2 inference feature matrix.

Stream-2 analogue of ``build_pipeline1_features_stream3.py`` — Hon+21 TESS ×
TIC v8.2 × Gaia DR3 asteroseismic giants (158k rows). No APOGEE labels.
GSP-Phot Teff/logg/[M/H] are selection-function inputs (NOT prediction
targets) — analogous to Andrae's role in Stream 3.

Joins (all on source_id):

- ``data/interim/stream2_tess_gaia.parquet`` — 158,450 rows. Already carries
  Lindegren+2021 parallax zpt (``parallax_corr``) and Riello+2021 G-mag
  (``phot_g_mean_mag_corr``).  Provides GSP-Phot/GSP-Spec stellar parameters,
  Hon+21 ν_max, GSP-Phot distance.
- ``data/raw/ir_photometry/stream2_ir.parquet`` — 2MASS + AllWISE join
  (158k rows, 95% IR-complete).
- ``data/interim/xp_sampled_corrected_stream2.parquet`` — Ye+2024 corrected
  XP sampled flux (Stream-2 delta only). Existing S1/S3 corrected XP files
  also queried for sources that overlap.

Distance / extinction stack:

- ``r_med_photogeo`` ← ``distance_gspphot`` (Stream 2 has no BJ21 chunk;
  GSP-Phot distance is reasonable for asteroseismic giants in the LV).
  ``r_lo_photogeo`` and ``r_hi_photogeo`` derived from
  ``distance_gspphot_lower``/``_upper``.
- ``av_sfd`` — SFD × 2.742 (all stars).
- ``av_lallement`` — Lallement+2022 cube (all stars with finite distance).
- ``av_edenhofer`` — NaN (cube evicted, identical policy to Stream 3).
- ``av_nbhd_median`` / ``av_nbhd_std`` — ``ag_gspphot`` K-D-tree median over
  75 pc heliocentric-Cartesian balls.

Output: ``data/processed/pipeline1_features_stream2.parquet`` — schema
matches Stream 3 features parquet so the inference driver and downstream
release-pipeline code can ingest it without branching on stream tag.

Selection cut applied LAST (matches Stream 1/3 RGB+RC validity per
Mészáros+2025): log g ∈ [0, 3.8] dex AND Teff ∈ [3500, 6500] K, using
GSP-Phot values. Stars outside the cut are dropped.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from arqueogal.data.dust_maps import neighborhood_av_features
from arqueogal.data.provenance import LocalSource, Provenance, write_sidecar
from arqueogal.data.selection_function import (
    SELECTION_PROB_CEIL,
    SELECTION_PROB_FLOOR,
    score_ir_completeness,
    score_selection_prob,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_pipeline1_features_stream2")

LALLEMENT_CUBE_PATH = Path("data/external/lallement2022/cube_ext.fits.gz")
NEIGHBORHOOD_RADIUS_PC = 75.0
MIN_NEIGHBORS_FOR_MEDIAN = 5
SFD_TO_AV = 2.742
COMPOUND_FLOOR = SELECTION_PROB_FLOOR * SELECTION_PROB_FLOOR

KIEL_TEFF = (3500.0, 6500.0)
KIEL_LOGG = (0.0, 3.8)

_IR_COLS = [
    "j_mag",
    "h_mag",
    "k_mag",
    "w1_mag",
    "w2_mag",
    "e_j_mag",
    "e_h_mag",
    "e_k_mag",
    "e_w1_mag",
    "e_w2_mag",
    "ir_missing_flag",
]


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _query_sfd(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    from dustmaps.config import config as dm_config

    repo = Path(__file__).resolve().parents[1]
    dm_config["data_dir"] = str(repo / "data" / "external" / "dustmaps")
    from dustmaps.sfd import SFDQuery

    sfd = SFDQuery()
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    ebv = np.asarray(sfd(coords), dtype=np.float64)
    return (ebv * SFD_TO_AV).astype(np.float32)


def _query_lallement(
    ra: np.ndarray,
    dec: np.ndarray,
    distance_pc: np.ndarray,
) -> np.ndarray:
    if not LALLEMENT_CUBE_PATH.exists():
        logger.warning("Lallement cube missing at %s — av_lallement all-NaN", LALLEMENT_CUBE_PATH)
        return np.full(len(ra), np.nan, dtype=np.float32)
    from arqueogal.data.dust_maps import lallement2022_query, load_lallement2022_cube

    cube = load_lallement2022_cube(LALLEMENT_CUBE_PATH)
    av = lallement2022_query(
        ra_deg=ra.astype(np.float64),
        dec_deg=dec.astype(np.float64),
        distance_pc=distance_pc.astype(np.float64),
        cube=cube,
    )
    del cube
    return av.astype(np.float32)


def _compute_selection_prob(
    b_deg: np.ndarray,
    g_mag: np.ndarray,
    teff: np.ndarray,
    logg: np.ndarray,
    parallax_over_error: np.ndarray,
    av_missing: np.ndarray,
    parallax_snr_min: float = 5.0,
) -> dict[str, np.ndarray]:
    p_ye = score_selection_prob(
        b_deg.astype(np.float64),
        g_mag.astype(np.float64),
    )
    p_ir = score_ir_completeness(
        b_deg.astype(np.float64),
        g_mag.astype(np.float64),
        teff.astype(np.float64),
        logg.astype(np.float64),
    )
    p_parallax = (
        np.isfinite(parallax_over_error) & (parallax_over_error >= parallax_snr_min)
    ).astype(np.float64)
    p_extinction = (~av_missing).astype(np.float64)
    product = p_ye * p_ir * p_parallax * p_extinction
    hard_zero = (p_parallax == 0.0) | (p_extinction == 0.0)
    p_compound = np.where(
        hard_zero,
        0.0,
        np.clip(product, COMPOUND_FLOOR, SELECTION_PROB_CEIL),
    )
    return {
        "p_ye_retained": p_ye.astype(np.float32),
        "p_ir_complete": p_ir.astype(np.float32),
        "p_parallax": p_parallax.astype(np.float32),
        "p_extinction": p_extinction.astype(np.float32),
        "selection_prob": p_compound.astype(np.float32),
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    s2_path = repo / "data" / "interim" / "stream2_tess_gaia.parquet"
    ir_path = repo / "data" / "raw" / "ir_photometry" / "stream2_ir.parquet"
    xp_s2 = repo / "data" / "interim" / "xp_sampled_corrected_stream2.parquet"
    xp_existing = repo / "data" / "interim" / "xp_sampled_corrected.parquet"
    xp_delta = repo / "data" / "interim" / "xp_sampled_corrected_delta.parquet"
    out_path = repo / "data" / "processed" / "pipeline1_features_stream2.parquet"

    for p in (s2_path, ir_path):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    logger.info("loading Stream 2 Gaia: %s", s2_path)
    s2 = pd.read_parquet(s2_path)
    logger.info("  %d rows", len(s2))

    # Compute Galactic latitude.
    coords = SkyCoord(
        ra=s2["ra"].to_numpy() * u.deg, dec=s2["dec"].to_numpy() * u.deg, frame="icrs"
    ).galactic
    s2["b_deg"] = coords.b.deg.astype(np.float32)

    # Rename Gaia columns to match Stream 3 schema.
    s2 = s2.rename(
        columns={
            "ra": "ra_deg",
            "dec": "dec_deg",
            "phot_g_mean_mag_corr": "g_mag",
            "phot_bp_mean_mag": "bp_mag",
            "phot_rp_mean_mag": "rp_mag",
            "teff_gspphot": "teff_gspphot_orig",
            "logg_gspphot": "logg_gspphot_orig",
            "mh_gspphot": "mh_gspphot_orig",
        }
    )
    s2["parallax_raw"] = s2["parallax"]
    s2["parallax"] = s2["parallax_corr"]
    s2["sample"] = "asteroseismic"

    # Use GSP-Phot teff/logg/mh as selection-function inputs; mirror the
    # Stream-3 ``teff_andrae`` / ``logg_andrae`` / ``mh_andrae`` schema.
    s2["teff_andrae"] = s2["teff_gspphot_orig"]
    s2["logg_andrae"] = s2["logg_gspphot_orig"]
    s2["mh_andrae"] = s2["mh_gspphot_orig"]
    s2["teff_gspphot"] = s2["teff_gspphot_orig"]
    s2["ag_gspphot_lower"] = s2.get("ag_gspphot_lower", pd.Series(np.nan, index=s2.index))
    s2["ag_gspphot_upper"] = s2.get("ag_gspphot_upper", pd.Series(np.nan, index=s2.index))

    # GSP-Phot distance triple → BJ21-style triple (triple slot used by
    # downstream release pipeline).
    s2["r_med_photogeo"] = s2["distance_gspphot"].astype(np.float32)
    s2["r_lo_photogeo"] = s2.get(
        "distance_gspphot_lower", pd.Series(np.nan, index=s2.index)
    ).astype(np.float32)
    s2["r_hi_photogeo"] = s2.get(
        "distance_gspphot_upper", pd.Series(np.nan, index=s2.index)
    ).astype(np.float32)
    s2["distance_pc"] = s2["distance_gspphot"].astype(np.float32)

    df = s2

    # Apply RGB+RC validity cut (Mészáros+2025).
    n_pre_cut = len(df)
    teff_a = df["teff_andrae"].to_numpy()
    logg_a = df["logg_andrae"].to_numpy()
    in_kiel = (
        np.isfinite(teff_a)
        & np.isfinite(logg_a)
        & (teff_a >= KIEL_TEFF[0])
        & (teff_a <= KIEL_TEFF[1])
        & (logg_a >= KIEL_LOGG[0])
        & (logg_a <= KIEL_LOGG[1])
    )
    df = df[in_kiel].reset_index(drop=True)
    logger.info(
        "RGB+RC cut: %d / %d retained (dropped %d)", len(df), n_pre_cut, n_pre_cut - len(df)
    )

    logger.info("loading IR: %s", ir_path)
    ir = pd.read_parquet(ir_path)[["source_id", *_IR_COLS]]
    df = df.merge(ir, on="source_id", how="left")
    logger.info("  IR-complete rows: %d / %d", int((~df["ir_missing_flag"]).sum()), len(df))

    # XP sampled flux: union of stream2-only delta + S1 + S3 corrected
    # (some S2 source_ids overlap with the existing XP corrected files).
    logger.info("loading Ye-corrected XP")
    xp_parts = []
    for p in (xp_s2, xp_existing, xp_delta):
        if p.exists():
            df_xp = pd.read_parquet(p)
            df_xp = df_xp[df_xp["ye2024_flag"] == 0]
            xp_parts.append(df_xp)
            logger.info("  %s: %d Ye-OK rows", p.name, len(df_xp))
    xp = pd.concat(xp_parts, ignore_index=True).drop_duplicates(subset="source_id", keep="first")
    logger.info("  XP Ye-OK union: %d", len(xp))

    df = df.merge(xp, on="source_id", how="inner")
    logger.info("  %d rows after XP inner-merge", len(df))

    # Coords for dust maps.
    ra = df["ra_deg"].to_numpy(dtype=np.float64)
    dec = df["dec_deg"].to_numpy(dtype=np.float64)
    distance = df["r_med_photogeo"].to_numpy(dtype=np.float64)

    logger.info("computing av_sfd")
    df["av_sfd"] = _query_sfd(ra, dec)
    if "a_v_sfd" in df.columns:
        df = df.drop(columns=["a_v_sfd"])

    logger.info("computing av_lallement")
    df["av_lallement"] = _query_lallement(ra, dec, distance)
    n_lall = int(np.isfinite(df["av_lallement"]).sum())
    logger.info("  valid av_lallement: %d / %d", n_lall, len(df))

    logger.info("av_edenhofer: NaN (cube evicted; same policy as Stream 3)")
    df["av_edenhofer"] = np.full(len(df), np.nan, dtype=np.float32)

    logger.info("computing nbhd-median A_V")
    nbhd = neighborhood_av_features(
        ra_deg=ra,
        dec_deg=dec,
        distance_pc=distance,
        ag_gspphot=df["ag_gspphot"].to_numpy(dtype=np.float64),
        radius_pc=NEIGHBORHOOD_RADIUS_PC,
        min_neighbors=MIN_NEIGHBORS_FOR_MEDIAN,
    )
    df["av_nbhd_median"] = nbhd.av_nbhd_median
    df["av_nbhd_std"] = nbhd.av_nbhd_std
    df["n_neighbors_75pc"] = nbhd.n_neighbors

    logger.info("computing compound selection_prob")
    plx = df["parallax"].to_numpy(dtype=np.float64)
    plx_err = df["parallax_error"].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        plx_snr = np.where(
            np.isfinite(plx) & np.isfinite(plx_err) & (plx_err > 0.0),
            plx / plx_err,
            np.nan,
        )
    av_missing = (
        ~np.isfinite(df["av_sfd"].to_numpy())
        & ~np.isfinite(df["av_lallement"].to_numpy())
        & ~np.isfinite(df["av_edenhofer"].to_numpy())
    )
    sel = _compute_selection_prob(
        b_deg=df["b_deg"].to_numpy(dtype=np.float64),
        g_mag=df["g_mag"].to_numpy(dtype=np.float64),
        teff=df["teff_andrae"].to_numpy(dtype=np.float64),
        logg=df["logg_andrae"].to_numpy(dtype=np.float64),
        parallax_over_error=plx_snr,
        av_missing=av_missing,
    )
    for k, v in sel.items():
        df[k] = v
    logger.info(
        "  selection_prob: min=%.4f mean=%.4f max=%.4f",
        float(df["selection_prob"].min()),
        float(df["selection_prob"].mean()),
        float(df["selection_prob"].max()),
    )

    keep = [
        "source_id",
        "sample",
        "ra_deg",
        "dec_deg",
        "b_deg",
        "teff_andrae",
        "logg_andrae",
        "mh_andrae",
        "g_mag",
        "bp_mag",
        "rp_mag",
        "bp_rp",
        "bp_g",
        "g_rp",
        "parallax",
        "parallax_error",
        "parallax_corr",
        "parallax_raw",
        "ruwe",
        "r_med_photogeo",
        "r_lo_photogeo",
        "r_hi_photogeo",
        "distance_pc",
        "j_mag",
        "h_mag",
        "k_mag",
        "w1_mag",
        "w2_mag",
        "e_j_mag",
        "e_h_mag",
        "e_k_mag",
        "e_w1_mag",
        "e_w2_mag",
        "ir_missing_flag",
        "av_edenhofer",
        "av_sfd",
        "av_lallement",
        "av_nbhd_median",
        "av_nbhd_std",
        "n_neighbors_75pc",
        "ag_gspphot",
        "ag_gspphot_lower",
        "ag_gspphot_upper",
        "teff_gspphot",
        "selection_prob",
        "p_ye_retained",
        "p_ir_complete",
        "p_parallax",
        "p_extinction",
        "corrected_flux",
        "ye2024_flag",
    ]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise SystemExit(f"columns missing from assembled DataFrame: {missing}")
    features = df[keep].copy()

    logger.info("writing %s (%d rows × %d cols)", out_path, len(features), len(features.columns))
    _write_parquet_atomic(features, out_path)
    size_mb = out_path.stat().st_size / 1024**2
    logger.info("  %.1f MB on disk", size_mb)

    sources = [
        LocalSource(
            name="Stream 2 TESS × Gaia DR3 corrected",
            path=str(s2_path.relative_to(repo)),
            sha256=_sha256_of(s2_path),
        ),
        LocalSource(
            name="Stream 2 IR 2MASS+AllWISE",
            path=str(ir_path.relative_to(repo)),
            sha256=_sha256_of(ir_path),
        ),
    ]
    for p in (xp_s2, xp_existing, xp_delta):
        if p.exists():
            sources.append(
                LocalSource(
                    name=f"Ye+2024 corrected XP ({p.name})",
                    path=str(p.relative_to(repo)),
                    sha256=_sha256_of(p),
                )
            )
    if LALLEMENT_CUBE_PATH.exists():
        sources.append(
            LocalSource(
                name="Lallement+2022 3D extinction cube",
                path=str(LALLEMENT_CUBE_PATH),
                sha256=_sha256_of(LALLEMENT_CUBE_PATH),
            )
        )

    prov = Provenance(
        output_file=str(out_path.relative_to(repo)),
        script="scripts/build_pipeline1_features_stream2.py",
        sources=sources,
        cuts_applied=[
            "RGB+RC validity (Mészáros+2025): logg ∈ [0, 3.8] dex, Teff ∈ [3500, 6500] K",
            "INNER JOIN stream2_tess_gaia × xp_sampled_corrected (Ye-OK)",
            "LEFT JOIN IR photometry",
        ],
        corrections=[
            "Lindegren+2021 parallax zero-point and Riello+2021 G-mag applied upstream",
            f"av_sfd = {SFD_TO_AV} × E(B-V)_SFD",
            "av_lallement from Lallement+2022 cube via trilinear LOS integration",
            "av_edenhofer = NaN (cube evicted; same policy as Stream 3)",
            f"av_nbhd_median via 3D K-D-tree on ag_gspphot within {NEIGHBORHOOD_RADIUS_PC} pc",
            "selection_prob via compound SF v1.1",
        ],
        row_count_before=int(len(s2)),
        row_count_after=int(len(features)),
        notes=(
            "Pipeline-1 INFERENCE feature matrix for Stream 2 (Hon+21 TESS × "
            "Gaia DR3 asteroseismic giants). No APOGEE labels. GSP-Phot "
            "Teff/logg/[M/H] populate the teff_andrae/logg_andrae/mh_andrae "
            "slots as selection-function inputs (NOT prediction targets). "
            "GSP-Phot distance populates r_med_photogeo as a BJ21 substitute. "
            "`corrected_flux` (Ye+2024 sampled flux) is reprojected to "
            "Hermite coefficients in stage B (scripts/emit_stream2_with_hermite.py)."
        ),
        extra={
            "n_rows_pre_kiel_cut": int(len(s2)),
            "kiel_teff_range": list(KIEL_TEFF),
            "kiel_logg_range": list(KIEL_LOGG),
            "feature_columns": list(features.columns),
            "phase": "Phase 3b stage A (Stream 2)",
        },
    )
    write_sidecar(prov)
    logger.info("wrote provenance sidecar")


if __name__ == "__main__":
    main()
