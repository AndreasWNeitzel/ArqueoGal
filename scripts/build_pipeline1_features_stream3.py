"""Assemble the Pipeline-1 Stream-3 inference feature matrix (Phase 3b stage A).

Stream-3 analogue of ``build_pipeline1_features_stream1.py`` — no APOGEE labels,
a larger source_id union (existing 168k + delta 449k Ye-OK = 613,939 rows), and
the Andrae+2023 Teff/logg/[M/H] as selection-function inputs (not targets).

Joins (inner on source_id, Ye-OK only):

- ``data/interim/stream3_expansion_union.parquet`` — 622,283 rows: source_id,
  sample tag (uniform / volume_limited), Andrae Teff/logg/[M/H], b_deg, ra/dec,
  BJ21 photogeometric distance (``distance_pc``).
- ``data/interim/stream3_gaia_dr3_corrected.parquet`` (168,099) +
  ``data/interim/stream3_delta_gaia_dr3_corrected.parquet`` (449,625) — AIP
  Gaia DR3 enrichment with Lindegren+2021 zpt + Riello+2021 G-mag applied.
- ``data/raw/ir_photometry/stream3_existing_ir.parquet`` (164,314) +
  ``data/raw/ir_photometry/stream3_delta_ir.parquet`` (449,625) — 2MASS + WISE.
- ``data/interim/xp_sampled_corrected.parquet`` (includes Stream-3 168k
  Ye-passed subset) + ``data/interim/xp_sampled_corrected_delta.parquet``
  (449,625 Ye-OK delta) — Ye+2024 corrected sampled flux.
- ``data/interim/bj21_andrae_chunks/chunk_*.parquet`` — BJ21 photogeometric
  distance triple (r_med/r_lo/r_hi) for the Andrae pool.

Extinction (matches the inference-driver's EXTINCTION_COLS contract):

- ``av_sfd`` — SFD × 2.742 (all stars).
- ``av_lallement`` — Lallement+2022 cube (all stars with r_med_photogeo).
- ``av_edenhofer`` — NaN (cube evicted at 2026-04-18 to stay within the 10 GB
  hard cap; re-downloading would push the repo to ~10.6 GB). The inference
  driver handles NaN-and-flagged extinction via ``aux_missing_any``, and 76%
  of Stream-1 training stars already had NaN ``av_edenhofer`` — so NaN here
  is IN-distribution for the ensemble.
- ``av_nbhd_median`` / ``av_nbhd_std`` — ``ag_gspphot`` K-D-tree median over
  75 pc heliocentric-Cartesian balls (§8.3).

Output: ``data/processed/pipeline1_features_stream3.parquet`` with
``corrected_flux`` (stage A) + all Stream-1 stage-A aux columns. Stage B
(``scripts/emit_stream3_with_hermite.py``) reprojects and writes the 3-tier
XP columns with RAW ``bp_c0_log``/``rp_c0_log`` (NOT z-scored); the inference
driver applies ``apply_frozen_zscore`` using Stream-1 provenance stats.
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
logger = logging.getLogger("build_pipeline1_features_stream3")

LALLEMENT_CUBE_PATH = Path("data/external/lallement2022/cube_ext.fits.gz")
NEIGHBORHOOD_RADIUS_PC = 75.0
MIN_NEIGHBORS_FOR_MEDIAN = 5
SFD_TO_AV = 2.742
COMPOUND_FLOOR = SELECTION_PROB_FLOOR * SELECTION_PROB_FLOOR  # 1e-4 bulk match

_IR_COLS = [
    "j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag",
    "e_j_mag", "e_h_mag", "e_k_mag", "e_w1_mag", "e_w2_mag",
    "ir_missing_flag",
]

_GAIA_KEEP = [
    "source_id",
    "parallax", "parallax_error", "parallax_corr",
    "ra_parallax_corr", "dec_parallax_corr",
    "parallax_pmra_corr", "parallax_pmdec_corr",
    "ruwe",
    "phot_g_mean_mag_corr", "phot_bp_mean_mag", "phot_rp_mean_mag",
    "bp_rp", "bp_g", "g_rp",
    "teff_gspphot",
    "ag_gspphot", "ag_gspphot_lower", "ag_gspphot_upper",
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


def _load_bj21_for_sources(chunk_dir: Path, source_ids: np.ndarray) -> pd.DataFrame:
    files = sorted(chunk_dir.glob("chunk_*.parquet"))
    logger.info("concatenating %d BJ21 chunks", len(files))
    full = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    logger.info("  BJ21 pool: %d rows; filtering to %d requested source_ids",
                len(full), len(source_ids))
    wanted = pd.Index(pd.unique(source_ids.astype("int64")))
    hit = full[full["source_id"].isin(wanted)].copy()
    logger.info("  BJ21 hits: %d / %d", len(hit), len(wanted))
    return hit


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
    ra: np.ndarray, dec: np.ndarray, distance_pc: np.ndarray,
) -> np.ndarray:
    if not LALLEMENT_CUBE_PATH.exists():
        logger.warning("Lallement cube missing at %s — av_lallement all-NaN",
                       LALLEMENT_CUBE_PATH)
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
        b_deg.astype(np.float64), g_mag.astype(np.float64),
    )
    p_ir = score_ir_completeness(
        b_deg.astype(np.float64), g_mag.astype(np.float64),
        teff.astype(np.float64), logg.astype(np.float64),
    )
    p_parallax = (
        np.isfinite(parallax_over_error) & (parallax_over_error >= parallax_snr_min)
    ).astype(np.float64)
    p_extinction = (~av_missing).astype(np.float64)
    product = p_ye * p_ir * p_parallax * p_extinction
    hard_zero = (p_parallax == 0.0) | (p_extinction == 0.0)
    p_compound = np.where(
        hard_zero, 0.0, np.clip(product, COMPOUND_FLOOR, SELECTION_PROB_CEIL),
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
    union_path = repo / "data" / "interim" / "stream3_expansion_union.parquet"
    gaia_existing = repo / "data" / "interim" / "stream3_gaia_dr3_corrected.parquet"
    gaia_delta = repo / "data" / "interim" / "stream3_delta_gaia_dr3_corrected.parquet"
    ir_existing = repo / "data" / "raw" / "ir_photometry" / "stream3_existing_ir.parquet"
    ir_delta = repo / "data" / "raw" / "ir_photometry" / "stream3_delta_ir.parquet"
    xp_existing = repo / "data" / "interim" / "xp_sampled_corrected.parquet"
    xp_delta = repo / "data" / "interim" / "xp_sampled_corrected_delta.parquet"
    ye_ok_existing = repo / "data" / "interim" / "stream3_ye_ok_source_ids.parquet"
    ye_ok_delta = repo / "data" / "interim" / "stream3_delta_ye_ok_source_ids.parquet"
    bj21_dir = repo / "data" / "interim" / "bj21_andrae_chunks"
    out_path = repo / "data" / "processed" / "pipeline1_features_stream3.parquet"

    for p in (union_path, gaia_existing, gaia_delta, ir_existing, ir_delta,
              xp_existing, xp_delta, ye_ok_existing, ye_ok_delta):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    logger.info("loading %s", union_path)
    union = pd.read_parquet(union_path)
    logger.info("  union: %d rows", len(union))
    union = union.rename(columns={"ra": "ra_deg", "dec": "dec_deg"})

    logger.info("loading Ye-OK source_ids (existing + delta)")
    ye_ok_ids_ex = pd.read_parquet(ye_ok_existing)["source_id"].astype("int64")
    ye_ok_ids_de = pd.read_parquet(ye_ok_delta)["source_id"].astype("int64")
    ye_ok_ids = pd.concat([ye_ok_ids_ex, ye_ok_ids_de]).drop_duplicates()
    logger.info("  Ye-OK union: %d (existing %d + delta %d)",
                len(ye_ok_ids), len(ye_ok_ids_ex), len(ye_ok_ids_de))

    logger.info("filtering union to Ye-OK only")
    n_pre_ye = len(union)
    union = union[union["source_id"].isin(ye_ok_ids)].reset_index(drop=True)
    logger.info("  %d rows after Ye-OK filter (dropped %d)",
                len(union), n_pre_ye - len(union))

    logger.info("loading Gaia corrected (existing + delta)")
    g_ex = pd.read_parquet(gaia_existing)[_GAIA_KEEP]
    g_de = pd.read_parquet(gaia_delta)[_GAIA_KEEP]
    logger.info("  existing %d + delta %d = %d", len(g_ex), len(g_de), len(g_ex) + len(g_de))
    gaia = pd.concat([g_ex, g_de], ignore_index=True).drop_duplicates(
        subset="source_id", keep="first",
    )
    gaia = gaia.rename(columns={
        "phot_g_mean_mag_corr": "g_mag",
        "phot_bp_mean_mag": "bp_mag",
        "phot_rp_mean_mag": "rp_mag",
    })
    gaia["parallax_raw"] = gaia["parallax"]
    gaia["parallax"] = gaia["parallax_corr"]
    logger.info("  gaia union rows: %d unique source_ids", len(gaia))

    logger.info("merging union × gaia on source_id (inner)")
    df = union.drop(columns=["g_mag"]).merge(gaia, on="source_id", how="inner")
    logger.info("  %d rows after gaia merge", len(df))

    logger.info("loading IR (existing + delta)")
    ir_ex = pd.read_parquet(ir_existing)[["source_id", *_IR_COLS]]
    ir_de = pd.read_parquet(ir_delta)[["source_id", *_IR_COLS]]
    ir = pd.concat([ir_ex, ir_de], ignore_index=True).drop_duplicates(
        subset="source_id", keep="first",
    )
    logger.info("  IR union rows: %d", len(ir))
    df = df.merge(ir, on="source_id", how="left")
    logger.info("  %d rows after IR left-merge", len(df))

    logger.info("loading Ye-corrected XP sampled flux (existing + delta)")
    xp_ex = pd.read_parquet(xp_existing)
    xp_de = pd.read_parquet(xp_delta)
    xp_ex = xp_ex[xp_ex["ye2024_flag"] == 0]
    xp_de = xp_de[xp_de["ye2024_flag"] == 0]
    xp = pd.concat([xp_ex, xp_de], ignore_index=True).drop_duplicates(
        subset="source_id", keep="first",
    )
    logger.info("  XP Ye-OK union rows: %d", len(xp))
    df = df.merge(xp, on="source_id", how="inner")
    logger.info("  %d rows after XP inner-merge (Ye-OK final)", len(df))

    logger.info("loading BJ21 photogeometric triple for %d source_ids", len(df))
    bj21 = _load_bj21_for_sources(bj21_dir, df["source_id"].to_numpy())
    bj21 = bj21.drop_duplicates(subset="source_id", keep="first")
    logger.info("  BJ21 resolved: %d / %d", len(bj21), len(df))
    df = df.merge(bj21, on="source_id", how="left")
    # Fallback: Phase 3a union carries BJ21 photogeometric distance as
    # distance_pc; use it when the per-chunk lookup missed the source_id
    # (some Andrae rows are absent from the current chunks).
    r_med = df["r_med_photogeo"].to_numpy(dtype=np.float64)
    dist_pc = df["distance_pc"].to_numpy(dtype=np.float64)
    use_fallback = ~np.isfinite(r_med) & np.isfinite(dist_pc)
    n_fill = int(use_fallback.sum())
    if n_fill:
        df.loc[use_fallback, "r_med_photogeo"] = dist_pc[use_fallback].astype(np.float32)
        # r_lo / r_hi left NaN; inference treats them as aux-missing.
        logger.info("  filled %d r_med_photogeo entries from union distance_pc",
                    n_fill)

    ra = df["ra_deg"].to_numpy(dtype=np.float64)
    dec = df["dec_deg"].to_numpy(dtype=np.float64)
    distance = df["r_med_photogeo"].to_numpy(dtype=np.float64)
    distance_for_lallement = np.where(
        np.isfinite(distance), distance, df["distance_pc"].to_numpy(dtype=np.float64),
    )

    logger.info("computing av_sfd via SFDQuery for %d stars", len(df))
    df["av_sfd"] = _query_sfd(ra, dec)
    if "a_v_sfd" in df.columns:
        df = df.drop(columns=["a_v_sfd"])

    logger.info("computing av_lallement via Lallement+2022 cube")
    df["av_lallement"] = _query_lallement(ra, dec, distance_for_lallement)
    n_lall = int(np.isfinite(df["av_lallement"]).sum())
    logger.info("  valid av_lallement: %d / %d", n_lall, len(df))

    logger.info("av_edenhofer: NaN (cube evicted 2026-04-18; nan_to_num at inference)")
    df["av_edenhofer"] = np.full(len(df), np.nan, dtype=np.float32)

    logger.info("computing §8.3 nbhd-median A_V (radius=%d pc)",
                int(NEIGHBORHOOD_RADIUS_PC))
    nbhd = neighborhood_av_features(
        ra_deg=ra, dec_deg=dec,
        distance_pc=distance_for_lallement,
        ag_gspphot=df["ag_gspphot"].to_numpy(dtype=np.float64),
        radius_pc=NEIGHBORHOOD_RADIUS_PC,
        min_neighbors=MIN_NEIGHBORS_FOR_MEDIAN,
    )
    df["av_nbhd_median"] = nbhd.av_nbhd_median
    df["av_nbhd_std"] = nbhd.av_nbhd_std
    df["n_neighbors_75pc"] = nbhd.n_neighbors
    logger.info("  valid nbhd-median: %d / %d",
                int(np.isfinite(nbhd.av_nbhd_median).sum()), len(df))

    logger.info("computing compound selection_prob v1.1")
    plx = df["parallax"].to_numpy(dtype=np.float64)
    plx_err = df["parallax_error"].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        plx_snr = np.where(
            np.isfinite(plx) & np.isfinite(plx_err) & (plx_err > 0.0),
            plx / plx_err, np.nan,
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
    logger.info("  selection_prob: min=%.4f mean=%.4f max=%.4f",
                float(df["selection_prob"].min()),
                float(df["selection_prob"].mean()),
                float(df["selection_prob"].max()))

    keep = [
        # Identifiers / audit
        "source_id", "sample", "ra_deg", "dec_deg", "b_deg",
        # Andrae Teff/logg/[M/H] (selection inputs, not targets)
        "teff_andrae", "logg_andrae", "mh_andrae",
        # Gaia photometry (Riello-corrected g, raw bp/rp)
        "g_mag", "bp_mag", "rp_mag", "bp_rp", "bp_g", "g_rp",
        # Gaia astrometry (Lindegren-corrected parallax)
        "parallax", "parallax_error", "parallax_corr", "parallax_raw", "ruwe",
        # BJ21 photogeometric triple
        "r_med_photogeo", "r_lo_photogeo", "r_hi_photogeo",
        "distance_pc",
        # IR photometry
        "j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag",
        "e_j_mag", "e_h_mag", "e_k_mag", "e_w1_mag", "e_w2_mag",
        "ir_missing_flag",
        # Extinction priors
        "av_edenhofer", "av_sfd", "av_lallement",
        "av_nbhd_median", "av_nbhd_std", "n_neighbors_75pc",
        "ag_gspphot", "ag_gspphot_lower", "ag_gspphot_upper",
        # Aux: teff_gspphot for stage-B Hermite flag stratification
        "teff_gspphot",
        # Selection function
        "selection_prob", "p_ye_retained", "p_ir_complete",
        "p_parallax", "p_extinction",
        # Sampled flux — replaced by Hermite coefficients in stage B
        "corrected_flux", "ye2024_flag",
    ]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise SystemExit(f"columns missing from assembled DataFrame: {missing}")
    features = df[keep].copy()

    sample_counts = features["sample"].value_counts().to_dict()
    logger.info("sample composition: %s", sample_counts)
    logger.info("writing %s (%d rows × %d cols)",
                out_path, len(features), len(features.columns))
    _write_parquet_atomic(features, out_path)
    size_mb = out_path.stat().st_size / 1024**2
    logger.info("  %.1f MB on disk", size_mb)

    sources = [
        LocalSource(
            name="Stream 3 Option-C expansion union (Phase 3a)",
            path=str(union_path.relative_to(repo)),
            sha256=_sha256_of(union_path),
        ),
        LocalSource(
            name="Stream 3 Gaia DR3 corrected (existing 168k)",
            path=str(gaia_existing.relative_to(repo)),
            sha256=_sha256_of(gaia_existing),
        ),
        LocalSource(
            name="Stream 3 Gaia DR3 corrected (delta 449k)",
            path=str(gaia_delta.relative_to(repo)),
            sha256=_sha256_of(gaia_delta),
        ),
        LocalSource(
            name="Stream 3 IR 2MASS+AllWISE (existing 164k)",
            path=str(ir_existing.relative_to(repo)),
            sha256=_sha256_of(ir_existing),
        ),
        LocalSource(
            name="Stream 3 IR 2MASS+AllWISE (delta 449k)",
            path=str(ir_delta.relative_to(repo)),
            sha256=_sha256_of(ir_delta),
        ),
        LocalSource(
            name="Ye+2024 corrected XP sampled flux (existing)",
            path=str(xp_existing.relative_to(repo)),
            sha256=_sha256_of(xp_existing),
        ),
        LocalSource(
            name="Ye+2024 corrected XP sampled flux (delta 449k)",
            path=str(xp_delta.relative_to(repo)),
            sha256=_sha256_of(xp_delta),
        ),
        LocalSource(
            name="BJ21 photogeometric distance chunks (Andrae pool)",
            path=str(bj21_dir.relative_to(repo)),
            sha256=None,
        ),
    ]
    if LALLEMENT_CUBE_PATH.exists():
        sources.append(LocalSource(
            name="Lallement+2022 3D extinction cube",
            path=str(LALLEMENT_CUBE_PATH),
            sha256=_sha256_of(LALLEMENT_CUBE_PATH),
        ))

    prov = Provenance(
        output_file=str(out_path.relative_to(repo)),
        script="scripts/build_pipeline1_features_stream3.py",
        sources=sources,
        cuts_applied=[
            "Ye-OK filter: union × stream3_ye_ok_source_ids ∪ stream3_delta_ye_ok_source_ids",
            "INNER JOIN expansion_union × gaia_dr3_corrected × xp_sampled_corrected",
            "LEFT JOIN IR photometry (existing + delta)",
            "LEFT JOIN BJ21 photogeometric triple from Andrae pool chunks",
        ],
        corrections=[
            "Lindegren+2021 parallax zero-point and Riello+2021 G-mag applied upstream",
            f"av_sfd = {SFD_TO_AV} × E(B-V)_SFD (Schlafly & Finkbeiner 2011)",
            "av_lallement from Lallement+2022 cube via trilinear LOS integration",
            "av_edenhofer = NaN (cube evicted 2026-04-18 to stay within 10 GB hard cap; "
            "76% of Stream-1 training also had NaN av_edenhofer so this is IN-distribution)",
            f"av_nbhd_median via 3D K-D-tree on ag_gspphot within {NEIGHBORHOOD_RADIUS_PC} pc",
            "selection_prob via compound SF v1.1 (Ye × IR × parallax gate × extinction gate)",
        ],
        row_count_before=int(len(union)),
        row_count_after=int(len(features)),
        notes=(
            "Pipeline-1 INFERENCE feature matrix, stage A of the two-stage emit "
            "(Phase 3b). No APOGEE labels. Andrae+2023 Teff/logg/[M/H] are "
            "selection-function inputs, not prediction targets. `corrected_flux` "
            "(Ye+2024 sampled flux) is reprojected to Hermite coefficients in "
            "stage B (scripts/emit_stream3_with_hermite.py) with raw (non-z-scored) "
            "bp_c0_log/rp_c0_log; the inference driver applies apply_frozen_zscore "
            "from Stream-1 provenance stats. Total Ye-OK union = 164,314 existing "
            "+ 449,625 delta = 613,939 rows spanning uniform and volume_limited arms."
        ),
        extra={
            "sample_composition": sample_counts,
            "n_ye_ok_existing": int(len(ye_ok_ids_ex)),
            "n_ye_ok_delta": int(len(ye_ok_ids_de)),
            "n_ye_ok_union": int(len(ye_ok_ids)),
            "n_rows_after_gaia_merge": int(len(features)),
            "nbhd_radius_pc": float(NEIGHBORHOOD_RADIUS_PC),
            "nbhd_min_neighbors": int(MIN_NEIGHBORS_FOR_MEDIAN),
            "sfd_to_av_coeff": SFD_TO_AV,
            "av_edenhofer_policy": "NaN (disk budget); inference nan_to_num",
            "feature_columns": list(features.columns),
            "phase": "Phase 3b stage A",
        },
    )
    write_sidecar(prov)
    logger.info("wrote provenance sidecar")


if __name__ == "__main__":
    main()
