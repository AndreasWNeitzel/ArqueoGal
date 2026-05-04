"""Assemble the Pipeline-1 training feature matrix from Stream 1 + Ye-corrected XP.

Stage A of the two-stage emit pipeline (stage B is
``scripts/emit_stream1_with_hermite.py``). Produces the feature matrix that
matches the frozen 2026-04-18 contract in
``src/arqueogal/xp_abundances/main/DESIGN.md`` — identifiers, APOGEE labels
renamed to the ``*_apogee`` convention, Gaia photometry + astrometry, IR
photometry, multi-column extinction priors, and the Ye+2024 corrected sampled
flux (replaced in stage B with Hermite coefficients + c0-z scalars).

Joins:

- ``data/interim/stream1_apogee_gaia.parquet`` — APOGEE DR19 labels +
  Gaia DR3 astrometry + 2MASS/WISE photometry + Bailer-Jones distances +
  Edenhofer+SFD E(B-V) + Gaia GSP-Phot Av (354,231 rows, 163 cols).
- ``data/interim/xp_sampled_corrected.parquet`` — Ye+2024-corrected
  sampled flux on ``np.geomspace(360, 990, 330)`` nm (Stream 1 ∪ Stream 3).

No XP normalisation happens here (still sampled flux at this stage); the c0
normalisation + z-scoring lands in stage B after Hermite reprojection.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute  # noqa: F401  (import side-effect: registers pa.compute)
from astropy import units as u
from astropy.coordinates import SkyCoord

from arqueogal.data.dust_maps import (
    neighborhood_av_features,
)
from arqueogal.data.provenance import LocalSource, Provenance, write_sidecar

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_pipeline1_features_stream1")

LALLEMENT_CUBE_PATH = Path("data/external/lallement2022/cube_ext.fits.gz")
NEIGHBORHOOD_RADIUS_PC = 75.0
MIN_NEIGHBORS_FOR_MEDIAN = 5
EBV_TO_AV = 3.1  # Standard R_V for diffuse ISM; used to convert Edenhofer E(B-V) → A_V

# Pipeline-1 scope-of-validity: APOGEE-DR19 × Gaia RGB window.
# At 2026-04-18 the emergent intersection of (ASPCAP flag_bad==0) ∩ (SNR>70) ∩
# (Gaia XP available) already lies inside this box — so the explicit cut drops
# zero stars on current data. It is retained as a named build-time cut so a
# future APOGEE DR20 or XP-release rebuild will surface a drifting Teff/logg
# distribution immediately (as a nonzero drop count) instead of weeks into
# training audit. See research_brief §3 / §7 for why the RGB window is the
# scope: [C/N]-age calibration is only valid for Teff 4200-5100 K and
# log g 1-3.5; we bracket this with a slightly looser 4000-5500 K envelope.
RGB_TEFF_MIN_K = 4000.0
RGB_TEFF_MAX_K = 5500.0
RGB_LOGG_MIN = 1.0
RGB_LOGG_MAX = 3.5


# APOGEE column → DESIGN.md canonical name. Applied after merge.
# Covers: Teff, logg (no suffix in source), M/H + α/M + per-element [X/H] (Astra
# ASPCAP names with ``*_atm`` suffix → ``*_apogee`` suffix).
_APOGEE_RENAMES: dict[str, str] = {
    "teff": "teff_apogee",
    "e_teff": "e_teff_apogee",
    "logg": "logg_apogee",
    "e_logg": "e_logg_apogee",
    "m_h_atm": "mh_apogee",
    "e_m_h_atm": "e_mh_apogee",
    "fe_h_atm": "fe_h_apogee",
    "e_fe_h_atm": "e_fe_h_apogee",
    "alpha_m_atm": "alpha_m_apogee",
    "e_alpha_m_atm": "e_alpha_m_apogee",
    "mg_h_atm": "mg_h_apogee",
    "e_mg_h_atm": "e_mg_h_apogee",
    "c_h_atm": "c_h_apogee",
    "e_c_h_atm": "e_c_h_apogee",
    "n_h_atm": "n_h_apogee",
    "e_n_h_atm": "e_n_h_apogee",
    "o_h_atm": "o_h_apogee",
    "e_o_h_atm": "e_o_h_apogee",
    "na_h_atm": "na_h_apogee",
    "e_na_h_atm": "e_na_h_apogee",
    "al_h_atm": "al_h_apogee",
    "e_al_h_atm": "e_al_h_apogee",
    "si_h_atm": "si_h_apogee",
    "e_si_h_atm": "e_si_h_apogee",
    "s_h_atm": "s_h_apogee",
    "e_s_h_atm": "e_s_h_apogee",
    "k_h_atm": "k_h_apogee",
    "e_k_h_atm": "e_k_h_apogee",
    "ca_h_atm": "ca_h_apogee",
    "e_ca_h_atm": "e_ca_h_apogee",
    "ti_h_atm": "ti_h_apogee",
    "e_ti_h_atm": "e_ti_h_apogee",
    "v_h_atm": "v_h_apogee",
    "e_v_h_atm": "e_v_h_apogee",
    "cr_h_atm": "cr_h_apogee",
    "e_cr_h_atm": "e_cr_h_apogee",
    "mn_h_atm": "mn_h_apogee",
    "e_mn_h_atm": "e_mn_h_apogee",
    "ni_h_atm": "ni_h_apogee",
    "e_ni_h_atm": "e_ni_h_apogee",
    "ce_h_atm": "ce_h_apogee",
    "e_ce_h_atm": "e_ce_h_apogee",
}


# Identifier / audit columns (DESIGN §Identifiers & audit).
_IDENT_COLS = [
    "source_id",
    "spectrum_pk",
    "apogee_id",
    "v_astra",
    "snr",
    "ra_deg",
    "dec_deg",
    # n_aspcap_tasks and b_deg are derived below
]

# APOGEE labels on disk in canonical *_apogee order (post-rename).
_APOGEE_LABELS_TIER1 = ["teff_apogee", "logg_apogee", "mh_apogee", "fe_h_apogee"]
_APOGEE_LABELS_TIER2 = ["alpha_m_apogee", "mg_h_apogee", "c_h_apogee", "n_h_apogee"]
_APOGEE_LABELS_TIER3 = [
    "o_h_apogee",
    "na_h_apogee",
    "al_h_apogee",
    "si_h_apogee",
    "s_h_apogee",
    "k_h_apogee",
    "ca_h_apogee",
    "ti_h_apogee",
    "v_h_apogee",
    "cr_h_apogee",
    "mn_h_apogee",
    "ni_h_apogee",
    "ce_h_apogee",
]
_APOGEE_LABELS_ALL = _APOGEE_LABELS_TIER1 + _APOGEE_LABELS_TIER2 + _APOGEE_LABELS_TIER3

# Gaia astrometry & quality (DESIGN §Gaia astrometry & quality).
_GAIA_ASTROMETRY_COLS = ["parallax", "parallax_error", "parallax_corr", "ruwe"]

# Gaia photometry (DESIGN §Gaia photometry). Colours are native columns in the
# source parquet (phot_bp_mean_mag - phot_rp_mean_mag equivalents).
_GAIA_PHOTOMETRY_COLS = ["g_mag", "bp_mag", "rp_mag", "bp_rp", "bp_g", "g_rp"]

# IR photometry — 2MASS + WISE with errors (DESIGN §IR photometry).
_IR_PHOTOMETRY_COLS = [
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
]

# Distance triple (DESIGN §Distance).
_DISTANCE_COLS = ["r_med_photogeo", "r_lo_photogeo", "r_hi_photogeo"]

# Extinction multi-column feature set (DESIGN §Extinction priors).
# av_edenhofer is derived (3.1 × ebv_edenhofer_2023). av_sfd is renamed from
# a_v_sfd in the source. av_lallement is renamed from av_lallement_xcheck
# after the Lallement lookup. av_nbhd_* is computed here.
_EXTINCTION_RAW_COLS = [
    "ebv_edenhofer_2023",
    "e_ebv_edenhofer_2023",
    "ebv_sfd",
    "e_ebv_sfd",
    "ag_gspphot",
    "ag_gspphot_lower",
    "ag_gspphot_upper",
]

# APOGEE quality flags (DESIGN §Flags).
_APOGEE_FLAG_COLS = ["flag_bad", "flag_warn"]


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


def _galactic_b_deg(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Return Galactic latitude (degrees) for an ICRS RA/Dec array."""
    sc = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    return sc.galactic.b.deg.astype(np.float32)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    s1_path = repo / "data" / "interim" / "stream1_apogee_gaia.parquet"
    xp_path = repo / "data" / "interim" / "xp_sampled_corrected.parquet"
    out_path = repo / "data" / "processed" / "pipeline1_features_stream1.parquet"

    for p in (s1_path, xp_path):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    logger.info("loading %s", s1_path)
    s1 = pd.read_parquet(s1_path)
    logger.info("  %d rows × %d cols", len(s1), len(s1.columns))

    # Memory-aware XP load via row-group streaming. The XP parquet has a
    # ``corrected_flux`` list<float32>(330) column; loading 700k+ rows into
    # pandas costs >10 GB of RAM and OOMs on a 9.7 GB box. Stream the
    # parquet row-group at a time, drop non-matching source_ids per group,
    # and accumulate only the survivors. The XP parquet is typically
    # written with row-groups of ~50k rows, so peak transient memory is
    # one row-group (~1 GB with corrected_flux) plus the accumulator
    # (the survivors, ~5 GB total for ~330k matched rows).
    import gc

    import pyarrow.parquet as _pq

    s1_ids_arr = s1["source_id"].to_numpy()
    s1_ids = pa.array(s1_ids_arr)
    s1_ids_set = pa.compute.SetLookupOptions(value_set=s1_ids)

    # CRITICAL: project corrected_flux OUT at parquet read time. The build
    # script does not need it; carrying the list<float32>(330) column into
    # pandas costs ~10 KB / row in Python-list overhead, blowing out memory
    # on the 717k-row XP parquet. emit_stream1_with_hermite.py reads
    # corrected_flux directly from xp_sampled_corrected.parquet via its own
    # row-group streaming filter; build only needs source_id, ye2024_flag,
    # and a_v_sfd here.
    XP_KEEP_COLS = ["source_id", "ye2024_flag", "a_v_sfd"]
    logger.info(
        "streaming %s row-groups, projection=%s, filtering to Stream 1 source_ids",
        xp_path,
        XP_KEEP_COLS,
    )
    xp_pf = _pq.ParquetFile(xp_path)
    n_rg = xp_pf.metadata.num_row_groups
    survivor_batches: list = []
    n_kept = 0
    n_seen = 0
    for rg_idx in range(n_rg):
        rg = xp_pf.read_row_group(rg_idx, columns=XP_KEEP_COLS)
        n_seen += rg.num_rows
        mask = pa.compute.is_in(rg.column("source_id"), options=s1_ids_set)
        kept = rg.filter(mask)
        if kept.num_rows:
            survivor_batches.append(kept)
            n_kept += kept.num_rows
        del rg, mask, kept
        if (rg_idx + 1) % 5 == 0 or rg_idx == n_rg - 1:
            logger.info(
                "  row-group %d/%d: kept %d / %d total",
                rg_idx + 1,
                n_rg,
                n_kept,
                n_seen,
            )
        gc.collect()
    xp_table = pa.concat_tables(survivor_batches)
    del survivor_batches
    gc.collect()
    logger.info("  XP rows matching Stream 1: %d (3-col projection only)", xp_table.num_rows)

    xp = xp_table.to_pandas()
    del xp_table
    gc.collect()
    logger.info("  XP DataFrame: %d rows × %d cols", len(xp), len(xp.columns))

    logger.info("inner-joining Stream 1 × Ye-corrected XP on source_id")
    merged = s1.merge(xp, on="source_id", how="inner")
    del s1, xp
    gc.collect()
    logger.info("  %d rows after join", len(merged))
    n_before = len(merged)

    logger.info("renaming APOGEE labels to *_apogee convention")
    present_renames = {k: v for k, v in _APOGEE_RENAMES.items() if k in merged.columns}
    missing_from_source = [k for k in _APOGEE_RENAMES if k not in merged.columns]
    if missing_from_source:
        raise SystemExit(
            f"source parquet is missing APOGEE columns that the DESIGN contract "
            f"requires: {missing_from_source}"
        )
    merged = merged.rename(columns=present_renames)

    # RGB scope cut on Teff / log g removed 2026-04-29: the OOD flags
    # (ood_mahalanobis_score, ood_disagreement_flag, regime_b_flag,
    # mode_ambiguous_flag) handle evolutionary-stage out-of-distribution stars
    # at inference time. The training set therefore retains the full APOGEE
    # post-quality-cut cohort regardless of Kiel position.
    logger.info(
        "RGB scope cut deprecated; retaining all %d rows post-APOGEE-quality-cut",
        len(merged),
    )

    logger.info("computing derived column: b_deg (Galactic latitude via astropy)")
    merged["b_deg"] = _galactic_b_deg(
        merged["ra_deg"].to_numpy(dtype=np.float64),
        merged["dec_deg"].to_numpy(dtype=np.float64),
    )

    logger.info("computing derived column: n_aspcap_tasks (rows per source_id)")
    tasks_per_star = merged.groupby("source_id").size()
    merged["n_aspcap_tasks"] = merged["source_id"].map(tasks_per_star).astype(np.int32)

    logger.info("computing derived column: av_edenhofer = %.2f × ebv_edenhofer_2023", EBV_TO_AV)
    merged["av_edenhofer"] = (EBV_TO_AV * merged["ebv_edenhofer_2023"]).astype(np.float32)

    logger.info("renaming a_v_sfd → av_sfd (DESIGN extinction contract)")
    if "a_v_sfd" not in merged.columns:
        raise SystemExit("expected `a_v_sfd` column from Stream 1 interim, not found")
    merged = merged.rename(columns={"a_v_sfd": "av_sfd"})

    logger.info(
        "computing §8.3 neighborhood-median A_V (radius=%d pc)",
        int(NEIGHBORHOOD_RADIUS_PC),
    )
    nbhd = neighborhood_av_features(
        ra_deg=merged["ra_deg"].to_numpy().astype(np.float64),
        dec_deg=merged["dec_deg"].to_numpy().astype(np.float64),
        distance_pc=merged["r_med_photogeo"].to_numpy().astype(np.float64),
        ag_gspphot=merged["ag_gspphot"].to_numpy().astype(np.float64),
        radius_pc=NEIGHBORHOOD_RADIUS_PC,
        min_neighbors=MIN_NEIGHBORS_FOR_MEDIAN,
    )
    merged["av_nbhd_median"] = nbhd.av_nbhd_median
    merged["av_nbhd_std"] = nbhd.av_nbhd_std
    merged["n_neighbors_75pc"] = nbhd.n_neighbors
    n_nbhd_valid = int(np.isfinite(nbhd.av_nbhd_median).sum())
    logger.info("  valid nbhd-median: %d / %d", n_nbhd_valid, len(merged))

    lallement_valid: int | None = None
    if LALLEMENT_CUBE_PATH.exists():
        from arqueogal.data.dust_maps import lallement2022_query, load_lallement2022_cube

        logger.info("loading Lallement+2022 cube for A_V feature")
        cube = load_lallement2022_cube(LALLEMENT_CUBE_PATH)
        av_lall = lallement2022_query(
            ra_deg=merged["ra_deg"].to_numpy().astype(np.float64),
            dec_deg=merged["dec_deg"].to_numpy().astype(np.float64),
            distance_pc=merged["r_med_photogeo"].to_numpy().astype(np.float64),
            cube=cube,
        )
        merged["av_lallement"] = av_lall.astype(np.float32)
        lallement_valid = int(np.isfinite(av_lall).sum())
        logger.info("  valid Lallement A_V: %d / %d", lallement_valid, len(merged))
        del cube
    else:
        logger.warning(
            "Lallement cube missing at %s — av_lallement will be all-NaN",
            LALLEMENT_CUBE_PATH,
        )
        merged["av_lallement"] = np.full(len(merged), np.nan, dtype=np.float32)

    # Error columns for APOGEE labels live under `e_*` names after rename.
    _APOGEE_ERROR_COLS = [f"e_{c}" for c in _APOGEE_LABELS_ALL]

    keep = [
        # Identifiers & audit (8 cols)
        *_IDENT_COLS,
        "n_aspcap_tasks",
        "b_deg",
        # APOGEE labels + errors (21 + 21 cols)
        *_APOGEE_LABELS_ALL,
        *_APOGEE_ERROR_COLS,
        # Gaia astrometry & quality
        *_GAIA_ASTROMETRY_COLS,
        # Gaia photometry (Riello+2021 g_mag correction applied upstream)
        *_GAIA_PHOTOMETRY_COLS,
        # IR photometry
        *_IR_PHOTOMETRY_COLS,
        # Distance
        *_DISTANCE_COLS,
        # Extinction priors — multi-column
        "av_edenhofer",
        "av_sfd",
        "av_lallement",
        "av_nbhd_median",
        "av_nbhd_std",
        "n_neighbors_75pc",
        *_EXTINCTION_RAW_COLS,
        # APOGEE flags (Ye + xp_fit flags attached in stage B)
        *_APOGEE_FLAG_COLS,
        # Stage-B auxiliary: teff_gspphot is the Teff source for the residual-
        # flag stratification (p99 thresholds in pre_emit_decisions.json were
        # computed against this column). Not an ML input; retained for audit.
        "teff_gspphot",
        # corrected_flux dropped from this stage 2026-04-29 (memory budget):
        # emit_stream1_with_hermite.py reads it directly from
        # data/interim/xp_sampled_corrected.parquet via row-group streaming.
        "ye2024_flag",
    ]
    missing = [c for c in keep if c not in merged.columns]
    if missing:
        raise SystemExit(f"feature-matrix columns missing from merged DataFrame: {missing}")

    features = merged[keep].copy()
    logger.info("writing %s (%d rows × %d cols)", out_path, len(features), len(features.columns))
    _write_parquet_atomic(features, out_path)
    size_mb = out_path.stat().st_size / 1024**2
    logger.info("  %.1f MB on disk", size_mb)

    prov = Provenance(
        output_file=str(out_path.relative_to(repo)),
        script="scripts/build_pipeline1_features_stream1.py",
        sources=[
            LocalSource(
                name="Stream 1 APOGEE × Gaia DR3 training table",
                path=str(s1_path.relative_to(repo)),
                sha256=_sha256_of(s1_path),
            ),
            LocalSource(
                name="Ye+2024 corrected XP sampled flux (Stream 1 ∪ Stream 3)",
                path=str(xp_path.relative_to(repo)),
                sha256=_sha256_of(xp_path),
            ),
            *(
                [
                    LocalSource(
                        name="Lallement+2022 3D extinction cube",
                        path=str(LALLEMENT_CUBE_PATH),
                        sha256=_sha256_of(LALLEMENT_CUBE_PATH),
                    )
                ]
                if LALLEMENT_CUBE_PATH.exists()
                else []
            ),
        ],
        cuts_applied=[
            "INNER JOIN stream1_apogee_gaia × xp_sampled_corrected on source_id",
            "RGB scope cut: REMOVED 2026-04-29 (OOD flags handle Kiel-outlier stars downstream)",
        ],
        corrections=[
            "Renamed APOGEE columns to *_apogee convention (teff → teff_apogee, "
            "logg → logg_apogee, m_h_atm → mh_apogee, alpha_m_atm → alpha_m_apogee, "
            "and all X_h_atm → X_h_apogee) per DESIGN 2026-04-18 feature contract",
            "Derived b_deg (Galactic latitude) via astropy SkyCoord ICRS → galactic",
            "Derived n_aspcap_tasks = count of rows per source_id (pre-dedup audit handle)",
            f"Derived av_edenhofer = {EBV_TO_AV} × ebv_edenhofer_2023 (standard R_V)",
            "Renamed a_v_sfd → av_sfd",
            "Renamed av_lallement_xcheck → av_lallement (first-class feature, not cross-check)",
            "Derived §8.3 neighborhood-median A_V from ag_gspphot + Gaia 3D "
            f"positions (radius={NEIGHBORHOOD_RADIUS_PC} pc, "
            f"min_neighbors={MIN_NEIGHBORS_FOR_MEDIAN})",
            *(
                ["Computed av_lallement from Lallement+2022 cube via trilinear LOS integration"]
                if LALLEMENT_CUBE_PATH.exists()
                else []
            ),
        ],
        row_count_before=n_before,
        row_count_after=int(len(features)),
        notes=(
            "Pipeline-1 TRAINING feature matrix, stage A of the two-stage emit. "
            "Shipped here: APOGEE DR19 labels (Mészáros+2025-corrected upstream) "
            "renamed to *_apogee, Gaia astrometry (Lindegren+2021 zpt-corrected) "
            "+ photometry (Riello+2021 g-corrected), IR photometry, multi-column "
            "extinction priors, and `corrected_flux` = Ye+2024-corrected sampled "
            "flux on np.geomspace(360, 990, 330) nm. Stage B "
            "(scripts/emit_stream1_with_hermite.py) reprojects `corrected_flux` "
            "onto the v1.0 Hermite basis, normalises coefficients by c0, z-scores "
            "log10(c0), and replaces `corrected_flux` with the 3-tier XP columns "
            "documented in src/arqueogal/xp_abundances/main/DESIGN.md."
        ),
        extra={
            "rgb_cut": {
                "status": "removed_2026_04_29",
                "note": (
                    "Teff / log g Kiel-validity cut dropped 2026-04-29; downstream "
                    "OOD flags (ood_mahalanobis_score, ood_disagreement_flag, "
                    "regime_b_flag, mode_ambiguous_flag) handle evolutionary outliers."
                ),
            },
            "nbhd_radius_pc": float(NEIGHBORHOOD_RADIUS_PC),
            "nbhd_min_neighbors": int(MIN_NEIGHBORS_FOR_MEDIAN),
            "n_nbhd_valid": n_nbhd_valid,
            "n_lallement_valid": lallement_valid,
            "ebv_to_av_R_V": EBV_TO_AV,
            "n_ye2024_ok": int((features["ye2024_flag"] == 0).sum()),
            "n_ye2024_no_synth_phot": int((features["ye2024_flag"] == 1).sum()),
            "n_ye2024_calibrate_fail": int((features["ye2024_flag"] == 2).sum()),
            "n_unique_source_ids": int(features["source_id"].nunique()),
            "n_duplicate_source_ids": int((features["source_id"].value_counts() > 1).sum()),
            "max_aspcap_tasks_per_star": int(features["n_aspcap_tasks"].max()),
            "feature_columns": list(features.columns),
        },
    )
    write_sidecar(prov)
    logger.info("wrote provenance sidecar")


if __name__ == "__main__":
    main()
