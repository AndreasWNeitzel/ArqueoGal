"""Release-side derivative artefact builders.

Per `reports/data_preparation_output.md`, the D-Cat-b release should ship not
just the full Parquet but also focused subsets that match common galactic-
archaeology consumer queries. Without these, every consumer reimplements the
same filter / aggregate code and the catalog feels inaccessible.

Five derivative artefacts:

1. **HRD-ready**: identifiers + (G, BP-RP, M_G, Teff, logg, [M/H]) for thin/thick
   disk plots. Drops everything not needed for a colour-magnitude or HR diagram.
2. **Kinematic-ready**: identifiers + astrometry + abundances + per-star tiers
   for thick-disk / halo-substructure analyses.
3. **Tier-1-only full**: subset of the full catalog filtered to ``release_tier == 1``
   and ``xp_abundance_type__<element> == 'spectrum_dominant'`` for high-confidence
   per-star use. The conservative-default filter that the methods paper recommends.
4. **Per-cell summary**: aggregated statistics (mean residual, σ, count) per
   regime cell. For methods-paper figures.
5. **Per-magnitude reliability**: by g_mag_bin × element, what's the coverage
   and bias.

Plus partitioning: when the full Parquet is large (>500MB), partition by
``g_mag_bin`` (3 subdirs) for efficient consumer batch reads.

This module is the library. The CLI wrapper lives in ``scripts/build_release_artefacts.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Per-element prediction column conventions, kept consistent with release.py.
_PRED_COLS = ("teff_pred", "logg_pred", "mh_pred", "alpha_m_pred", "mg_h_pred")
_SIGMA_COLS = ("teff_sigma", "logg_sigma", "mh_sigma", "alpha_m_sigma", "mg_h_sigma")
_ABUNDANCE_ELEMENTS = ("teff", "logg", "mh", "alpha_m", "mg_h")
_AUX_ASSISTED_ELEMENTS = ("alpha_m", "mg_h")

_HRD_COLS = (
    "source_id",
    "ra",
    "dec",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "parallax",
    "parallax_error",
    "teff_pred",
    "teff_sigma",
    "logg_pred",
    "logg_sigma",
    "mh_pred",
    "mh_sigma",
    "release_tier",
    "release_tier__teff",
    "release_tier__logg",
    "release_tier__mh",
    "g_mag_bin",
)


_KINEMATIC_COLS = (
    "source_id",
    "ra",
    "dec",
    "parallax",
    "parallax_error",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "phot_g_mean_mag",
    "teff_pred",
    "teff_sigma",
    "logg_pred",
    "logg_sigma",
    "mh_pred",
    "mh_sigma",
    "alpha_m_pred",
    "alpha_m_sigma",
    "mg_h_pred",
    "mg_h_sigma",
    "release_tier",
    "release_tier__teff",
    "release_tier__logg",
    "release_tier__mh",
    "release_tier__alpha_m",
    "release_tier__mg_h",
    "kin_ood_flag",
    "dist_prior_dominated",
    "g_mag_bin",
    "xp_abundance_type__alpha_m",
    "xp_abundance_type__mg_h",
)


def _project_columns(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    """Return a copy of ``df`` keeping only columns in ``cols`` that actually exist."""
    keep = [c for c in cols if c in df.columns]
    return df[keep].copy()


def build_hrd_ready_subset(
    parquet_path: Path,
    output_path: Path,
    *,
    tier_filter: int | None = 1,
) -> dict[str, int | str]:
    """HRD-ready subset of a release Parquet.

    Filters to the columns relevant for HR-diagram and chemical-bimodality
    plots, optionally restricted to ``release_tier <= tier_filter``.
    """
    df = pd.read_parquet(parquet_path)
    if tier_filter is not None and "release_tier" in df.columns:
        df = df[df["release_tier"] <= tier_filter].copy()
    out = _project_columns(df, _HRD_COLS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path)
    return {
        "artefact": "hrd_ready",
        "input": str(parquet_path),
        "output": str(output_path),
        "n_rows": int(len(out)),
        "n_cols": int(out.shape[1]),
        "tier_filter": "all" if tier_filter is None else f"<={tier_filter}",
    }


def build_kinematic_ready_subset(
    parquet_path: Path,
    output_path: Path,
    *,
    tier_filter: int | None = 2,
) -> dict[str, int | str]:
    """Kinematic-ready subset for galactic-archaeology consumers.

    Includes per-element tiers, kin_ood_flag, and dist_prior_dominated so
    consumers can filter on the disc-kinematics-prior boundaries when
    using aux-assisted abundances.
    """
    df = pd.read_parquet(parquet_path)
    if tier_filter is not None and "release_tier" in df.columns:
        df = df[df["release_tier"] <= tier_filter].copy()
    out = _project_columns(df, _KINEMATIC_COLS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path)
    return {
        "artefact": "kinematic_ready",
        "input": str(parquet_path),
        "output": str(output_path),
        "n_rows": int(len(out)),
        "n_cols": int(out.shape[1]),
        "tier_filter": "all" if tier_filter is None else f"<={tier_filter}",
    }


def build_tier1_only_subset(
    parquet_path: Path,
    output_path: Path,
    *,
    drop_aux_assisted_columns: bool = True,
) -> dict[str, int | str | bool | list[str]]:
    """Conservative Tier-1-only subset (spectrum-dominant per-star release).

    Filters rows to ``release_tier == 1`` (strictest composite tier). When
    ``drop_aux_assisted_columns`` is True (default), additionally drops the
    prediction, sigma, per-element-tier, and abundance-type columns for every
    aux-assisted element ([α/M] and [Mg/H]) — the consumer using this subset
    will not see aux-prior-driven values at all. The subset becomes the
    conservative "spectrum-dominant per-star" release.

    Earlier behaviour (deprecated): the function row-filtered on
    ``xp_abundance_type__<element> == "spectrum_dominant"`` for every element,
    which always returned 0 rows because [α/M] and [Mg/H] are always tagged
    ``aux_assisted``. The smoke test caught this; the new column-dropping
    behaviour preserves usable rows while still excluding aux-assisted output.
    """
    df = pd.read_parquet(parquet_path)
    if "release_tier" in df.columns:
        df = df[df["release_tier"] == 1].copy()

    cols_dropped: list[str] = []
    if drop_aux_assisted_columns:
        drop_patterns = []
        for elem in _AUX_ASSISTED_ELEMENTS:
            drop_patterns.extend(
                [
                    f"{elem}_pred",
                    f"{elem}_sigma",
                    f"release_tier__{elem}",
                    f"xp_abundance_type__{elem}",
                ]
            )
        for col in drop_patterns:
            if col in df.columns:
                df = df.drop(columns=col)
                cols_dropped.append(col)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    return {
        "artefact": "tier1_only_full",
        "input": str(parquet_path),
        "output": str(output_path),
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "drop_aux_assisted_columns": bool(drop_aux_assisted_columns),
        "columns_dropped": cols_dropped,
    }


def build_per_cell_summary(
    parquet_path: Path,
    output_path: Path,
    *,
    cell_columns: tuple[str, ...] = ("g_mag_bin",),
) -> dict[str, int | str | list[str]]:
    """Aggregated per-cell summary table for methods-paper figures.

    For each (cell) group defined by ``cell_columns`` × element, computes:
    - count
    - mean prediction
    - mean sigma
    - per-tier count (Tier 1, 2, 3)
    - fraction with kin_ood_flag = True
    - fraction with dist_prior_dominated = True

    Default ``cell_columns = ("g_mag_bin",)`` produces one row per
    magnitude bin. Pass a longer tuple (e.g. ``("g_mag_bin", "regime_b_flag")``)
    for finer stratification.
    """
    df = pd.read_parquet(parquet_path)

    available_cell_cols = [c for c in cell_columns if c in df.columns]
    if not available_cell_cols:
        # Fallback: a single-row summary.
        df = df.assign(_all="all")
        available_cell_cols = ["_all"]

    summary_rows: list[dict[str, object]] = []
    for cell_values, sub in df.groupby(list(available_cell_cols), dropna=False):
        # cell_values may be a tuple or scalar depending on len(available_cell_cols).
        if not isinstance(cell_values, tuple):
            cell_values = (cell_values,)
        row: dict[str, object] = {}
        for k, v in zip(available_cell_cols, cell_values, strict=False):
            row[k] = v
        row["n_rows"] = int(len(sub))
        # Per-element aggregates.
        for elem, pred_col, sigma_col in zip(
            _ABUNDANCE_ELEMENTS, _PRED_COLS, _SIGMA_COLS, strict=False
        ):
            if pred_col in sub.columns:
                row[f"mean_{elem}_pred"] = float(np.nanmean(sub[pred_col]))
            if sigma_col in sub.columns:
                row[f"mean_{elem}_sigma"] = float(np.nanmean(sub[sigma_col]))
            tier_col = f"release_tier__{elem}"
            if tier_col in sub.columns:
                tiers = sub[tier_col].astype("Int8")
                for t in (1, 2, 3):
                    row[f"n_tier{t}_{elem}"] = int((tiers == t).sum())
        # Per-row flag fractions.
        if "kin_ood_flag" in sub.columns:
            row["frac_kin_ood"] = float(sub["kin_ood_flag"].fillna(False).mean())
        if "dist_prior_dominated" in sub.columns:
            row["frac_dist_prior_dominated"] = float(
                sub["dist_prior_dominated"].fillna(False).mean()
            )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_parquet(output_path)
    return {
        "artefact": "per_cell_summary",
        "input": str(parquet_path),
        "output": str(output_path),
        "n_rows": int(len(summary_df)),
        "cell_columns": list(available_cell_cols),
    }


def build_per_magnitude_reliability(
    parquet_path: Path,
    output_path: Path,
) -> dict[str, int | str]:
    """Per-magnitude × element reliability table.

    For each (g_mag_bin × element) combination, computes:
    - n_tier1, n_tier2, n_tier3
    - mean_predicted_sigma
    - frac_aux_assisted (for elements that have xp_abundance_type__<element>)

    This is the per-magnitude reliability table the methods paper uses to
    document magnitude-dependent calibration quality.
    """
    df = pd.read_parquet(parquet_path)

    if "g_mag_bin" not in df.columns:
        # Without the binning column, fall back to a single global row.
        df = df.assign(g_mag_bin="all")

    rows: list[dict[str, object]] = []
    for (mag_bin,), sub in df.groupby(["g_mag_bin"], dropna=False):
        for elem, pred_col, sigma_col in zip(
            _ABUNDANCE_ELEMENTS, _PRED_COLS, _SIGMA_COLS, strict=False
        ):
            row: dict[str, object] = {
                "g_mag_bin": mag_bin,
                "element": elem,
                "n_total": int(len(sub)),
            }
            tier_col = f"release_tier__{elem}"
            if tier_col in sub.columns:
                tiers = sub[tier_col].astype("Int8")
                for t in (1, 2, 3):
                    row[f"n_tier{t}"] = int((tiers == t).sum())
            if sigma_col in sub.columns:
                row["mean_sigma"] = float(np.nanmean(sub[sigma_col]))
            if pred_col in sub.columns:
                row["frac_predicted"] = float(sub[pred_col].notna().mean())
            type_col = f"xp_abundance_type__{elem}"
            if type_col in sub.columns:
                row["frac_aux_assisted"] = float(
                    (sub[type_col] == "aux_assisted").fillna(False).mean(),
                )
            rows.append(row)

    out_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path)
    return {
        "artefact": "per_magnitude_reliability",
        "input": str(parquet_path),
        "output": str(output_path),
        "n_rows": int(len(out_df)),
    }


def partition_by_g_mag_bin(
    parquet_path: Path,
    output_dir: Path,
    *,
    compression: str = "zstd",
    compression_level: int = 10,
    row_group_size: int = 25_000,
) -> dict[str, int | str | dict[str, int]]:
    """Partition a release Parquet by ``g_mag_bin`` (3 subdirs: bright, mid, faint).

    Pyarrow native partitioned write: each partition becomes its own file
    under ``output_dir/g_mag_bin=<value>/part-0.parquet``. Consumers can
    use ``pyarrow.parquet.read_table(output_dir, filters=[...])`` for
    predicate pushdown to read only the relevant partition.

    Defaults:
    - zstd compression at level 10 (better ratio than snappy; modern
      decompression speed is excellent).
    - 25k-row row groups (matches consumer batch sizes; allows partial reads).

    For D-Cat-d (~20M rows) the same approach extends to HEALPix Nside=7
    spatial partitioning; see data_preparation_output.md for the roadmap.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(
            "Partitioning requires pyarrow. Install via uv add pyarrow.",
        ) from e

    df = pd.read_parquet(parquet_path)
    if "g_mag_bin" not in df.columns:
        raise ValueError(
            "input parquet has no g_mag_bin column; run release.annotate_parquet first.",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=str(output_dir),
        partition_cols=["g_mag_bin"],
        compression=compression,
        compression_level=compression_level,
        row_group_size=row_group_size,
        existing_data_behavior="overwrite_or_ignore",
    )

    # Count rows per partition for the manifest.
    counts: dict[str, int] = {}
    for mag_bin, sub in df.groupby("g_mag_bin", dropna=False):
        counts[str(mag_bin)] = int(len(sub))

    manifest = {
        "artefact": "partitioned_by_g_mag_bin",
        "input": str(parquet_path),
        "output_dir": str(output_dir),
        "compression": compression,
        "compression_level": int(compression_level),
        "row_group_size": int(row_group_size),
        "partition_counts": counts,
        "n_rows_total": int(len(df)),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


__all__ = [
    "build_hrd_ready_subset",
    "build_kinematic_ready_subset",
    "build_per_cell_summary",
    "build_per_magnitude_reliability",
    "build_tier1_only_subset",
    "partition_by_g_mag_bin",
]
