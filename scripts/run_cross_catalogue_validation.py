"""Run §3.3 Test 6 — cross-catalogue consistency validation.

Driver that takes the ArqueoGal Pipeline-1 release parquet plus a set of
already-cross-matched reference catalogues and emits

- a long-form per-cell statistics table (CSV + JSON);
- the seven diagnostic-plot families from
  :mod:`arqueogal.xp_abundances.main.cross_catalogue_plots`;
- a methods-paper-ready pass/fail summary printed to stdout and stored in
  the JSON sidecar.

Reference-catalogue cross-matching is **out of scope** for this driver
because each catalogue (AspGap, SHBoost, Guiglion+2024, Andrae+2023,
Zhang+2023, Fallows-Sanders 2024, GALAH DR4) has its own source-id
convention. The expected workflow is:

    # 1. Cross-match externally (pyvo / TAP / VizieR), produce parquet
    # files where each row maps 1:1 to a row of the release parquet,
    # carrying the external columns + a ``source_id`` column for sanity.
    # 2. Run this driver:
    PYTHONPATH=src python scripts/run_cross_catalogue_validation.py \\
        --release release/D-Cat-b/predictions_with_features.parquet \\
        --catalogue aspgap=external/aspgap_xmatch.parquet \\
        --catalogue shboost=external/shboost_xmatch.parquet \\
        --binding configs/main/cross_catalogue_bindings.yaml \\
        --out reports/cross_catalogue/

The bindings YAML maps each catalogue tag to its ``column_for`` and
``sigma_for`` mappings; an example file is shipped at
``configs/main/cross_catalogue_bindings.yaml``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import yaml

from arqueogal.xp_abundances.main.cross_catalogue import (
    CatalogueBinding,
    compute_cross_catalogue_report,
    matched_sigma_subsample,
    rank_summary,
    report_to_long_dataframe,
)
from arqueogal.xp_abundances.main.cross_catalogue_plots import render_all

logger = logging.getLogger(__name__)


def _parse_catalogue_kwarg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--catalogue expects 'name=path/to/parquet'; got {value!r}"
        )
    name, path = value.split("=", 1)
    return name.strip(), Path(path.strip())


def _load_bindings(path: Path) -> dict[str, CatalogueBinding]:
    """Load per-catalogue column-bindings YAML.

    Expected schema (one entry per catalogue):

    .. code-block:: yaml

        aspgap:
          citation: "Li+2024 (AspGap)"
          column_for:
            teff: "Teff"
            logg: "logg"
            mh: "FeH"
          sigma_for:
            teff: "e_Teff"
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    out: dict[str, CatalogueBinding] = {}
    for name, entry in data.items():
        column_for = dict(entry.get("column_for") or {})
        sigma_for = dict(entry.get("sigma_for") or {})
        citation = str(entry.get("citation") or "")
        out[name] = CatalogueBinding(
            name=name, column_for=column_for, sigma_for=sigma_for, citation=citation
        )
    return out


def _print_summary(long: pd.DataFrame) -> None:
    """Print a compact pass/fail table to stdout."""
    if long.empty:
        print("(no per-cell statistics produced; check overlap)")
        return
    cols = [
        "label",
        "catalogue",
        "mag_bin",
        "n",
        "bias",
        "scatter",
        "sigma_ratio",
        "passed",
    ]
    print(long[cols].to_string(index=False, float_format=lambda x: f"{x:>+.4g}"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run §3.3 Test 6 cross-catalogue consistency.",
    )
    parser.add_argument("--release", type=Path, required=True, help="Pipeline-1 release parquet.")
    parser.add_argument(
        "--catalogue",
        action="append",
        default=[],
        type=_parse_catalogue_kwarg,
        metavar="NAME=PATH",
        help="External catalogue parquet (already cross-matched 1:1 with --release).",
    )
    parser.add_argument(
        "--binding", type=Path, required=True, help="YAML mapping catalogue tag → column-bindings."
    )
    parser.add_argument(
        "--out", type=Path, default=Path("reports/cross_catalogue"), help="Output directory."
    )
    parser.add_argument(
        "--release-slice",
        choices=("tier1", "tier1_plus_tier2", "all"),
        default="tier1_plus_tier2",
        help=(
            "Which release-tier slice to evaluate. Methods paper Test 6 "
            "evaluates Tier 1 + Tier 2 by default."
        ),
    )
    parser.add_argument(
        "--matched-sigma-quantile",
        type=float,
        default=None,
        help=(
            "If set, also produce a 'matched-σ' run: only stars below the "
            "given σ percentile per element. Use 0.5 to mimic AspGap's "
            "low-σ-only release in the comparison."
        ),
    )
    parser.add_argument(
        "--g-mag-col", default="g_mag", help="Column in --release used for magnitude binning."
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    release = pd.read_parquet(args.release)
    logger.info("Loaded release with %d rows from %s", len(release), args.release)

    if args.release_slice == "tier1" and "release_tier" in release.columns:
        release = release[release["release_tier"] == 1].reset_index(drop=True)
    elif args.release_slice == "tier1_plus_tier2" and "release_tier" in release.columns:
        release = release[release["release_tier"] <= 2].reset_index(drop=True)
    logger.info("After tier filter %s: %d rows", args.release_slice, len(release))

    bindings = _load_bindings(args.binding)
    catalogues: dict[str, pd.DataFrame] = {}
    for name, path in args.catalogue:
        df = pd.read_parquet(path)
        if len(df) != len(release):
            raise ValueError(
                f"{name}: {path} has {len(df)} rows but release has {len(release)}; "
                "cross-match must be 1:1."
            )
        catalogues[name] = df
        logger.info("Loaded %s catalogue with %d rows from %s", name, len(df), path)

    if not catalogues:
        logger.error("No --catalogue supplied; nothing to validate against.")
        return 2

    args.out.mkdir(parents=True, exist_ok=True)

    for tag, slice_release, slice_catalogues in _iterate_slices(
        release, catalogues, matched_sigma_quantile=args.matched_sigma_quantile
    ):
        logger.info("Running slice %s with %d release rows", tag, len(slice_release))
        report = compute_cross_catalogue_report(
            slice_release,
            slice_catalogues,
            bindings,
            g_mag_col=args.g_mag_col,
        )
        out_dir = args.out / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        # Long-form CSV + JSON.
        long = report_to_long_dataframe(report)
        long.to_csv(out_dir / "cells.csv", index=False)
        rank_summary(report).to_csv(out_dir / "rank_summary.csv", index=False)
        report.to_json(out_dir / "report.json")
        plot_paths = render_all(report, slice_release, slice_catalogues, bindings, out_dir)
        # Persist plot manifest.
        manifest = {family: [str(p) for p in paths] for family, paths in plot_paths.items()}
        with (out_dir / "plot_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        # Stdout summary.
        print(f"\n=== Slice {tag} ===")
        _print_summary(long)

    return 0


def _iterate_slices(
    release: pd.DataFrame,
    catalogues: dict[str, pd.DataFrame],
    *,
    matched_sigma_quantile: float | None,
):
    """Yield ``(tag, release_slice, catalogue_slices)`` tuples.

    Always yields the full slice. If ``matched_sigma_quantile`` is set,
    also yields a "matched_sigma_{q:.2f}" slice where each element's σ is
    below the per-element quantile. The matched-σ subsample is the methods-
    paper sensitivity check that controls for σ-inflation Tier-2 demotion.
    """
    yield "full", release, catalogues
    if matched_sigma_quantile is None:
        return
    mask = matched_sigma_subsample(release, sigma_quantile=matched_sigma_quantile)
    if not mask.any():
        logger.warning("matched-σ subsample is empty; skipping")
        return
    sliced_release = release[mask].reset_index(drop=True)
    sliced_catalogues = {
        name: df[mask.to_numpy()].reset_index(drop=True) for name, df in catalogues.items()
    }
    tag = f"matched_sigma_{matched_sigma_quantile:.2f}"
    yield tag, sliced_release, sliced_catalogues


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
