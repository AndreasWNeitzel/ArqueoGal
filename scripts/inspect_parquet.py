#!/usr/bin/env python
"""Inspect a parquet file (or a directory of parquet chunks).

Static-CLI replacement for ad-hoc ``python -c "import pandas..."`` one-liners;
keeps dataframe inspection reproducible via argparse flags rather than shell
substitution (``$()``, globbing, heredocs).

Examples
--------
Basic shape + dtypes::

    python scripts/inspect_parquet.py data/interim/stream1_apogee_gaia.parquet --dtypes

Keyword-matched column subset::

    python scripts/inspect_parquet.py data/interim/stream1_apogee_gaia.parquet \\
        --match mag parallax ruwe photogeo sfd edenhofer lallement ag_gsp \\
        --exact bp_rp bp_g g_rp source_id ra_deg dec_deg b_deg

Inspect the first chunk in a directory (alphabetical)::

    python scripts/inspect_parquet.py --chunk-dir data/interim/bj21_andrae_chunks/

Inspect the newest chunk (by mtime), with a head preview::

    python scripts/inspect_parquet.py --chunk-dir data/interim/bj21_andrae_chunks/ \\
        --chunk-sort mtime --head 5

Summary stats over matched columns::

    python scripts/inspect_parquet.py data/interim/stream1_apogee_gaia.parquet \\
        --match mag parallax --describe
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd


def resolve_target(
    path: Path | None,
    chunk_dir: Path | None,
    glob: str,
    chunk_index: int,
    chunk_sort: str,
) -> Path:
    """Resolve which parquet file to read.

    Exactly one of ``path`` or ``chunk_dir`` must be provided.
    """
    if (path is None) == (chunk_dir is None):
        raise SystemExit("Provide exactly one of <path> or --chunk-dir.")

    if path is not None:
        if not path.exists():
            raise SystemExit(f"File not found: {path}")
        return path

    assert chunk_dir is not None  # for type checkers
    if not chunk_dir.is_dir():
        raise SystemExit(f"Not a directory: {chunk_dir}")

    chunks = list(chunk_dir.glob(glob))
    if not chunks:
        raise SystemExit(f"No files matching {glob!r} in {chunk_dir}")

    if chunk_sort == "alpha":
        chunks.sort()
    elif chunk_sort == "mtime":
        chunks.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    elif chunk_sort == "size":
        chunks.sort(key=lambda p: p.stat().st_size, reverse=True)
    else:
        raise SystemExit(f"Unknown --chunk-sort: {chunk_sort}")

    if not (0 <= chunk_index < len(chunks)):
        raise SystemExit(f"--chunk-index {chunk_index} out of range (found {len(chunks)} chunks).")

    target = chunks[chunk_index]
    print(
        f"[chunk-dir] {chunk_dir}  ({len(chunks)} files, sort={chunk_sort}, index={chunk_index})",
        file=sys.stderr,
    )
    print(f"[chunk-dir] -> {target.name}", file=sys.stderr)
    return target


def select_columns(
    df: pd.DataFrame,
    match: Sequence[str],
    exact: Sequence[str],
    regex: str | None,
) -> list[str]:
    """Return the subset of ``df.columns`` matching the given filters.

    If no filter is supplied, returns all columns.
    """
    if not (match or exact or regex):
        return list(df.columns)

    cols: list[str] = []
    lowered = [(c, c.lower()) for c in df.columns]
    match_lower = [m.lower() for m in match]

    for col, col_lower in lowered:
        if col in exact:
            cols.append(col)
            continue
        if any(m in col_lower for m in match_lower):
            cols.append(col)
            continue

    if regex:
        import re

        pattern = re.compile(regex)
        cols.extend(c for c in df.columns if pattern.search(c) and c not in cols)

    # Preserve original column order.
    col_set = set(cols)
    return [c for c in df.columns if c in col_set]


def print_header(target: Path, df: pd.DataFrame) -> None:
    size_mb = target.stat().st_size / 1024**2
    print(f"file    : {target}")
    print(f"size    : {size_mb:,.2f} MB")
    print(f"shape   : {df.shape[0]:,} rows x {df.shape[1]:,} cols")
    print(f"memory  : {df.memory_usage(deep=False).sum() / 1024**2:,.2f} MB (shallow)")


def print_columns(cols: list[str], total_cols: int, as_json: bool) -> None:
    print(f"matched : {len(cols)} / {total_cols} columns")
    if as_json:
        print(json.dumps(cols, indent=2))
    else:
        for c in cols:
            print(f"  {c}")


def print_dtypes(df: pd.DataFrame, cols: list[str]) -> None:
    print("\n-- dtypes --")
    dtypes = df[cols].dtypes
    width = max(len(c) for c in cols) if cols else 0
    for c, dt in dtypes.items():
        print(f"  {c:<{width}}  {dt}")


def print_nulls(df: pd.DataFrame, cols: list[str]) -> None:
    print("\n-- null counts --")
    n = len(df)
    nulls = df[cols].isna().sum()
    width = max(len(c) for c in cols) if cols else 0
    for c, k in nulls.items():
        pct = (k / n * 100) if n else 0.0
        print(f"  {c:<{width}}  {k:>10,}  ({pct:5.2f}%)")


def print_head(df: pd.DataFrame, cols: list[str], n: int) -> None:
    print(f"\n-- head({n}) --")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df[cols].head(n))


def print_describe(df: pd.DataFrame, cols: list[str]) -> None:
    numeric = df[cols].select_dtypes(include="number").columns.tolist()
    if not numeric:
        print("\n-- describe -- (no numeric columns in selection)")
        return
    print(f"\n-- describe ({len(numeric)} numeric cols) --")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df[numeric].describe())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Inspect parquet files without shell gymnastics. "
            "Accepts either a direct <path> or a --chunk-dir."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Target selection -------------------------------------------------------
    p.add_argument(
        "path",
        type=Path,
        nargs="?",
        help="Parquet file to inspect. Omit when using --chunk-dir.",
    )
    p.add_argument(
        "--chunk-dir",
        type=Path,
        help="Directory of parquet chunks; one is selected via --chunk-index.",
    )
    p.add_argument(
        "--glob",
        default="*.parquet",
        help="Glob pattern used with --chunk-dir (default: '*.parquet').",
    )
    p.add_argument(
        "--chunk-index",
        type=int,
        default=0,
        help="Which chunk to read after sorting (default: 0 = first).",
    )
    p.add_argument(
        "--chunk-sort",
        choices=("alpha", "mtime", "size"),
        default="alpha",
        help="Sort order for chunks: alpha (default), mtime (newest first), size (largest first).",
    )

    # Column filters ---------------------------------------------------------
    p.add_argument(
        "--match",
        nargs="*",
        default=[],
        metavar="SUBSTR",
        help="Case-insensitive substrings to match against column names.",
    )
    p.add_argument(
        "--exact",
        nargs="*",
        default=[],
        metavar="COL",
        help="Exact column names to always include.",
    )
    p.add_argument(
        "--regex",
        default=None,
        help="Python regex to match against column names (case-sensitive).",
    )

    # Output controls --------------------------------------------------------
    p.add_argument(
        "--dtypes",
        action="store_true",
        help="Print dtypes of matched columns.",
    )
    p.add_argument(
        "--nulls",
        action="store_true",
        help="Print null counts and percentages for matched columns.",
    )
    p.add_argument(
        "--head",
        type=int,
        default=0,
        metavar="N",
        help="Show first N rows of matched columns (default: 0 = skip).",
    )
    p.add_argument(
        "--describe",
        action="store_true",
        help="Print describe() over numeric matched columns.",
    )
    p.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit the matched column list as JSON (for piping to jq, etc.).",
    )
    p.add_argument(
        "--columns-only",
        action="store_true",
        help="Read only the matched columns (requires pyarrow). "
        "Useful for very wide tables; loads everything first to resolve names.",
    )

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    target = resolve_target(
        path=args.path,
        chunk_dir=args.chunk_dir,
        glob=args.glob,
        chunk_index=args.chunk_index,
        chunk_sort=args.chunk_sort,
    )

    # First pass: read schema only (fast) to resolve matched columns.
    try:
        import pyarrow.parquet as pq

        schema_cols = pq.ParquetFile(target).schema.names
    except Exception:
        # Fallback: load fully; pandas will handle any engine.
        schema_cols = list(pd.read_parquet(target).columns)

    schema_df = pd.DataFrame(columns=schema_cols)
    cols = select_columns(schema_df, args.match, args.exact, args.regex)

    if args.columns_only and cols:
        df = pd.read_parquet(target, columns=cols)
    else:
        df = pd.read_parquet(target)

    print_header(target, df)
    print_columns(cols, total_cols=len(schema_cols), as_json=args.as_json)

    if args.dtypes and cols:
        print_dtypes(df, cols)
    if args.nulls and cols:
        print_nulls(df, cols)
    if args.head and cols:
        print_head(df, cols, args.head)
    if args.describe and cols:
        print_describe(df, cols)

    return 0


if __name__ == "__main__":
    sys.exit(main())
