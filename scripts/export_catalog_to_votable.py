"""Export a Pipeline 1 release Parquet to VOTable format for VizieR / VO discovery.

Why VOTable
-----------

VizieR (CDS) and the broader Virtual Observatory (IVOA) ecosystem consume
VOTable as the canonical interchange format. Without a VOTable export the
D-Cat-b release cannot be formally registered at CDS and is invisible to
TAP-based discovery tools like Aladin's catalog query and TOPCAT's VO
service tab (META_META §14.4 P0; data_preparation_output.md CRITICAL).

This script reads a release-annotated Parquet and writes a VOTable v1.4
file with per-column UCDs, units, and descriptions. The PARAM section
records release metadata.

Usage
-----

    python scripts/export_catalog_to_votable.py \\
        --input data/processed/pipeline1_inference_v1.parquet \\
        --output release/D-Cat-b/D-Cat-b_v1.0.vot \\
        --release-tag pipeline1-v1-2026-04-19

The output is a binary-encoded VOTable (TABLEDATA is human-readable but
slow for >1M rows; BINARY2 is dense and TOPCAT-supported).

References
----------

- IVOA VOTable v1.4: https://www.ivoa.net/documents/VOTable/20191021/
- IVOA UCD1+ v1.5: https://www.ivoa.net/documents/UCD1+/20180527/
- VizieR submission: https://cds.unistra.fr/vizier/submit/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess

# Reuse UCD/unit dictionaries from the FITS exporter to keep the catalog
# metadata consistent across formats.
import sys
from datetime import UTC, datetime
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

# Lazy imports of metadata constants from the FITS exporter to avoid
# duplicating them. The FITS exporter is in the same dir; importing it
# pulls in lazy astropy paths only when called.
import contextlib

from export_catalog_to_fits import (  # type: ignore[import-not-found]
    _COLUMN_DESCRIPTIONS,
    _COLUMN_UCDS,
    _COLUMN_UNITS,
)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_to_votable(
    parquet_path: Path,
    votable_path: Path,
    *,
    release_tag: str,
    schema_version: int = 3,
    frozen_stats_fingerprint: str | None = None,
    table_format: str = "binary2",
) -> dict[str, str | int]:
    """Export a release Parquet to a VOTable v1.4 file.

    Parameters
    ----------
    parquet_path : Path
        Input Parquet (release-annotated).
    votable_path : Path
        Output VOTable path; parent dir auto-created.
    release_tag : str
        Git tag for provenance PARAM.
    schema_version : int
        Catalog schema version for provenance PARAM.
    frozen_stats_fingerprint : str, optional
        Hermite basis fingerprint for provenance PARAM.
    table_format : str
        One of ``"binary2"`` (default; dense, recommended for large
        catalogs), ``"binary"``, or ``"tabledata"``.

    Returns
    -------
    dict
        Summary similar to the FITS exporter.

    Notes
    -----
    Astropy is required. The VOTable is built from an astropy.table.Table,
    augmented with per-Field UCDs, units, descriptions, and PARAM records
    for provenance metadata.
    """
    try:
        import pandas as pd
        from astropy.io.votable.tree import (  # type: ignore[import-not-found]
            Param,
            VOTableFile,
        )
        from astropy.table import Table
    except ImportError as e:
        raise ImportError(
            "VOTable export requires astropy. Install via `uv add astropy` or `pip install astropy`.",
        ) from e

    votable_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    table = Table.from_pandas(df)

    # Apply per-column metadata to the astropy Table, then convert to VOTable.
    for col_name in table.colnames:
        if col_name in _COLUMN_UNITS:
            with contextlib.suppress(Exception):
                table[col_name].unit = _COLUMN_UNITS[col_name]
        if col_name in _COLUMN_DESCRIPTIONS:
            table[col_name].description = _COLUMN_DESCRIPTIONS[col_name]

    votable = VOTableFile.from_table(table)
    resource = votable.resources[0]
    vo_table = resource.tables[0]

    # Apply UCDs to fields.
    for field in vo_table.fields:
        col_name = field.name
        if col_name in _COLUMN_UCDS:
            field.ucd = _COLUMN_UCDS[col_name]

    # PARAM records for provenance.
    resource.params.append(
        Param(
            votable,
            name="release_tag",
            datatype="char",
            arraysize="*",
            value=release_tag,
        ),
    )
    resource.params.append(
        Param(
            votable,
            name="schema_version",
            datatype="int",
            value=str(schema_version),
        ),
    )
    resource.params.append(
        Param(
            votable,
            name="git_sha",
            datatype="char",
            arraysize="*",
            value=_git_sha()[:12],
        ),
    )
    resource.params.append(
        Param(
            votable,
            name="parquet_sha256",
            datatype="char",
            arraysize="*",
            value=_file_sha256(parquet_path)[:16],
        ),
    )
    if frozen_stats_fingerprint is not None:
        resource.params.append(
            Param(
                votable,
                name="frozen_stats_fingerprint",
                datatype="char",
                arraysize="*",
                value=frozen_stats_fingerprint[:16],
            ),
        )
    resource.params.append(
        Param(
            votable,
            name="export_timestamp_utc",
            datatype="char",
            arraysize="*",
            value=datetime.now(tz=UTC).isoformat(),
        ),
    )

    # Set encoding format on the table itself.
    if table_format not in {"binary2", "binary", "tabledata"}:
        raise ValueError(
            f"unknown table_format '{table_format}'; expected 'binary2', 'binary', or 'tabledata'",
        )
    vo_table.format = table_format

    votable.to_xml(str(votable_path))

    return {
        "n_rows": int(len(df)),
        "votable_sha256": _file_sha256(votable_path),
        "git_sha": _git_sha(),
        "release_tag": release_tag,
        "format": table_format,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--input", type=Path, required=True, help="Input Parquet path.")
    parser.add_argument("--output", type=Path, required=True, help="Output VOTable path.")
    parser.add_argument("--release-tag", type=str, required=True, help="Git tag identifier.")
    parser.add_argument(
        "--schema-version",
        type=int,
        default=3,
        help="Catalog schema version (default 3).",
    )
    parser.add_argument(
        "--frozen-stats-fingerprint",
        type=str,
        default=None,
        help="Hermite basis fingerprint (optional).",
    )
    parser.add_argument(
        "--table-format",
        choices=("binary2", "binary", "tabledata"),
        default="binary2",
        help="VOTable encoding format (default binary2).",
    )
    args = parser.parse_args()

    summary = export_to_votable(
        parquet_path=args.input,
        votable_path=args.output,
        release_tag=args.release_tag,
        schema_version=args.schema_version,
        frozen_stats_fingerprint=args.frozen_stats_fingerprint,
        table_format=args.table_format,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
