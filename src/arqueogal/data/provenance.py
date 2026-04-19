"""Provenance sidecar writer for ArqueoGal data artefacts.

Every Parquet / FITS file produced by an ingestion script must ship with a
companion ``*.provenance.json`` describing *where it came from*, *what was
applied to it*, and *when*. Without this sidecar, the artefact is not
reproducible — see data_acquisition.md §14.4 / §15 for the full rationale.

Usage
-----
.. code-block:: python

    from arqueogal.data.provenance import (
        Provenance, HttpSource, TapSource, write_sidecar,
    )

    prov = Provenance(
        output_file="data/interim/stream1_apogee_gaia.parquet",
        script="src/arqueogal/data/apogee_dr19.py",
        sources=[
            HttpSource(
                name="APOGEE DR19 ASPCAP summary",
                url="https://dr19.sdss.org/sas/.../astraAllStarASPCAP-0.6.0.fits.gz",
                size_bytes=info.size_bytes,
                sha256=info.sha256,
            ),
            TapSource(
                name="AIP Gaia DR3 TAP enrichment",
                endpoint="https://gaia.aip.de/tap",
                query=adql_template,
                n_batches=70,
                batch_size=10_000,
            ),
        ],
        cuts_applied=["flag_bad == 0", "snr > 70", "teff in [4000, 5500]"],
        row_count_before=964_989,
        row_count_after=682_344,
        corrections=["Meszaros+2025 Teff-trend", "Lindegren+2021 zpt"],
    )
    write_sidecar(prov)  # -> data/interim/stream1_apogee_gaia.provenance.json
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROVENANCE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class HttpSource:
    """A file fetched over HTTPS. SHA-256 is mandatory for reproducibility."""

    name: str
    url: str
    size_bytes: int
    sha256: str
    kind: str = "http"


@dataclass(frozen=True, slots=True)
class TapSource:
    """A TAP query. ``query`` should be the *template* for batched runs."""

    name: str
    endpoint: str
    query: str
    n_batches: int = 1
    batch_size: int | None = None
    kind: str = "tap"


@dataclass(frozen=True, slots=True)
class LocalSource:
    """An existing local artefact consumed as input (e.g. an interim Parquet)."""

    name: str
    path: str
    sha256: str
    kind: str = "local"


Source = HttpSource | TapSource | LocalSource


@dataclass(slots=True)
class Provenance:
    """One provenance record = one output artefact's full lineage."""

    output_file: str
    script: str
    sources: list[Source] = field(default_factory=list)
    cuts_applied: list[str] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)
    row_count_before: int | None = None
    row_count_after: int | None = None
    random_seed: int | None = None
    notes: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    schema_version: int = PROVENANCE_SCHEMA_VERSION
    git_sha: str = field(default_factory=lambda: git_sha(strict=False))
    timestamp_utc: str = field(default_factory=lambda: _utc_now_iso())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sources"] = [asdict(s) for s in self.sources]
        return d


def write_sidecar(prov: Provenance, path: Path | str | None = None) -> Path:
    """Write ``prov`` to its sidecar JSON.

    If ``path`` is omitted, derive it from ``prov.output_file`` by replacing
    any suffix (including ``.gz``-style double suffixes) with
    ``.provenance.json`` — so ``stream1.parquet`` → ``stream1.provenance.json``.
    The output directory is created if missing, and the file is written
    atomically (temp + rename) so a crash mid-write never leaves a partial
    sidecar next to a finished Parquet.
    """
    if path is None:
        path = sidecar_path(prov.output_file)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(prov.to_dict(), indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(path)
    logger.info("wrote provenance: %s", path)
    return path


_KNOWN_SUFFIXES = frozenset(
    {".gz", ".bz2", ".xz", ".zst", ".fits", ".parquet", ".csv", ".json", ".yaml", ".yml", ".npz"}
)


def sidecar_path(output_file: Path | str) -> Path:
    """Return the conventional sidecar path for an artefact.

    Strips only known data suffixes from the right so multi-suffix artefacts
    (``foo.fits.gz``) produce a single clean ``.provenance.json`` while
    version-dotted filenames (``astraAllStarASPCAP-0.6.0.fits.gz``) keep
    their version intact.
    """
    p = Path(output_file)
    while p.suffix.lower() in _KNOWN_SUFFIXES:
        p = p.with_suffix("")
    return p.with_name(f"{p.name}.provenance.json")


def git_sha(repo: Path | str | None = None, *, strict: bool = True) -> str:
    """Return short git SHA of ``repo`` (default: package root).

    Parameters
    ----------
    strict : bool
        When True, raise on git failure. When False (default for Provenance
        auto-capture), return ``"unknown"`` so provenance writing never blocks
        a successful ingestion run.
    """
    cwd = Path(repo) if repo is not None else Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--short=7", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        if strict:
            raise
        logger.debug("git_sha lookup failed (%r); using 'unknown'", exc)
        return "unknown"


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
