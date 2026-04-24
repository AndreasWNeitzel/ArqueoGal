"""HTTPS downloads with streaming, checksumming, and atomic replacement.

All file downloads in ArqueoGal go through this module. Large files (FITS
summaries, catalogues) must never be fetched via ``r.content`` — stream to a
temp path and rename on success for atomicity, so a crashed download never
leaves a truncated file at the destination path.

Usage
-----
.. code-block:: python

    from arqueogal.data.downloads import download

    info = download(
        "https://dr19.sdss.org/sas/.../astraAllStarASPCAP-0.6.0.fits.gz",
        Path("data/raw/apogee_dr19/astraAllStarASPCAP-0.6.0.fits.gz"),
    )
    # info.size_bytes, info.sha256 → drop into provenance sidecar

See data_acquisition.md §14.2 for the design rationale.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_BYTES = 1024 * 1024  # 1 MiB
DEFAULT_TIMEOUT_SEC = 60


@dataclass(frozen=True, slots=True)
class DownloadInfo:
    """Outcome of a successful download — feeds directly into provenance."""

    url: str
    dest: Path
    size_bytes: int
    sha256: str


def download(  # noqa: PLR0913 — keyword-only tuning knobs with safe defaults
    url: str,
    dest: Path | str,
    *,
    chunk_size: int = DEFAULT_CHUNK_BYTES,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    expected_sha256: str | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> DownloadInfo:
    """Stream ``url`` to ``dest`` atomically, returning size + SHA-256.

    Writes to ``dest.with_suffix(dest.suffix + ".part")`` and ``os.replace``s
    on success — a crash mid-stream leaves ``.part`` behind, never a
    truncated ``dest``.

    Parameters
    ----------
    expected_sha256 : str, optional
        If provided, the computed digest must match (hex, case-insensitive) or
        the temp file is deleted and ``ValueError`` raised.
    overwrite : bool
        When False (default) and ``dest`` already exists, return the existing
        file's info without re-downloading (SHA-256 recomputed so provenance
        stays accurate). Set True to force re-download.
    progress : bool
        Show tqdm progress bar. Disable for non-interactive runs.
    """
    dest = Path(dest)

    if dest.exists() and not overwrite:
        logger.info("download exists, reusing: %s", dest)
        return _describe_existing(url, dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    hasher = hashlib.sha256()
    size = 0
    try:
        with requests.get(url, stream=True, timeout=timeout_sec) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0)) or None
            bar = tqdm(total=total, unit="B", unit_scale=True, disable=not progress, desc=dest.name)
            with tmp.open("wb") as f, bar:
                for chunk in r.iter_content(chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    hasher.update(chunk)
                    size += len(chunk)
                    bar.update(len(chunk))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    digest = hasher.hexdigest()
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        tmp.unlink(missing_ok=True)
        raise ValueError(f"SHA-256 mismatch for {url}: got {digest}, expected {expected_sha256}")

    os.replace(tmp, dest)
    logger.info("downloaded %s (%d bytes, sha256=%s)", dest, size, digest)
    return DownloadInfo(url=url, dest=dest, size_bytes=size, sha256=digest)


def sha256_file(path: Path | str, chunk_size: int = DEFAULT_CHUNK_BYTES) -> str:
    """Compute SHA-256 of a local file (streamed, no full-file load)."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _describe_existing(url: str, dest: Path) -> DownloadInfo:
    return DownloadInfo(
        url=url, dest=dest, size_bytes=dest.stat().st_size, sha256=sha256_file(dest)
    )
