"""Offline tests for arqueogal.data.provenance."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from arqueogal.data.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    HttpSource,
    LocalSource,
    Provenance,
    TapSource,
    git_sha,
    sidecar_path,
    write_sidecar,
)


def test_sidecar_path_single_suffix() -> None:
    assert sidecar_path("data/interim/stream1.parquet") == Path(
        "data/interim/stream1.provenance.json"
    )


def test_sidecar_path_fits_gz_keeps_version_dots() -> None:
    assert sidecar_path("data/raw/apogee/astraAllStarASPCAP-0.6.0.fits.gz") == Path(
        "data/raw/apogee/astraAllStarASPCAP-0.6.0.provenance.json"
    )


def test_sidecar_path_strips_known_compression() -> None:
    assert sidecar_path("x/y/file.parquet").name == "file.provenance.json"
    assert sidecar_path("x/y/file.csv.gz").name == "file.provenance.json"


def test_sidecar_path_leaves_unknown_extension_alone() -> None:
    # Unknown suffix like ``.tar`` is not stripped — conservative default.
    assert sidecar_path("x/y/bundle.tar").name == "bundle.tar.provenance.json"


def test_write_sidecar_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "interim" / "stream1.parquet"
    prov = Provenance(
        output_file=str(out),
        script="src/arqueogal/data/apogee_dr19.py",
        sources=[
            HttpSource(
                name="APOGEE DR19",
                url="https://dr19.sdss.org/sas/.../file.fits.gz",
                size_bytes=524_288_000,
                sha256="abc123",
            ),
            TapSource(
                name="AIP Gaia DR3 TAP",
                endpoint="https://gaia.aip.de/tap",
                query="SELECT g.source_id FROM gaiadr3.gaia_source WHERE source_id IN (__batch__)",
                n_batches=70,
                batch_size=10_000,
            ),
        ],
        cuts_applied=["flag_bad == 0", "snr > 70"],
        corrections=["Meszaros+2025"],
        row_count_before=964_989,
        row_count_after=682_344,
    )
    path = write_sidecar(prov)

    assert path == tmp_path / "interim" / "stream1.provenance.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["output_file"] == str(out)
    assert data["schema_version"] == PROVENANCE_SCHEMA_VERSION
    assert data["row_count_before"] == 964_989
    assert len(data["sources"]) == 2
    assert data["sources"][0]["kind"] == "http"
    assert data["sources"][1]["kind"] == "tap"
    assert "timestamp_utc" in data and data["timestamp_utc"].endswith("Z")


def test_write_sidecar_explicit_path(tmp_path: Path) -> None:
    custom = tmp_path / "custom.json"
    prov = Provenance(output_file="whatever.parquet", script="s.py")
    returned = write_sidecar(prov, path=custom)
    assert returned == custom
    assert custom.is_file()


def test_write_sidecar_creates_parent(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b" / "c" / "stream.parquet"
    prov = Provenance(output_file=str(deep), script="s.py")
    path = write_sidecar(prov)
    assert path.is_file()
    assert path.parent == deep.parent


def test_write_sidecar_no_partfile_left(tmp_path: Path) -> None:
    prov = Provenance(output_file=str(tmp_path / "x.parquet"), script="s.py")
    write_sidecar(prov)
    assert list(tmp_path.glob("*.part")) == []


def test_local_source_serialises(tmp_path: Path) -> None:
    prov = Provenance(
        output_file=str(tmp_path / "x.parquet"),
        script="s.py",
        sources=[LocalSource(name="interim", path="data/interim/y.parquet", sha256="deadbeef")],
    )
    path = write_sidecar(prov)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["sources"][0]["kind"] == "local"
    assert data["sources"][0]["sha256"] == "deadbeef"


def test_git_sha_non_strict_returns_unknown_outside_repo(tmp_path: Path) -> None:
    assert git_sha(tmp_path, strict=False) == "unknown"


def test_git_sha_strict_raises_outside_repo(tmp_path: Path) -> None:
    with pytest.raises(subprocess.SubprocessError):
        git_sha(tmp_path, strict=True)


def test_provenance_auto_captures_timestamp_and_sha() -> None:
    prov = Provenance(output_file="x.parquet", script="s.py")
    assert prov.timestamp_utc.endswith("Z")
    assert len(prov.timestamp_utc) == 20  # YYYY-MM-DDTHH:MM:SSZ
    assert isinstance(prov.git_sha, str) and prov.git_sha  # either 7-hex or "unknown"
