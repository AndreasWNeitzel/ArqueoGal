"""Offline tests for arqueogal.data.downloads.

Network is stubbed via a fake ``requests.get`` — no real HTTP calls. These
tests cover atomic write, SHA-256, checksum mismatch, partial-file cleanup,
and the reuse-on-exists fast path.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from arqueogal.data import downloads
from arqueogal.data.downloads import DownloadInfo, download, sha256_file


class _FakeResponse:
    def __init__(self, payload: bytes, status: int = 200, content_length: bool = True) -> None:
        self._payload = payload
        self.status_code = status
        self.headers = {"content-length": str(len(payload))} if content_length else {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        for i in range(0, len(self._payload), chunk_size):
            yield self._payload[i : i + chunk_size]

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


@pytest.fixture
def fake_get(monkeypatch: pytest.MonkeyPatch):
    """Patch ``downloads.requests.get`` with a configurable fake."""

    state: dict[str, object] = {"payload": b"hello world", "calls": 0}

    def fake(url: str, stream: bool = False, timeout: float = 0) -> _FakeResponse:  # noqa: ARG001
        state["calls"] = int(state["calls"]) + 1  # type: ignore[arg-type]
        return _FakeResponse(bytes(state["payload"]))  # type: ignore[arg-type]

    monkeypatch.setattr(downloads.requests, "get", fake)
    return state


def test_download_writes_file_and_returns_sha(fake_get, tmp_path: Path) -> None:
    dest = tmp_path / "sub" / "file.bin"
    info = download("https://example/x", dest, progress=False)

    assert dest.is_file()
    assert dest.read_bytes() == b"hello world"
    assert info.size_bytes == len(b"hello world")
    assert info.sha256 == hashlib.sha256(b"hello world").hexdigest()
    assert info.url == "https://example/x"
    assert info.dest == dest


def test_download_atomic_part_file_removed_on_success(fake_get, tmp_path: Path) -> None:
    dest = tmp_path / "file.bin"
    download("https://example/x", dest, progress=False)
    assert list(tmp_path.glob("*.part")) == []


def test_download_checksum_mismatch_deletes_temp(fake_get, tmp_path: Path) -> None:
    dest = tmp_path / "file.bin"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download("https://example/x", dest, expected_sha256="deadbeef", progress=False)
    assert not dest.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_download_checksum_match_accepted(fake_get, tmp_path: Path) -> None:
    dest = tmp_path / "file.bin"
    expected = hashlib.sha256(b"hello world").hexdigest()
    info = download("https://example/x", dest, expected_sha256=expected.upper(), progress=False)
    assert info.sha256 == expected


def test_download_reuses_existing_without_redownload(fake_get, tmp_path: Path) -> None:
    dest = tmp_path / "file.bin"
    dest.write_bytes(b"hello world")
    info = download("https://example/x", dest, progress=False)

    assert fake_get["calls"] == 0  # no HTTP call issued
    assert info.size_bytes == len(b"hello world")
    assert info.sha256 == hashlib.sha256(b"hello world").hexdigest()


def test_download_overwrite_forces_redownload(fake_get, tmp_path: Path) -> None:
    dest = tmp_path / "file.bin"
    dest.write_bytes(b"stale")
    download("https://example/x", dest, overwrite=True, progress=False)
    assert fake_get["calls"] == 1
    assert dest.read_bytes() == b"hello world"


def test_download_raises_cleans_part_on_http_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake(url: str, stream: bool = False, timeout: float = 0) -> _FakeResponse:  # noqa: ARG001
        return _FakeResponse(b"", status=500)

    monkeypatch.setattr(downloads.requests, "get", fake)

    dest = tmp_path / "file.bin"
    with pytest.raises(RuntimeError, match="HTTP 500"):
        download("https://example/x", dest, progress=False)
    assert not dest.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    p = tmp_path / "blob.bin"
    p.write_bytes(b"abc" * 10_000)
    assert sha256_file(p) == hashlib.sha256(b"abc" * 10_000).hexdigest()


def test_download_info_is_frozen() -> None:
    info = DownloadInfo(url="u", dest=Path("/tmp/x"), size_bytes=1, sha256="s")
    with pytest.raises((AttributeError, Exception)):
        info.size_bytes = 2  # type: ignore[misc]
