"""Offline tests for arqueogal.data.credentials."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from arqueogal.data.credentials import (
    AIP_TOKEN_ENV_VAR,
    ENV_VAR,
    Credentials,
    ServiceCredentials,
    TokenCredentials,
    load_aip_token_from_env,
    load_credentials,
    resolve_path,
)


@pytest.fixture
def creds_file(tmp_path: Path) -> Path:
    path = tmp_path / "credentials.yaml"
    path.write_text(
        "aip:\n"
        "  user: testuser\n"
        "  password: testpass\n"
        "esa:\n"
        "  user: esauser\n"
        "  password: esapass\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def test_load_full(creds_file: Path) -> None:
    creds = load_credentials(creds_file)
    assert creds == Credentials(
        aip=ServiceCredentials(user="testuser", password="testpass"),
        esa=ServiceCredentials(user="esauser", password="esapass"),
    )


def test_load_aip_only(tmp_path: Path) -> None:
    path = tmp_path / "credentials.yaml"
    path.write_text("aip:\n  user: u\n  password: p\n", encoding="utf-8")
    path.chmod(0o600)
    creds = load_credentials(path)
    assert creds.aip == ServiceCredentials(user="u", password="p")
    assert creds.esa is None


def test_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "credentials.yaml"
    path.write_text("", encoding="utf-8")
    path.chmod(0o600)
    creds = load_credentials(path)
    assert creds == Credentials(aip=None, esa=None)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_credentials(tmp_path / "does-not-exist.yaml")


def test_permissions_too_wide(tmp_path: Path) -> None:
    path = tmp_path / "credentials.yaml"
    path.write_text("aip:\n  user: u\n  password: p\n", encoding="utf-8")
    path.chmod(0o644)
    with pytest.raises(PermissionError, match="0o644"):
        load_credentials(path)


def test_permissions_0400_is_ok(tmp_path: Path) -> None:
    path = tmp_path / "credentials.yaml"
    path.write_text("aip:\n  user: u\n  password: p\n", encoding="utf-8")
    path.chmod(0o400)
    load_credentials(path)  # read-only by owner is tighter than 0600 — accepted


def test_malformed_top_level(tmp_path: Path) -> None:
    path = tmp_path / "credentials.yaml"
    path.write_text("- just a list\n- not a mapping\n", encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(ValueError, match="must be a mapping"):
        load_credentials(path)


def test_service_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "credentials.yaml"
    path.write_text("aip:\n  user: u\n", encoding="utf-8")  # no password
    path.chmod(0o600)
    with pytest.raises(ValueError, match="missing fields"):
        load_credentials(path)


def test_env_var_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "from_env.yaml"
    monkeypatch.setenv(ENV_VAR, str(target))
    assert resolve_path() == target


def test_explicit_path_beats_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "from_env.yaml"))
    explicit = tmp_path / "explicit.yaml"
    assert resolve_path(explicit) == explicit


def test_tilde_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "~/custom.yaml")
    assert resolve_path() == Path(os.path.expanduser("~/custom.yaml"))


def test_load_aip_token_from_env_returns_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(AIP_TOKEN_ENV_VAR, "abc123")
    creds = load_aip_token_from_env()
    assert creds == TokenCredentials(token="abc123")


def test_load_aip_token_from_env_unset_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(AIP_TOKEN_ENV_VAR, raising=False)
    assert load_aip_token_from_env() is None


def test_load_aip_token_from_env_whitespace_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(AIP_TOKEN_ENV_VAR, "   \t  ")
    assert load_aip_token_from_env() is None


def test_load_aip_token_strips_surrounding_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(AIP_TOKEN_ENV_VAR, "  padded-token\n")
    creds = load_aip_token_from_env()
    assert creds is not None
    assert creds.token == "padded-token"
