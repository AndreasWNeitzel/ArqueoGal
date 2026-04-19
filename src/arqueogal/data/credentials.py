"""Credential loader for ArqueoGal data services.

Expected file: ``~/.arqueogal/credentials.yaml`` with ``0600`` permissions. Never
commit it; never hardcode values. Override the path via the
``ARQUEOGAL_CREDENTIALS_PATH`` environment variable if you need to point at a
different file (useful for CI or HPC).

Schema
------
.. code-block:: yaml

    aip:
      user: "your_aip_login"
      password: "your_aip_password"

    # Optional — ESA Gaia Archive. Public access works without, so this is only
    # needed if you have a Gaia Archive account and want per-user quotas.
    esa:
      user: "your_esa_login"
      password: "your_esa_password"

GAVO, VizieR, SDSS DR19, and MAST are public and need no credentials.

AIP token fallback
------------------
If the YAML file has no ``aip`` block, :func:`load_aip_token_from_env` reads
the ``GAIA_AIP_TOKEN`` environment variable and returns a
:class:`TokenCredentials`. ``tap.aip_service`` consumes this fallback
automatically — YAML wins when both are present.

The token is sent as an ``Authorization: Token <token>`` header (Daiquiri /
DRF-style TokenAuthentication, which AIP's TAP service uses).
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PATH = Path.home() / ".arqueogal" / "credentials.yaml"
ENV_VAR = "ARQUEOGAL_CREDENTIALS_PATH"
AIP_TOKEN_ENV_VAR = "GAIA_AIP_TOKEN"


@dataclass(frozen=True, slots=True)
class ServiceCredentials:
    user: str
    password: str


@dataclass(frozen=True, slots=True)
class TokenCredentials:
    """API-token auth. Sent as ``Authorization: Token <token>`` on each request."""

    token: str


@dataclass(frozen=True, slots=True)
class Credentials:
    """Parsed credential bundle. Fields are ``None`` when not configured."""

    aip: ServiceCredentials | None = None
    esa: ServiceCredentials | None = None


def load_aip_token_from_env() -> TokenCredentials | None:
    """Return a :class:`TokenCredentials` when ``GAIA_AIP_TOKEN`` is set.

    Used as a fallback by :func:`arqueogal.data.tap.aip_service` when the
    YAML file has no ``aip`` block. Whitespace-only values are treated as
    unset so that ``export GAIA_AIP_TOKEN=`` in a shell session disables
    the fallback cleanly.
    """
    token = os.environ.get(AIP_TOKEN_ENV_VAR, "").strip()
    return TokenCredentials(token=token) if token else None


def resolve_path(path: Path | str | None = None) -> Path:
    """Resolve the credentials file path, respecting the env-var override."""
    if path is not None:
        return Path(path).expanduser()
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env).expanduser()
    return DEFAULT_PATH


def load_credentials(path: Path | str | None = None) -> Credentials:
    """Load and validate the credentials YAML.

    Raises
    ------
    FileNotFoundError
        If the credentials file does not exist.
    PermissionError
        If the file has any group/other permission bits set (i.e. wider than
        ``0600``). Tighten with ``chmod 600 <path>``.
    ValueError
        If the YAML is malformed or a service block is missing required fields.
    """
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"credentials file not found: {resolved}\n"
            f"create it with schema documented in arqueogal.data.credentials"
        )

    _check_permissions(resolved)

    with resolved.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{resolved}: top-level YAML must be a mapping, got {type(raw).__name__}")

    return Credentials(
        aip=_parse_service(raw, "aip", resolved),
        esa=_parse_service(raw, "esa", resolved),
    )


def _check_permissions(path: Path) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise PermissionError(
            f"{path} has permissions {oct(mode)}; expected 0o600 (owner only). "
            f"fix with: chmod 600 {path}"
        )


def _parse_service(raw: dict, key: str, source: Path) -> ServiceCredentials | None:
    if key not in raw:
        return None
    block = raw[key]
    if not isinstance(block, dict):
        raise ValueError(f"{source}: '{key}' block must be a mapping")
    missing = {"user", "password"} - block.keys()
    if missing:
        raise ValueError(f"{source}: '{key}' block missing fields: {sorted(missing)}")
    return ServiceCredentials(user=str(block["user"]), password=str(block["password"]))
