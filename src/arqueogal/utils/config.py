"""YAML config loader with validation — utils/DESIGN.md.

Design goals:

- **Nested dataclasses** — configs are typed, not ``dict[str, Any]``.
- **Validation on load** — numeric ranges, type checks, raise on invalid.
- **Unknown-key warnings** — silent ignores hide typos.
- **Path resolution** — relative paths resolved against the config file.
- **Round-trip** — ``to_yaml`` for checkpoint embedding.

The loader is generic: callers pass the dataclass schema and get an
instance back. It recurses through nested dataclasses and lists of
dataclasses.
"""

from __future__ import annotations

import dataclasses
import types
import warnings
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

_UNION_ORIGINS: tuple[object, ...] = (Union, types.UnionType)


class ConfigValidationError(ValueError):
    """Raised when a config file fails validation."""


def _resolve_paths(
    value: Any,
    base_dir: Path,
    field_type: Any,
) -> Any:
    """Resolve relative paths against ``base_dir`` if field expects Path."""
    if value is None:
        return value
    origin = get_origin(field_type)
    args = get_args(field_type)
    # Unwrap Optional[X] → X (covers both typing.Union and PEP 604 X | None).
    if origin in _UNION_ORIGINS and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _resolve_paths(value, base_dir, non_none[0])
    if field_type is Path or field_type == "Path":
        p = Path(value)
        return p if p.is_absolute() else (base_dir / p).resolve()
    return value


def _coerce_and_validate(  # noqa: PLR0911, PLR0912 — dispatch over type-system variants
    value: Any,
    field_type: Any,
    field_name: str,
    base_dir: Path,
) -> Any:
    """Coerce ``value`` to ``field_type`` and run basic type validation.

    Handles: primitive types, Path, Optional[X], list[X], dict[K, V], and
    nested dataclasses. Raises :class:`ConfigValidationError` on type
    mismatches.
    """
    origin = get_origin(field_type)
    args = get_args(field_type)

    # Optional[X] / Union[X, None] (covers typing.Union and PEP 604 X | None).
    if origin in _UNION_ORIGINS and type(None) in args:
        if value is None:
            return None
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _coerce_and_validate(value, non_none[0], field_name, base_dir)

    # list[X]
    if origin in (list,):
        if not isinstance(value, list):
            raise ConfigValidationError(
                f"{field_name}: expected list, got {type(value).__name__}",
            )
        inner = args[0] if args else Any
        return [
            _coerce_and_validate(v, inner, f"{field_name}[{i}]", base_dir)
            for i, v in enumerate(value)
        ]

    # dict[K, V]
    if origin in (dict,):
        if not isinstance(value, dict):
            raise ConfigValidationError(
                f"{field_name}: expected dict, got {type(value).__name__}",
            )
        return value

    # tuple[X, ...] — YAML lists coerced.
    if origin in (tuple,):
        if not isinstance(value, (list, tuple)):
            raise ConfigValidationError(
                f"{field_name}: expected sequence, got {type(value).__name__}",
            )
        return tuple(value)

    # Nested dataclass.
    if dataclasses.is_dataclass(field_type):
        if not isinstance(value, dict):
            raise ConfigValidationError(
                f"{field_name}: expected mapping for nested "
                f"{field_type.__name__}, got {type(value).__name__}",
            )
        return _from_dict(field_type, value, base_dir, field_name)

    # Path
    if field_type is Path:
        return _resolve_paths(value, base_dir, field_type)

    # Primitives: bool, int, float, str. YAML already decodes these — but we
    # still do light type checks.
    if field_type in (int, float, str, bool):
        # bool is a subclass of int in Python — avoid accepting True/False
        # for an int field silently.
        if field_type is int and isinstance(value, bool):
            raise ConfigValidationError(
                f"{field_name}: expected int, got bool",
            )
        if field_type is float and isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        if not isinstance(value, field_type):
            raise ConfigValidationError(
                f"{field_name}: expected {field_type.__name__}, got {type(value).__name__}",
            )
        return value

    # Unknown / Any — pass through.
    return value


def _from_dict[T](
    cls: type[T],
    data: dict[str, Any],
    base_dir: Path,
    prefix: str = "",
) -> T:
    """Build a dataclass instance from a dict with unknown-key warnings."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    # Resolve PEP 563 string annotations into real types. Must include the
    # caller-module's globals so nested dataclasses referred to by name
    # resolve — ``get_type_hints`` handles that automatically.
    try:
        resolved = get_type_hints(cls)
    except Exception:  # noqa: BLE001 — fall back to raw annotations if resolution fails
        resolved = {f.name: f.type for f in dataclasses.fields(cls)}
    field_map = {f.name: f for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in field_map:
            loc = f"{prefix}.{key}" if prefix else key
            warnings.warn(
                f"Unknown config key '{loc}' for {cls.__name__}; ignoring",
                stacklevel=3,
            )
            continue
        full = f"{prefix}.{key}" if prefix else key
        field_type = resolved.get(key, field_map[key].type)
        kwargs[key] = _coerce_and_validate(value, field_type, full, base_dir)
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigValidationError(
            f"{cls.__name__}: {exc}",
        ) from exc


def load_config[T](path: str | Path, schema: type[T]) -> T:
    """Load a YAML config as an instance of ``schema`` (a dataclass).

    Relative paths in the YAML are resolved against the YAML file's
    directory. Unknown keys emit a warning; type mismatches raise
    :class:`ConfigValidationError`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigValidationError(
            f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}",
        )
    return _from_dict(schema, raw, base_dir=path.parent)


def to_yaml(obj: Any) -> str:
    """Dump a (possibly nested) dataclass instance to a YAML string.

    Paths are serialised as strings. Tuples as lists.
    """
    return yaml.safe_dump(_to_plain(obj), sort_keys=False)


def _to_plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


__all__ = [
    "ConfigValidationError",
    "load_config",
    "to_yaml",
]
