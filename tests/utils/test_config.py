"""Tests for utils.config — YAML + dataclass validation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from arqueogal.utils.config import ConfigValidationError, load_config, to_yaml


@dataclass
class TrainCfg:
    lr: float = 1e-3
    epochs: int = 10
    optimizer: str = "adam"


@dataclass
class TopCfg:
    name: str = "run"
    train: TrainCfg = field(default_factory=TrainCfg)
    outputs: Path | None = None
    seeds: list[int] = field(default_factory=list)


def _write_yaml(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_load_config_basic(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path / "c.yaml", """
name: exp1
train:
  lr: 0.01
  epochs: 5
  optimizer: sgd
seeds: [0, 1, 2]
""")
    cfg = load_config(p, TopCfg)
    assert cfg.name == "exp1"
    assert cfg.train.lr == 0.01
    assert cfg.train.epochs == 5
    assert cfg.train.optimizer == "sgd"
    assert cfg.seeds == [0, 1, 2]


def test_load_config_resolves_relative_path(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path / "c.yaml", """
outputs: out/x
""")
    cfg = load_config(p, TopCfg)
    assert cfg.outputs is not None
    assert cfg.outputs.is_absolute()
    assert cfg.outputs.name == "x"


def test_load_config_warns_on_unknown_key(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path / "c.yaml", "bogus_key: 1\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_config(p, TopCfg)
    assert any("bogus_key" in str(x.message) for x in w)


def test_load_config_rejects_type_mismatch(tmp_path: Path) -> None:
    # epochs expects int, pass string.
    p = _write_yaml(tmp_path / "c.yaml", "train:\n  epochs: not-an-int\n")
    with pytest.raises(ConfigValidationError, match="epochs"):
        load_config(p, TopCfg)


def test_load_config_rejects_bool_for_int(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path / "c.yaml", "train:\n  epochs: true\n")
    with pytest.raises(ConfigValidationError, match="epochs"):
        load_config(p, TopCfg)


def test_load_config_coerces_int_to_float(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path / "c.yaml", "train:\n  lr: 1\n")
    cfg = load_config(p, TopCfg)
    assert isinstance(cfg.train.lr, float)
    assert cfg.train.lr == 1.0


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml", TopCfg)


def test_load_config_top_level_must_be_mapping(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path / "c.yaml", "- 1\n- 2\n")
    with pytest.raises(ConfigValidationError, match="mapping"):
        load_config(p, TopCfg)


def test_to_yaml_round_trip(tmp_path: Path) -> None:
    cfg = TopCfg(name="r", train=TrainCfg(lr=0.2, epochs=3),
                 outputs=tmp_path / "o", seeds=[0])
    s = to_yaml(cfg)
    assert "name: r" in s
    assert "lr: 0.2" in s
    # Path serialised as string.
    assert str(tmp_path / "o") in s


def test_to_yaml_handles_tuple() -> None:
    @dataclass
    class X:
        t: tuple[int, ...] = (1, 2, 3)

    s = to_yaml(X())
    assert "- 1" in s
