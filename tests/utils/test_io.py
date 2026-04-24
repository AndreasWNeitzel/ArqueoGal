"""Tests for utils.io — Parquet + checkpoint helpers."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from arqueogal.utils.io import (
    CHECKPOINT_VERSION,
    ArqueoGalCheckpointError,
    _strip_orig_mod_prefix,
    load_checkpoint,
    load_parquet,
    save_checkpoint,
    save_parquet,
    streaming_parquet_reader,
)

# --- Parquet round-trip --------------------------------------------------


def test_parquet_round_trip(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    out = save_parquet(df, tmp_path / "t.parquet")
    assert out.exists()
    back = load_parquet(out)
    pd.testing.assert_frame_equal(df, back)


def test_parquet_column_subset(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["x", "y"]})
    path = save_parquet(df, tmp_path / "t.parquet")
    sub = load_parquet(path, columns=["a", "c"])
    assert list(sub.columns) == ["a", "c"]


def test_save_parquet_is_atomic(tmp_path: Path) -> None:
    """No .tmp file should survive after save."""
    df = pd.DataFrame({"a": [1]})
    save_parquet(df, tmp_path / "atomic.parquet")
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_streaming_parquet_reader_batches(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": np.arange(10)})
    path = save_parquet(df, tmp_path / "s.parquet")
    batches = list(streaming_parquet_reader(path, batch_size=4))
    assert sum(len(b) for b in batches) == 10


def test_streaming_parquet_reader_rejects_nonpositive_batch(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1]})
    path = save_parquet(df, tmp_path / "s.parquet")
    with pytest.raises(ValueError, match="batch_size"):
        list(streaming_parquet_reader(path, batch_size=0))


# --- Checkpoint round-trip ----------------------------------------------


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    state = {"weight": torch.zeros(3), "epoch": 7}
    path = save_checkpoint(tmp_path / "ckpt.pt", **state)
    loaded = load_checkpoint(path)
    assert loaded["epoch"] == 7
    assert loaded["version"] == CHECKPOINT_VERSION
    torch.testing.assert_close(loaded["weight"], state["weight"])


def test_checkpoint_strip_orig_mod_prefix() -> None:
    sd = {"_orig_mod.layer.weight": torch.zeros(2), "_orig_mod.layer.bias": torch.zeros(1)}
    stripped = _strip_orig_mod_prefix(sd)
    assert set(stripped.keys()) == {"layer.weight", "layer.bias"}


def test_checkpoint_strip_requires_all_prefixed() -> None:
    sd = {"_orig_mod.a": 1, "b": 2}
    out = _strip_orig_mod_prefix(sd)
    # Partial prefix → no-op (avoid silent corruption).
    assert out == sd


def test_checkpoint_load_warns_on_version_mismatch(tmp_path: Path) -> None:
    # Write a fake checkpoint with a stale version.
    p = tmp_path / "ck.pt"
    torch.save({"version": 0, "weight": torch.zeros(1)}, p)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_checkpoint(p)
    assert any("version" in str(x.message) for x in w)


def test_checkpoint_load_strict_version_raises(tmp_path: Path) -> None:
    p = tmp_path / "ck.pt"
    torch.save({"version": 0, "weight": torch.zeros(1)}, p)
    with pytest.raises(ArqueoGalCheckpointError, match="version"):
        load_checkpoint(p, strict_version=True)


def test_checkpoint_load_strips_state_dict_prefix(tmp_path: Path) -> None:
    sd = {"_orig_mod.w": torch.zeros(2)}
    save_checkpoint(tmp_path / "c.pt", state_dict=sd)
    loaded = load_checkpoint(tmp_path / "c.pt")
    assert "w" in loaded["state_dict"]
    assert "_orig_mod.w" not in loaded["state_dict"]


def test_checkpoint_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt")
