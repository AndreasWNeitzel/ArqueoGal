"""Tests for utils.gpu."""

from __future__ import annotations

import pytest
import torch

from arqueogal.utils.gpu import (
    check_vram,
    get_device,
    get_hdbscan_class,
    get_umap_class,
)


def test_get_device_auto_returns_torch_device() -> None:
    dev = get_device("auto")
    assert isinstance(dev, torch.device)


def test_get_device_cpu_forces_cpu() -> None:
    dev = get_device("cpu")
    assert dev.type == "cpu"


def test_get_device_rejects_invalid_prefer() -> None:
    with pytest.raises(ValueError, match="prefer must be"):
        get_device("gpu")


def test_get_device_cuda_raises_if_unavailable() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA available — cannot test fallback raise")
    with pytest.raises(RuntimeError, match="CUDA requested"):
        get_device("cuda")


def test_check_vram_returns_true_for_cpu() -> None:
    assert check_vram(required_mb=99_999_999, device=torch.device("cpu"))


def test_get_umap_class_returns_callable() -> None:
    cls = get_umap_class()
    assert callable(cls)
    # Has at least fit/transform interface.
    assert hasattr(cls, "fit")


def test_get_hdbscan_class_returns_callable() -> None:
    cls = get_hdbscan_class()
    assert callable(cls)
    assert hasattr(cls, "fit")
