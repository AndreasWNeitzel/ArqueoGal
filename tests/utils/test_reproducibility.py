"""Tests for utils.reproducibility."""

from __future__ import annotations

import random

import numpy as np
import pytest

from arqueogal.utils.reproducibility import set_global_seed


def test_set_global_seed_returns_generator() -> None:
    rng = set_global_seed(42)
    assert isinstance(rng, np.random.Generator)


def test_set_global_seed_is_reproducible_across_calls() -> None:
    set_global_seed(123)
    a_np = np.random.rand(5)
    a_py = [random.random() for _ in range(3)]
    set_global_seed(123)
    b_np = np.random.rand(5)
    b_py = [random.random() for _ in range(3)]
    np.testing.assert_array_equal(a_np, b_np)
    assert a_py == b_py


def test_set_global_seed_returns_deterministic_generator() -> None:
    rng_a = set_global_seed(7)
    vals_a = rng_a.standard_normal(10)
    rng_b = set_global_seed(7)
    vals_b = rng_b.standard_normal(10)
    np.testing.assert_array_equal(vals_a, vals_b)


def test_set_global_seed_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        set_global_seed(-1)


def test_torch_seed_applied_if_torch_imported() -> None:
    import torch

    set_global_seed(99)
    a = torch.randn(5)
    set_global_seed(99)
    b = torch.randn(5)
    assert torch.equal(a, b)
