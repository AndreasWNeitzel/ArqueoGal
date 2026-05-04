"""Unit tests for the extinction-recipe ablation harness.

Scope: pure numeric helpers (slope recovery, verdict logic). The
``_build_synthetic_ablation_fixture`` function and its full-pipeline tests
were removed on 2026-04-29 when synthetic data fabrication was banned
across the ArqueoGal codebase. End-to-end ablation tests now run against
real Stream-1 holdout predictions, which live outside the unit-test layer.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from ablation_extinction_recipe import (  # noqa: E402
    AblationConfig,
    _residual_vs_av_slope,
    _verdict,
)


def test_residual_vs_av_slope_recovers_known_slope():
    """Pure-math helper test: feed a linear residual-vs-Av relation and
    verify the slope-fit recovers the slope. Inputs are numeric arrays
    designed only to exercise the function — not stellar data."""
    rng = np.random.default_rng(0)
    av = rng.uniform(0, 1.5, 5000)
    residual = 0.10 * av + rng.standard_normal(5000) * 0.02
    out = _residual_vs_av_slope(residual, av, n_bins=10, av_max_quantile=0.99)
    assert out["slope"] == pytest.approx(0.10, abs=1e-2)
    assert out["x_centre"].size == 10


def test_residual_vs_av_slope_handles_short_input():
    out = _residual_vs_av_slope(
        np.array([0.0, 0.1]), np.array([0.0, 0.1]), n_bins=4, av_max_quantile=0.99
    )
    assert np.isnan(out["slope"])
    assert out["x_centre"].size == 0


def test_verdict_pass_when_hybrid_slope_smaller():
    slopes = {
        "teff": {
            "baseline_slope": 0.5,
            "hybrid_slope": 0.05,
            "baseline_slope_se": 0.01,
            "hybrid_slope_se": 0.01,
        },
        "mh": {
            "baseline_slope": -0.10,
            "hybrid_slope": -0.01,
            "baseline_slope_se": 0.001,
            "hybrid_slope_se": 0.001,
        },
    }
    out = _verdict(slopes, AblationConfig())
    assert out["verdict"] == "hybrid-D wins"
    assert all(v["passes"] for v in out["per_element"].values())


def test_verdict_inconclusive_when_no_improvement():
    slopes = {
        "teff": {
            "baseline_slope": 0.5,
            "hybrid_slope": 0.49,
            "baseline_slope_se": 0.01,
            "hybrid_slope_se": 0.01,
        },
    }
    out = _verdict(slopes, AblationConfig())
    assert out["verdict"] == "inconclusive"
    assert not out["per_element"]["teff"]["passes"]
