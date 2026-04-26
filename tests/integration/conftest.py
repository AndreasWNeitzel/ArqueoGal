"""Integration-test conftest: opt-in switch for stress-battery tests.

Use ``pytest tests/integration --run-stress`` to enable the heavy stress
battery (5-fold CV, leakage check, multi-spectrum, etc.). By default these
tests are skipped because they require:

- a CUDA device with at least 6 GB VRAM,
- the production strong-contrastive-v2 ensemble on disk,
- the full Stream-1 and Stream-3 feature parquets.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-stress",
        action="store_true",
        default=False,
        help="Run the hybrid stress-battery integration tests (slow, GPU).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-stress"):
        return
    skip_stress = pytest.mark.skip(reason="opt-in only; pass --run-stress to enable")
    for item in items:
        if "stress" in item.keywords:
            item.add_marker(skip_stress)
