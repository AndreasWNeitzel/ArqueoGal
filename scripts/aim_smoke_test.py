"""Phase 5 smoke test: demonstrate aim experiment tracking on a toy fit.

Runs three short trials with different hyperparameters so the aim UI has
something comparable. Not part of the real training pipeline. Safe to delete
once the aim integration pattern is ported into ``main/training.py``.
"""

from __future__ import annotations

import aim
import numpy as np


def fit_one(lr: float, n_steps: int, seed: int) -> None:
    """Minimal gradient-descent fit of ``y = 2 x + 1 + noise``, tracked via aim."""

    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=256)
    y = 2.0 * x + 1.0 + rng.normal(0.0, 0.1, size=256)

    run = aim.Run(experiment="aim-smoke-test")
    run["hparams"] = {"lr": lr, "n_steps": n_steps, "seed": seed}

    w, b = 0.0, 0.0
    for step in range(n_steps):
        y_pred = w * x + b
        resid = y_pred - y
        loss = float(np.mean(resid**2))
        w -= lr * float(np.mean(2.0 * resid * x))
        b -= lr * float(np.mean(2.0 * resid))
        run.track(loss, name="train_loss", step=step)
        run.track(w, name="w_hat", step=step)
        run.track(b, name="b_hat", step=step)

    print(f"lr={lr}  steps={n_steps}  seed={seed}  final w={w:.3f}  b={b:.3f}  loss={loss:.4f}")


if __name__ == "__main__":
    for lr, n_steps, seed in [(0.05, 200, 0), (0.01, 500, 1), (0.10, 100, 2)]:
        fit_one(lr, n_steps, seed)
