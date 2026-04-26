# Protocol: empirical-Bayes shrinkage τ-hyperparameter sweep

**Status:** Scoped. Computation pending.
**Authoring trigger:** bayesian_uq.md MAJOR-1; META_META §8 P1-5.

## 1 Why this protocol

The production τ = 50 in the empirical-Bayes shrinkage calibration
(`uncertainty.shrunken_per_cell_per_label_scale`; ADR-0003) is hardcoded.
A τ-sensitivity sweep gives the methods paper a defensible justification:
"τ = 50 was empirically optimal across τ ∈ {10, 20, 30, 50, 100, 150}
under the validation-set ECE metric". Without the sweep, a hostile
referee asks "why 50 and not 100?" and the answer is "intuition".
a weak position.

## 2 Plan

### Step A: τ grid

τ ∈ {10, 20, 30, 50, 100, 150}. Six runs total.

### Step B: per-τ calibration

For each τ:
1. Load the trained Pipeline 1 v1 ensemble (no retraining needed.
   calibration is post-training).
2. Run inference on Stream 2 (validation).
3. Apply `shrunken_per_cell_per_label_scale(τ=τ)` to per-cell σ.
4. Compute per-element ECE (Expected Calibration Error) on the
   validation set: ECE = Σ_b |p_b − q_b| × (n_b / N), where p_b is
   the predicted-σ band and q_b the empirical-σ band.
5. Compute per-element coverage at 68 / 95 / 99 % confidence levels.
   Acceptance: coverage at nominal σ is within ±5 percentage points
   for τ to be acceptable.

### Step C: methods-paper figure

Plot per-element ECE vs τ on a 5-panel grid; mark τ = 50 with a
vertical line; identify the optimum.

## 3 Effort estimate

1 week:
- 2 days to instrument the calibration code with the τ grid.
- 1 day per τ for inference + calibration on Stream 2 (parallelisable).
- 2 days for ECE computation + figure + writing.

Computationally cheap (one forward pass per τ on Stream 2 ~150k rows).
Total ~1 GPU-day or less.

## 4 Outcome contract

The methods paper §3.3 cites this sweep with the optimal τ. If the
optimum differs from 50 by more than the ECE-equivalent of ±0.5
percentage points, the production code is updated and a v1.1 catalog
is emitted. Otherwise, τ = 50 stays and the sweep figure is the
empirical justification.
