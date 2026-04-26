# ADR-0002: Per-coefficient z-scoring of Hermite coefs at feature-matrix emit

**Date**: 2026-04-19 · **Status**: Accepted, in production

## Context

Initial Pipeline 1 feature matrix normalised Hermite coefficients by coefficient 0
(c_i/c_0 ratio per band), then z-scored c_0 separately. This preserves shape
information as ratios but does NOT standardise the ratios themselves.

Empirically, c_i/c_0 for i = 1..54 spans approximately 17 orders of magnitude on the
orthonormalised Hermite basis, low-order modes are O(0.1–1), high-order modes are
O(10⁻¹⁷). Standard ReLU/GELU networks with Glorot init effectively ignore the
high-order tail: the high-order coefs contribute ~0 to the first linear layer and
receive ~0 gradient.

## Decision

**Per-coefficient z-scoring at feature-matrix emit time.** Fit `(μ_i, σ_i)` for each
of the 110 coefs on the normal-pop training subset, apply in-place at emit, persist
stats in provenance sidecar with a deterministic basis fingerprint (`0d34b565...` for
v1). Stream 3 and any future inference MUST load these frozen stats, never refit.

## Rationale

Tested at 2026-04-19: per-coefficient z-scoring collapsed per-cell μ-bias on Teff by
79% and on [M/H] by 78% relative to the β=0.5 / non-z-scored baseline. This was the
**dominant** calibration fix of the sprint, far larger than 5-label vs 21-label,
larger than the β change, larger than any loss-function adjustment tried.

The input-side pathology was the root cause of structural miscalibration we'd been
attributing to head/architecture/loss choices through three diagnostic iterations.

## Alternatives rejected

- **Log-scale normalisation**: would handle the order-of-magnitude spread but would
  compress physical meaning and obscure small-delta modes important for subtle
  spectral features.
- **Learnable affine per-coefficient inside the network**: another 220 parameters
  competing for gradient, at a stage where a simpler deterministic fix works.
- **Truncated 43-D basis only**: reduces the problem but doesn't eliminate it (even
  within 43-D the magnitude spread is substantial). And a higher-dim basis stays
  useful for experimental-arm work.

## Consequences

- Stream 3 inference must load frozen v1 stats. This is a hard contract, any refit
  on Stream 3 silently invalidates the inference because the network's first-layer
  weights assume specific scaling.
- Integrity check required at Stream 3 build time: verify basis fingerprint matches
  `0d34b565...`. Halt with clear error on mismatch.
- This is the single most important input-preprocessing decision of Pipeline 1 v1.
  Methods paper should feature it.

## Methodology note

The initial framing pointed at the 21-D jump or the loss function; the real issue
was input-side magnitude disparity, which the diagnostic sequence should have
surfaced earlier. Noted as a failure mode to watch for in future: when calibration
iterations aren't converging, re-check input-side preprocessing before adjusting
architecture or loss.
