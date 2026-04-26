# ADR-0008: 4-block physics-motivated Cholesky covariance structure

**Date**: ~2026-04-18 · **Status**: Accepted; 5-label production head simplified to
single 5×5 block

## Context

Pipeline 1 v1 original design targeted 21-label output with a full 21×21 covariance.
Full-cov Cholesky has 231 lower-triangular entries (21 diagonal + 210 off-diagonal),
which is capacity-hungry, numerically unstable at tails, and physically over-expressive
given the 21 labels' natural grouping.

## Decision

**4-block physics-motivated structure**: atmospheric {Teff, log g, [M/H]} + α-process
{Mg, Si, Ca, Ti} + Fe-peak {Fe, Mn, Ni, Cr} + light {C, N, O, Na, Al, K}. Diagonal-only
tail for 4 remaining labels. Cross-block entries enforced to zero by zero-init + scatter
writes to within-block positions (not via mask multiplication at forward time).

`CovarianceBlockLayout` dataclass carries both `label_order_block` (internal model
order, with within-block contiguity) and `label_order_human` (Tier-order, for external
use), with named conversion methods (`reorder_block_to_human`, `reorder_human_to_block`)
and unit-tested permutation roundtrips.

**For 5-label production head** (ADR-0001): simplified to a single 5×5 full-Cholesky
block, the 4-block structure only matters for 21-label configurations.

## Rationale

- Physics covers the correlations: atmospheric parameters are genuinely covarying
  (Teff-log g-[M/H] degeneracies); α elements covary as a group (shared production
  pathway); Fe-peak elements covary internally; light elements are a loose group.
- 4-block gives 6+10+10+21+4 = 51 Cholesky params vs 231 for full, 4.5× fewer,
  enormously more trainable with heterogeneous label-completeness.
- Zero-init + in-place writes (rather than mask-at-forward) makes cross-block zeros
  exact at the data-structure level; cannot leak through forward passes.
- Dual label ordering prevents silent bugs where reliability diagrams get the wrong
  label mapping. The contract is explicit and tested.

## Alternatives rejected

- **Full 21×21 covariance**: too capacity-hungry on the 6 GB RTX 3060 and on DR19
  label noise.
- **Tier-based blocks** (Tier 1 dense, Tier 2 dense, Tier 3 diagonal), was the
  original design before the refactor but was wrong for the new spec; physics-
  motivated blocks are more coherent.
- **Full-cov with structure implied by regularisation**: doesn't hold at numerical
  edge cases; PD-enforcement harder.

## Consequences

- Retained in code even after 5-label simplification; supports methods-paper
  comparison work and any future 21-label (or other-count) experiments.
- Any future widening of blocks should be evidence-backed (information-content audit
  showing cross-block covariances are material).

## Peer-review note

No disagreement; the refactor from tier-based to physics-motivated blocks surfaced
several subtle correctness requirements (label ordering, zero-init vs masking,
cross-block-zero enforcement) that the refactor handled cleanly.
