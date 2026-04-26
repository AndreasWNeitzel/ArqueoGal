# ADR-0012: NaN-safe inference driver with aux-missingness flag system

**Date**: 2026-04-19 · **Status**: Accepted, Phase 2 complete

## Context

Pre-flight halt before Stream 3 inference (2026-04-19) surfaced a latent bug in
`run_pipeline1_inference.py`: it did not mirror the `np.nan_to_num(..., nan=0.0)`
step that `training.py` applies at the data-loader boundary. Result: a single NaN
in any aux feature → NaN trunk → NaN μ → NaN Σ → NaN predictions, with no OOD flag
raised (Mahalanobis covers the 108-D XP block only).

## Decision

1. **Mirror training's NaN handling at inference time**: inference driver applies
   `nan_to_num(nan=0.0)` at the same boundary.
2. **Aux-missingness flag system**: per-star boolean flags in Pipeline 1 output:
   - `ir_missing_flag` (any of j/h/k/w1/w2 missing pre-imputation)
   - `parallax_missing_flag` (parallax missing or abnormally high error)
   - `extinction_missing_flag` (all three dust maps failed for the sky position)
3. **Documentation contract**: D-Cat-b release explicitly states which aux features
   are consumed at inference, what happens when any are missing, and which flags
   raise.

## Rationale

- Without NaN sanitisation at inference, any missing aux feature silently corrupts
  predictions without an OOD flag. This is a latent data-quality bug that would
  have surfaced in production.
- The aux-missingness flags close the coverage gap of the existing Mahalanobis OOD
  (XP-only) by making aux-feature unreliability a first-class per-star attribute.
- Downstream users can filter or weight by these flags without guessing.

## Alternatives rejected

- **NaN sanitisation without flags**: silent degradation. Violates honesty-under-
  uncertainty.
- **Extend Mahalanobis OOD to cover aux block**: more architecturally invasive than
  boolean flags; adds opportunity for error without clear marginal value.

## Consequences

- `scripts/run_pipeline1_inference.py` is the production inference driver with
  NaN-safe handling and aux-missingness flags. 18/18 tests passing.
- Release-documentation task expanded: D-Cat-b release explicitly describes each
  aux-missingness flag and its downstream implication.
- The bug's pre-flight discovery (via halt-and-ratify process on Stream 3 launch)
  is itself a process vindication, halting before fetch caught a production bug
  that would have cost a full fetch + re-run otherwise.

## Methodology note

The bug was caught by the pre-flight audit; the scope expansion to flags for IR,
parallax, and extinction (not only IR) fell out of the same audit. The
halt-and-ratify cadence before fetch launches proved valuable here.
