# Pipeline 1 inference-driver hardening — `scripts/run_pipeline1_inference.py`

Pre-release hardening for D-Cat-b (Aug 2026). Two tasks:

1. Close the train-vs-inference `nan_to_num` mismatch that let a single NaN
   in any aux column silently propagate to NaN predictions with no gate.
2. Extend the output schema with a general auxiliary-missingness flag
   system, separate from the existing Mahalanobis/ensemble OOD channel.

No retraining, no checkpoint mutation. Diff is scoped to the driver, the
module-level constants, the provenance block, and the driver test file.

---

## 1. `nan_to_num` boundary in `training.py`

- **File:line**: `src/arqueogal/xp_abundances/main/training.py:153`
- **Call**: `np.nan_to_num(arrs["X"], copy=False, nan=0.0, posinf=0.0, neginf=0.0)`
- **Columns**: applies to the full **`(N, input_dim)`** feature matrix
  `arrs["X"]`, where `input_dim = len(bp_coef_cols) + len(rp_coef_cols) +
  len(xp_scalar_cols) + len(residual_cols) + len(aux_cols)` per
  `FeatureLayout.input_dim` (production: 139-D = 54 BP + 54 RP + 2 c0
  scalars + 3 residuals + 26 aux).
- **Ordering**: at training this is the last step before Dataset/DataLoader
  construction. It runs AFTER the XP-NaN row drop (training.py:143–151)
  drops any row with a non-finite coefficient in the 108-D XP block, so in
  practice it imputes NaN only in residuals and aux priors — "stars
  outside a given 3D dust-map's coverage" is the canonical example called
  out in the in-code comment (training.py:136–138).

## 2. Mirror implementation in the inference driver

- **File:line**: `scripts/run_pipeline1_inference.py:819`
- **Call**: identical signature — `np.nan_to_num(X, copy=False, nan=0.0,
  posinf=0.0, neginf=0.0)` on the assembled flat feature matrix.
- **Ordering**: runs AFTER `_assemble_feature_matrix` and AFTER
  `_compute_aux_missingness_flags` (flags must read the RAW frame,
  see §4), and BEFORE `_build_loader` + `predict_ensemble`.
- **Difference from training**: inference does NOT drop rows with NaN in
  the 108-D XP block — those rows are kept, their Mahalanobis OOD score
  returns NaN as the documented "XP-non-finite" gate, and downstream
  consumers can filter on that. Aux-missingness is handled by the new
  flags; the NaN is imputed to 0 so the forward pass is finite.

## 3. Aux-missingness flag definitions

Driver-level module constants (`scripts/run_pipeline1_inference.py`):

```
IR_COLS              = ("j_mag", "h_mag", "k_mag", "w1_mag", "w2_mag")
PARALLAX_COL         = "parallax"
PARALLAX_ERROR_COL   = "parallax_error"
PARALLAX_OVER_ERROR_MIN = 5.0
EXTINCTION_COLS      = ("av_edenhofer", "av_sfd", "av_lallement")
```

All column names are drawn from `FeatureLayout.aux_cols`
(`src/arqueogal/xp_abundances/main/data.py:73–87`, `DEFAULT_AUX_COLS`).

Flag rules (all read the INPUT frame BEFORE `nan_to_num`):

| Flag | Rule |
|------|------|
| `ir_missing_flag` | True if ANY of `IR_COLS` is NaN for that row. |
| `parallax_missing_flag` | True if `parallax` is NaN OR `parallax_error` is NaN OR `parallax / parallax_error < 5.0` (low-S/N treated as informationally missing). Division by zero also flags True. |
| `extinction_missing_flag` | True if ALL THREE of `EXTINCTION_COLS` are NaN. One successful dust-map entry keeps the flag False. |
| `aux_missing_any` | Logical OR of the three above. |

Cols missing from a particular `FeatureLayout.aux_cols` contribute
`False` to their respective channel — lets the driver run against trimmed
layouts without spurious True flags. The provenance sidecar records
which configured cols the layout actually carries.

**Independence from OOD**: per §Scope/Task 2, aux-missingness is NOT
folded into `ood_joint_flag`. `ood_joint_flag` continues to be driven
exclusively by Mahalanobis (108-D XP block) + ensemble-disagreement
ratio. Rationale: aux-missingness is a data-availability signal, not
evidence that the XP spectrum is OOD. Downstream consumers combine as
their policy dictates.

## 4. Output schema delta

Four new boolean columns appended to the Parquet output, after
`selection_prob`:

- `ir_missing_flag` (bool)
- `parallax_missing_flag` (bool)
- `extinction_missing_flag` (bool)
- `aux_missing_any` (bool)

Provenance sidecar gains an `aux_missingness` block with sub-fields
`definitions`, `layout_resolution`, `flag_rates`, `flag_counts`,
`independence_note`, and `nan_handling`. The `flag_rates` / `flag_counts`
are the run's observed rates.

Driver INFO log line added at feature-assembly time:

```
aux-missingness: ir=0.xxxx parallax=0.xxxx extinction=0.xxxx any=0.xxxx
```

## 5. Test status

- **Before**: `tests/scripts/test_run_pipeline1_inference.py` — 10/10
  passing.
- **After**: 18/18 passing. Added 8 new tests:
  - `test_nan_to_num_regression_produces_finite_predictions` — injects
    NaN in each of the 10 aux cols one row at a time, asserts every μ and
    σ is finite. This is the core regression test for the training-vs-
    inference mismatch.
  - `test_ir_missing_flag_truth_table`
  - `test_parallax_missing_flag_truth_table` — covers NaN parallax, NaN
    error, low S/N (=1.0), healthy S/N (=20), boundary at exactly 5.0,
    and divide-by-zero.
  - `test_extinction_missing_flag_requires_all_three`
  - `test_compound_aux_missing_any_is_logical_or` — exhaustive truth-table
    across all seven non-trivial subsets.
  - `test_ood_joint_flag_independent_of_aux_missingness` — sets all aux
    NaN with clean XP, asserts Mahalanobis score stays finite (i.e.
    aux-missingness does NOT contaminate the XP-distribution OOD metric).
  - `test_output_schema_includes_aux_missingness_columns`
  - `test_provenance_records_aux_missingness_definitions`

Broader test sweep: `tests/xp_abundances/` + `tests/scripts/` —
**241/241 passing**.

## 6. Downstream consumer notes

- Existing callers reading `ood_joint_flag`, `ood_disagreement_flag`,
  `ood_mahalanobis_score`, `regime_b_flag`, `selection_prob`, the per-
  label μ/σ/epi/cov blocks: **no change**. All columns and their
  semantics are preserved byte-for-byte.
- New consumers who want a "predictions with incomplete aux inputs"
  filter can use `aux_missing_any`. Consumers who want a specific
  sub-channel (e.g. "RGB stars lacking WISE photometry") can use
  `ir_missing_flag` directly.
- The 2–5 % expected auxiliary-feature incompleteness at Stream 3 scale
  is now gated, not silenced: instead of 2–5 % NaN predictions with no
  flag, we get 100 % finite predictions with the 2–5 % marked via
  `aux_missing_any`. Downstream D-Cat-b aggregation can drop, down-
  weight, or passthrough at its discretion.
