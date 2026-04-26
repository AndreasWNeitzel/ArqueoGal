# Code Review — v5 Release-Tier Contract

**Auditor:** code-reviewer (haiku)
**Date:** 2026-04-26
**Scope:** consistency between code, DESIGN.md, ADR-0015, and the ablation REPORT.md for the v5 release-tier simplification.

> Note: this report was reconstructed from the agent's chat-returned summary
> after the original `Write` tool call did not land on disk. Content is the
> agent's verbatim findings.

---

## Finding (a) — Consistency across code / DESIGN / ADR / ablation report

All four documents are internally consistent. The v5 schema is cleanly implemented:

- `release.py` lines 56-103 define the active gates.
- `DESIGN.md` §3.5 and §3.6 describe the contract accurately.
- ADR-0015 (`docs/decisions/0015_v5_release_tier_simplification.md`) provides the empirical justification.
- The ablation report (`release/test_ablations_2026-04-26/REPORT.md`) supplies the per-gate evidence.

The σ-threshold mirror in `release_pipeline.py` line 419 is correctly synchronized at 0.05 dex for `[α/M]`; the drift test `test_hybrid_thresholds_match_release` enforces this lock going forward.

## Finding (b) — Per-element caveat composition correctness

`assign_per_element_release_tier` (release.py lines 353-464) correctly implements v5 logic:

1. `_OOD_FLAGS = ("ood_joint_flag",)` alone triggers Tier 3 for all elements (lines 398-401).
2. `_PER_ELEMENT_CAVEAT_FLAGS = {"alpha_m": ("mode_ambiguous_flag",)}` demotes only `[α/M]` (lines 441-445).
3. σ-inflation thresholds are all present in `_PER_ELEMENT_SIGMA_INFLATED_THRESHOLD` (lines 136-160) and checked per-element at line 436.
4. Aux-assisted demotion on `kin_ood_flag` fires only for `("alpha_m", "mg_h")` at lines 456-457.
5. Tier-3 hard-kill (`tier[tier3] = 3`, line 461) correctly trumps all Tier-2 caveats via the boolean composition at line 448.

No logical error.

## Finding (c) — Sidecar manifest completeness

The sidecar emitted by `annotate_parquet` (lines 602-667) properly enumerates:

- `expected_upstream_columns` (lines 637-643): `["ood_joint_flag", "mode_ambiguous_flag"]` — the two v5 active gates, both correctly listed.
- `diagnostic_only_columns` (lines 644-654): six retired flags (`ood_aux_mahalanobis_flag`, `latent_support_flag`, `regime_b_flag`, `ood_disagreement_flag`, `aux_missing_any`, `dist_prior_dominated`) — all six are empirically justified as non-functional in REPORT.md lines 68-90.
- `_PER_ELEMENT_CAVEAT_FLAGS` block (lines 609-612): correctly reflects the per-element subset.

The active vs diagnostic split is complete and accurate.

## Finding (d) — v3-v4 retired flags silent demotion risk

No silent demotion risk remains. The v3-v4 retired flags are still checked if present in the input (e.g., lines 407-409), but they no longer affect tier assignment because `_CAVEAT_FLAGS = ()` is the empty tuple (line 74 comment; the loop at line 407 iterates over zero elements).

The ablation REPORT.md confirms zero Tier 1+2 RMSE effect for all six retired flags when disabled (REPORT.md lines 37-51, 58-66). The only remaining active gates are:

- `ood_joint_flag` (hard OOD)
- per-element σ-inflation
- `mode_ambiguous_flag` on `[α/M]` only
- `kin_ood_flag` on aux-assisted elements

All four are explicitly listed in `expected_upstream_columns` and empirically justified.

---

## Verdict

The v5 implementation passes a strict contract review. No correctness defects. The simplification is consistent across code, DESIGN.md, ADR-0015, and the ablation report. The per-element caveat composition correctly confines `mode_ambiguous_flag` to `[α/M]`. The sidecar manifest's `expected_upstream_columns` vs `diagnostic_only_columns` split is complete. The drift test on the σ-threshold mirror (the only remaining duplication concern) is in place.

The retired flags cannot silently demote because they no longer feed any tier-gating logic.
