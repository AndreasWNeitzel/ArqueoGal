# Phase 05. Release packaging

**Status: Underspecified. Deliverable deadlines are firm; release-format decisions are
not.**

## The three releases

| Code | Title | Due | Format expectations |
|---|---|---|---|
| D-Cat-b | XP abundance catalogue | Aug 2026 | Parquet + accompanying README + methodology |
| D5.1 | Population-classifier ML tool | Dec 2026 | Starfold (separate repo), release logistics handled there |
| D-Cat-d | Population membership probabilities | Feb 2027 | Parquet appended to the team all-sky catalogue; produced by Starfold from this repo's Pipeline 1 predictions |

## What's decided

- D-Cat-b release contains per-star: Tier 1 labels (5) + full covariance + OOD flags
  + Regime B flag + selection_prob + aux_missingness flags + tier marker (T1 or
  T1-caveat for log g).
- D-Cat-b Tier 1 = {Teff, log g (with caveat), [M/H], [α/M], [Mg/H]}. No other labels
  released per-star.
- D-Cat-b release documents the selection function (Ye retention × IR completeness),
  non-uniform σ by parameter region (cool-giant inflation up to 1.5×), Regime B
  exclusion (population-only).
- D-Cat-b release ships the v1 ensemble checkpoint and frozen preprocessing stats
  (per-coefficient Hermite z-score fingerprint) for reproducibility.

## What's NOT decided

- **D5.1 release logistics** (license, scope, install story), now Starfold's concern,
  not this repo's.
- **D-Cat-d integration with team catalogue.** Format and interface with the
  ArqueoGal all-sky catalogue unspecified in this workspace (team-side decision;
  Starfold produces the content, team integrates).
- **Methods paper(s)**: scope ambiguous now that Pipeline 2 methodology has moved
  out of tree. See `docs/plan/06_methods_paper.md`.

## Needs clarification

- D-Cat-d format and integration, may need coordination with PI Campante and
  Starfold.
- Whether `data/` provenance sidecars (currently mostly unpopulated for Pipeline 1
  ensemble outputs) need to be backfilled before release, or whether the release
  report is enough.
