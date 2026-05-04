# Contributing to ArqueoGal

This repository is the personal development workspace of Andreas Neitzel
(Co-I, FCT 2024.15303.PEX) and is the canonical source of code for the
Pipeline 1 (`xp_abundances`) deliverable. External contributions are
welcome but coordination first is appreciated, given the project's
publication-bound deliverable schedule.

## Before you start

1. Read [`docs/research_brief.md`](docs/research_brief.md) for the
   scientific context, the tier-promotion protocol (§3.3), and the
   information-content audit framework (§9.2).
2. Read [`docs/data_acquisition.md`](docs/data_acquisition.md) for the
   data-pipeline contract: TAP endpoints, Gaia DR3 corrections, Ye+2024
   XP preprocessing, dust-map composition, provenance sidecars.
3. Skim [`docs/decisions/`](docs/decisions/) for the architectural
   decisions on offer, especially ADR-0015 (the v5 release-tier
   simplification).
4. Skim the per-module `DESIGN.md` files under `src/arqueogal/`. They
   are the contracts.

## Environment

The README "Environment" section documents both the CPU-only contributor
setup (the same dep set CI installs) and the maintainer's full RAPIDS / CUDA
stack. Use the CPU-only path unless you are building or running training
checkpoints. The CI workflow exercises the full ``tests/xp_abundances/main/``
tree and ``tests/data/test_release_pipeline.py``, filtered with
``-m "not gpu and not slow and not stress"`` so it stays portable; passing
``ruff check`` and ``pytest`` locally with that marker filter implies passing
on push.

## Development workflow

- Branch off `main` for any non-trivial change.
- Format and lint with `ruff format` + `ruff check` before committing.
- Test with `pytest` (the CI suite is in
  [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).
- Any change to a Parquet column shape, name, or dtype must update the
  matching `DESIGN.md` in the **same commit** (invariant 15).
- Any new data-layer module must ship with a production-size stratified
  smoke test; 100-row tests on `gaiaxpy` or other external libraries
  are integration placebo (AGENTS.md "Testing expectations").

## Pull requests

- Keep PRs focused: one logical change per PR.
- Reference the relevant ADR or `docs/plan/*` file in the PR
  description.
- Include benchmark numbers for any performance-relevant change. The
  RTX 3060 6 GB VRAM limit is a hard constraint for training-loop
  changes.
- For research-flavoured changes (new tier promotion, new calibration
  recipe, new OOD detector), surface the change in a dedicated issue or
  PR comment so it can get its own review. Don't bundle research
  decisions with infrastructure changes.

## Reporting issues

- For reproducibility issues, attach the relevant `*.provenance.json`
  sidecar from the parquet you are working with.
- For inference-time NaN propagation, enable
  `XpFeatureAdapter(assert_finite=True)` to localize the offending
  column before reporting.
- For TAP-related failures (AIP, GAVO, ESA), include the query, the
  endpoint, and whether the inline-IN payload exceeded the AIP 100 KB
  ceiling (use `tap.batched_upload_fetch_df()` for large payloads).

## Citation

A `CITATION.cff` ships with the repo at version `1.0.0-pre`. It does not yet
carry a DOI: the Zenodo mint will accompany the v1 methods-paper submission.
Until then, cite by repository URL and commit SHA.

## License

[MIT](LICENSE).
