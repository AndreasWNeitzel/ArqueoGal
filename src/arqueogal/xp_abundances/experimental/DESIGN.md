# `arqueogal.xp_abundances.experimental`. Design

## Status

**Segregated exploration arm**. Does NOT contribute to D-Cat-b unless promoted to main via
the bar in research_brief §8.6. Does NOT import from `main/` and is not imported by it.
Shared code goes to `arqueogal.utils`.

All experimental work gets its own subdirectory below, its own configs under
`configs/experimental/`, its own tests under `tests/xp_abundances/experimental/`, and its
own model checkpoints under `models/experimental/`.

## Planned subdirectories

See research_brief §8 for full rationale.

```
arqueogal.xp_abundances.experimental
├── normalising_flows/, §8.1 Conditional NF over label vector (zuko / nflows).
│                          Non-Gaussian, multi-modal label posteriors; valuable for
│                          transition-region [α/M] and boundary stars.
├── transformer/, §8.2 Transformer-on-spectra with masked prediction
│                          (StellarPerceptron-style; Leung & Bovy 2024, arXiv:2411.04750).
│                          Missing-input tolerance, natural multi-task.
├── physics_informed/, §8.3 Forward-modelled XP spectrum (PHOENIX/MARCS convolved to XP
│                          response) as prior regulariser. Only path to honest OOD detection.
│                          Gateway to SBI.
├── diffusion/, §8.4 Conditional diffusion for label density estimation
│                          (Rouhiainen+2024 style). B&S 2024 flag as natural next step from
│                          their CL framework.
└── multimodal_cl/, §8.5 Buck & Schwarz 2024-style CLIP on RVS + XP.
                            KEY CONSTRAINT: RVS is training-time only; inference is XP-only
                            (preserves full G ≤ 17.65 applicability). Test on DR19 overlap
                            (~700 k, 15× B&S's 44 780). Replace B&S's 1-layer MLP XP encoder
                            with attention; replace k-NN with heteroscedastic regression
                            head + ensemble. Honest baseline: XP-only main must match or
                            exceed B&S's R²(Teff)≈0.987, R²([α/M])≈0.849 on same k-NN
                            protocol before the multimodal variant is declared "better".
```

## Promotion rule (research_brief §8.6)

An experimental method is promoted to `main/` only when all four hold:

1. Passes every validation test `main/` passes (research_brief §9 in full).
2. Beats `main/` by ≥ 0.02 dex on ≥ one Tier-1 label, OR ≥ 0.03 dex on ≥ one Tier-2 label,
   on the hold-out test set.
3. Passes the §9.2 information-content audit at least as cleanly as `main/`.
4. Matches or exceeds `main/`'s calibration quality (per-cell reliability diagram error
   ≤ main's).

Promotion is a conscious, documented act, a PR with the comparison report card, not a
refactor commit.

## Hard rules

- Absolutely no cross-imports with `main/`. Shared code → `arqueogal.utils`.
- Experimental configs under `configs/experimental/`. Experimental notebooks under
  `notebooks/xp_abundances/experimental/`. Experimental tests under
  `tests/xp_abundances/experimental/`.
- Each experimental subdir owns its own `README.md` or sub-`DESIGN.md` when non-trivial.
- Pre-registration: before running a new experimental arm, note the hypothesis, success
  criteria, and pass/fail thresholds. Post-hoc declarations of success are a code-review
  rejection.
