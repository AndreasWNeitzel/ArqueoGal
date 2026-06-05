# Protocol: interstellar extinction & reddening correction

**Status:** Implemented (2026-04-29). Module: `src/arqueogal/data/extinction.py`. Tests: `tests/data/test_extinction.py` (26 tests). Gallery: `scripts/gallery/plot_28_extinction_corrections.py` → `reports/gallery/28_extinction_corrections/`.
**Authoring trigger:** literature review of 2026-04-29 (literature-grounder + galactic-archaeology reviewer; full transcript in conversation log).

## 1. Why this protocol

The Hon+2021 TESS asteroseismic-giant priority target reaches G ≈ 16 and into the Galactic plane (|b| < 10°), where A_V can exceed 3 mag. Without an explicit dereddening recipe applied identically at training and inference, the ML model conflates intrinsic colour with extinction and the [Fe/H] / [α/M] residual develops a slope vs A_V — the failure mode Hattori+2024 (arXiv:2404.01269) explicitly recommends filtering on E(B−V) < 0.1 to avoid.

This protocol pins the dereddening recipe at v1 frozen-stats time so train and inference apply byte-identical transforms. The protocol is part of the v1 contract; any change requires re-deriving the basis fingerprint and is treated as a methods-paper-level revision.

## 2. The recipe (Option D, hybrid)

| Surface | Treatment | Reference |
|---|---|---|
| 2MASS J / H / Ks | Explicit dereddening: `mag_dered = mag − A_V · (A_λ/A_V)` | Yuan, Liu & Xiang 2013, MNRAS 430, 2188 |
| AllWISE W1 / W2 | Same recipe | Yuan+2013 |
| Gaia BP / RP Hermite coefficients | **Not** re-dereddened. Ye+2024 instrumental flux correction (mandatory upstream) absorbs wavelength-dependent systematics; A_V kept as residual auxiliary feature | Ye+2024 (arXiv:2411.19105) |
| A_V source | Fused: Edenhofer+2024 (d ≤ 1.25 kpc) → Lallement+2022 (1.25 < d ≤ 3 kpc) → SFD (d > 3 kpc) → neighbourhood-median fallback | Edenhofer+2024 (arXiv:2308.01295), Lallement+2022 (A&A 664 A9), Schlegel+1998 (ApJ 500, 525) |
| Extinction law | Cardelli, Clayton & Mathis 1989 (CCM89) at fixed R_V = 3.1 | Cardelli+1989 ApJ 345, 245 |

**Frozen ratios (Yuan+2013):**

```
A_J  / A_V = 0.276
A_H  / A_V = 0.176
A_Ks / A_V = 0.112
A_W1 / A_V = 0.063
A_W2 / A_V = 0.050
```

These are exposed as `arqueogal.data.extinction.YUAN2013_AV_RATIOS` (`MappingProxyType`, runtime-immutable). The full law is `arqueogal.data.extinction.DEFAULT_EXTINCTION_LAW`, a frozen dataclass whose `.fingerprint()` method round-trips to a JSON-able dict for sidecar provenance.

**Why not coefficient-level CCM89 on the Hermite basis?** Both 2026-04-29 reviewers confirmed no published per-coefficient extinction operator on the 55+55 BP/RP Hermite basis exists. Ye+2024 already absorbs wavelength-dependent flux systematics; applying CCM89 to the Hermite coefficients in addition is double-counting. Av is retained as a *feature* alongside the dereddened broadbands so the encoder still sees the residual XP extinction signal — but the *transform* is broadband-only.

**Why fixed R_V = 3.1?** Schlafly+2016 (ApJ 821, 78) measures σ(R_V) ≈ 0.18 across sightlines; the propagated bias on [Fe/H] for E(B−V) < 0.5 is ≤ 0.02 dex, well below the 0.05 dex precision target. AspGap, SHBoost, Ye+2024, and Hattori+2024 all fix R_V = 3.1.

## 3. Trust flags

Three booleans emitted alongside `av_los`:

- `av_is_neighborhood_fallback`: True iff `av_los_source == 3` (no per-sightline 3D dust map fired).
- `av_distance_prior_dominated`: True iff `parallax_over_error < 5`. The Bailer-Jones distance is then Galactic-prior-dominated and the dust-map A_V inherits that uncertainty.
- `av_neighbourhood_high_dispersion`: True iff `av_nbhd_std > 0.5 mag`. Patchy-extinction sightline; dust-map A_V less reliable.

These do **not** demote the per-element release tier (consistent with the v5 simplification that retired `dist_prior_dominated` from gating). They are diagnostic-only, exposed for consumer-side filtering and methods-paper plots.

## 4. Public API

```python
from arqueogal.data.extinction import apply_extinction_corrections

df = apply_extinction_corrections(df)  # adds av_los, av_los_source, three trust flags,
                                       # j_mag_dered, h_mag_dered, k_mag_dered,
                                       # w1_mag_dered, w2_mag_dered.
```

The function is the single entry point for callers (Stream 1 ingestion, Stream 2 enrichment, Stream 3 inference). It does not mutate the input by default.

## 5. Validation diagnostics

Three plot families (`reports/gallery/28_extinction_corrections/`):

1. **`av_provenance.pdf`** — distance × |b| scatter coloured by which dust-map layer fired; A_V vs distance per source. Confirms the priority logic.
2. **`dereddening_lever_arms.pdf`** — per-band raw vs dereddened scatter coloured by A_V, with the Yuan+2013 reference slope overlaid. Deviation from the reference slope at any band is a regression.
3. **`av_trust_flags.pdf`** — stacked bar of the three trust flags' firing rates.

For methods-paper §5, the additional three diagnostics that prove the dereddening worked on real data:

- **Residual [M/H] (and [α/M]) vs A_V**, binned in A_V quartiles, on the Stream-1 holdout. Hybrid-D wins iff the residual slope flattens vs the "A_V as feature only" baseline.
- **Per-Galactic-quadrant residual heatmap**: a model leaking extinction shows structured residual at low |b|.
- **(BP − RP)_0 vs predicted [α/M]**: intrinsic colour after dereddening should be uncorrelated with [α/M]; any correlation is the smoking gun for under-dereddening.

These three are the methods-paper-figure-5 set. They cannot be produced until the v2 ensemble runs both configs side-by-side; that ablation is the next item once Stream-2 plumbing lands.

## 6. Frozen-stats interaction

The dereddened broadband columns (`j_mag_dered` … `w2_mag_dered`) replace the raw broadbands in `DEFAULT_AUX_COLS`. The frozen-stats z-score is therefore computed on the dereddened columns. **Refitting frozen stats without re-running this dereddening step is a contract violation** — the basis fingerprint will change and `assert_frozen_stats_match()` at inference will fail loud. This is the intended invariant.

## 7. References (sourced 2026-04-29)

- Cardelli, Clayton & Mathis 1989, ApJ 345, 245 — CCM89 extinction law.
- Yuan, Liu & Xiang 2013, MNRAS 430, 2188 — IR A_λ/A_V ratios.
- Wang & Chen 2019, ApJ 877, 116 — alternative IR ratios (within 0.01 of Yuan+2013); supplied as a comparison dataclass.
- Schlafly+2016, ApJ 821, 78 — R_V variance bound.
- Edenhofer+2024, A&A 685 A82 (arXiv:2308.01295) — 3D dust < 1.25 kpc.
- Lallement+2022, A&A 664 A9 — 3D dust 1.25–3 kpc.
- Schlegel, Finkbeiner & Davis 1998, ApJ 500, 525 — SFD beyond 3 kpc.
- Bailer-Jones+2021, AJ 161, 147 — photogeometric distances.
- Ye+2024, arXiv:2411.19105 — XP instrumental flux correction (NOT extinction).
- Khalatyan+2024 (SHBoost), arXiv:2407.06963 — Av-as-feature precedent.
- Li+2024 (AspGap), ApJ 974, 42 / arXiv:2309.14294 — hybrid dereddening precedent.
- Hattori 2024, arXiv:2404.01269 — explicit E(B−V) < 0.1 filter recommendation.
- Zhang & Green 2023, MNRAS 524, 1855 / arXiv:2303.03420 — joint posterior baseline.
