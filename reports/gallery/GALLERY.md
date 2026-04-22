# ArqueoGal Visual Gallery

**Purpose.** Every stage of the ArqueoGal pipeline has a visualization here.
Walk top-to-bottom and you see the full data → model → science story without
reading code. Used for reporting (FCT deliverables, methods paper, internal
review) and for honest self-inspection during development.

**Version tracking.** Figures are labeled with the release they correspond to:
`v1` (frozen 2026-04-19, tag `pipeline1-v1-2026-04-19`), `v1.1` (2026-04-21,
inverse-frequency [M/H] weighting, pending tag), `v1.2` (2026-04-21,
σ-gate on downstream feature matrix, pending tag). Superseded-but-preserved
figures sit under `../pipeline*/run_a/` and are labeled as historical.

**Build.** Each stage has figures produced by `scripts/gallery/plot_<stage>.py`.
Re-emit the whole gallery via `scripts/gallery/build_all.sh` (to be added in
batch 7).

---

## Table of contents

| # | Stage | What it shows | README | Status |
|--:|---|---|---|---|
| 00 | [Data sources](00_data_sources/README.md) | Streams 1/2/3, APOGEE, Andrae+23, Hon+21 — sky footprint, counts, HRD | `00_data_sources/` | scaffold |
| 01 | [Gaia XP (raw)](01_gaia_xp_raw/README.md) | XP coef distributions, reconstructed SEDs by HRD cell, c0 vs G | `01_gaia_xp_raw/` | scaffold |
| 02 | [Ye+2024 correction](02_ye_correction/README.md) | Before/after SED, Δcoef distribution, NO_SYNTH_PHOT flag map | `02_ye_correction/` | scaffold |
| 03 | [Hermite reprojection](03_hermite_reprojection/README.md) | 110-coef reprojection fidelity, per-coef z-score pre/post | `03_hermite_reprojection/` | scaffold |
| 04 | [Extinction](04_extinction/README.md) | Edenhofer / Lallement / SFD / nbhd-median A_V side-by-side | `04_extinction/` | scaffold |
| 05 | [IR photometry](05_ir_photometry/README.md) | 2MASS JHK + AllWISE W1W2 coverage, missingness map | `05_ir_photometry/` | scaffold |
| 06 | [Selection function](06_selection_function/README.md) | ω_i = Ye × IR × ϖ × A_V — per-factor decomposition | `06_selection_function/` | scaffold |
| 07 | [APOGEE labels](07_apogee_labels/README.md) | Label coverage matrix, Meszaros+25 Δ per element, NaN rates | `07_apogee_labels/` | scaffold |
| 08 | [Kinematics](08_kinematics/README.md) | E–L_z, action plane, eccentricity, orbit-family fractions | `08_kinematics/` | scaffold |
| 09 | [Feature matrix](09_feature_matrix/README.md) | NN input (110 XP + 43 aux): schema + distributions | `09_feature_matrix/` | scaffold |
| 10 | [Contrastive pretraining](10_contrastive_pretraining/README.md) | Loss trajectory, positive-pair examples, halfway + final embeddings | `10_contrastive_pretraining/` | partial (halfway UMAPs exist) |
| 11 | [Supervised training](11_supervised_training/README.md) | Per-seed curves, grad norms, LR schedule, per-label NLL, inv-freq weights | `11_supervised_training/` | partial |
| 12 | [Pipeline-1 validation](12_pipeline1_validation/README.md) | Pred-vs-truth, residual Gaussians, σ reliability, coverage, residual stratified | `12_pipeline1_validation/` | partial (batch 2 adds stratified) |
| 13 | [Ensemble + uncertainty](13_ensemble_uncertainty/README.md) | Aleatoric vs epistemic, OOD Mahalanobis, disagreement, regime-B | `13_ensemble_uncertainty/` | scaffold |
| 14 | [Pipeline-1 inference](14_pipeline1_inference/README.md) | Stream-3 predictions: HRD, sky, chemistry, OOD rate map | `14_pipeline1_inference/` | scaffold |
| 17 | [Pipeline-1 regime diagnostics](17_pipeline1_regime_diagnostics/README.md) | `release_tier` composition vs G / Av / distance / sky / HR / chem + σ by tier + flag mix | `17_pipeline1_regime_diagnostics/` | v1 |
| 99 | [Methods-paper subset](99_methods_paper/README.md) | Curated paper-ready subset with final captions + PDF | `99_methods_paper/` | batch 7 |

Stages 15–16 (population-classifier features + UMAP/HDBSCAN classification)
were moved to `reports/gallery/archive/` on 2026-04-22 when population
classification spun out into the separate **Starfold** repository.

Legend: **scaffold** = directory + README only, no figures yet; **partial** =
some figures exist, more planned in later batches; **done** = all planned
figures exist and are current.

---

## Narrative read-through

**Stage 00 — sources.** Three streams feed the pipeline. Stream 1 is the
training corpus: APOGEE DR19 × Gaia DR3 (354 k RGB after cuts). Stream 2 is
TESS/Kepler red giants with asteroseismic ages, used for information-content
audit and as a held-out natural experiment; not in the training loop. Stream 3
is the deployment sample: Andrae+2023 RGB × Gaia DR3 XP (Option-C: 800 k
uniform + 500 k volume-limited arms).

**Stage 01 — raw XP.** Each Gaia DR3 RVS/XP star supplies 55 BP + 55 RP Hermite
coefficients. Coef 0 carries the absolute flux scale; coefs 1–54 are the shape.
Stars at fainter G have noisier high-order coefs — this drives the SNR floor
at G ≈ 17.

**Stage 02 — Ye+2024.** Flux-level correction against synthetic photometry
for flux-zero-point drifts. Applied to all stars; those with `NO_SYNTH_PHOT`
set are flagged OOD and excluded from training. Correction effect is typically
a few %, concentrated at the blue end.

**Stage 03 — Hermite reprojection.** Ye-corrected *flux samples* are
reprojected onto a 110-dim Hermite basis consistent with Gaia's internal
representation. Residual RMS is well below the per-coef noise floor.
Per-coefficient z-score stats are **frozen at v1 fit** (basis fingerprint
`0d34b565…`); Stream 3 inference reuses them — do not refit.

**Stage 04 — extinction.** A_V per star composed from: Edenhofer+2024
(d < 1.25 kpc, highest precision), Lallement+2022 (1.25–3 kpc), SFD asymptotic
beyond 3 kpc, GSP-Phot neighborhood-median for stars missing all three.
Provenance per star in `stream3_av.provenance.json`.

**Stage 05 — IR photometry.** 2MASS JHK + AllWISE W1 W2 from VizieR
cross-match. Missingness tracked via `ir_missing_flag`; bright stars can
saturate W1 so coverage is non-monotone in G.

**Stage 06 — selection function.** ω_i = ω_Ye · ω_IR · ω_ϖ · ω_Av. Used
as per-star weight in downstream density estimates and as a diagnostic
overlay; does NOT enter training as a sample weight (research_brief §11.2).

**Stage 07 — APOGEE labels.** Training targets after Meszaros+25 [X/M]/Teff
corrections. Element-level NaN rates vary (V ~5.3%, Mg/Fe ~1.6%, α/M 0%);
`beta_nll_block_cholesky` handles them via the `mask=` path.

**Stage 08 — kinematics.** Actions (J_R, J_z, L_z), energy E, eccentricity ε
from Galpy/agama with MWPotential14. Downstream consumers (Starfold, or any
utility module that reuses the kinematics output) join on `source_id`; NOT
fed to Pipeline 1.

**Stage 09 — feature matrix.** Pipeline-1 input = 110 XP coefs + 43 aux
(A_V, parallax, IR magnitudes, flags). Schema contract lives in
`data/DESIGN.md`.

**Stages 10–11 — training.** Contrastive pretraining on XP+aux yields a
trunk encoder; halfway-UMAP inspection is in `reports/pipeline1/halfway/`.
Supervised fine-tuning with β-NLL block-Cholesky loss over 5 label-channels
(Teff, logg, [M/H], [α/M], [Mg/H]), inverse-frequency [M/H] weighting
(v1.1 onwards), 5-seed ensemble.

**Stages 12–13 — validation + uncertainty.** Ensemble moment-match per star:
Σ̄ = mean(Σ_k) + between(μ_k). OOD rejection = Mahalanobis on XP block ∪
ensemble disagreement on any of (Teff, logg, [M/H], [α/M], [Mg/H]).
Regime-B envelope excludes warm-upper-RGB / |b|<5° stars from per-star Tier-1.

**Stage 14 — inference.** Stream-3 volume-limited and uniform arms; 613 k
predictions union with OOD/regime flags per star. This is where this repo's
chain ends; the predictions (plus Stream-3 kinematics) are the inputs
Starfold consumes.

---

## Known absences

- **SHAP per-star saliency** (audit test 3 stub). Permutation-importance bar
  ships in batch 3 as the interim proxy.
- **Cross-catalogue consistency** (tier-promotion test 6 stub). No figure
  until that gate is actually wired.
- **FIRE-2 method-validation track** (Subtask 5.1) — lives under
  `data/fire2/`; strictly separated from real-data science per project invariant.
  Its visuals will land in a separate `reports/fire2/` tree, not this gallery.

---

## How to re-emit everything

(Will be added in batch 7 as `scripts/gallery/build_all.sh`.)
