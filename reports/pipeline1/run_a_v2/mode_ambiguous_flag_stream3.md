# Stream-3 `mode_ambiguous_flag` capture — v2 ensemble, 2026-04-22

Closes the implementation arm of task #216 (option-3 remediation for
bimodal-target collapse at intermediate [M/H]). See ADR-0015-draft for the
design contract.

## Grid

Built from Stream-1 training parquet (`pipeline1_features_stream1.parquet`,
SHA-256 `4a98caa7…`, 324 054 rows, all finite on the four keys).

- Edges: Teff 150 K in [3500, 6000]; log g 0.3 in [0, 4]; [M/H] 0.2 in [-3, +0.5]
- Cells: 3 536 total, 276 evaluated (N ≥ 50), 103 flagged bimodal
- Bimodality criteria: 2-vs-1 GaussianMixture ΔBIC ≥ 4 ∧ minor-weight ≥ 0.15
  ∧ mean separation ≥ 0.08 dex
- Stars in bimodal cells (training): 169 074 / 323 074 ≈ 52.3 %
- Spatial concentration: 52 cells at [M/H] ∈ [-0.5, 0) and 31 cells at
  [M/H] ∈ [-1.0, -0.5) — matches Hayden+2015 α-sequence doubling.

Artefact: `data/processed/mode_ambiguous_grid.npz` (+ provenance sidecar).
Builder: `scripts/build_mode_ambiguous_mask.py`.

## Inference pass

`run_pipeline1_inference.py` now queries the grid with predicted
`(Teff, log g, [M/H])` and emits two new bool columns:

- `mode_ambiguous_flag` — `in_grid_flag | (~in_grid)` (conservative: also
  flags out-of-grid predictions)
- `mode_ambiguous_in_grid` — whether the query landed inside the grid span

Re-run on the v2 ensemble (`20260421_38a993e_712b774_ensemble_5label`) and
the same Stream-3 union input as the pre-flag v2 predictions
(`pipeline1_features_stream3.parquet`, SHA-256 `a0b1888b…`, 613 939 stars).

## Flag rates

| Corpus            | N       | flagged | in-grid bimodal | out-of-grid |
|-------------------|--------:|--------:|----------------:|------------:|
| Stream-3 union    | 613 939 | 48.03 % |         47.89 % |      0.14 % |
| Volume-limited    | 249 092 | 33.66 % |         33.47 % |      0.19 % |
| Uniform (Option C)| 364 847 | 57.84 % |         57.73 % |      0.11 % |

The uniform arm is more flagged because it oversamples the [M/H] ∈ [-0.5, 0)
overlap zone by construction; the volume-limited sample is local-disc-heavy
with more thin-disc stars in unimodal cells.

## Stripe capture ([α/M]_pred ∈ [+0.09, +0.13] ∧ [M/H]_pred ∈ [-1.5, -0.3])

The +0.11 attractor — the collapsed-mean predictor Andreas flagged
pre-compaction.

| Corpus            | stripe N | flagged | leaks past |
|-------------------|---------:|--------:|-----------:|
| Stream-3 union    | 129 445  | 72.21 % |    27.79 % |
| Volume-limited    |  17 616  | 67.19 % |    32.81 % |
| Uniform           | 111 829  | 73.00 % |    27.00 % |

The ~30 % leak is almost entirely at [M/H]_pred ∈ [-1.5, -1.0) — where
training has few bimodal cells (only 20 evaluated, 0 flagged in that band).
Physically, the α-sequence bifurcation is weaker at the metal-poor end, and
many stripe stars there are genuine halo α-enhanced stars whose α/M ≈ +0.11
is a reasonable prediction, not a collapse artefact.

## Confirmation that option 3 recovers the bimodality — volume-limited corpus

Conditional on the clean slice (no OOD, no Regime B, no aux-missing; N = 229 201):

### [M/H]_pred ∈ [-0.5, 0.0) (n = 120 261, the thin/thick overlap)

|                | p05    | p25    | p50    | p75    | p95    | α>0.15 | α∈[0.08,0.14] |
|----------------|-------:|-------:|-------:|-------:|-------:|-------:|--------------:|
| ALL            | +0.001 | +0.029 | +0.060 | +0.107 | +0.197 |  12.5% |        22.7%  |
| **Unflagged**  | −0.003 | +0.019 | +0.040 | +0.068 | +0.123 |   2.9% |        14.1%  |

The α/M > 0.15 shoulder (thick disc + collapsed-mean contamination) drops
4× in the unflagged remainder. The distribution re-concentrates around the
thin-disc value +0.04. This is the intended behaviour — option 3 throws out
the stars whose disc identity is not recoverable from XP at this [M/H], and
what remains is a cleaner thin-disc-only per-star sample at this [M/H].

### [M/H]_pred ∈ [-1.0, -0.5) (n = 21 923, the α-bimodality tail)

|                | p05    | p25    | p50    | p75    | p95    | α>0.15 |
|----------------|-------:|-------:|-------:|-------:|-------:|-------:|
| ALL            | +0.096 | +0.120 | +0.162 | +0.232 | +0.273 |  53.4% |
| Unflagged      | +0.115 | +0.129 | +0.154 | +0.250 | +0.280 |  51.2% |

Barely changed — the grid flags only 70 % of this band because few cells
are bimodal at [M/H] < -0.5. These predictions are mostly genuine α-enhanced
halo/thick-disc stars and the option-3 flag leaves them intact.

## Release-contract consequence

- The per-star α/M release on Stream-3 volume-limited drops from 249 092 to
  roughly 149 000 stars (clean slice ∩ ¬mode_ambiguous) — a ≈ 34 %
  attrition on top of OOD/Regime-B losses.
- Pipeline-2 consumes the full Stream-3 catalogue unmodified; downstream
  clustering is over stars, not per-star α/M, and the mode-collapse does
  not propagate into UMAP+HDBSCAN structure.
- Any paper or catalogue-level α-vs-[M/H] plot must be drawn from the
  unflagged subset, with a caption noting the 34 % refusal rate at
  intermediate [M/H].

## Artefacts

- `data/processed/pipeline1_predictions_stream3_v2.parquet` — 613 939 × 42 cols
- `data/processed/pipeline1_predictions_stream3_volume_v2.parquet` — 249 092 × 41 cols
- `data/processed/pipeline1_predictions_stream3_uniform_v2.parquet` — 364 847 × 41 cols
- Each carries a provenance sidecar with the grid SHA-256, flag counts, and
  the `mode_ambiguous` criteria block.

## Figures

- `mode_ambiguous_diagnosis_v2.png` — 2 × 3 panel: α/M vs [M/H] density for
  ALL / UNFLAGGED / FLAGGED on volume-limited clean slice; 1-D α/M histograms
  in three [M/H] bands showing the +0.11 shoulder collapsing in the unflagged
  subset at [M/H] ∈ [-0.5, 0) and the intrinsic Hayden+2015 bimodality
  surviving intact at [M/H] ∈ [-1.0, -0.5).
- `attractor_stream3_v1_v11_v2.png` — 4-panel α/M vs [M/H] density:
  v1 → v1.1 → v2 (all with the attractor stripe at +0.11) → v2 release
  (¬mode_ambiguous) where the stripe is gone. Release_ok drops from 92.3 %
  to 60.1 %.

## What this does NOT resolve

Option 3 is release-gating only. The root cause — Gaussian-NLL head
collapsing to the conditional mean when the target is bimodal given XP —
is still present in the ensemble. Two open threads:

1. Task #218 — supersede ADR-0014 with the bimodal-target-collapse
   diagnosis as the canonical root cause.
2. Task #217 — XP-only smoke retrain to test whether the aux stack is
   further flattening the posterior (the TESS_ML prototype reached a
   cleaner bimodality at low [M/H] with *fewer* inputs — suspicious).
