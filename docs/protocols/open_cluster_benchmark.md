# Protocol: held-out open-cluster benchmark (test 3 of §3.3)

**Status:** Scoped. Computation pending.
**Authoring trigger:** overfitting_mitigation.md CRITICAL #2; META_META §8 P1-4.

## 1 Why this protocol

The §3.3 promotion protocol's test 3 (open-cluster precision floor) is
currently a stub. The released catalog ships at 5/6 coverage. Running
test 3 properly turns "5/6 with stubs" into "5/6 with one test
deferred for compute reasons", a stronger position.

The test rationale: open-cluster members are a population of stars
known to be co-eval, co-spatial, and co-chemical (modulo well-
characterised internal scatter). Per-cluster intra-cluster σ on
predicted [Fe/H] etc. is a direct empirical floor on Pipeline 1's
per-star precision. If the intra-cluster σ exceeds 1.5× the APOGEE
intra-cluster σ on the same cluster, the model has not added value
beyond noise.

## 2 Cluster sample

Target: 6–10 nearby open clusters with ≥ 30 APOGEE members each, well-
covered by Gaia DR3, and spread across [Fe/H] from −0.5 to +0.4. The
candidate list (sourced from Cantat-Gaudin et al. 2020, A&A 633, A99
and APOGEE-2 Donor et al. 2020):

- M67 (NGC 2682), solar-metallicity benchmark, ~150 APOGEE members.
- NGC 6791, old, metal-rich ([Fe/H] ~ +0.4), ~50 APOGEE members.
- NGC 2420, intermediate-age, metal-poor ([Fe/H] ~ −0.2), ~30 members.
- NGC 6819, APOKASC-3 calibrator, ~50 members.
- NGC 2477, moderately metal-rich ([Fe/H] ~ +0.1), ~40 members.
- NGC 5822, open cluster well-studied in DR17.
- NGC 2243, metal-poor outlier ([Fe/H] ~ −0.4).
- NGC 6253, metal-rich ([Fe/H] ~ +0.2).
- IC 4651, older intermediate-age cluster.
- M71 (NGC 6838), globular-cluster-edge candidate; document if used.

## 3 Plan

### Step A: training-set exclusion

Re-build Stream 1 with cluster members (cross-matched by source_id from
the candidate-cluster member lists) excluded. Document the excluded
source_id list in `reports/pipeline1/audit/open_cluster_holdout.md`.

### Step B: retrain Pipeline 1 on the cluster-excluded Stream 1

Same architecture, same hyperparameters, new ensemble seeds (so the
checkpoint is independent of the v1 release). 5–10 ensemble members on
RTX 3060 takes ~1 day per member; 7–14 days total wall-clock. On
Deucalion this collapses to ~1 day total.

### Step C: predict per-cluster member abundances

For each cluster's members, predict via the cluster-excluded ensemble.
Compute per-cluster per-element intra-cluster σ.

### Step D: compare to APOGEE intra-cluster σ

For each cluster × element, compute the ratio
intra_cluster_σ_pipeline1 / intra_cluster_σ_apogee. Threshold: ratio
≤ 1.5 → test 3 PASS; ratio > 1.5 → FAIL with explicit per-element-per-
cluster note.

## 4 Methods-paper figure

A 5-panel × 10-cluster heatmap showing the ratio. Methods paper
references it as "Figure X (open-cluster precision floor)".

## 5 Effort estimate

2 weeks:
- Days 1–3: cluster-membership cross-match and training-set exclusion.
- Days 4–11: retrain on Deucalion (or 14–21 days on laptop).
- Days 12–13: predict + intra-cluster σ + figure.
- Day 14: methods-paper §X writing.

This is the longest single deferred deliverable. It is recommended
post-Deucalion-access (target July 2026).

## 6 Outcome contract

The methods paper §3.3 cites this benchmark explicitly. Test 3 moves
from "stub" to "performed; pass on N/N clusters; fail on K clusters
with documented reasons". The release tier-promotion code's
`STUBBED_TESTS` set drops Test 3 once this protocol completes; only
Test 6 (cross-catalogue) remains stubbed at that point.
