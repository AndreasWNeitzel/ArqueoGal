# ADR-0010: Ensemble members share pretrained encoder; vary only head init

**Date**: ~2026-04-18 · **Status**: Accepted, in production

## Context

Pipeline 1 ensemble design targets 5–10 members for epistemic-uncertainty estimation.
Training cost is a hard budget constraint on RTX 3060 (each member's supervised
fine-tune is ~1–2 h; contrastive pretrain is ~6 h).

## Decision

All ensemble members share the single contrastive-pretrained encoder checkpoint. Vary
only the supervised-head initialization (per-seed random init). Do NOT vary data
splits across members.

## Rationale

1. **Training cost**: sharing the pretrained encoder means 5–10 members × 1–2 h
   supervised fine-tune, not 5–10 × (6 h + 1–2 h) = 35–80 h.
2. **Clean epistemic**: varying data splits confounds epistemic uncertainty with
   split-dependent label coverage. Sharing the encoder and data split isolates
   ensemble disagreement to head-init stochasticity only.
3. **Runs A / B / C comparisons** (108-D vs 43-D pretraining/fine-tune) must share
   the pretrained encoder when pretraining inputs are identical (Runs A and B);
   this forces reproducibility across experimental comparisons.

## Alternatives rejected

- **Vary data splits across members** — confounds epistemic and data coverage.
- **Independent ensemble members (fresh pretrain each)** — budget-busting.
- **Bayesian NN / MC dropout** — would eliminate ensemble ambiguity but is a much
  larger architectural change.

## Consequences

- Persist pretrained encoder checkpoint deterministically with basis fingerprint
  stability.
- Identical load into each supervised head; only head init seed varies (0, 1, 2, 3, 4
  for v1).
- Cross-run A/B comparisons reuse the same pretrained encoder rather than retraining
  per run.

## Peer-review note

No disagreement. This is industry-standard for contrastive-pretrain + supervised-head
ensembling (Chen+2020-style).
