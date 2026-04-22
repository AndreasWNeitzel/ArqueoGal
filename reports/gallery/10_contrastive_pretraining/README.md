# 10 — (Retired) Contrastive pretraining

**Architecture change (2026-04-22).** The two-stage pipeline (SupCon pretrain →
β-NLL fine-tune) has been retired in favour of a single-stage joint-loss
recipe ported from TESS_ML. Training now runs
`supcon + beta_nll + barlow_twins` in one pass with `encoder_lr_ratio=1.0`,
no frozen-encoder stage.

**Why.** The two-stage v1 / v1.1 / v2 pipeline produced an [α/M]≈+0.11
attractor-stripe on metal-poor stars (chemistry collapse to a single mode).
The joint-loss recipe preserves the low-α / high-α disc bifurcation on the
same Stream-1 validation split (attractor-stripe fraction 6.27% vs v2 ~72%).
See `docs/plan/01_pipeline1_v1.md` for the full story and bug catalogue.

**This directory is intentionally empty.** No pretrain stage exists — all
training diagnostics live under `11_supervised_training/`. The halfway-UMAP
plots are still useful trunk-quality diagnostics but are now produced during
the joint-loss epoch midpoint rather than after a distinct pretrain stage;
they live under `../../pipeline1/halfway/`.

## Why keep the stage directory

For git-history continuity. The numbering of `11_*`, `12_*`, etc. is
referenced in external notes and in `GALLERY.md`; renumbering everything
would create more churn than the cost of an empty stage.
