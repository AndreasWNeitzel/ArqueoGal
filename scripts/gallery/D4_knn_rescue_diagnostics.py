"""D4: kNN rescue diagnostics on Stream 1 holdout.

STUB — real post-kNN-rescue predictions for Stream 1 holdout are not on disk
as of 2026-04-29. The available kNN parquets are:

  data/processed/pipeline1_knn_rescue.parquet            (Stream 3, summary stats only)
  data/processed/pipeline1_knn_rescue_stream2.parquet    (Stream 2, summary stats only)

Neither carries per-element post-rescue final predictions for Stream 1
holdout test stars, so a pre-vs-post comparison cannot be made without
fabricating the "post" branch.

To populate this plot, the following artefact must exist on disk:

  data/processed/pipeline1_predictions_stream1_v1_post_knn_rescue.parquet

with columns ``source_id``, ``<elem>_pred``, ``<elem>_sigma`` for the five
v1 labels {Teff, log g, [M/H], [alpha/M], [Mg/H]} after the kNN rescue
composer has been applied.

DO NOT fabricate "post" residuals or pretend an RMSE improvement that has
not been measured.
"""

from __future__ import annotations

print(__doc__)
