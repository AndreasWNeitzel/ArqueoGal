"""Pipeline 2 (``population_classifier.main``) — UMAP + HDBSCAN main pipeline.

Builds towards D5.1 (Dec 2026) and D-Cat-d (Feb 2027). The public surface is
the feature-builder, embedder, clusterer, hyperparameter selector, diagnostic
stack, and MC-ensemble orchestrator. Experimental methods live in a parallel
``experimental/`` tree.
"""

from arqueogal.population_classifier.main.clustering import (
    ClusteringResult,
    HDBSCANConfig,
    cluster_hdbscan,
)
from arqueogal.population_classifier.main.diagnostics import (
    BootstrapStabilityReport,
    DiagnosticStackReport,
    FeatureCausalReport,
    HeldOutFeatureReport,
    LiteratureCrossReferenceReport,
    NullModelReport,
    bootstrap_cluster_stability,
    held_out_feature_consistency,
    literature_cross_reference,
    null_model_comparison,
    permutation_feature_causal,
)
from arqueogal.population_classifier.main.embedding import (
    ParametricUMAP,
    ParametricUMAPConfig,
    ParametricUMAPEncoder,
)
from arqueogal.population_classifier.main.features import (
    BASELINE_NEITZEL2025_COLUMNS,
    MAIN_FEATURE_COLUMNS,
    FeatureMatrix,
    FeatureSpec,
    apply_c_n_gate,
    build_feature_matrix,
    standardize,
)
from arqueogal.population_classifier.main.hare_hounds import (
    HareHoundsReport,
    compute_hare_hounds_metrics,
)
from arqueogal.population_classifier.main.hyperparameter import (
    GridCell,
    HyperparameterGrid,
    UMAPHyperparams,
    dbcv_score,
    grid_search,
)
from arqueogal.population_classifier.main.mc_ensemble import (
    MCEnsembleConfig,
    MCEnsembleResult,
    run_mc_ensemble,
    sample_feature_posteriors,
)

__all__ = [
    "BASELINE_NEITZEL2025_COLUMNS",
    "MAIN_FEATURE_COLUMNS",
    "BootstrapStabilityReport",
    "ClusteringResult",
    "DiagnosticStackReport",
    "FeatureCausalReport",
    "FeatureMatrix",
    "FeatureSpec",
    "GridCell",
    "HDBSCANConfig",
    "HareHoundsReport",
    "HeldOutFeatureReport",
    "HyperparameterGrid",
    "LiteratureCrossReferenceReport",
    "MCEnsembleConfig",
    "MCEnsembleResult",
    "NullModelReport",
    "ParametricUMAP",
    "ParametricUMAPConfig",
    "ParametricUMAPEncoder",
    "UMAPHyperparams",
    "apply_c_n_gate",
    "bootstrap_cluster_stability",
    "build_feature_matrix",
    "cluster_hdbscan",
    "compute_hare_hounds_metrics",
    "dbcv_score",
    "grid_search",
    "held_out_feature_consistency",
    "literature_cross_reference",
    "null_model_comparison",
    "permutation_feature_causal",
    "run_mc_ensemble",
    "sample_feature_posteriors",
    "standardize",
]
