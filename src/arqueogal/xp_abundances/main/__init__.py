"""Pipeline 1 (``xp_abundances.main``) — production pipeline for D-Cat-b.

The public surface re-exports the module-level dataclasses and entry points
callers typically need — training scripts and inference CLI. Internal helpers
stay private to their modules.
"""

from arqueogal.xp_abundances.main.adapter import (
    XpFeatureAdapter,
    reorder_labels_human_to_block,
)
from arqueogal.xp_abundances.main.audit import (
    AuditReport,
    audit_report,
    conditional_mi_ksg,
    decorrelated_subsample,
    leave_one_coeff_out,
    mutual_information_ksg,
    permutation_feature_importance,
    shuffled_spectrum_null,
)
from arqueogal.xp_abundances.main.config import LossWeights, TrainingConfig
from arqueogal.xp_abundances.main.data import (
    DEFAULT_AUX_COLS,
    FeatureLayout,
    LabelTiers,
    XpAbundanceDataset,
    load_arrays,
    stratified_split_ids,
)
from arqueogal.xp_abundances.main.inference import (
    EnsembleMember,
    EnsemblePrediction,
    load_ensemble,
    predict_ensemble,
)
from arqueogal.xp_abundances.main.losses import (
    beta_nll_block_cholesky,
    mahalanobis_residual,
    supcon_soft_positive,
)
from arqueogal.xp_abundances.main.model import (
    BlockCholeskyHead,
    CovarianceBlockLayout,
    Encoder,
    ModelConfig,
    XpAbundanceModel,
    default_pipeline1_layout,
)
from arqueogal.xp_abundances.main.tier_promotion import (
    TestResult,
    TierPromotionReport,
    audit_gate,
    cluster_precision,
    conditional_mi_bootstrap,
    cross_catalogue_consistency,
    holdout_rmse,
    physical_gate,
    tier_promotion_report,
)
from arqueogal.xp_abundances.main.training import (
    CHECKPOINT_VERSION,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    train_ensemble,
    train_model,
)
from arqueogal.xp_abundances.main.uncertainty import (
    CalibrationArtifacts,
    apply_calibration,
    bin_by_cells,
    coverage_at_levels,
    fit_calibration,
)

__all__ = [
    "CHECKPOINT_VERSION",
    "AuditReport",
    "BlockCholeskyHead",
    "CalibrationArtifacts",
    "CovarianceBlockLayout",
    "DEFAULT_AUX_COLS",
    "Encoder",
    "EnsembleMember",
    "EnsemblePrediction",
    "FeatureLayout",
    "LabelTiers",
    "LossWeights",
    "ModelConfig",
    "TestResult",
    "TierPromotionReport",
    "TrainingConfig",
    "XpAbundanceDataset",
    "XpAbundanceModel",
    "XpFeatureAdapter",
    "apply_calibration",
    "audit_gate",
    "audit_report",
    "beta_nll_block_cholesky",
    "bin_by_cells",
    "cluster_precision",
    "conditional_mi_bootstrap",
    "conditional_mi_ksg",
    "coverage_at_levels",
    "cross_catalogue_consistency",
    "decorrelated_subsample",
    "default_pipeline1_layout",
    "fit_calibration",
    "holdout_rmse",
    "leave_one_coeff_out",
    "load_arrays",
    "load_checkpoint",
    "load_ensemble",
    "mahalanobis_residual",
    "mutual_information_ksg",
    "permutation_feature_importance",
    "physical_gate",
    "predict_ensemble",
    "reorder_labels_human_to_block",
    "save_checkpoint",
    "seed_everything",
    "shuffled_spectrum_null",
    "stratified_split_ids",
    "supcon_soft_positive",
    "tier_promotion_report",
    "train_ensemble",
    "train_model",
]
