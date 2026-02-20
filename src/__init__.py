"""
Interpretability-guided token pooling for visual document retrieval.

This package provides importance estimation methods and pooling strategies
that use interpretability signals to improve compression of multi-vector
document embeddings from ColPali-family models.
"""

from src.importance_estimation import (
    AttentionImportanceEstimator,
    BaseImportanceEstimator,
    CentroidDistanceImportanceEstimator,
    ImportanceOutput,
    ProbeImportanceEstimator,
    SelfSimilarityImportanceEstimator,
    SVDImportanceEstimator,
    get_importance_map,
)
from src.importance_guided_pooling import (
    AdaptivePoolFactorTokenPooler,
    ImportanceWeightedDistancePooler,
    ImportanceWeightedHierarchicalTokenPooler,
    ImportanceWeightedKMeansTokenPooler,
    ProtectAndPoolTokenPooler,
    SplitAndAllocateTokenPooler,
    TopKTokenPooler,
)

__all__ = [
    # Importance estimation
    "AttentionImportanceEstimator",
    "BaseImportanceEstimator",
    "CentroidDistanceImportanceEstimator",
    "ImportanceOutput",
    "ProbeImportanceEstimator",
    "SelfSimilarityImportanceEstimator",
    "SVDImportanceEstimator",
    "get_importance_map",
    # Pooling strategies
    "AdaptivePoolFactorTokenPooler",
    "ImportanceWeightedDistancePooler",
    "ImportanceWeightedHierarchicalTokenPooler",
    "ImportanceWeightedKMeansTokenPooler",
    "ProtectAndPoolTokenPooler",
    "SplitAndAllocateTokenPooler",
    "TopKTokenPooler",
]
