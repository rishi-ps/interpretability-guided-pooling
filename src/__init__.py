"""
Interpretability-guided token pooling for visual document retrieval.

This package provides importance estimation methods and pooling strategies
that use interpretability signals to improve compression of multi-vector
document embeddings from ColPali-family models.
"""

from src.importance_estimation import (
    BaseImportanceEstimator,
    CentroidDistanceImportanceEstimator,
    ImportanceOutput,
    ProbeImportanceEstimator,
    SelfSimilarityImportanceEstimator,
    get_importance_map,
)
from src.importance_guided_pooling import (
    AdaptivePoolFactorTokenPooler,
    ImportanceWeightedDistancePooler,
    ImportanceWeightedHierarchicalTokenPooler,
    ProtectAndPoolTokenPooler,
    SplitAndAllocateTokenPooler,
    TopKTokenPooler,
)

__all__ = [
    # Importance estimation
    "BaseImportanceEstimator",
    "CentroidDistanceImportanceEstimator",
    "ImportanceOutput",
    "ProbeImportanceEstimator",
    "SelfSimilarityImportanceEstimator",
    "get_importance_map",
    # Pooling strategies
    "AdaptivePoolFactorTokenPooler",
    "ImportanceWeightedDistancePooler",
    "ImportanceWeightedHierarchicalTokenPooler",
    "ProtectAndPoolTokenPooler",
    "SplitAndAllocateTokenPooler",
    "TopKTokenPooler",
]
