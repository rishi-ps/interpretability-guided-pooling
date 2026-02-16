"""Tests for importance-guided token pooling strategies."""

import pytest
import torch

from src.importance_estimation import (
    CentroidDistanceImportanceEstimator,
    SelfSimilarityImportanceEstimator,
)
from src.importance_guided_pooling import (
    AdaptivePoolFactorTokenPooler,
    ImportanceWeightedHierarchicalTokenPooler,
    ProtectAndPoolTokenPooler,
    TopKTokenPooler,
)


# --- Fixtures ---


@pytest.fixture
def embeddings_and_scores():
    """Generate random embeddings and importance scores."""
    torch.manual_seed(42)
    emb = torch.randn(100, 128)
    emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    est = SelfSimilarityImportanceEstimator()
    scores = est.estimate([emb]).scores
    return [emb], scores


@pytest.fixture
def batch_embeddings_and_scores():
    """Generate batch of embeddings with importance scores."""
    torch.manual_seed(42)
    emb1 = torch.nn.functional.normalize(torch.randn(80, 128), p=2, dim=-1)
    emb2 = torch.nn.functional.normalize(torch.randn(120, 128), p=2, dim=-1)
    embeddings = [emb1, emb2]
    est = CentroidDistanceImportanceEstimator()
    scores = est.estimate(embeddings).scores
    return embeddings, scores


@pytest.fixture
def sample_embedding() -> torch.Tensor:
    """Single embedding for simple tests."""
    torch.manual_seed(42)
    emb = torch.randn(50, 128)
    return torch.nn.functional.normalize(emb, p=2, dim=-1)


# --- TopKTokenPooler ---


class TestTopKTokenPooler:
    def test_output_shape(self, embeddings_and_scores):
        embs, scores = embeddings_and_scores
        pooler = TopKTokenPooler(importance_scores=scores)
        result = pooler.pool_embeddings(embs, pool_factor=4)
        assert result[0].shape == (25, 128)

    def test_pool_factor_2(self, embeddings_and_scores):
        embs, scores = embeddings_and_scores
        pooler = TopKTokenPooler(importance_scores=scores)
        result = pooler.pool_embeddings(embs, pool_factor=2)
        assert result[0].shape == (50, 128)

    def test_preserves_original_vectors(self, embeddings_and_scores):
        """TopK should keep original vectors, not averaged versions."""
        embs, scores = embeddings_and_scores
        pooler = TopKTokenPooler(importance_scores=scores)
        result = pooler.pool_embeddings(embs, pool_factor=2)
        # Each output row should exist in the input
        for row in result[0]:
            diffs = (embs[0] - row).abs().sum(dim=-1)
            assert diffs.min() < 1e-5

    def test_preserves_order(self, embeddings_and_scores):
        """Selected tokens should maintain their original order."""
        embs, scores = embeddings_and_scores
        pooler = TopKTokenPooler(importance_scores=scores)
        output = pooler.pool_embeddings(embs, pool_factor=4, return_dict=True)
        cluster_map = output.cluster_id_to_indices[0]
        indices = [cluster_map[k][0].item() for k in sorted(cluster_map.keys())]
        assert indices == sorted(indices)

    def test_batch_processing(self, batch_embeddings_and_scores):
        embs, scores = batch_embeddings_and_scores
        pooler = TopKTokenPooler(importance_scores=scores)
        result = pooler.pool_embeddings(embs, pool_factor=4)
        assert result[0].shape == (20, 128)
        assert result[1].shape == (30, 128)

    def test_raises_without_scores(self, embeddings_and_scores):
        embs, _ = embeddings_and_scores
        pooler = TopKTokenPooler()
        with pytest.raises(ValueError, match="importance_scores"):
            pooler.pool_embeddings(embs, pool_factor=2)

    def test_scores_at_call_time(self, embeddings_and_scores):
        embs, scores = embeddings_and_scores
        pooler = TopKTokenPooler()
        result = pooler.pool_embeddings(embs, pool_factor=4, importance_scores=scores)
        assert result[0].shape == (25, 128)

    def test_pool_factor_1_returns_all(self, embeddings_and_scores):
        embs, scores = embeddings_and_scores
        pooler = TopKTokenPooler(importance_scores=scores)
        result = pooler.pool_embeddings(embs, pool_factor=1)
        assert result[0].shape == embs[0].shape


# --- ImportanceWeightedHierarchicalTokenPooler ---


class TestWeightedHierarchical:
    def test_output_shape(self, embeddings_and_scores):
        embs, scores = embeddings_and_scores
        pooler = ImportanceWeightedHierarchicalTokenPooler(importance_scores=scores)
        result = pooler.pool_embeddings(embs, pool_factor=4)
        # Should have ~25 clusters (100 // 4)
        assert result[0].shape[0] == 25
        assert result[0].shape[1] == 128

    def test_outputs_are_normalized(self, embeddings_and_scores):
        embs, scores = embeddings_and_scores
        pooler = ImportanceWeightedHierarchicalTokenPooler(importance_scores=scores)
        result = pooler.pool_embeddings(embs, pool_factor=2)
        norms = result[0].float().norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_differs_from_uniform_hierarchical(self, embeddings_and_scores):
        """Weighted pooling should produce different results from uniform."""
        from colpali_engine.compression import HierarchicalTokenPooler

        embs, scores = embeddings_and_scores
        uniform_pooler = HierarchicalTokenPooler()
        weighted_pooler = ImportanceWeightedHierarchicalTokenPooler(importance_scores=scores)

        uniform_result = uniform_pooler.pool_embeddings(embs, pool_factor=3)
        weighted_result = weighted_pooler.pool_embeddings(embs, pool_factor=3)

        # Results should generally differ (unless all weights are equal, which is unlikely)
        diff = (uniform_result[0].float() - weighted_result[0].float()).abs().sum()
        assert diff > 0.01

    def test_return_dict(self, embeddings_and_scores):
        embs, scores = embeddings_and_scores
        pooler = ImportanceWeightedHierarchicalTokenPooler(importance_scores=scores)
        output = pooler.pool_embeddings(embs, pool_factor=4, return_dict=True)
        assert output.cluster_id_to_indices is not None
        assert len(output.cluster_id_to_indices) == 1
        assert len(output.cluster_id_to_indices[0]) == 25

    def test_pool_factor_1(self, embeddings_and_scores):
        embs, scores = embeddings_and_scores
        pooler = ImportanceWeightedHierarchicalTokenPooler(importance_scores=scores)
        result = pooler.pool_embeddings(embs, pool_factor=1)
        assert result[0].shape == embs[0].shape


# --- ProtectAndPoolTokenPooler ---


class TestProtectAndPool:
    def test_output_shape(self, embeddings_and_scores):
        embs, scores = embeddings_and_scores
        pooler = ProtectAndPoolTokenPooler(protect_fraction=0.25, importance_scores=scores)
        result = pooler.pool_embeddings(embs, pool_factor=4)
        # Target output = 100 // 4 = 25 tokens
        assert result[0].shape[0] == 25
        assert result[0].shape[1] == 128

    def test_protected_tokens_preserved(self, embeddings_and_scores):
        """The top-p% tokens should appear unmodified in the output."""
        embs, scores = embeddings_and_scores
        n_protect = max(int(100 * 0.25), 1)
        n_protect = min(n_protect, 25)  # capped by target output

        pooler = ProtectAndPoolTokenPooler(protect_fraction=0.25, importance_scores=scores)
        result = pooler.pool_embeddings(embs, pool_factor=4)

        # The first n_protect rows should match original vectors
        protected = result[0][:n_protect]
        for row in protected:
            diffs = (embs[0] - row).abs().sum(dim=-1)
            assert diffs.min() < 1e-5

    def test_invalid_protect_fraction(self):
        with pytest.raises(ValueError, match="protect_fraction"):
            ProtectAndPoolTokenPooler(protect_fraction=0.0)
        with pytest.raises(ValueError, match="protect_fraction"):
            ProtectAndPoolTokenPooler(protect_fraction=1.0)

    def test_batch_processing(self, batch_embeddings_and_scores):
        embs, scores = batch_embeddings_and_scores
        pooler = ProtectAndPoolTokenPooler(protect_fraction=0.3, importance_scores=scores)
        result = pooler.pool_embeddings(embs, pool_factor=4)
        assert len(result) == 2
        assert result[0].shape[0] == 20  # 80 // 4
        assert result[1].shape[0] == 30  # 120 // 4


# --- AdaptivePoolFactorTokenPooler ---


class TestAdaptivePoolFactor:
    def test_output_has_variable_sizes(self, batch_embeddings_and_scores):
        """Different importance distributions → different effective pool factors."""
        embs, scores = batch_embeddings_and_scores
        pooler = AdaptivePoolFactorTokenPooler(
            min_pool_factor=2, max_pool_factor=8, importance_scores=scores
        )
        result = pooler.pool_embeddings(embs)
        # Both should be compressed, but potentially by different amounts
        assert result[0].shape[0] < 80
        assert result[1].shape[0] < 120

    def test_outputs_are_normalized(self, batch_embeddings_and_scores):
        embs, scores = batch_embeddings_and_scores
        pooler = AdaptivePoolFactorTokenPooler(
            min_pool_factor=2, max_pool_factor=6, importance_scores=scores
        )
        result = pooler.pool_embeddings(embs)
        for r in result:
            norms = r.float().norm(dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_entropy_computation(self):
        pooler = AdaptivePoolFactorTokenPooler(min_pool_factor=2, max_pool_factor=8)

        # Uniform scores → high entropy → low pool factor (close to min)
        uniform_scores = torch.ones(100)
        pf_uniform = pooler._compute_effective_pool_factor(uniform_scores)

        # Concentrated scores → low entropy → high pool factor (close to max)
        concentrated_scores = torch.zeros(100)
        concentrated_scores[0] = 100.0  # One dominant token
        pf_concentrated = pooler._compute_effective_pool_factor(concentrated_scores)

        assert pf_concentrated > pf_uniform
        assert pf_uniform >= 2  # Should be at least min_pool_factor

    def test_invalid_params(self):
        with pytest.raises(ValueError, match="min_pool_factor"):
            AdaptivePoolFactorTokenPooler(min_pool_factor=0)
        with pytest.raises(ValueError, match="max_pool_factor"):
            AdaptivePoolFactorTokenPooler(min_pool_factor=5, max_pool_factor=3)

    def test_return_dict(self, batch_embeddings_and_scores):
        embs, scores = batch_embeddings_and_scores
        pooler = AdaptivePoolFactorTokenPooler(
            min_pool_factor=2, max_pool_factor=6, importance_scores=scores
        )
        output = pooler.pool_embeddings(embs, return_dict=True)
        assert output.cluster_id_to_indices is not None
        assert len(output.cluster_id_to_indices) == 2
