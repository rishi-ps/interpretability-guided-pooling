"""Tests for importance-guided token pooling strategies."""

import pytest
import torch

from src.importance_estimation import (
    CentroidDistanceImportanceEstimator,
    SelfSimilarityImportanceEstimator,
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
        # After fix: n_protect = min(25, max(int(25 * 0.75), 1)) = 18
        target_output = 100 // 4  # = 25
        max_protect = max(int(target_output * 0.75), 1)  # = 18
        n_protect = min(max(int(100 * 0.25), 1), max_protect)  # = 18

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


# --- SplitAndAllocateTokenPooler ---


class TestSplitAndAllocate:
    def test_output_shape(self, embeddings_and_scores):
        embs, scores = embeddings_and_scores
        pooler = SplitAndAllocateTokenPooler(importance_scores=scores)
        result = pooler.pool_embeddings(embs, pool_factor=4)
        # 100 tokens // 4 = 25 output clusters
        assert result[0].shape == (25, 128)

    def test_same_token_count_as_hierarchical(self, embeddings_and_scores):
        """Must produce the same number of output tokens as standard hierarchical."""
        from colpali_engine.compression import HierarchicalTokenPooler

        embs, scores = embeddings_and_scores
        hier = HierarchicalTokenPooler()
        sa = SplitAndAllocateTokenPooler(importance_scores=scores)

        for pf in [2, 3, 4, 6, 8]:
            hier_result = hier.pool_embeddings(embs, pool_factor=pf)
            sa_result = sa.pool_embeddings(embs, pool_factor=pf)
            assert sa_result[0].shape[0] == hier_result[0].shape[0], (
                f"PF={pf}: split_allocate has {sa_result[0].shape[0]} tokens, "
                f"hierarchical has {hier_result[0].shape[0]}"
            )

    def test_outputs_are_normalized(self, embeddings_and_scores):
        embs, scores = embeddings_and_scores
        pooler = SplitAndAllocateTokenPooler(importance_scores=scores)
        result = pooler.pool_embeddings(embs, pool_factor=4)
        norms = result[0].float().norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_differs_from_hierarchical(self, embeddings_and_scores):
        """Should produce different results from standard hierarchical."""
        from colpali_engine.compression import HierarchicalTokenPooler

        embs, scores = embeddings_and_scores
        hier = HierarchicalTokenPooler()
        sa = SplitAndAllocateTokenPooler(importance_scores=scores)

        hier_result = hier.pool_embeddings(embs, pool_factor=4)
        sa_result = sa.pool_embeddings(embs, pool_factor=4)

        # Results should differ because cluster budget is allocated differently
        # (compare as sets of vectors since order may differ)
        diff = (hier_result[0].float().sort(dim=0).values - sa_result[0].float().sort(dim=0).values).abs().sum()
        assert diff > 0.01

    def test_batch_processing(self, batch_embeddings_and_scores):
        embs, scores = batch_embeddings_and_scores
        pooler = SplitAndAllocateTokenPooler(importance_scores=scores)
        result = pooler.pool_embeddings(embs, pool_factor=4)
        assert result[0].shape[0] == 20  # 80 // 4
        assert result[1].shape[0] == 30  # 120 // 4

    def test_pool_factor_1(self, embeddings_and_scores):
        embs, scores = embeddings_and_scores
        pooler = SplitAndAllocateTokenPooler(importance_scores=scores)
        result = pooler.pool_embeddings(embs, pool_factor=1)
        assert result[0].shape == embs[0].shape

    def test_return_dict(self, embeddings_and_scores):
        embs, scores = embeddings_and_scores
        pooler = SplitAndAllocateTokenPooler(importance_scores=scores)
        output = pooler.pool_embeddings(embs, pool_factor=4, return_dict=True)
        assert output.cluster_id_to_indices is not None
        assert len(output.cluster_id_to_indices) == 1

    def test_invalid_params(self):
        with pytest.raises(ValueError, match="split_fraction"):
            SplitAndAllocateTokenPooler(split_fraction=0.0)
        with pytest.raises(ValueError, match="budget_fraction"):
            SplitAndAllocateTokenPooler(budget_fraction=1.0)

    def test_scores_at_call_time(self, embeddings_and_scores):
        embs, scores = embeddings_and_scores
        pooler = SplitAndAllocateTokenPooler()
        result = pooler.pool_embeddings(embs, pool_factor=4, importance_scores=scores)
        assert result[0].shape == (25, 128)

    def test_custom_split_and_budget(self, embeddings_and_scores):
        """Custom split/budget fractions should still produce correct output size."""
        embs, scores = embeddings_and_scores
        pooler = SplitAndAllocateTokenPooler(
            importance_scores=scores, split_fraction=0.3, budget_fraction=0.6
        )
        result = pooler.pool_embeddings(embs, pool_factor=4)
        assert result[0].shape == (25, 128)


# --- Protect-and-Pool regression test ---


class TestProtectAndPoolRegression:
    def test_does_not_collapse_to_topk_at_high_pf(self, embeddings_and_scores):
        """At PF=4 with 100 tokens, protect-and-pool must NOT degrade to top-k."""
        embs, scores = embeddings_and_scores
        pp = ProtectAndPoolTokenPooler(protect_fraction=0.25, importance_scores=scores)
        topk = TopKTokenPooler(importance_scores=scores)

        pp_result = pp.pool_embeddings(embs, pool_factor=4)
        topk_result = topk.pool_embeddings(embs, pool_factor=4)

        # If they match exactly, the bug is back
        diff = (pp_result[0].float() - topk_result[0].float()).abs().sum()
        assert diff > 0.01, "protect_and_pool collapsed to topk at PF=4"

    def test_has_both_protected_and_pooled_tokens(self, embeddings_and_scores):
        """At PF=4, output should have some exact tokens AND some averaged ones."""
        embs, scores = embeddings_and_scores
        pooler = ProtectAndPoolTokenPooler(protect_fraction=0.25, importance_scores=scores)
        result = pooler.pool_embeddings(embs, pool_factor=4)

        n_exact = 0
        for row in result[0]:
            diffs = (embs[0] - row).abs().sum(dim=-1)
            if diffs.min() < 1e-5:
                n_exact += 1

        n_total = result[0].shape[0]  # 25
        assert n_exact > 0, "No protected tokens found"
        assert n_exact < n_total, "All tokens are protected — no pooling happening"


# --- ImportanceWeightedDistancePooler ---


class TestImportanceWeightedDistance:
    def test_output_shape(self, embeddings_and_scores):
        embs, scores = embeddings_and_scores
        pooler = ImportanceWeightedDistancePooler(importance_scores=scores)
        result = pooler.pool_embeddings(embs, pool_factor=4)
        assert result[0].shape == (25, 128)

    def test_same_token_count_as_hierarchical(self, embeddings_and_scores):
        """Must produce the same number of output tokens as standard hierarchical."""
        from colpali_engine.compression import HierarchicalTokenPooler

        embs, scores = embeddings_and_scores
        hier = HierarchicalTokenPooler()
        iwd = ImportanceWeightedDistancePooler(importance_scores=scores)

        for pf in [2, 4, 8]:
            hier_result = hier.pool_embeddings(embs, pool_factor=pf)
            iwd_result = iwd.pool_embeddings(embs, pool_factor=pf)
            assert iwd_result[0].shape[0] == hier_result[0].shape[0], (
                f"PF={pf}: iwd has {iwd_result[0].shape[0]} tokens, "
                f"hierarchical has {hier_result[0].shape[0]}"
            )

    def test_outputs_are_normalized(self, embeddings_and_scores):
        embs, scores = embeddings_and_scores
        pooler = ImportanceWeightedDistancePooler(importance_scores=scores)
        result = pooler.pool_embeddings(embs, pool_factor=4)
        norms = result[0].float().norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_differs_from_hierarchical(self, embeddings_and_scores):
        """Should produce different cluster assignments than standard hierarchical."""
        from colpali_engine.compression import HierarchicalTokenPooler

        embs, scores = embeddings_and_scores
        hier = HierarchicalTokenPooler()
        iwd = ImportanceWeightedDistancePooler(importance_scores=scores)

        hier_result = hier.pool_embeddings(embs, pool_factor=4)
        iwd_result = iwd.pool_embeddings(embs, pool_factor=4)

        diff = (hier_result[0].float().sort(dim=0).values - iwd_result[0].float().sort(dim=0).values).abs().sum()
        assert diff > 0.01

    def test_alpha_zero_consistent(self, embeddings_and_scores):
        """With alpha=0, two runs should produce identical results."""
        embs, scores = embeddings_and_scores
        iwd1 = ImportanceWeightedDistancePooler(importance_scores=scores, alpha=0.0)
        iwd2 = ImportanceWeightedDistancePooler(importance_scores=scores, alpha=0.0)

        result1 = iwd1.pool_embeddings(embs, pool_factor=4)
        result2 = iwd2.pool_embeddings(embs, pool_factor=4)

        diff = (result1[0].float() - result2[0].float()).abs().max()
        assert diff < 1e-5, f"Two runs with alpha=0 should match, max diff={diff}"

    def test_batch_processing(self, batch_embeddings_and_scores):
        embs, scores = batch_embeddings_and_scores
        pooler = ImportanceWeightedDistancePooler(importance_scores=scores)
        result = pooler.pool_embeddings(embs, pool_factor=4)
        assert result[0].shape[0] == 20  # 80 // 4
        assert result[1].shape[0] == 30  # 120 // 4

    def test_pool_factor_1(self, embeddings_and_scores):
        embs, scores = embeddings_and_scores
        pooler = ImportanceWeightedDistancePooler(importance_scores=scores)
        result = pooler.pool_embeddings(embs, pool_factor=1)
        assert result[0].shape == embs[0].shape

    def test_return_dict(self, embeddings_and_scores):
        embs, scores = embeddings_and_scores
        pooler = ImportanceWeightedDistancePooler(importance_scores=scores)
        output = pooler.pool_embeddings(embs, pool_factor=4, return_dict=True)
        assert output.cluster_id_to_indices is not None
        assert len(output.cluster_id_to_indices) == 1

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            ImportanceWeightedDistancePooler(alpha=-1.0)

    def test_scores_at_call_time(self, embeddings_and_scores):
        embs, scores = embeddings_and_scores
        pooler = ImportanceWeightedDistancePooler()
        result = pooler.pool_embeddings(embs, pool_factor=4, importance_scores=scores)
        assert result[0].shape == (25, 128)

    def test_higher_alpha_stronger_effect(self, embeddings_and_scores):
        """Higher alpha should produce results that differ more from standard hierarchical."""
        from colpali_engine.compression import HierarchicalTokenPooler

        embs, scores = embeddings_and_scores
        hier = HierarchicalTokenPooler()
        hier_result = hier.pool_embeddings(embs, pool_factor=4)

        diff_a1 = 0.0
        diff_a5 = 0.0
        for alpha, diff_ref in [(1.0, None), (5.0, None)]:
            iwd = ImportanceWeightedDistancePooler(importance_scores=scores, alpha=alpha)
            iwd_result = iwd.pool_embeddings(embs, pool_factor=4)
            diff = (hier_result[0].float().sort(dim=0).values - iwd_result[0].float().sort(dim=0).values).abs().sum().item()
            if alpha == 1.0:
                diff_a1 = diff
            else:
                diff_a5 = diff

        assert diff_a5 >= diff_a1, "Higher alpha should diverge more from hierarchical"


# --- ImportanceWeightedKMeansTokenPooler ---


class TestImportanceWeightedKMeans:
    def test_output_shape(self, embeddings_and_scores):
        embs, scores = embeddings_and_scores
        pooler = ImportanceWeightedKMeansTokenPooler(importance_scores=scores)
        result = pooler.pool_embeddings(embs, pool_factor=4)
        assert result[0].shape == (25, 128)

    def test_same_token_count_as_hierarchical(self, embeddings_and_scores):
        """Must produce the same number of output tokens as standard hierarchical."""
        from colpali_engine.compression import HierarchicalTokenPooler

        embs, scores = embeddings_and_scores
        hier = HierarchicalTokenPooler()
        kmeans = ImportanceWeightedKMeansTokenPooler(importance_scores=scores)

        for pf in [2, 4, 8]:
            hier_result = hier.pool_embeddings(embs, pool_factor=pf)
            kmeans_result = kmeans.pool_embeddings(embs, pool_factor=pf)
            assert kmeans_result[0].shape[0] == hier_result[0].shape[0], (
                f"PF={pf}: kmeans has {kmeans_result[0].shape[0]} tokens, "
                f"hierarchical has {hier_result[0].shape[0]}"
            )

    def test_outputs_are_normalized(self, embeddings_and_scores):
        embs, scores = embeddings_and_scores
        pooler = ImportanceWeightedKMeansTokenPooler(importance_scores=scores)
        result = pooler.pool_embeddings(embs, pool_factor=4)
        norms = result[0].float().norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_differs_from_hierarchical(self, embeddings_and_scores):
        """K-means should produce different results from hierarchical clustering."""
        from colpali_engine.compression import HierarchicalTokenPooler

        embs, scores = embeddings_and_scores
        hier = HierarchicalTokenPooler()
        kmeans = ImportanceWeightedKMeansTokenPooler(importance_scores=scores)

        hier_result = hier.pool_embeddings(embs, pool_factor=4)
        kmeans_result = kmeans.pool_embeddings(embs, pool_factor=4)

        diff = (hier_result[0].float().sort(dim=0).values - kmeans_result[0].float().sort(dim=0).values).abs().sum()
        assert diff > 0.01

    def test_alpha_zero_is_uniform_kmeans(self, embeddings_and_scores):
        """With alpha=0, weights are all 1.0 → uniform k-means."""
        embs, scores = embeddings_and_scores
        pooler = ImportanceWeightedKMeansTokenPooler(importance_scores=scores, alpha=0.0)
        result = pooler.pool_embeddings(embs, pool_factor=4)
        assert result[0].shape == (25, 128)
        norms = result[0].float().norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_deterministic_across_runs(self, embeddings_and_scores):
        """With fixed seed, two runs should produce identical results."""
        embs, scores = embeddings_and_scores
        pooler1 = ImportanceWeightedKMeansTokenPooler(importance_scores=scores)
        pooler2 = ImportanceWeightedKMeansTokenPooler(importance_scores=scores)

        result1 = pooler1.pool_embeddings(embs, pool_factor=4)
        result2 = pooler2.pool_embeddings(embs, pool_factor=4)

        diff = (result1[0].float() - result2[0].float()).abs().max()
        assert diff < 1e-5, f"Two runs should match, max diff={diff}"

    def test_batch_processing(self, batch_embeddings_and_scores):
        embs, scores = batch_embeddings_and_scores
        pooler = ImportanceWeightedKMeansTokenPooler(importance_scores=scores)
        result = pooler.pool_embeddings(embs, pool_factor=4)
        assert result[0].shape[0] == 20  # 80 // 4
        assert result[1].shape[0] == 30  # 120 // 4

    def test_pool_factor_1(self, embeddings_and_scores):
        embs, scores = embeddings_and_scores
        pooler = ImportanceWeightedKMeansTokenPooler(importance_scores=scores)
        result = pooler.pool_embeddings(embs, pool_factor=1)
        assert result[0].shape == embs[0].shape

    def test_return_dict(self, embeddings_and_scores):
        embs, scores = embeddings_and_scores
        pooler = ImportanceWeightedKMeansTokenPooler(importance_scores=scores)
        output = pooler.pool_embeddings(embs, pool_factor=4, return_dict=True)
        assert output.cluster_id_to_indices is not None
        assert len(output.cluster_id_to_indices) == 1

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            ImportanceWeightedKMeansTokenPooler(alpha=-1.0)

    def test_invalid_max_iters(self):
        with pytest.raises(ValueError, match="max_iters"):
            ImportanceWeightedKMeansTokenPooler(max_iters=0)

    def test_invalid_n_restarts(self):
        with pytest.raises(ValueError, match="n_restarts"):
            ImportanceWeightedKMeansTokenPooler(n_restarts=0)

    def test_raises_without_scores(self, embeddings_and_scores):
        embs, _ = embeddings_and_scores
        pooler = ImportanceWeightedKMeansTokenPooler()
        with pytest.raises(ValueError, match="importance_scores"):
            pooler.pool_embeddings(embs, pool_factor=2)

    def test_scores_at_call_time(self, embeddings_and_scores):
        embs, scores = embeddings_and_scores
        pooler = ImportanceWeightedKMeansTokenPooler()
        result = pooler.pool_embeddings(embs, pool_factor=4, importance_scores=scores)
        assert result[0].shape == (25, 128)

    def test_higher_alpha_stronger_effect(self, embeddings_and_scores):
        """Higher alpha should produce results that differ more from uniform k-means."""
        embs, scores = embeddings_and_scores
        uniform = ImportanceWeightedKMeansTokenPooler(importance_scores=scores, alpha=0.0)
        uniform_result = uniform.pool_embeddings(embs, pool_factor=4)

        diff_a1 = 0.0
        diff_a5 = 0.0
        for alpha in [1.0, 5.0]:
            weighted = ImportanceWeightedKMeansTokenPooler(importance_scores=scores, alpha=alpha)
            weighted_result = weighted.pool_embeddings(embs, pool_factor=4)
            diff = (uniform_result[0].float().sort(dim=0).values - weighted_result[0].float().sort(dim=0).values).abs().sum().item()
            if alpha == 1.0:
                diff_a1 = diff
            else:
                diff_a5 = diff

        assert diff_a5 >= diff_a1, "Higher alpha should diverge more from uniform k-means"

    def test_kmeans_plus_plus_init_shape(self):
        """K-means++ init should return k centroids of correct shape."""
        torch.manual_seed(42)
        emb = torch.nn.functional.normalize(torch.randn(50, 128), p=2, dim=-1)
        weights = torch.ones(50)
        centroids = ImportanceWeightedKMeansTokenPooler._kmeans_plus_plus_init(emb, k=10, importance_weights=weights)
        assert centroids.shape == (10, 128)

    def test_kmeans_plus_plus_init_unique(self):
        """K-means++ init should return distinct centroids."""
        torch.manual_seed(42)
        emb = torch.nn.functional.normalize(torch.randn(50, 128), p=2, dim=-1)
        weights = torch.ones(50)
        centroids = ImportanceWeightedKMeansTokenPooler._kmeans_plus_plus_init(emb, k=10, importance_weights=weights)
        # All centroids should be different
        for i in range(10):
            for j in range(i + 1, 10):
                diff = (centroids[i] - centroids[j]).abs().sum()
                assert diff > 1e-6, f"Centroids {i} and {j} are identical"
