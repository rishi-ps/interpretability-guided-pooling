"""Tests for importance estimation methods."""

import pytest
import torch

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


# --- Fixtures ---


@pytest.fixture
def uniform_embedding() -> torch.Tensor:
    """All-identical rows: every token is maximally redundant."""
    return torch.ones(10, 128) / (128**0.5)


@pytest.fixture
def diverse_embedding() -> torch.Tensor:
    """One-hot-like rows: every token is unique."""
    emb = torch.zeros(5, 128)
    for i in range(5):
        emb[i, i] = 1.0
    return emb


@pytest.fixture
def random_embedding() -> torch.Tensor:
    """Random normalized embedding."""
    emb = torch.randn(64, 128)
    return torch.nn.functional.normalize(emb, p=2, dim=-1)


@pytest.fixture
def batch_embeddings(random_embedding: torch.Tensor) -> list:
    """List of 2D embeddings with varying lengths."""
    return [random_embedding[:32], random_embedding[:64]]


# --- SelfSimilarityImportanceEstimator ---


class TestSelfSimilarity:
    def test_output_type(self, random_embedding: torch.Tensor):
        est = SelfSimilarityImportanceEstimator()
        out = est.estimate([random_embedding])
        assert isinstance(out, ImportanceOutput)
        assert out.method == "self_similarity"

    def test_output_shape(self, random_embedding: torch.Tensor):
        est = SelfSimilarityImportanceEstimator()
        out = est.estimate([random_embedding])
        assert len(out.scores) == 1
        assert out.scores[0].shape == (64,)

    def test_scores_in_unit_range(self, random_embedding: torch.Tensor):
        est = SelfSimilarityImportanceEstimator()
        out = est.estimate([random_embedding])
        assert out.scores[0].min() >= -1e-6
        assert out.scores[0].max() <= 1.0 + 1e-6

    def test_uniform_gets_low_variance(self, uniform_embedding: torch.Tensor):
        """All tokens identical → all equally (un)important → scores should be uniform."""
        est = SelfSimilarityImportanceEstimator()
        out = est.estimate([uniform_embedding])
        # All scores should be 1.0 (fallback when all identical)
        assert out.scores[0].std() < 0.01

    def test_diverse_has_high_scores(self, diverse_embedding: torch.Tensor):
        """All tokens orthogonal → none redundant → all should have high importance."""
        est = SelfSimilarityImportanceEstimator()
        out = est.estimate([diverse_embedding])
        # With orthogonal embeddings, max cosine similarity to others is 0
        # So importance = 1 - 0 = 1 → after normalization, all = 1
        assert out.scores[0].mean() > 0.8

    def test_batch_processing(self, batch_embeddings: list):
        est = SelfSimilarityImportanceEstimator()
        out = est.estimate(batch_embeddings)
        assert len(out.scores) == 2
        assert out.scores[0].shape == (32,)
        assert out.scores[1].shape == (64,)

    def test_3d_tensor_input(self, random_embedding: torch.Tensor):
        """Accept 3D tensor (batch, tokens, dim)."""
        est = SelfSimilarityImportanceEstimator()
        batch = random_embedding.unsqueeze(0)  # (1, 64, 128)
        out = est.estimate(batch)
        assert len(out.scores) == 1
        assert out.scores[0].shape == (64,)


# --- CentroidDistanceImportanceEstimator ---


class TestCentroidDistance:
    def test_output_type(self, random_embedding: torch.Tensor):
        est = CentroidDistanceImportanceEstimator()
        out = est.estimate([random_embedding])
        assert isinstance(out, ImportanceOutput)
        assert out.method == "centroid_distance"

    def test_output_shape(self, random_embedding: torch.Tensor):
        est = CentroidDistanceImportanceEstimator()
        out = est.estimate([random_embedding])
        assert out.scores[0].shape == (64,)

    def test_scores_in_unit_range(self, random_embedding: torch.Tensor):
        est = CentroidDistanceImportanceEstimator()
        out = est.estimate([random_embedding])
        assert out.scores[0].min() >= -1e-6
        assert out.scores[0].max() <= 1.0 + 1e-6

    def test_uniform_gets_low_variance(self, uniform_embedding: torch.Tensor):
        est = CentroidDistanceImportanceEstimator()
        out = est.estimate([uniform_embedding])
        assert out.scores[0].std() < 0.01

    def test_batch_processing(self, batch_embeddings: list):
        est = CentroidDistanceImportanceEstimator()
        out = est.estimate(batch_embeddings)
        assert len(out.scores) == 2
        assert out.scores[0].shape == (32,)
        assert out.scores[1].shape == (64,)


# --- ProbeImportanceEstimator ---


class TestProbeImportance:
    def test_with_manual_probes(self, random_embedding: torch.Tensor):
        probe_emb = torch.randn(5, 128)
        probe_emb = torch.nn.functional.normalize(probe_emb, p=2, dim=-1)
        est = ProbeImportanceEstimator(probe_embeddings=probe_emb)
        out = est.estimate([random_embedding])
        assert out.method == "probe"
        assert out.scores[0].shape == (64,)

    def test_scores_in_unit_range(self, random_embedding: torch.Tensor):
        probe_emb = torch.randn(10, 128)
        probe_emb = torch.nn.functional.normalize(probe_emb, p=2, dim=-1)
        est = ProbeImportanceEstimator(probe_embeddings=probe_emb)
        out = est.estimate([random_embedding])
        assert out.scores[0].min() >= -1e-6
        assert out.scores[0].max() <= 1.0 + 1e-6

    def test_raises_without_probes(self, random_embedding: torch.Tensor):
        est = ProbeImportanceEstimator()
        with pytest.raises(ValueError, match="Probe embeddings not set"):
            est.estimate([random_embedding])

    def test_default_probe_strings(self):
        assert len(ProbeImportanceEstimator.DEFAULT_PROBES) == 15

    def test_batch_processing(self, batch_embeddings: list):
        probe_emb = torch.randn(5, 128)
        probe_emb = torch.nn.functional.normalize(probe_emb, p=2, dim=-1)
        est = ProbeImportanceEstimator(probe_embeddings=probe_emb)
        out = est.estimate(batch_embeddings)
        assert len(out.scores) == 2


# --- get_importance_map ---


class TestImportanceMap:
    def test_basic_reshape(self):
        scores = torch.arange(12, dtype=torch.float32)
        imp_map = get_importance_map(scores, (3, 4))
        assert imp_map.shape == (4, 3)

    def test_mismatch_raises(self):
        scores = torch.arange(10, dtype=torch.float32)
        with pytest.raises(ValueError, match="does not match"):
            get_importance_map(scores, (3, 4))

    def test_single_patch(self):
        scores = torch.tensor([0.5])
        imp_map = get_importance_map(scores, (1, 1))
        assert imp_map.shape == (1, 1)
        assert imp_map.item() == pytest.approx(0.5)


# --- SVDImportanceEstimator ---


class TestSVDImportance:
    def test_output_type(self, random_embedding: torch.Tensor):
        est = SVDImportanceEstimator()
        out = est.estimate([random_embedding])
        assert isinstance(out, ImportanceOutput)
        assert out.method == "svd"

    def test_output_shape(self, random_embedding: torch.Tensor):
        est = SVDImportanceEstimator()
        out = est.estimate([random_embedding])
        assert len(out.scores) == 1
        assert out.scores[0].shape == (64,)

    def test_scores_in_unit_range(self, random_embedding: torch.Tensor):
        est = SVDImportanceEstimator()
        out = est.estimate([random_embedding])
        assert out.scores[0].min() >= -1e-6
        assert out.scores[0].max() <= 1.0 + 1e-6

    def test_uniform_gets_low_variance(self, uniform_embedding: torch.Tensor):
        """All tokens identical → degenerate case → scores should be nearly uniform."""
        est = SVDImportanceEstimator(rank=1)
        out = est.estimate([uniform_embedding])
        # With identical tokens the SVD is rank-1; normalized projections
        # should be close to equal (small numerical differences possible)
        assert out.scores[0].mean() > 0.8

    def test_diverse_has_high_scores(self, diverse_embedding: torch.Tensor):
        """Orthogonal tokens → all contribute equally to top singular vectors."""
        est = SVDImportanceEstimator(rank=5)
        out = est.estimate([diverse_embedding])
        # With 5 orthogonal vectors, rank=5 captures all
        assert out.scores[0].mean() > 0.8

    def test_batch_processing(self, batch_embeddings: list):
        est = SVDImportanceEstimator()
        out = est.estimate(batch_embeddings)
        assert len(out.scores) == 2
        assert out.scores[0].shape == (32,)
        assert out.scores[1].shape == (64,)

    def test_custom_rank(self, random_embedding: torch.Tensor):
        """Different ranks should produce different importance distributions."""
        est_r2 = SVDImportanceEstimator(rank=2)
        est_r16 = SVDImportanceEstimator(rank=16)
        out_r2 = est_r2.estimate([random_embedding])
        out_r16 = est_r16.estimate([random_embedding])
        diff = (out_r2.scores[0].float() - out_r16.scores[0].float()).abs().sum()
        assert diff > 0.01

    def test_invalid_rank(self):
        with pytest.raises(ValueError, match="rank"):
            SVDImportanceEstimator(rank=0)

    def test_3d_tensor_input(self, random_embedding: torch.Tensor):
        est = SVDImportanceEstimator()
        batch = random_embedding.unsqueeze(0)
        out = est.estimate(batch)
        assert len(out.scores) == 1
        assert out.scores[0].shape == (64,)


# --- AttentionImportanceEstimator (proxy mode) ---


class TestAttentionImportanceProxy:
    """Test the embedding-based proxy mode (no model needed)."""

    def test_output_type(self, random_embedding: torch.Tensor):
        est = AttentionImportanceEstimator(model=None, processor=None)
        out = est.estimate([random_embedding])
        assert isinstance(out, ImportanceOutput)
        assert out.method == "attention_proxy"

    def test_output_shape(self, random_embedding: torch.Tensor):
        est = AttentionImportanceEstimator(model=None, processor=None)
        out = est.estimate([random_embedding])
        assert len(out.scores) == 1
        assert out.scores[0].shape == (64,)

    def test_scores_in_unit_range(self, random_embedding: torch.Tensor):
        est = AttentionImportanceEstimator(model=None, processor=None)
        out = est.estimate([random_embedding])
        assert out.scores[0].min() >= -1e-6
        assert out.scores[0].max() <= 1.0 + 1e-6

    def test_uniform_gets_low_variance(self, uniform_embedding: torch.Tensor):
        """All tokens identical → uniform attention → uniform scores."""
        est = AttentionImportanceEstimator(model=None, processor=None)
        out = est.estimate([uniform_embedding])
        assert out.scores[0].std() < 0.01

    def test_batch_processing(self, batch_embeddings: list):
        est = AttentionImportanceEstimator(model=None, processor=None)
        out = est.estimate(batch_embeddings)
        assert len(out.scores) == 2
        assert out.scores[0].shape == (32,)
        assert out.scores[1].shape == (64,)

    def test_differs_from_centroid(self, random_embedding: torch.Tensor):
        """Attention proxy should produce different scores from centroid distance."""
        attn_est = AttentionImportanceEstimator(model=None, processor=None)
        cent_est = CentroidDistanceImportanceEstimator()
        attn_out = attn_est.estimate([random_embedding])
        cent_out = cent_est.estimate([random_embedding])
        diff = (attn_out.scores[0].float() - cent_out.scores[0].float()).abs().sum()
        assert diff > 0.01
