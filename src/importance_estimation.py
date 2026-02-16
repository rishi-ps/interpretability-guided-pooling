"""
Importance estimation methods for multi-vector document embeddings.

These methods estimate per-token importance scores for document patch embeddings,
enabling interpretability-guided token pooling. All methods are query-agnostic
(computed at indexing time, not at query time).

Three strategies are provided:
1. **Self-similarity redundancy**: Patches that are redundant (similar to many others) are less important.
2. **Centroid distance**: Patches far from the mean embedding carry more distinctive information.
3. **Probe-based importance**: Similarity to a set of semantic probe queries reveals content-bearing patches.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


@dataclass
class ImportanceOutput:
    """
    Output of an importance estimation method.

    Attributes:
        scores: Per-token importance scores. Either a list of 1D tensors (one per document,
                variable length) or a 2D tensor of shape (batch_size, max_tokens) with 0-padding.
        method: Name of the importance estimation method that produced these scores.
    """

    scores: Union[List[torch.Tensor], torch.Tensor]
    method: str


class BaseImportanceEstimator(ABC):
    """Abstract base class for importance estimation methods."""

    @abstractmethod
    def estimate(
        self,
        embeddings: Union[torch.Tensor, List[torch.Tensor]],
    ) -> ImportanceOutput:
        """
        Estimate per-token importance scores.

        Args:
            embeddings: Document embeddings. Either a list of 2D tensors
                        (token_length, embedding_dim) or a 3D tensor
                        (batch_size, token_length, embedding_dim).

        Returns:
            ImportanceOutput with per-token scores in [0, 1].
        """
        pass

    def _prepare_embeddings(
        self,
        embeddings: Union[torch.Tensor, List[torch.Tensor]],
    ) -> List[torch.Tensor]:
        """Convert input to a list of 2D tensors."""
        if isinstance(embeddings, torch.Tensor) and embeddings.dim() == 3:
            return list(embeddings.unbind(dim=0))
        if isinstance(embeddings, list):
            return embeddings
        raise ValueError("Embeddings must be a list of 2D tensors or a 3D tensor.")

    @staticmethod
    def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
        """Min-max normalize scores to [0, 1]."""
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val < 1e-10:
            return torch.ones_like(scores)
        return (scores - min_val) / (max_val - min_val)


class SelfSimilarityImportanceEstimator(BaseImportanceEstimator):
    """
    Importance based on self-similarity redundancy.

    For each patch embedding, compute the maximum cosine similarity to any other patch
    in the same document. Patches with *low* max-similarity to others are distinctive
    (non-redundant) and thus more important.

    importance(i) = 1 - max_{j != i} cos(e_i, e_j)

    Intuition: Background/margin patches are all similar to each other (high redundancy,
    low importance). Text, figures, and tables produce distinctive embeddings.
    """

    def estimate(
        self,
        embeddings: Union[torch.Tensor, List[torch.Tensor]],
    ) -> ImportanceOutput:
        embedding_list = self._prepare_embeddings(embeddings)
        scores_list: List[torch.Tensor] = []

        for emb in embedding_list:
            # emb: (token_length, dim)
            emb_normalized = F.normalize(emb.float(), p=2, dim=-1)
            # Pairwise cosine similarity: (token_length, token_length)
            sim_matrix = torch.mm(emb_normalized, emb_normalized.t())
            # Mask diagonal (self-similarity)
            sim_matrix.fill_diagonal_(-float("inf"))
            # Max similarity to any other token
            max_sim, _ = sim_matrix.max(dim=-1)  # (token_length,)
            # Importance = 1 - redundancy
            importance = 1.0 - max_sim
            importance = self._normalize_scores(importance)
            scores_list.append(importance.to(emb.dtype))

        return ImportanceOutput(scores=scores_list, method="self_similarity")


class CentroidDistanceImportanceEstimator(BaseImportanceEstimator):
    """
    Importance based on distance from the centroid embedding.

    For each patch, compute 1 - cosine_similarity(patch, mean_embedding).
    Patches far from the average are more informative.

    importance(i) = 1 - cos(e_i, mean(e))

    Intuition: The centroid represents the "average" content of the page.
    Distinctive regions (tables, figures, titles) deviate from this average.
    """

    def estimate(
        self,
        embeddings: Union[torch.Tensor, List[torch.Tensor]],
    ) -> ImportanceOutput:
        embedding_list = self._prepare_embeddings(embeddings)
        scores_list: List[torch.Tensor] = []

        for emb in embedding_list:
            # emb: (token_length, dim)
            emb_float = emb.float()
            centroid = emb_float.mean(dim=0, keepdim=True)  # (1, dim)
            centroid = F.normalize(centroid, p=2, dim=-1)
            emb_normalized = F.normalize(emb_float, p=2, dim=-1)
            # Cosine similarity to centroid: (token_length,)
            cos_sim = (emb_normalized * centroid).sum(dim=-1)
            importance = 1.0 - cos_sim
            importance = self._normalize_scores(importance)
            scores_list.append(importance.to(emb.dtype))

        return ImportanceOutput(scores=scores_list, method="centroid_distance")


class ProbeImportanceEstimator(BaseImportanceEstimator):
    """
    Importance based on similarity to a set of semantic probe queries.

    Encode a predefined set of document element descriptions (e.g., "title", "table",
    "figure") as probe embeddings. For each patch, compute the maximum similarity
    across all probes.

    importance(i) = max_q sim(e_i, probe_q)

    Intuition: Patches that strongly match any semantic probe are content-bearing.
    This method requires probe embeddings to be computed externally (typically by
    running the model's text encoder on probe strings).

    Default probes cover common document elements:
    - title, abstract, heading, paragraph, table, figure, chart, equation,
      caption, footnote, page number, author, date, reference, logo
    """

    # Default probe strings covering common document elements
    DEFAULT_PROBES: List[str] = [
        "title",
        "abstract",
        "heading",
        "paragraph text",
        "table with data",
        "figure or image",
        "chart or graph",
        "mathematical equation",
        "caption",
        "footnote",
        "page number",
        "author name",
        "date",
        "reference or citation",
        "logo or icon",
    ]

    def __init__(
        self,
        probe_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            probe_embeddings: Pre-computed probe embeddings of shape (num_probes, dim).
                              Must be L2-normalized. If None, must be set before calling estimate().
        """
        self.probe_embeddings = probe_embeddings

    @classmethod
    def from_processor_and_model(
        cls,
        model: torch.nn.Module,
        processor,
        probe_strings: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ) -> "ProbeImportanceEstimator":
        """
        Create a ProbeImportanceEstimator by encoding probe strings through the model.

        Args:
            model: A ColPali-family model with a forward() method that produces multi-vector embeddings.
            processor: The corresponding processor with process_queries() method.
            probe_strings: List of probe query strings. Uses DEFAULT_PROBES if None.
            device: Device to run inference on. Inferred from model if None.

        Returns:
            ProbeImportanceEstimator with pre-computed probe embeddings.
        """
        if probe_strings is None:
            probe_strings = cls.DEFAULT_PROBES

        if device is None:
            device = next(model.parameters()).device

        # Encode probes through the model's query pipeline
        with torch.no_grad():
            batch = processor.process_queries(probe_strings).to(device)
            # Output: (num_probes, query_tokens, dim)
            probe_multi_embeddings = model(**batch)
            # Mean-pool across query tokens to get a single vector per probe
            # Use attention mask if available
            if "attention_mask" in batch:
                mask = batch["attention_mask"].unsqueeze(-1).float()  # (num_probes, query_tokens, 1)
                probe_embeddings = (probe_multi_embeddings * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                probe_embeddings = probe_multi_embeddings.mean(dim=1)
            probe_embeddings = F.normalize(probe_embeddings.float(), p=2, dim=-1)  # (num_probes, dim)

        return cls(probe_embeddings=probe_embeddings)

    def estimate(
        self,
        embeddings: Union[torch.Tensor, List[torch.Tensor]],
    ) -> ImportanceOutput:
        if self.probe_embeddings is None:
            raise ValueError(
                "Probe embeddings not set. Use from_processor_and_model() or provide "
                "probe_embeddings at initialization."
            )

        embedding_list = self._prepare_embeddings(embeddings)
        scores_list: List[torch.Tensor] = []
        probe_emb = self.probe_embeddings.float()  # (num_probes, dim)

        for emb in embedding_list:
            # emb: (token_length, dim)
            emb_normalized = F.normalize(emb.float(), p=2, dim=-1)
            # Similarity to all probes: (token_length, num_probes)
            sim_to_probes = torch.mm(emb_normalized, probe_emb.to(emb.device).t())
            # Max across probes: (token_length,)
            max_sim, _ = sim_to_probes.max(dim=-1)
            importance = self._normalize_scores(max_sim)
            scores_list.append(importance.to(emb.dtype))

        return ImportanceOutput(scores=scores_list, method="probe")


def get_importance_map(
    importance_scores: torch.Tensor,
    n_patches: Tuple[int, int],
) -> torch.Tensor:
    """
    Reshape flat importance scores into a 2D spatial map for visualization.

    Args:
        importance_scores: 1D tensor of shape (n_patches_x * n_patches_y,).
        n_patches: Tuple (n_patches_x, n_patches_y) giving the spatial grid dimensions.

    Returns:
        2D tensor of shape (n_patches_x, n_patches_y) with importance values.
    """
    n_x, n_y = n_patches
    if importance_scores.numel() != n_x * n_y:
        raise ValueError(
            f"Number of importance scores ({importance_scores.numel()}) does not match "
            f"patch grid ({n_x} x {n_y} = {n_x * n_y})."
        )
    return importance_scores.view(n_y, n_x)
