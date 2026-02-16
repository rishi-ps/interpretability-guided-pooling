"""
Importance-guided token pooling strategies.

These pooling strategies use per-token importance scores to make informed decisions
about which tokens to keep, merge, or discard during compression. They extend
the BaseTokenPooler interface from the ColPali engine.

Four strategies are provided:
1. **TopKTokenPooler**: Keep the k most important tokens.
2. **ImportanceWeightedHierarchicalTokenPooler**: Hierarchical clustering with
   importance-weighted mean pooling within clusters.
3. **ProtectAndPoolTokenPooler**: Protect top-p% tokens, apply hierarchical
   pooling to the rest.
4. **AdaptivePoolFactorTokenPooler**: Vary compression per document based on
   importance distribution entropy.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage

from colpali_engine.compression.token_pooling.base_token_pooling import BaseTokenPooler


class TopKTokenPooler(BaseTokenPooler):
    """
    Keep the top-k most important tokens based on precomputed importance scores.

    The simplest importance-guided strategy: rank tokens by importance, keep the
    top k = token_length // pool_factor, discard the rest. Retained tokens are
    returned in their original order (preserving spatial/sequential structure).

    Example:

    ```python
    import torch
    from src.importance_estimation import SelfSimilarityImportanceEstimator

    embeddings = [torch.randn(1024, 128)]
    estimator = SelfSimilarityImportanceEstimator()
    importance = estimator.estimate(embeddings)

    pooler = TopKTokenPooler(importance_scores=importance.scores)
    pooled = pooler.pool_embeddings(embeddings, pool_factor=4)
    # pooled[0].shape == (256, 128)
    ```
    """

    def __init__(self, importance_scores: Union[List[torch.Tensor], None] = None):
        """
        Args:
            importance_scores: Precomputed per-token importance scores.
                               List of 1D tensors, one per document.
                               If None, must be passed via pool_kwargs.
        """
        self.importance_scores = importance_scores

    def _pool_embeddings_impl(
        self,
        embeddings: List[torch.Tensor],
        pool_factor: int = 2,
        importance_scores: Optional[List[torch.Tensor]] = None,
        num_workers: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], Optional[List[Dict[int, Tuple[torch.Tensor]]]]]:
        scores = importance_scores or self.importance_scores
        if scores is None:
            raise ValueError("importance_scores must be provided either at init or at call time.")

        args = list(zip(embeddings, scores, [pool_factor] * len(embeddings)))

        if num_workers and num_workers > 1:
            with ThreadPoolExecutor(num_workers) as executor:
                results = list(executor.map(lambda a: self._pool_single(*a), args))
        else:
            results = [self._pool_single(*a) for a in args]

        pooled = [r[0] for r in results]
        mappings = [r[1] for r in results]
        return pooled, mappings

    @staticmethod
    def _pool_single(
        embedding: torch.Tensor,
        scores: torch.Tensor,
        pool_factor: int,
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor]]]:
        """Select top-k tokens by importance, preserving original order."""
        token_length = embedding.size(0)
        k = max(token_length // pool_factor, 1)

        # Get top-k indices, then sort to preserve order
        _, topk_indices = scores.to(embedding.device).topk(k, dim=0)
        topk_indices_sorted, _ = topk_indices.sort()

        pooled = embedding[topk_indices_sorted]  # (k, dim)

        # Build cluster mapping: each kept token is its own "cluster"
        cluster_map: Dict[int, Tuple[torch.Tensor]] = {}
        for cluster_id, idx in enumerate(topk_indices_sorted):
            cluster_map[cluster_id] = (idx.unsqueeze(0),)

        return pooled, cluster_map


class ImportanceWeightedHierarchicalTokenPooler(BaseTokenPooler):
    """
    Hierarchical clustering with importance-weighted mean pooling.

    Uses the same agglomerative clustering as HierarchicalTokenPooler, but when
    computing the representative vector for each cluster, uses importance scores
    as weights instead of a uniform mean. This biases the pooled representation
    toward the most important tokens within each cluster.

    pooled_vector(cluster_c) = normalize( sum_i( importance(i) * e_i ) for i in cluster_c )

    Example:

    ```python
    import torch
    from src.importance_estimation import CentroidDistanceImportanceEstimator

    embeddings = [torch.randn(1024, 128)]
    estimator = CentroidDistanceImportanceEstimator()
    importance = estimator.estimate(embeddings)

    pooler = ImportanceWeightedHierarchicalTokenPooler(importance_scores=importance.scores)
    pooled = pooler.pool_embeddings(embeddings, pool_factor=3)
    ```
    """

    def __init__(self, importance_scores: Union[List[torch.Tensor], None] = None):
        self.importance_scores = importance_scores

    def _pool_embeddings_impl(
        self,
        embeddings: List[torch.Tensor],
        pool_factor: int = 2,
        importance_scores: Optional[List[torch.Tensor]] = None,
        num_workers: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[Dict[int, Tuple[torch.Tensor]]]]:
        scores = importance_scores or self.importance_scores
        if scores is None:
            raise ValueError("importance_scores must be provided either at init or at call time.")

        args = list(zip(embeddings, scores, [pool_factor] * len(embeddings)))

        if num_workers and num_workers > 1:
            with ThreadPoolExecutor(num_workers) as executor:
                results = list(executor.map(lambda a: self._pool_single(*a), args))
        else:
            results = [self._pool_single(*a) for a in args]

        pooled = [r[0] for r in results]
        mappings = [r[1] for r in results]
        return pooled, mappings

    @staticmethod
    def _pool_single(
        embedding: torch.Tensor,
        scores: torch.Tensor,
        pool_factor: int,
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor]]]:
        token_length = embedding.size(0)
        if token_length <= 1:
            raise ValueError("Input must have more than one token.")

        if pool_factor == 1:
            cluster_map = {0: (torch.arange(token_length),)}
            return embedding, cluster_map

        dtype = embedding.dtype
        device = embedding.device
        embedding_f = embedding.to(torch.float32).cpu()
        scores_f = scores.to(torch.float32).cpu()

        # Hierarchical clustering (same as HierarchicalTokenPooler)
        similarities = torch.mm(embedding_f, embedding_f.t())
        distances = 1 - similarities.numpy()
        Z = linkage(distances, metric="euclidean", method="ward")  # noqa: N806
        max_clusters = max(token_length // pool_factor, 1)
        cluster_labels: NDArray[np.int32] = fcluster(Z, t=max_clusters, criterion="maxclust") - 1

        cluster_map: Dict[int, Tuple[torch.Tensor]] = {}
        pooled_list: List[torch.Tensor] = []

        with torch.no_grad():
            for cluster_id in range(max_clusters):
                cluster_indices = cast(
                    Tuple[torch.Tensor],
                    torch.where(torch.tensor(cluster_labels == cluster_id)),
                )
                cluster_map[cluster_id] = cluster_indices

                if cluster_indices[0].numel() > 0:
                    cluster_embs = embedding_f[cluster_indices]  # (n, dim)
                    cluster_weights = scores_f[cluster_indices[0]]  # (n,)

                    # Ensure weights sum to > 0; fall back to uniform if all zero
                    weight_sum = cluster_weights.sum()
                    if weight_sum < 1e-10:
                        cluster_weights = torch.ones_like(cluster_weights)
                        weight_sum = cluster_weights.sum()

                    # Weighted mean
                    weights_normalized = cluster_weights / weight_sum  # (n,)
                    pooled = (cluster_embs * weights_normalized.unsqueeze(-1)).sum(dim=0)
                    pooled = F.normalize(pooled, p=2, dim=-1)
                    pooled_list.append(pooled)

        pooled_embeddings = torch.stack(pooled_list, dim=0).to(device).to(dtype)
        return pooled_embeddings, cluster_map


class ProtectAndPoolTokenPooler(BaseTokenPooler):
    """
    Protect the most important tokens and pool the rest.

    Two-phase strategy:
    1. **Protect**: Keep the top-p fraction of tokens by importance, untouched.
    2. **Pool**: Apply hierarchical clustering to the remaining tokens.

    The final output concatenates protected tokens (in original order) with
    pooled representations of the remaining tokens.

    Args:
        protect_fraction: Fraction of tokens to protect (default: 0.25 = top 25%).
        importance_scores: Precomputed importance scores.

    Example:

    ```python
    pooler = ProtectAndPoolTokenPooler(protect_fraction=0.25, importance_scores=scores)
    pooled = pooler.pool_embeddings(embeddings, pool_factor=4)
    ```
    """

    def __init__(
        self,
        protect_fraction: float = 0.25,
        importance_scores: Union[List[torch.Tensor], None] = None,
    ):
        if not 0.0 < protect_fraction < 1.0:
            raise ValueError(f"protect_fraction must be in (0, 1), got {protect_fraction}")
        self.protect_fraction = protect_fraction
        self.importance_scores = importance_scores

    def _pool_embeddings_impl(
        self,
        embeddings: List[torch.Tensor],
        pool_factor: int = 2,
        importance_scores: Optional[List[torch.Tensor]] = None,
        num_workers: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], Optional[List[Dict[int, Tuple[torch.Tensor]]]]]:
        scores = importance_scores or self.importance_scores
        if scores is None:
            raise ValueError("importance_scores must be provided either at init or at call time.")

        args = list(
            zip(embeddings, scores, [pool_factor] * len(embeddings), [self.protect_fraction] * len(embeddings))
        )

        if num_workers and num_workers > 1:
            with ThreadPoolExecutor(num_workers) as executor:
                results = list(executor.map(lambda a: self._pool_single(*a), args))
        else:
            results = [self._pool_single(*a) for a in args]

        pooled = [r[0] for r in results]
        mappings = [r[1] for r in results]
        return pooled, mappings

    @staticmethod
    def _pool_single(
        embedding: torch.Tensor,
        scores: torch.Tensor,
        pool_factor: int,
        protect_fraction: float,
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor]]]:
        token_length = embedding.size(0)
        target_output = max(token_length // pool_factor, 1)

        # Determine how many tokens to protect
        n_protect = max(int(token_length * protect_fraction), 1)
        n_protect = min(n_protect, target_output)  # Can't protect more than target output

        # Get protected indices (sorted to preserve order)
        _, topk_indices = scores.to(embedding.device).topk(n_protect, dim=0)
        protect_indices_sorted, _ = topk_indices.sort()
        protect_mask = torch.zeros(token_length, dtype=torch.bool, device=embedding.device)
        protect_mask[protect_indices_sorted] = True

        protected_embs = embedding[protect_indices_sorted]  # (n_protect, dim)

        # Pool the remaining tokens
        remaining_indices = torch.where(~protect_mask)[0]
        n_remaining_output = target_output - n_protect

        cluster_map: Dict[int, Tuple[torch.Tensor]] = {}

        # Add protected tokens to cluster map
        for i, idx in enumerate(protect_indices_sorted):
            cluster_map[i] = (idx.unsqueeze(0),)

        if n_remaining_output > 0 and remaining_indices.numel() > 1:
            remaining_embs = embedding[remaining_indices]  # (n_remaining, dim)
            dtype = remaining_embs.dtype
            device_orig = remaining_embs.device
            remaining_f = remaining_embs.to(torch.float32).cpu()

            if remaining_indices.numel() <= n_remaining_output:
                # Not enough remaining tokens to cluster — keep them all
                pooled_remaining_tensor = remaining_f
                for i, idx in enumerate(remaining_indices):
                    cluster_map[n_protect + i] = (idx.unsqueeze(0),)
            else:
                # Hierarchical pooling on remaining tokens
                similarities = torch.mm(remaining_f, remaining_f.t())
                distances = 1 - similarities.numpy()

                max_clusters = max(n_remaining_output, 1)
                Z = linkage(distances, metric="euclidean", method="ward")  # noqa: N806
                cluster_labels: NDArray[np.int32] = fcluster(Z, t=max_clusters, criterion="maxclust") - 1

                pooled_remaining_list: List[torch.Tensor] = []
                with torch.no_grad():
                    for c_id in range(max_clusters):
                        c_indices_local = cast(
                            Tuple[torch.Tensor],
                            torch.where(torch.tensor(cluster_labels == c_id)),
                        )
                        # Map back to original indices
                        c_indices_global = remaining_indices[c_indices_local[0]]
                        cluster_map[n_protect + c_id] = (c_indices_global,)

                        if c_indices_local[0].numel() > 0:
                            pooled = remaining_f[c_indices_local].mean(dim=0)
                            pooled = F.normalize(pooled, p=2, dim=-1)
                            pooled_remaining_list.append(pooled)

                pooled_remaining_tensor = torch.stack(pooled_remaining_list, dim=0) if pooled_remaining_list else None

            if pooled_remaining_tensor is not None:
                result = torch.cat([protected_embs, pooled_remaining_tensor.to(device_orig).to(dtype)], dim=0)
            else:
                result = protected_embs
        elif remaining_indices.numel() == 1 and n_remaining_output > 0:
            # Single remaining token — just append it
            cluster_map[n_protect] = (remaining_indices,)
            result = torch.cat([protected_embs, embedding[remaining_indices]], dim=0)
        else:
            result = protected_embs

        return result, cluster_map


class AdaptivePoolFactorTokenPooler(BaseTokenPooler):
    """
    Vary pool factor per document based on importance distribution entropy.

    Documents with concentrated importance (few important patches, e.g., a single
    table) can be compressed more. Documents with distributed importance (many
    important patches) should be compressed less.

    The effective pool factor for each document is computed as:

        entropy = -sum(p * log(p))  where p = softmax(importance_scores)
        max_entropy = log(token_length)
        normalized_entropy = entropy / max_entropy  (in [0, 1])
        effective_pool_factor = min_pool_factor + (max_pool_factor - min_pool_factor) * (1 - normalized_entropy)

    Low entropy (concentrated) → high compression. High entropy (distributed) → low compression.

    The total storage budget across the batch is approximately equivalent to using
    a fixed pool factor of (min_pool_factor + max_pool_factor) / 2.

    Example:

    ```python
    pooler = AdaptivePoolFactorTokenPooler(
        min_pool_factor=2, max_pool_factor=8,
        importance_scores=scores,
    )
    pooled = pooler.pool_embeddings(embeddings)
    ```
    """

    def __init__(
        self,
        min_pool_factor: int = 2,
        max_pool_factor: int = 8,
        importance_scores: Union[List[torch.Tensor], None] = None,
    ):
        if min_pool_factor < 1:
            raise ValueError(f"min_pool_factor must be >= 1, got {min_pool_factor}")
        if max_pool_factor < min_pool_factor:
            raise ValueError(
                f"max_pool_factor ({max_pool_factor}) must be >= min_pool_factor ({min_pool_factor})"
            )
        self.min_pool_factor = min_pool_factor
        self.max_pool_factor = max_pool_factor
        self.importance_scores = importance_scores

    def _compute_effective_pool_factor(self, scores: torch.Tensor) -> int:
        """Compute effective pool factor from importance score entropy."""
        scores_f = scores.float()
        # Normalize scores to a probability distribution (they are already in [0,1])
        # Add small epsilon to avoid zeros, then normalize to sum to 1
        probs = scores_f + 1e-10
        probs = probs / probs.sum()
        # Shannon entropy
        entropy = -(probs * torch.log(probs)).sum()
        max_entropy = np.log(scores.numel())
        normalized_entropy = (entropy.item() / max_entropy) if max_entropy > 0 else 1.0
        normalized_entropy = min(max(normalized_entropy, 0.0), 1.0)

        # Low entropy → high pool factor (more compression)
        effective = self.min_pool_factor + (self.max_pool_factor - self.min_pool_factor) * (
            1.0 - normalized_entropy
        )
        return max(int(round(effective)), self.min_pool_factor)

    def _pool_embeddings_impl(
        self,
        embeddings: List[torch.Tensor],
        importance_scores: Optional[List[torch.Tensor]] = None,
        num_workers: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[Dict[int, Tuple[torch.Tensor]]]]:
        scores = importance_scores or self.importance_scores
        if scores is None:
            raise ValueError("importance_scores must be provided either at init or at call time.")

        results = []
        for emb, sc in zip(embeddings, scores):
            pf = self._compute_effective_pool_factor(sc)
            result = self._pool_single_hierarchical(emb, pf)
            results.append(result)

        pooled = [r[0] for r in results]
        mappings = [r[1] for r in results]
        return pooled, mappings

    @staticmethod
    def _pool_single_hierarchical(
        embedding: torch.Tensor,
        pool_factor: int,
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor]]]:
        """Standard hierarchical pooling for a single embedding (same as HierarchicalTokenPooler)."""
        token_length = embedding.size(0)
        if token_length <= 1:
            return embedding, {0: (torch.arange(token_length),)}

        if pool_factor == 1:
            return embedding, {0: (torch.arange(token_length),)}

        dtype = embedding.dtype
        device = embedding.device
        embedding_f = embedding.to(torch.float32).cpu()

        similarities = torch.mm(embedding_f, embedding_f.t())
        distances = 1 - similarities.numpy()
        Z = linkage(distances, metric="euclidean", method="ward")  # noqa: N806
        max_clusters = max(token_length // pool_factor, 1)
        cluster_labels: NDArray[np.int32] = fcluster(Z, t=max_clusters, criterion="maxclust") - 1

        cluster_map: Dict[int, Tuple[torch.Tensor]] = {}
        pooled_list: List[torch.Tensor] = []

        with torch.no_grad():
            for c_id in range(max_clusters):
                c_indices = cast(
                    Tuple[torch.Tensor],
                    torch.where(torch.tensor(cluster_labels == c_id)),
                )
                cluster_map[c_id] = c_indices
                if c_indices[0].numel() > 0:
                    pooled = embedding_f[c_indices].mean(dim=0)
                    pooled = F.normalize(pooled, p=2, dim=-1)
                    pooled_list.append(pooled)

        pooled_embeddings = torch.stack(pooled_list, dim=0).to(device).to(dtype)
        return pooled_embeddings, cluster_map
