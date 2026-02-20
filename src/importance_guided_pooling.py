"""
Importance-guided token pooling strategies.

These pooling strategies use per-token importance scores to make informed decisions
about which tokens to keep, merge, or discard during compression. They extend
the BaseTokenPooler interface from the ColPali engine.

Seven strategies are provided:
1. **TopKTokenPooler**: Keep the k most important tokens.
2. **ImportanceWeightedHierarchicalTokenPooler**: Hierarchical clustering with
   importance-weighted mean pooling within clusters.
3. **ProtectAndPoolTokenPooler**: Protect top-p% tokens, apply hierarchical
   pooling to the rest.
4. **AdaptivePoolFactorTokenPooler**: Vary compression per document based on
   importance distribution entropy.
5. **SplitAndAllocateTokenPooler**: Split tokens into importance tiers and
   allocate more cluster budget to the important tier.
6. **ImportanceWeightedDistancePooler**: Scale embeddings by importance before
   computing distances for clustering, so important tokens resist merging.
7. **ImportanceWeightedKMeansTokenPooler**: K-means clustering with importance-weighted
   centroids and importance-biased initialization.
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

        # Determine how many tokens to protect.
        # Reserve at least 25% of the output budget for pooling the remaining tokens,
        # to avoid collapsing to pure top-k selection at high pool factors.
        max_protect = max(int(target_output * 0.75), 1)
        n_protect = max(int(token_length * protect_fraction), 1)
        n_protect = min(n_protect, max_protect)

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
        # Use softmax with temperature to create a sharper probability distribution.
        # Raw min-max normalized scores are nearly uniform for large token counts,
        # which makes entropy uninformative. Temperature < 1 amplifies differences.
        temperature = 0.1
        probs = torch.softmax(scores_f / temperature, dim=0)
        # Shannon entropy
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
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


class SplitAndAllocateTokenPooler(BaseTokenPooler):
    """
    Importance-guided cluster budget allocation.

    Splits tokens into two tiers by importance, then allocates more of the fixed
    cluster budget to the important tier. Both tiers are compressed via
    hierarchical clustering independently, then concatenated.

    For a pool_factor of 4 with 1000 tokens (250 output clusters):
      - Top 50% (500 tokens) → 188 clusters (75% of budget, effective PF ≈ 2.7)
      - Bottom 50% (500 tokens) → 62 clusters (25% of budget, effective PF ≈ 8.1)
      - Total: 250 clusters (same as standard hierarchical)

    This gives important tokens (text, tables, figures) 3× the representation
    fidelity of unimportant tokens (margins, backgrounds), at the same total
    storage cost as standard hierarchical pooling.

    Args:
        importance_scores: Precomputed importance scores.
        split_fraction: Fraction of tokens in the "important" tier (default: 0.5).
        budget_fraction: Fraction of cluster budget allocated to the important tier (default: 0.75).
    """

    def __init__(
        self,
        importance_scores: Union[List[torch.Tensor], None] = None,
        split_fraction: float = 0.5,
        budget_fraction: float = 0.75,
    ):
        if not 0.0 < split_fraction < 1.0:
            raise ValueError(f"split_fraction must be in (0, 1), got {split_fraction}")
        if not 0.0 < budget_fraction < 1.0:
            raise ValueError(f"budget_fraction must be in (0, 1), got {budget_fraction}")
        self.importance_scores = importance_scores
        self.split_fraction = split_fraction
        self.budget_fraction = budget_fraction

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

        args = list(
            zip(
                embeddings,
                scores,
                [pool_factor] * len(embeddings),
                [self.split_fraction] * len(embeddings),
                [self.budget_fraction] * len(embeddings),
            )
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
        split_fraction: float,
        budget_fraction: float,
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor]]]:
        token_length = embedding.size(0)
        if token_length <= 1:
            raise ValueError("Input must have more than one token.")

        if pool_factor == 1:
            cluster_map = {0: (torch.arange(token_length),)}
            return embedding, cluster_map

        total_clusters = max(token_length // pool_factor, 1)

        # Split tokens into important / unimportant tiers
        n_important = max(int(token_length * split_fraction), 1)
        n_important = min(n_important, token_length - 1)  # Ensure at least 1 unimportant

        _, sorted_indices = scores.to(embedding.device).sort(descending=True)
        important_indices = sorted_indices[:n_important].sort().values
        unimportant_indices = sorted_indices[n_important:].sort().values

        # Allocate cluster budget
        clusters_important = max(int(total_clusters * budget_fraction), 1)
        clusters_unimportant = max(total_clusters - clusters_important, 1)
        # Adjust if a tier has fewer tokens than clusters
        clusters_important = min(clusters_important, n_important)
        clusters_unimportant = min(clusters_unimportant, token_length - n_important)
        # Reallocate any surplus
        surplus = total_clusters - clusters_important - clusters_unimportant
        if surplus > 0:
            if clusters_important < n_important:
                add = min(surplus, n_important - clusters_important)
                clusters_important += add
                surplus -= add
            if surplus > 0 and clusters_unimportant < (token_length - n_important):
                clusters_unimportant += surplus

        dtype = embedding.dtype
        device = embedding.device
        embedding_f = embedding.to(torch.float32).cpu()

        cluster_map: Dict[int, Tuple[torch.Tensor]] = {}
        pooled_list: List[torch.Tensor] = []
        cluster_offset = 0

        # Pool each tier independently
        for tier_indices, n_clusters in [
            (important_indices, clusters_important),
            (unimportant_indices, clusters_unimportant),
        ]:
            tier_embs = embedding_f[tier_indices]  # (n_tier, dim)
            n_tier = tier_embs.size(0)

            if n_tier <= n_clusters:
                # Fewer tokens than clusters — keep all tokens as-is
                for i, idx in enumerate(tier_indices):
                    cluster_map[cluster_offset + i] = (idx.unsqueeze(0),)
                    pooled_list.append(F.normalize(tier_embs[i], p=2, dim=-1))
                cluster_offset += n_tier
            else:
                # Hierarchical clustering within the tier
                similarities = torch.mm(tier_embs, tier_embs.t())
                distances = 1 - similarities.numpy()
                Z = linkage(distances, metric="euclidean", method="ward")  # noqa: N806
                cluster_labels: NDArray[np.int32] = fcluster(Z, t=n_clusters, criterion="maxclust") - 1

                with torch.no_grad():
                    for c_id in range(n_clusters):
                        c_local = cast(
                            Tuple[torch.Tensor],
                            torch.where(torch.tensor(cluster_labels == c_id)),
                        )
                        # Map back to original document-level indices
                        c_global = tier_indices[c_local[0]]
                        cluster_map[cluster_offset + c_id] = (c_global,)

                        if c_local[0].numel() > 0:
                            pooled = tier_embs[c_local].mean(dim=0)
                            pooled = F.normalize(pooled, p=2, dim=-1)
                            pooled_list.append(pooled)

                cluster_offset += n_clusters

        pooled_embeddings = torch.stack(pooled_list, dim=0).to(device).to(dtype)
        return pooled_embeddings, cluster_map


class ImportanceWeightedDistancePooler(BaseTokenPooler):
    """
    Scale embeddings by importance before computing distances for clustering.

    Important tokens are scaled up, making them appear "further" from other tokens
    in the distance space used for hierarchical clustering. This causes Ward linkage
    to preferentially merge unimportant tokens first, giving important tokens smaller
    (more precise) clusters.

    Critically, the cluster *representatives* are computed from the original
    (unscaled) embeddings — the scaling only affects the merge order, not the
    output vectors. This preserves the embedding space geometry for MaxSim scoring.

    Scaling formula per token i:
        e_i_scaled = e_i * (1 + alpha * importance_i_normalized)

    where importance_i_normalized is min-max scaled to [0, 1] within the document.

    Args:
        importance_scores: Precomputed per-token importance scores.
        alpha: Scaling strength (default: 1.0). Higher values make importance
               have a stronger effect on cluster formation.
    """

    def __init__(
        self,
        importance_scores: Union[List[torch.Tensor], None] = None,
        alpha: float = 1.0,
    ):
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        self.importance_scores = importance_scores
        self.alpha = alpha

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

        args = list(
            zip(
                embeddings,
                scores,
                [pool_factor] * len(embeddings),
                [self.alpha] * len(embeddings),
            )
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
        alpha: float,
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

        # Min-max normalize importance to [0, 1]
        s_min = scores_f.min()
        s_max = scores_f.max()
        if s_max - s_min > 1e-10:
            scores_norm = (scores_f - s_min) / (s_max - s_min)
        else:
            scores_norm = torch.ones_like(scores_f)

        # Scale embeddings: important tokens get larger magnitude
        # This makes them "further" from others in Euclidean space
        scale = 1.0 + alpha * scores_norm  # (token_length,)
        scaled_embs = embedding_f * scale.unsqueeze(-1)  # (token_length, dim)

        # Cluster on SCALED embeddings using Euclidean distance
        # Important tokens have larger magnitude → larger Euclidean distances
        # → they resist merging and get their own smaller clusters
        Z = linkage(scaled_embs.numpy(), metric="euclidean", method="ward")  # noqa: N806
        max_clusters = max(token_length // pool_factor, 1)
        cluster_labels: NDArray[np.int32] = fcluster(Z, t=max_clusters, criterion="maxclust") - 1

        # Compute representatives from ORIGINAL embeddings (not scaled)
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


class ImportanceWeightedKMeansTokenPooler(BaseTokenPooler):
    """
    K-means clustering with importance-weighted centroids and importance-biased
    initialization (k-means++ with D² sampling weighted by importance).

    Compared to hierarchical clustering (Ward linkage), k-means:
    - Optimizes reconstruction error directly (important for MaxSim fidelity)
    - Supports importance-weighted centroids: cluster representatives are biased
      toward their most important members
    - Uses importance-biased D²-sampling for initialization: important tokens
      are preferentially chosen as initial seeds, giving content-rich regions
      finer coverage from the start

    The centroid for cluster c is:

        centroid_c = normalize( sum_i w_i * e_i  for i in cluster_c )
        where w_i = 1 + alpha * importance_i_normalized

    Args:
        importance_scores: Precomputed per-token importance scores.
        alpha: Weight strength for centroids (default: 1.0). 0 = uniform k-means.
        max_iters: Maximum k-means iterations (default: 20).
        n_restarts: Number of random restarts, best by inertia (default: 3).
    """

    def __init__(
        self,
        importance_scores: Union[List[torch.Tensor], None] = None,
        alpha: float = 1.0,
        max_iters: int = 20,
        n_restarts: int = 3,
    ):
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if max_iters < 1:
            raise ValueError(f"max_iters must be >= 1, got {max_iters}")
        if n_restarts < 1:
            raise ValueError(f"n_restarts must be >= 1, got {n_restarts}")
        self.importance_scores = importance_scores
        self.alpha = alpha
        self.max_iters = max_iters
        self.n_restarts = n_restarts

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

        args = list(
            zip(
                embeddings,
                scores,
                [pool_factor] * len(embeddings),
                [self.alpha] * len(embeddings),
                [self.max_iters] * len(embeddings),
                [self.n_restarts] * len(embeddings),
            )
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
    def _kmeans_plus_plus_init(
        embeddings: torch.Tensor,
        k: int,
        importance_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Importance-biased K-means++ initialization.

        Standard D² sampling is multiplied by importance weights, so important
        tokens are more likely to be chosen as seeds. This gives content-rich
        regions finer coverage from the start.

        Args:
            embeddings: (n, dim) float tensor, L2-normalized.
            k: Number of clusters.
            importance_weights: (n,) float tensor, positive.

        Returns:
            (k, dim) initial centroids.
        """
        n = embeddings.size(0)
        chosen_indices: List[int] = []

        # First centroid: sample proportional to importance
        probs = importance_weights / importance_weights.sum()
        idx = torch.multinomial(probs, 1).item()
        chosen_indices.append(idx)

        for _ in range(1, k):
            # Distance to nearest existing centroid (cosine distance)
            stacked = embeddings[chosen_indices]  # (current_k, dim)
            sims = torch.mm(embeddings, stacked.t())  # (n, current_k)
            min_dist = (1.0 - sims.max(dim=1).values).clamp(min=0)  # (n,)
            # D² × importance weighting
            weighted_dist = min_dist * min_dist * importance_weights
            total = weighted_dist.sum()
            if total < 1e-10:
                # Fallback: pick random unchosen point
                unchosen = list(set(range(n)) - set(chosen_indices))
                idx = unchosen[torch.randint(len(unchosen), (1,)).item()] if unchosen else 0
            else:
                probs = weighted_dist / total
                idx = torch.multinomial(probs, 1).item()
            chosen_indices.append(idx)

        return embeddings[chosen_indices].clone()

    @staticmethod
    def _pool_single(
        embedding: torch.Tensor,
        scores: torch.Tensor,
        pool_factor: int,
        alpha: float,
        max_iters: int,
        n_restarts: int,
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor]]]:
        token_length = embedding.size(0)
        if token_length <= 1:
            raise ValueError("Input must have more than one token.")

        if pool_factor == 1:
            cluster_map = {0: (torch.arange(token_length),)}
            return embedding, cluster_map

        dtype = embedding.dtype
        device = embedding.device
        k = max(token_length // pool_factor, 1)

        embedding_f = F.normalize(embedding.to(torch.float32).cpu(), p=2, dim=-1)
        scores_f = scores.to(torch.float32).cpu()

        # Min-max normalize importance to [0, 1]
        s_min = scores_f.min()
        s_max = scores_f.max()
        if s_max - s_min > 1e-10:
            scores_norm = (scores_f - s_min) / (s_max - s_min)
        else:
            scores_norm = torch.ones_like(scores_f)

        # Importance-based weights for centroid computation
        weights = 1.0 + alpha * scores_norm  # (token_length,)

        best_centroids = None
        best_labels = None
        best_inertia = float("inf")

        for restart in range(n_restarts):
            torch.manual_seed(42 + restart)

            # Importance-biased K-means++ init
            centroids = ImportanceWeightedKMeansTokenPooler._kmeans_plus_plus_init(
                embedding_f, k, weights,
            )

            labels = torch.zeros(token_length, dtype=torch.long)

            for _ in range(max_iters):
                # Assignment: each token → nearest centroid (cosine similarity)
                sims = torch.mm(embedding_f, centroids.t())  # (n, k)
                new_labels = sims.argmax(dim=1)  # (n,)

                if torch.equal(new_labels, labels):
                    break
                labels = new_labels

                # Update: importance-weighted centroid computation
                new_centroids = torch.zeros_like(centroids)
                for c_id in range(k):
                    mask = labels == c_id
                    if mask.sum() == 0:
                        # Dead cluster: reinit to farthest point from all centroids
                        sims_all = torch.mm(embedding_f, centroids.t())
                        min_sim_per_point = sims_all.max(dim=1).values
                        new_centroids[c_id] = embedding_f[min_sim_per_point.argmin()]
                    else:
                        cluster_embs = embedding_f[mask]
                        cluster_weights = weights[mask]
                        w_norm = cluster_weights / cluster_weights.sum()
                        new_centroids[c_id] = (cluster_embs * w_norm.unsqueeze(-1)).sum(dim=0)

                centroids = F.normalize(new_centroids, p=2, dim=-1)

            # Inertia: importance-weighted sum of cosine distances
            sims = torch.mm(embedding_f, centroids.t())
            best_sim = sims.max(dim=1).values
            inertia = (weights * (1.0 - best_sim)).sum().item()

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids.clone()
                best_labels = labels.clone()

        # Build output from best run
        cluster_map: Dict[int, Tuple[torch.Tensor]] = {}
        pooled_list: List[torch.Tensor] = []

        with torch.no_grad():
            for c_id in range(k):
                c_indices = cast(
                    Tuple[torch.Tensor],
                    torch.where(best_labels == c_id),
                )
                cluster_map[c_id] = c_indices
                if c_indices[0].numel() > 0:
                    cluster_embs = embedding_f[c_indices]
                    cluster_weights = weights[c_indices[0]]
                    w_norm = cluster_weights / cluster_weights.sum()
                    pooled = (cluster_embs * w_norm.unsqueeze(-1)).sum(dim=0)
                    pooled = F.normalize(pooled, p=2, dim=-1)
                    pooled_list.append(pooled)

        pooled_embeddings = torch.stack(pooled_list, dim=0).to(device).to(dtype)
        return pooled_embeddings, cluster_map
