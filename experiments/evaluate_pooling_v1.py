#!/usr/bin/env python3
"""
Evaluate retrieval quality with different pooling strategies and compression ratios.

This script:
1. Loads a ColModernVBert (or ColIdefics3) model
2. Embeds all documents and queries from a ViDoRe-format dataset
3. Applies different pooling strategies at various compression ratios
4. Computes NDCG@5 and Recall@5 for each configuration
5. Saves results as CSV + plots

Usage:
    python experiments/evaluate_pooling.py \
        --model ModernVBERT/colmodernvbert \
        --dataset vidore/docvqa_test_subsampled \
        --pool-factors 1 2 3 4 6 8 \
        --output-dir experiments/results

Requirements:
    pip install -e ".[eval]"
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# ColPali imports (installed via pip)
from colpali_engine.compression import HierarchicalTokenPooler

# Our imports
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


def compute_maxsim_scores(
    query_embeddings: List[torch.Tensor],
    doc_embeddings: List[torch.Tensor],
) -> torch.Tensor:
    """
    Compute MaxSim scores between all query-document pairs.

    Args:
        query_embeddings: List of tensors, each (query_tokens, dim).
        doc_embeddings: List of tensors, each (doc_tokens, dim).

    Returns:
        Score matrix of shape (num_queries, num_docs).
    """
    num_queries = len(query_embeddings)
    num_docs = len(doc_embeddings)
    scores = torch.zeros(num_queries, num_docs)

    for q_idx in tqdm(range(num_queries), desc="Computing MaxSim", leave=False):
        q_emb = query_embeddings[q_idx].float()  # (qt, dim)
        for d_idx in range(num_docs):
            d_emb = doc_embeddings[d_idx].float()  # (dt, dim)
            # MaxSim: for each query token, find max similarity across doc tokens
            sim = torch.mm(q_emb, d_emb.t())  # (qt, dt)
            max_sim_per_query_token = sim.max(dim=1).values  # (qt,)
            scores[q_idx, d_idx] = max_sim_per_query_token.sum()

    return scores


def compute_ndcg_at_k(
    scores: torch.Tensor,
    relevance: torch.Tensor,
    k: int = 5,
) -> float:
    """
    Compute NDCG@k.

    Args:
        scores: (num_queries, num_docs) score matrix.
        relevance: (num_queries, num_docs) binary relevance matrix.
        k: cutoff.

    Returns:
        Mean NDCG@k across queries.
    """
    num_queries = scores.size(0)
    ndcg_values = []

    for q in range(num_queries):
        # Rank documents by score
        _, ranked_indices = scores[q].sort(descending=True)
        ranked_rel = relevance[q][ranked_indices[:k]]

        # DCG
        positions = torch.arange(1, k + 1, dtype=torch.float32)
        dcg = (ranked_rel.float() / torch.log2(positions + 1)).sum().item()

        # Ideal DCG
        ideal_rel, _ = relevance[q].float().sort(descending=True)
        ideal_rel = ideal_rel[:k]
        idcg = (ideal_rel / torch.log2(positions[: len(ideal_rel)] + 1)).sum().item()

        ndcg_values.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcg_values))


def compute_recall_at_k(
    scores: torch.Tensor,
    relevance: torch.Tensor,
    k: int = 5,
) -> float:
    """
    Compute Recall@k.

    Args:
        scores: (num_queries, num_docs) score matrix.
        relevance: (num_queries, num_docs) binary relevance matrix.
        k: cutoff.

    Returns:
        Mean Recall@k across queries.
    """
    num_queries = scores.size(0)
    recall_values = []

    for q in range(num_queries):
        _, ranked_indices = scores[q].sort(descending=True)
        top_k_rel = relevance[q][ranked_indices[:k]]
        total_rel = relevance[q].sum().item()
        recall_values.append(top_k_rel.sum().item() / total_rel if total_rel > 0 else 0.0)

    return float(np.mean(recall_values))


def embed_documents(
    model,
    processor,
    images: List[Image.Image],
    batch_size: int = 4,
    device: torch.device = torch.device("cpu"),
) -> List[torch.Tensor]:
    """Embed all document images, returning list of 2D tensors."""
    all_embeddings = []
    model = model.to(device)

    for i in tqdm(range(0, len(images), batch_size), desc="Embedding documents"):
        batch_imgs = images[i : i + batch_size]
        batch = processor.process_images(batch_imgs).to(device)
        with torch.no_grad():
            embs = model(**batch)  # (B, tokens, dim)
        # Unbind to list of 2D tensors
        for j in range(embs.size(0)):
            # Remove padding (zero vectors)
            mask = embs[j].abs().sum(dim=-1) > 0
            all_embeddings.append(embs[j][mask].cpu())

    return all_embeddings


def embed_queries(
    model,
    processor,
    queries: List[str],
    batch_size: int = 8,
    device: torch.device = torch.device("cpu"),
) -> List[torch.Tensor]:
    """Embed all queries, returning list of 2D tensors."""
    all_embeddings = []
    model = model.to(device)

    for i in tqdm(range(0, len(queries), batch_size), desc="Embedding queries"):
        batch_texts = queries[i : i + batch_size]
        batch = processor.process_queries(batch_texts).to(device)
        with torch.no_grad():
            embs = model(**batch)  # (B, tokens, dim)
        for j in range(embs.size(0)):
            mask = embs[j].abs().sum(dim=-1) > 0
            all_embeddings.append(embs[j][mask].cpu())

    return all_embeddings


def apply_pooling(
    doc_embeddings: List[torch.Tensor],
    method: str,
    pool_factor: int,
    importance_scores: Optional[List[torch.Tensor]] = None,
) -> List[torch.Tensor]:
    """
    Apply a pooling strategy to document embeddings.

    Args:
        doc_embeddings: List of (tokens, dim) tensors.
        method: One of "none", "random", "hierarchical", "topk", "weighted_hierarchical",
                "protect_and_pool", "adaptive".
        pool_factor: Compression factor.
        importance_scores: Required for importance-guided methods.

    Returns:
        List of pooled (compressed_tokens, dim) tensors.
    """
    if method == "none" or pool_factor == 1:
        return doc_embeddings

    if method == "random":
        pooled = []
        for emb in doc_embeddings:
            k = max(emb.size(0) // pool_factor, 1)
            indices = torch.randperm(emb.size(0))[:k].sort().values
            pooled.append(emb[indices])
        return pooled

    if method == "hierarchical":
        pooler = HierarchicalTokenPooler()
        result = pooler.pool_embeddings(doc_embeddings, pool_factor=pool_factor)
        return result  # type: ignore

    # All importance-guided methods need scores
    if importance_scores is None:
        raise ValueError(f"Method '{method}' requires importance_scores.")

    if method == "topk":
        pooler_topk = TopKTokenPooler(importance_scores=importance_scores)
        return pooler_topk.pool_embeddings(doc_embeddings, pool_factor=pool_factor)  # type: ignore

    if method == "weighted_hierarchical":
        pooler_wh = ImportanceWeightedHierarchicalTokenPooler(importance_scores=importance_scores)
        return pooler_wh.pool_embeddings(doc_embeddings, pool_factor=pool_factor)  # type: ignore

    if method == "protect_and_pool":
        pooler_pp = ProtectAndPoolTokenPooler(protect_fraction=0.25, importance_scores=importance_scores)
        return pooler_pp.pool_embeddings(doc_embeddings, pool_factor=pool_factor)  # type: ignore

    if method == "adaptive":
        pooler_adaptive = AdaptivePoolFactorTokenPooler(
            min_pool_factor=max(pool_factor // 2, 1),
            max_pool_factor=pool_factor * 2,
            importance_scores=importance_scores,
        )
        return pooler_adaptive.pool_embeddings(doc_embeddings)  # type: ignore

    raise ValueError(f"Unknown pooling method: {method}")


def run_evaluation(
    model_name: str,
    dataset_name: str,
    pool_factors: List[int],
    output_dir: str,
    batch_size: int = 4,
    max_docs: Optional[int] = None,
    device: str = "cuda",
):
    """Main evaluation loop."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Detect device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"
    dev = torch.device(device)

    # Load model
    print(f"Loading model: {model_name}")
    from colpali_engine.models import ColModernVBert, ColModernVBertProcessor

    processor = ColModernVBertProcessor.from_pretrained(model_name)
    model = ColModernVBert.from_pretrained(model_name, torch_dtype=torch.float16).to(dev)
    model.eval()

    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="test")
    if max_docs is not None:
        dataset = dataset.select(range(min(max_docs, len(dataset))))

    images: List[Image.Image] = [sample["image"] for sample in dataset]
    queries: List[str] = [sample["query"] for sample in dataset]

    # Build relevance matrix (for ViDoRe format, each query's relevant doc is at same index)
    num_samples = len(queries)
    relevance = torch.zeros(num_samples, num_samples)
    for i in range(num_samples):
        relevance[i, i] = 1.0

    print(f"Dataset: {num_samples} query-document pairs")

    # Embed all documents and queries
    print("\n--- Embedding Phase ---")
    doc_embeddings = embed_documents(model, processor, images, batch_size=batch_size, device=dev)
    query_embeddings = embed_queries(model, processor, queries, batch_size=batch_size * 2, device=dev)

    avg_tokens_per_doc = np.mean([e.size(0) for e in doc_embeddings])
    print(f"Average tokens per document: {avg_tokens_per_doc:.1f}")
    print(f"Embedding dim: {doc_embeddings[0].size(1)}")

    # Compute importance scores with all estimation methods
    print("\n--- Importance Estimation Phase ---")
    estimators = {
        "self_similarity": SelfSimilarityImportanceEstimator(),
        "centroid_distance": CentroidDistanceImportanceEstimator(),
    }
    importance_cache: Dict[str, List[torch.Tensor]] = {}

    for est_name, estimator in estimators.items():
        print(f"  Computing {est_name} importance...")
        t0 = time.time()
        output = estimator.estimate(doc_embeddings)
        elapsed = time.time() - t0
        importance_cache[est_name] = output.scores  # type: ignore
        print(f"    Done in {elapsed:.2f}s")

    # Define all experiment configurations
    methods = [
        ("none", None),
        ("random", None),
        ("hierarchical", None),
        ("topk", "self_similarity"),
        ("topk", "centroid_distance"),
        ("weighted_hierarchical", "self_similarity"),
        ("weighted_hierarchical", "centroid_distance"),
        ("protect_and_pool", "self_similarity"),
        ("protect_and_pool", "centroid_distance"),
        ("adaptive", "self_similarity"),
        ("adaptive", "centroid_distance"),
    ]

    results = []

    # Evaluate each configuration
    print("\n--- Evaluation Phase ---")
    for pool_factor in pool_factors:
        for method_name, importance_method in methods:
            if method_name == "none" and pool_factor > 1:
                continue  # Only run "none" once
            if method_name != "none" and pool_factor == 1:
                continue  # All methods are identical at pool_factor=1

            label = method_name
            if importance_method:
                label = f"{method_name}_{importance_method}"

            print(f"\n  [{label}] pool_factor={pool_factor}")

            # Get importance scores if needed
            scores = importance_cache.get(importance_method) if importance_method else None

            # Apply pooling
            t0 = time.time()
            pooled_docs = apply_pooling(doc_embeddings, method_name, pool_factor, scores)
            pool_time = time.time() - t0

            # Compute average compressed tokens
            avg_pooled_tokens = np.mean([e.size(0) for e in pooled_docs])
            compression_ratio = 1.0 - (avg_pooled_tokens / avg_tokens_per_doc)

            # Compute retrieval scores
            t0 = time.time()
            score_matrix = compute_maxsim_scores(query_embeddings, pooled_docs)
            score_time = time.time() - t0

            ndcg5 = compute_ndcg_at_k(score_matrix, relevance, k=5)
            recall5 = compute_recall_at_k(score_matrix, relevance, k=5)

            result = {
                "method": method_name,
                "importance_method": importance_method or "n/a",
                "pool_factor": pool_factor,
                "ndcg@5": ndcg5,
                "recall@5": recall5,
                "avg_tokens": avg_pooled_tokens,
                "compression_ratio": compression_ratio,
                "pool_time_s": pool_time,
                "score_time_s": score_time,
            }
            results.append(result)
            print(
                f"    NDCG@5={ndcg5:.4f}  Recall@5={recall5:.4f}  "
                f"tokens={avg_pooled_tokens:.0f}  compression={compression_ratio:.1%}  "
                f"pool_time={pool_time:.2f}s  score_time={score_time:.2f}s"
            )

    # Save results
    df = pd.DataFrame(results)
    csv_path = output_path / "pooling_evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVED] Results CSV: {csv_path}")

    # Also save as JSON for easier programmatic access
    json_path = output_path / "pooling_evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVED] Results JSON: {json_path}")

    # Generate summary plot
    try:
        _plot_results(df, output_path)
    except Exception as e:
        print(f"[WARN] Could not generate plot: {e}")

    return df


def _plot_results(df: pd.DataFrame, output_path: Path):
    """Generate NDCG@5 vs compression ratio plot."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Group by method+importance combination
    df["label"] = df.apply(
        lambda r: f"{r['method']}" if r["importance_method"] == "n/a" else f"{r['method']}_{r['importance_method']}",
        axis=1,
    )

    markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "p", "<"]
    for idx, label in enumerate(df["label"].unique()):
        subset = df[df["label"] == label]
        if label == "none":
            ax.axhline(y=subset["ndcg@5"].iloc[0], color="black", linestyle="--", alpha=0.5, label="no pooling")
            continue
        marker = markers[idx % len(markers)]
        ax.plot(
            subset["compression_ratio"],
            subset["ndcg@5"],
            marker=marker,
            label=label,
            linewidth=1.5,
            markersize=7,
        )

    ax.set_xlabel("Compression Ratio", fontsize=12)
    ax.set_ylabel("NDCG@5", fontsize=12)
    ax.set_title("Retrieval Quality vs Compression", fontsize=14)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = output_path / "ndcg_vs_compression.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate importance-guided token pooling for document retrieval")
    parser.add_argument("--model", type=str, default="ModernVBERT/colmodernvbert", help="HuggingFace model name")
    parser.add_argument(
        "--dataset", type=str, default="vidore/docvqa_test_subsampled", help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--pool-factors", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8], help="Pool factors to evaluate"
    )
    parser.add_argument("--output-dir", type=str, default="experiments/results", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for embedding")
    parser.add_argument("--max-docs", type=int, default=None, help="Max documents to evaluate (for quick testing)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    run_evaluation(
        model_name=args.model,
        dataset_name=args.dataset,
        pool_factors=args.pool_factors,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_docs=args.max_docs,
        device=args.device,
    )


if __name__ == "__main__":
    main()
