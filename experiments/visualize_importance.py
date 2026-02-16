#!/usr/bin/env python3
"""
Visualize importance maps and compare pooling strategies on sample documents.

Generates:
1. Per-document importance heatmaps from all estimation methods
2. Side-by-side comparisons (self-similarity vs centroid distance vs probe)
3. Before/after pooling similarity maps showing preserved regions

Usage:
    python experiments/visualize_importance.py \
        --model ModernVBERT/colmodernvbert \
        --output-dir experiments/results/visualizations

    # Quick test with a blank image (no model download needed for layout checks):
    python experiments/visualize_importance.py --demo
"""

import argparse
import uuid
from pathlib import Path
from typing import Any, List, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from einops import rearrange
from PIL import Image

# ColPali imports (installed via pip)
from colpali_engine.interpretability.similarity_map_utils import normalize_similarity_map

# Our imports
from src.importance_estimation import (
    CentroidDistanceImportanceEstimator,
    SelfSimilarityImportanceEstimator,
    get_importance_map,
)


def plot_importance_map(
    image: Image.Image,
    importance_map: torch.Tensor,
    title: str = "",
    figsize: Tuple[int, int] = (8, 8),
    show_colorbar: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Overlay importance heatmap on a document image.

    Args:
        image: Original document image.
        importance_map: 2D tensor (n_patches_x, n_patches_y), values in [0, 1].
        title: Plot title.
        figsize: Figure size.
        show_colorbar: Whether to show colorbar.

    Returns:
        (fig, ax) tuple.
    """
    img_array = np.array(image.convert("RGBA"))

    # Normalize and reshape for display
    imp_normalized = normalize_similarity_map(importance_map).to(torch.float32).cpu().numpy()
    imp_display = rearrange(imp_normalized, "h w -> w h")

    imp_image = Image.fromarray((imp_display * 255).astype("uint8")).resize(image.size, Image.Resampling.BICUBIC)

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img_array)
        im = ax.imshow(
            imp_image,
            cmap=sns.color_palette("mako", as_cmap=True),
            alpha=0.5,
        )
        if show_colorbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if title:
            ax.set_title(title, fontsize=14)
        ax.set_axis_off()
        fig.tight_layout()

    return fig, ax


def plot_importance_comparison(
    image: Image.Image,
    importance_maps: dict,
    figsize: Tuple[int, int] = (24, 8),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Side-by-side comparison of importance maps from different estimation methods.

    Args:
        image: Original document image.
        importance_maps: Dict mapping method name → 2D importance tensor.
        figsize: Figure size.

    Returns:
        (fig, axes) tuple.
    """
    n_methods = len(importance_maps)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=figsize)

    # Original image
    axes[0].imshow(np.array(image.convert("RGB")))
    axes[0].set_title("Original Document", fontsize=12)
    axes[0].set_axis_off()

    img_array = np.array(image.convert("RGBA"))

    for idx, (method_name, imp_map) in enumerate(importance_maps.items()):
        ax = axes[idx + 1]
        imp_normalized = normalize_similarity_map(imp_map).to(torch.float32).cpu().numpy()
        imp_display = rearrange(imp_normalized, "h w -> w h")
        imp_image = Image.fromarray((imp_display * 255).astype("uint8")).resize(image.size, Image.Resampling.BICUBIC)

        ax.imshow(img_array)
        ax.imshow(
            imp_image,
            cmap=sns.color_palette("mako", as_cmap=True),
            alpha=0.5,
        )
        ax.set_title(method_name.replace("_", " ").title(), fontsize=12)
        ax.set_axis_off()

    fig.suptitle("Importance Map Comparison", fontsize=16, y=1.02)
    fig.tight_layout()
    return fig, axes


def visualize_single_document(
    model,
    processor,
    image: Image.Image,
    query: str,
    output_dir: Path,
    doc_idx: int = 0,
    device: torch.device = torch.device("cpu"),
):
    """Generate all visualizations for a single document."""
    print(f"\n--- Document {doc_idx} ---")
    print(f"Query: '{query}'")

    # Process image
    batch_images = processor.process_images([image]).to(device)
    batch_queries = processor.process_queries([query]).to(device)

    with torch.no_grad():
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)

    # Get image mask and extract image-only embeddings
    image_mask = processor.get_local_image_mask(cast(Any, batch_images))
    masked_embs = image_embeddings[0][image_mask[0]]  # (n_image_tokens, dim)

    # Get patch grid dimensions
    n_patches = processor.get_n_patches((image.size[1], image.size[0]))
    print(f"Patch grid: {n_patches[0]} x {n_patches[1]} = {n_patches[0] * n_patches[1]} patches")
    print(f"Image tokens: {masked_embs.size(0)}")

    # Compute importance scores
    estimators = {
        "self_similarity": SelfSimilarityImportanceEstimator(),
        "centroid_distance": CentroidDistanceImportanceEstimator(),
    }

    importance_maps = {}
    for est_name, estimator in estimators.items():
        output = estimator.estimate([masked_embs.cpu()])
        scores = output.scores[0]  # (n_image_tokens,)
        imp_map = get_importance_map(scores, n_patches)
        importance_maps[est_name] = imp_map
        print(f"  {est_name}: min={scores.min():.3f}, max={scores.max():.3f}, mean={scores.mean():.3f}")

    # Save individual importance maps
    doc_dir = output_dir / f"doc_{doc_idx:04d}"
    doc_dir.mkdir(parents=True, exist_ok=True)

    # Save original image
    image.save(doc_dir / "original.png")

    for method_name, imp_map in importance_maps.items():
        fig, ax = plot_importance_map(image, imp_map, title=f"{method_name.replace('_', ' ').title()} Importance")
        fig.savefig(doc_dir / f"importance_{method_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: importance_{method_name}.png")

    # Save comparison plot
    fig_cmp, _ = plot_importance_comparison(image, importance_maps)
    fig_cmp.savefig(doc_dir / "importance_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig_cmp)
    print(f"  Saved: importance_comparison.png")

    # Generate similarity maps for the query
    sim_maps = processor.get_similarity_maps_from_embeddings(
        image_embeddings=image_embeddings,
        query_embeddings=query_embeddings,
        n_patches=n_patches,
        image_mask=image_mask,
    )
    sim_map = sim_maps[0]  # (query_tokens, n_patches_x, n_patches_y)

    # Aggregate similarity (max across query tokens) for comparison
    max_sim_map = sim_map.max(dim=0).values  # (n_patches_x, n_patches_y)
    fig_sim, ax_sim = plot_importance_map(image, max_sim_map, title=f"Query MaxSim: '{query[:50]}...'")
    fig_sim.savefig(doc_dir / "query_maxsim_aggregate.png", dpi=150, bbox_inches="tight")
    plt.close(fig_sim)
    print(f"  Saved: query_maxsim_aggregate.png")

    # Correlation analysis: importance vs query relevance
    print("\n  Correlation (importance vs query MaxSim):")
    flat_max_sim = max_sim_map.flatten().float().cpu()
    for method_name, imp_map in importance_maps.items():
        flat_imp = imp_map.flatten().float().cpu()
        correlation = torch.corrcoef(torch.stack([flat_imp, flat_max_sim]))[0, 1].item()
        print(f"    {method_name}: r = {correlation:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize importance maps for document retrieval")
    parser.add_argument("--model", type=str, default="ModernVBERT/colmodernvbert", help="HuggingFace model name")
    parser.add_argument(
        "--dataset", type=str, default="vidore/docvqa_test_subsampled", help="HuggingFace dataset"
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments/results/visualizations", help="Output directory"
    )
    parser.add_argument("--num-docs", type=int, default=5, help="Number of documents to visualize")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--demo", action="store_true", help="Run with a blank demo image (no model needed)")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.demo:
        print("=== Demo Mode (blank image, random embeddings) ===")
        # Create a simple demo without needing the model
        image = Image.new("RGB", (800, 600), color="white")
        n_patches = (16, 12)  # fake grid
        fake_emb = torch.randn(n_patches[0] * n_patches[1], 128)
        fake_emb = torch.nn.functional.normalize(fake_emb, p=2, dim=-1)

        estimators = {
            "self_similarity": SelfSimilarityImportanceEstimator(),
            "centroid_distance": CentroidDistanceImportanceEstimator(),
        }
        importance_maps = {}
        for name, est in estimators.items():
            out = est.estimate([fake_emb])
            scores = out.scores[0]
            importance_maps[name] = get_importance_map(scores, n_patches)

        fig, _ = plot_importance_comparison(image, importance_maps)
        save_path = output_path / "demo_comparison.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved demo to: {save_path}")
        return

    # Full mode with model
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    print(f"Loading model: {args.model}")
    from colpali_engine.models import ColModernVBert, ColModernVBertProcessor

    processor = ColModernVBertProcessor.from_pretrained(args.model)
    model = ColModernVBert.from_pretrained(args.model, torch_dtype=torch.float16).to(device)
    model.eval()

    print(f"Loading dataset: {args.dataset}")
    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split="test")
    num_samples = min(args.num_docs, len(dataset))

    for i in range(num_samples):
        sample = dataset[i]
        visualize_single_document(
            model=model,
            processor=processor,
            image=sample["image"],
            query=sample["query"],
            output_dir=output_path,
            doc_idx=i,
            device=device,
        )

    print(f"\n[DONE] All visualizations saved to: {output_path}")


if __name__ == "__main__":
    main()
