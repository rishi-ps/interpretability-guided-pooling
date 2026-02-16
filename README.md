# Interpretability-Guided Token Pooling for Visual Document Retrieval

Importance-aware compression strategies for multi-vector vision-language retrieval models (ColPali family). This project introduces query-agnostic importance estimation methods and pooling strategies that use interpretability signals to improve token compression quality.

## Overview

Late-interaction models like ColPali encode document pages as 1000+ patch embeddings per page. This project provides:

- **3 importance estimation methods** — identify which patches carry the most semantic information (self-similarity, centroid distance, probe-based)
- **4 importance-guided pooling strategies** — use importance scores to make smarter compression decisions (top-k, weighted hierarchical, protect-and-pool, adaptive)

Key finding: importance-weighted hierarchical pooling at 75% compression **outperforms the uncompressed baseline** (NDCG@5 = 0.679 vs 0.660), acting as a form of noise reduction.

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install with all dependencies (evaluation + dev)
pip install -e ".[dev]"
```

Requires Python 3.10+ and a CUDA-capable GPU (tested on RTX 3070, 8GB VRAM).

## Project Structure

```
src/
    importance_estimation.py          # 3 importance estimators
    importance_guided_pooling.py      # 4 pooling strategies (extend ColPali's BaseTokenPooler)

experiments/
    evaluate_pooling.py               # Full evaluation pipeline (NDCG@5, Recall@5)
    visualize_importance.py           # Importance heatmap generation
    results/                          # Generated results (CSV, JSON, plots, heatmaps)

tests/
    test_importance_estimation.py     # 20 tests
    test_importance_guided_pooling.py # 22 tests
```

## Quick Start

```python
import torch
from src.importance_estimation import CentroidDistanceImportanceEstimator
from src.importance_guided_pooling import ImportanceWeightedHierarchicalTokenPooler

# Assume doc_embeddings is a list of (tokens, 128) tensors from a ColPali model
estimator = CentroidDistanceImportanceEstimator()
importance = estimator.estimate(doc_embeddings)

pooler = ImportanceWeightedHierarchicalTokenPooler(importance_scores=importance.scores)
compressed = pooler.pool_embeddings(doc_embeddings, pool_factor=4)  # 75% compression
```

## Running Experiments

```bash
# Full evaluation (requires GPU + model download)
python experiments/evaluate_pooling.py \
    --model ModernVBERT/colmodernvbert \
    --dataset vidore/docvqa_test_subsampled \
    --pool-factors 1 2 3 4 6 8 \
    --batch-size 4 --device cuda

# Visualization (importance heatmaps)
python experiments/visualize_importance.py \
    --model ModernVBERT/colmodernvbert \
    --num-docs 5 --device cuda

# Quick demo (no GPU needed)
python experiments/visualize_importance.py --demo
```

## Running Tests

```bash
pytest tests/ -v
```

## Dependencies

This project builds on top of [ColPali Engine](https://github.com/illuin-tech/colpali) (installed as a pip dependency). The pooling strategies extend ColPali's `BaseTokenPooler` interface and are compatible with any ColPali-family model.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{interpretability-guided-pooling-2026,
    title={Interpretability-Guided Token Pooling for Visual Document Retrieval},
    year={2026},
}
```
