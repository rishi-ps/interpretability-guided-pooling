#!/usr/bin/env python3
"""
Evaluate retrieval quality with different pooling strategies and compression ratios.

Enhanced version supporting:
- Model-agnostic loading (ColModernVBERT, ColIdefics3, ColQwen2, etc.)
- Multiple ViDoRe datasets in a single run
- Probe-based importance estimation
- Multi-query-per-document datasets (e.g., TabFQuAD)
- Random baseline with multiple seeds for statistical robustness
- Per-dataset and aggregate results

Usage:
    # Single model, multiple datasets
    python experiments/evaluate_pooling.py \
        --model ModernVBERT/colmodernvbert \
        --datasets vidore/docvqa_test_subsampled vidore/infovqa_test_subsampled \
        --pool-factors 1 2 3 4 6 8 \
        --output-dir experiments/results

    # Second model
    python experiments/evaluate_pooling.py \
        --model vidore/colSmolVLM-256M-base \
        --datasets vidore/docvqa_test_subsampled \
        --pool-factors 1 2 4 8

    # Full benchmark run
    python experiments/evaluate_pooling.py \
        --model ModernVBERT/colmodernvbert \
        --datasets vidore/docvqa_test_subsampled vidore/infovqa_test_subsampled \
                   vidore/tabfquad_test_subsampled vidore/arxivqa_test_subsampled \
        --pool-factors 1 2 3 4 6 8 \
        --random-seeds 42 123 456 \
        --include-probe

Requirements:
    pip install -e ".[eval]"
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type, cast

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# ColPali imports
from colpali_engine.compression import HierarchicalTokenPooler

# Our imports
from src.importance_estimation import (
    CentroidDistanceImportanceEstimator,
    ProbeImportanceEstimator,
    SelfSimilarityImportanceEstimator,
)
from src.importance_guided_pooling import (
    AdaptivePoolFactorTokenPooler,
    ImportanceWeightedDistancePooler,
    ImportanceWeightedHierarchicalTokenPooler,
    ProtectAndPoolTokenPooler,
    SplitAndAllocateTokenPooler,
    TopKTokenPooler,
)


# ---------------------------------------------------------------------------
# Progress tracking & early stopping
# ---------------------------------------------------------------------------


class ProgressTracker:
    """
    Tracks evaluation progress, estimates remaining time, and writes a live
    status file that can be monitored from another terminal.

    Usage from another terminal:
        watch -n5 cat experiments/results/status.txt
        # or: tail -f experiments/results/eval.log
    """

    def __init__(self, total_configs: int, total_datasets: int, status_file: Path, log_file: Path):
        self.total_configs = total_configs
        self.total_datasets = total_datasets
        self.completed_configs = 0
        self.completed_datasets = 0
        self.current_dataset = ""
        self.current_method = ""
        self.start_time = time.time()
        self.config_times: List[float] = []  # time per config
        self.status_file = status_file
        self.log_file = log_file
        self.baseline_ndcg: Dict[str, float] = {}  # dataset -> baseline NDCG@5
        self.best_results: Dict[str, Dict[str, float]] = {}  # dataset -> {method: ndcg}
        self._last_config_start = time.time()

        # Initialize log file
        with open(self.log_file, "w") as f:
            f.write(f"[{self._now()}] Evaluation started\n")
        self._write_status()

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h{m:02d}m"

    def start_config(self, dataset: str, method: str, pool_factor: int):
        self.current_dataset = dataset
        self.current_method = f"{method} PF={pool_factor}"
        self._last_config_start = time.time()
        self._write_status()

    def finish_config(self, ndcg: float, method_label: str):
        elapsed = time.time() - self._last_config_start
        self.config_times.append(elapsed)
        self.completed_configs += 1

        # Track best results per dataset
        ds = self.current_dataset
        if ds not in self.best_results:
            self.best_results[ds] = {}
        self.best_results[ds][method_label] = ndcg

        self._write_status()

    def finish_dataset(self, dataset: str):
        self.completed_datasets += 1
        self.log(f"Finished dataset: {dataset} ({self.completed_datasets}/{self.total_datasets})")
        self._write_status()

    def set_baseline(self, dataset: str, ndcg: float):
        self.baseline_ndcg[dataset] = ndcg
        self.log(f"Baseline NDCG@5 for {dataset}: {ndcg:.4f}")

    def log(self, msg: str):
        line = f"[{self._now()}] {msg}"
        print(line)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")

    def get_eta(self) -> str:
        if self.completed_configs == 0 or len(self.config_times) == 0:
            return "estimating..."
        avg_time = np.mean(self.config_times)
        if np.isnan(avg_time) or np.isinf(avg_time):
            return "estimating..."
        remaining = self.total_configs - self.completed_configs
        eta_seconds = avg_time * remaining
        finish_time = datetime.now() + timedelta(seconds=int(eta_seconds))
        return f"{self._fmt_duration(eta_seconds)} (finish ~{finish_time.strftime('%H:%M')})"

    def _write_status(self):
        elapsed = time.time() - self.start_time
        pct = (self.completed_configs / self.total_configs * 100) if self.total_configs > 0 else 0

        lines = [
            f"=== Evaluation Status ({self._now()}) ===",
            f"Elapsed: {self._fmt_duration(elapsed)}",
            f"Progress: {self.completed_configs}/{self.total_configs} configs ({pct:.1f}%)",
            f"Datasets: {self.completed_datasets}/{self.total_datasets}",
            f"ETA: {self.get_eta()}",
            f"Currently: {self.current_dataset} / {self.current_method}",
            "",
        ]

        # Show baselines
        if self.baseline_ndcg:
            lines.append("--- Baselines (no pooling) ---")
            for ds, ndcg in self.baseline_ndcg.items():
                lines.append(f"  {ds}: NDCG@5={ndcg:.4f}")
            lines.append("")

        # Show progress bar
        bar_width = 40
        filled = int(bar_width * pct / 100)
        bar = "#" * filled + "-" * (bar_width - filled)
        lines.append(f"[{bar}] {pct:.1f}%")

        try:
            with open(self.status_file, "w") as f:
                f.write("\n".join(lines) + "\n")
        except Exception:
            pass  # Don't crash on status write failure


class EarlyStopError(Exception):
    """Raised when early stopping is triggered due to catastrophically bad results."""
    pass


def check_early_stop(
    baseline_ndcg: float,
    current_ndcg: float,
    method_label: str,
    pool_factor: int,
    threshold: float,
    tracker: Optional["ProgressTracker"] = None,
) -> bool:
    """
    Check if results are catastrophically bad and we should abort.

    Returns True if we should stop. The logic:
    - If baseline NDCG@5 < 0.05, embeddings are probably broken
    - If a method's NDCG@5 drops below (baseline * threshold) at moderate
      compression (PF <= 4), that's a red flag but we just warn
    - We never auto-abort on individual method results (some methods like topk
      are expected to degrade) — only on baseline issues
    """
    if baseline_ndcg < 0.05:
        msg = (
            f"EARLY STOP: Baseline NDCG@5={baseline_ndcg:.4f} is near-zero. "
            f"Model embeddings may be broken. Aborting to save time."
        )
        if tracker:
            tracker.log(msg)
        else:
            print(msg)
        return True

    if pool_factor <= 4 and current_ndcg < baseline_ndcg * threshold:
        # Warn but don't abort — individual methods can be bad
        msg = (
            f"  WARNING: [{method_label}] PF={pool_factor} NDCG@5={current_ndcg:.4f} "
            f"is below {threshold:.0%} of baseline ({baseline_ndcg:.4f}). "
            f"This method may not be working well on this dataset."
        )
        if tracker:
            tracker.log(msg)
        else:
            print(msg)

    return False


# ---------------------------------------------------------------------------
# Model registry — maps model name patterns to (model_class, processor_class)
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Tuple[str, str]] = {
    # (model_class_name, processor_class_name) — imported from colpali_engine.models
    "colmodernvbert": ("ColModernVBert", "ColModernVBertProcessor"),
    "colidefics3": ("ColIdefics3", "ColIdefics3Processor"),
    "colsmolvlm": ("ColIdefics3", "ColIdefics3Processor"),  # SmolVLM uses Idefics3 architecture
    "colpali": ("ColPali", "ColPaliProcessor"),
    "colqwen2_5": ("ColQwen2_5", "ColQwen2_5_Processor"),
    "colqwen2": ("ColQwen2", "ColQwen2Processor"),
    "colqwen3": ("ColQwen3", "ColQwen3Processor"),
    "colgemma3": ("ColGemma3", "ColGemmaProcessor3"),
}


def detect_model_classes(model_name: str) -> Tuple[Any, Any]:
    """
    Auto-detect model and processor classes from the model name.

    Matches against known patterns in the model name string.
    Falls back to ColModernVBert if no match is found.

    Returns:
        (model_class, processor_class) tuple.
    """
    import colpali_engine.models as models_module

    model_name_lower = model_name.lower()

    for key, (model_cls_name, proc_cls_name) in MODEL_REGISTRY.items():
        if key in model_name_lower:
            model_cls = getattr(models_module, model_cls_name)
            proc_cls = getattr(models_module, proc_cls_name)
            return model_cls, proc_cls

    # Default fallback
    print(f"[WARN] Could not detect model type from '{model_name}', falling back to ColModernVBert.")
    from colpali_engine.models import ColModernVBert, ColModernVBertProcessor

    return ColModernVBert, ColModernVBertProcessor


def load_model_and_processor(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> Tuple[Any, Any]:
    """Load model and processor, auto-detecting the correct classes."""
    model_cls, proc_cls = detect_model_classes(model_name)
    print(f"  Model class: {model_cls.__name__}")
    print(f"  Processor class: {proc_cls.__name__}")

    processor = proc_cls.from_pretrained(model_name)
    model = model_cls.from_pretrained(model_name, dtype=dtype).to(device)
    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


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
            sim = torch.mm(q_emb, d_emb.t())  # (qt, dt)
            max_sim_per_query_token = sim.max(dim=1).values  # (qt,)
            scores[q_idx, d_idx] = max_sim_per_query_token.sum()

    return scores


def compute_ndcg_at_k(
    scores: torch.Tensor,
    relevance: torch.Tensor,
    k: int = 5,
) -> float:
    """Compute NDCG@k. Relevance can be multi-hot (multiple relevant docs per query)."""
    num_queries = scores.size(0)
    num_docs = scores.size(1)
    effective_k = min(k, num_docs)
    ndcg_values = []

    for q in range(num_queries):
        _, ranked_indices = scores[q].sort(descending=True)
        ranked_rel = relevance[q][ranked_indices[:effective_k]]

        positions = torch.arange(1, effective_k + 1, dtype=torch.float32)
        dcg = (ranked_rel.float() / torch.log2(positions + 1)).sum().item()

        ideal_rel, _ = relevance[q].float().sort(descending=True)
        ideal_rel = ideal_rel[:effective_k]
        idcg = (ideal_rel / torch.log2(positions[: len(ideal_rel)] + 1)).sum().item()

        ndcg_values.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcg_values))


def compute_recall_at_k(
    scores: torch.Tensor,
    relevance: torch.Tensor,
    k: int = 5,
) -> float:
    """Compute Recall@k. Each query may have multiple relevant documents."""
    num_queries = scores.size(0)
    num_docs = scores.size(1)
    effective_k = min(k, num_docs)
    recall_values = []

    for q in range(num_queries):
        _, ranked_indices = scores[q].sort(descending=True)
        top_k_rel = relevance[q][ranked_indices[:effective_k]]
        total_rel = relevance[q].sum().item()
        recall_values.append(top_k_rel.sum().item() / total_rel if total_rel > 0 else 0.0)

    return float(np.mean(recall_values))


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


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
        for j in range(embs.size(0)):
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


# ---------------------------------------------------------------------------
# Dataset loading with proper relevance matrix
# ---------------------------------------------------------------------------


# All 10 ViDoRe v1 benchmark datasets (from ColPali paper, Table 1)
# ---------------------------------------------------------------------------
# GPU thermal management
# ---------------------------------------------------------------------------

def get_gpu_temp() -> Optional[int]:
    """Get current GPU temperature via nvidia-smi. Returns None if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return int(result.stdout.strip())
    except Exception:
        return None


def thermal_cooldown(threshold: int = 72, target: int = 60, mandatory_pause: int = 15, tracker=None):
    """If GPU temp exceeds threshold, sleep until it drops to target. Always pause mandatory_pause seconds."""
    if mandatory_pause > 0:
        time.sleep(mandatory_pause)
    temp = get_gpu_temp()
    if temp is None or temp < threshold:
        return
    msg = f"GPU temp {temp}°C exceeds {threshold}°C, cooling down to {target}°C..."
    if tracker:
        tracker.log(msg)
    else:
        print(msg)
    while True:
        time.sleep(15)
        temp = get_gpu_temp()
        if temp is None or temp <= target:
            cool_msg = f"GPU cooled to {temp}°C, resuming."
            if tracker:
                tracker.log(cool_msg)
            else:
                print(cool_msg)
            break


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_completed_configs(output_path: Path, model_short: str) -> Set[Tuple[str, str, str, int]]:
    """Load already-completed (dataset, method, importance_method, pool_factor) tuples from CSV."""
    csv_path = output_path / f"results_{model_short}_all.csv"
    completed = set()
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                imp = str(row["importance_method"])
                if imp == "nan" or imp == "":
                    imp = "n/a"
                completed.add((
                    str(row["dataset"]),
                    str(row["method"]),
                    imp,
                    int(row["pool_factor"]),
                ))
        except Exception:
            pass
    return completed


def load_existing_results(output_path: Path, model_short: str) -> List[Dict[str, Any]]:
    """Load existing result rows from CSV for resume."""
    csv_path = output_path / f"results_{model_short}_all.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            return df.to_dict("records")
        except Exception:
            pass
    return []


VIDORE_V1_DATASETS = [
    # Academic Tasks
    "vidore/docvqa_test_subsampled",           # 500q / 500d, EN
    "vidore/infovqa_test_subsampled",           # 500q / 500d, EN
    "vidore/tatdqa_test",                       # 1600q / 1600d (277 unique imgs), EN
    "vidore/arxivqa_test_subsampled",           # 500q / 500d, EN
    "vidore/tabfquad_test_subsampled",          # 210q / 210d (70 unique imgs), FR
    # Practical Tasks (100 queries, ~1000 document pages as corpus)
    "vidore/syntheticDocQA_energy_test",        # 100q / 1000d, EN
    "vidore/syntheticDocQA_government_reports_test",  # 100q / 1000d, EN
    "vidore/syntheticDocQA_healthcare_industry_test", # 100q / 1000d, EN
    "vidore/syntheticDocQA_artificial_intelligence_test",  # 100q / 1000d, EN
    "vidore/shiftproject_test",                 # 100q / 1000d, FR
]


def load_vidore_dataset(
    dataset_name: str,
    max_docs: Optional[int] = None,
) -> Tuple[List[Image.Image], List[str], torch.Tensor, List[int]]:
    """
    Load a ViDoRe-format dataset.

    Handles:
    - Multi-query-per-document datasets (e.g., TabFQuAD, TAT-DQA) by
      deduplicating images and building the correct relevance matrix.
    - Practical task datasets (e.g., syntheticDocQA_*) where most rows are
      document-only distractors with null/empty queries.

    Args:
        dataset_name: HuggingFace dataset name (e.g., 'vidore/docvqa_test_subsampled').
        max_docs: Maximum number of unique documents to include (all queries for
                  included documents are kept).

    Returns:
        images: List of unique document images (deduplicated).
        queries: List of query strings (only non-null queries).
        relevance: Binary relevance matrix of shape (num_queries, num_docs).
        query_to_doc: List mapping each query index to its relevant document index.
    """
    print(f"  Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="test")

    # Deduplicate images by image_filename (collect ALL unique docs)
    image_filenames = dataset["image_filename"]
    unique_filenames = []
    filename_to_idx: Dict[str, int] = {}
    images: List[Image.Image] = []

    for i, fn in enumerate(image_filenames):
        if fn not in filename_to_idx:
            filename_to_idx[fn] = len(unique_filenames)
            unique_filenames.append(fn)
            images.append(dataset[i]["image"])

    # Apply max_docs limit on unique documents
    if max_docs is not None and max_docs < len(images):
        images = images[:max_docs]
        kept_filenames = set(unique_filenames[:max_docs])
    else:
        kept_filenames = set(unique_filenames)

    # Build query list and relevance matrix
    # Filter out null/empty queries (practical task datasets have document-only
    # rows that serve as distractors in the retrieval corpus)
    queries: List[str] = []
    query_to_doc: List[int] = []
    skipped_null = 0

    for i in range(len(dataset)):
        fn = image_filenames[i]
        if fn not in kept_filenames:
            continue
        query = dataset[i]["query"]
        if query is None or not query.strip():
            skipped_null += 1
            continue
        queries.append(query)
        query_to_doc.append(filename_to_idx[fn])

    num_queries = len(queries)
    num_docs = len(images)
    relevance = torch.zeros(num_queries, num_docs)
    for q_idx, d_idx in enumerate(query_to_doc):
        relevance[q_idx, d_idx] = 1.0

    print(f"    {num_queries} queries, {num_docs} unique documents")
    if skipped_null > 0:
        print(f"    (skipped {skipped_null} document-only rows with no query)")
    if num_queries != num_docs:
        ratio = num_queries / num_docs if num_docs > 0 else 0
        if ratio > 1:
            print(f"    (multi-query: avg {ratio:.1f} queries per doc)")
        else:
            print(f"    (sparse retrieval: {num_queries} queries over {num_docs} doc corpus)")

    return images, queries, relevance, query_to_doc


# ---------------------------------------------------------------------------
# Pooling application
# ---------------------------------------------------------------------------


def apply_pooling(
    doc_embeddings: List[torch.Tensor],
    method: str,
    pool_factor: int,
    importance_scores: Optional[List[torch.Tensor]] = None,
    random_seed: Optional[int] = None,
) -> List[torch.Tensor]:
    """
    Apply a pooling strategy to document embeddings.

    Args:
        doc_embeddings: List of (tokens, dim) tensors.
        method: One of "none", "random", "hierarchical", "topk", "weighted_hierarchical",
                "protect_and_pool", "adaptive", "split_allocate", "importance_weighted_distance".
        pool_factor: Compression factor.
        importance_scores: Required for importance-guided methods.
        random_seed: Seed for random method (for reproducibility / variance estimation).

    Returns:
        List of pooled (compressed_tokens, dim) tensors.
    """
    if method == "none" or pool_factor == 1:
        return doc_embeddings

    if method == "random":
        if random_seed is not None:
            rng = torch.Generator().manual_seed(random_seed)
        pooled = []
        for emb in doc_embeddings:
            k = max(emb.size(0) // pool_factor, 1)
            if random_seed is not None:
                indices = torch.randperm(emb.size(0), generator=rng)[:k].sort().values
            else:
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

    if method == "split_allocate":
        pooler_sa = SplitAndAllocateTokenPooler(importance_scores=importance_scores)
        return pooler_sa.pool_embeddings(doc_embeddings, pool_factor=pool_factor)  # type: ignore

    if method == "importance_weighted_distance":
        pooler_iwd = ImportanceWeightedDistancePooler(importance_scores=importance_scores)
        return pooler_iwd.pool_embeddings(doc_embeddings, pool_factor=pool_factor)  # type: ignore

    raise ValueError(f"Unknown pooling method: {method}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate_single_dataset(
    model,
    processor,
    dataset_name: str,
    pool_factors: List[int],
    include_probe: bool = False,
    random_seeds: Optional[List[int]] = None,
    batch_size: int = 4,
    max_docs: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    tracker: Optional[ProgressTracker] = None,
    early_stop_threshold: float = 0.3,
    allowed_methods: Optional[Set[str]] = None,
    completed_configs: Optional[Set[Tuple[str, str, str, int]]] = None,
    gpu_temp_threshold: int = 78,
) -> List[Dict[str, Any]]:
    """
    Run full evaluation on a single dataset.

    Returns list of result dicts (one per configuration).
    """
    # Load dataset
    images, queries, relevance, query_to_doc = load_vidore_dataset(dataset_name, max_docs)

    # Embed documents and queries
    print("  Embedding documents...")
    doc_embeddings = embed_documents(model, processor, images, batch_size=batch_size, device=device)
    print("  Embedding queries...")
    query_embeddings = embed_queries(model, processor, queries, batch_size=batch_size * 2, device=device)

    avg_tokens_per_doc = np.mean([e.size(0) for e in doc_embeddings])
    print(f"  Avg tokens/doc: {avg_tokens_per_doc:.1f}, embedding dim: {doc_embeddings[0].size(1)}")

    # Compute importance scores
    print("  Computing importance scores...")
    estimators: Dict[str, Any] = {
        "self_similarity": SelfSimilarityImportanceEstimator(),
        "centroid_distance": CentroidDistanceImportanceEstimator(),
    }

    if include_probe:
        print("    Initializing probe-based estimator (encoding probes through model)...")
        probe_estimator = ProbeImportanceEstimator.from_processor_and_model(
            model=model,
            processor=processor,
            device=device,
        )
        estimators["probe"] = probe_estimator

    importance_cache: Dict[str, List[torch.Tensor]] = {}
    for est_name, estimator in estimators.items():
        t0 = time.time()
        output = estimator.estimate(doc_embeddings)
        elapsed = time.time() - t0
        importance_cache[est_name] = output.scores  # type: ignore
        print(f"    {est_name}: {elapsed:.2f}s")

    # Define experiment configurations
    importance_methods = list(importance_cache.keys())

    # All importance-guided method names
    ALL_IMPORTANCE_METHODS = [
        "topk", "weighted_hierarchical", "protect_and_pool",
        "adaptive", "split_allocate", "importance_weighted_distance",
    ]
    # Filter by allowed methods if specified
    active_importance_methods = [
        m for m in ALL_IMPORTANCE_METHODS
        if allowed_methods is None or m in allowed_methods
    ]

    methods = [
        ("none", None),
        ("random", None),
        ("hierarchical", None),
    ]
    for imp_method in importance_methods:
        methods.extend([
            (m, imp_method) for m in active_importance_methods
        ])

    # Determine random seeds
    if random_seeds is None:
        random_seeds_for_random = [None]  # Single run, no fixed seed
    else:
        random_seeds_for_random = random_seeds

    results: List[Dict[str, Any]] = []
    dataset_short = dataset_name.split("/")[-1]

    # Count total configs for this dataset (for progress tracking)
    n_configs = 0
    for pf in pool_factors:
        for mn, im in methods:
            if mn == "none" and pf > 1:
                continue
            if mn != "none" and pf == 1:
                continue
            if mn == "random":
                n_configs += len(random_seeds_for_random)
            else:
                n_configs += 1

    print(f"  Evaluating {len(pool_factors)} pool factors x {len(methods)} methods ({n_configs} configs)...")

    baseline_ndcg = 0.0

    for pool_factor in pool_factors:
        for method_name, importance_method in methods:
            if method_name == "none" and pool_factor > 1:
                continue
            if method_name != "none" and pool_factor == 1:
                continue

            # Skip already-completed configs (resume support)
            config_key = (dataset_short, method_name, importance_method or "n/a", pool_factor)
            if completed_configs and config_key in completed_configs:
                if tracker:
                    tracker.log(f"    [SKIP] {method_name}{'_' + importance_method if importance_method else ''} PF={pool_factor} (already done)")
                    tracker.completed_configs += 1
                continue

            # Thermal cooldown before each config (mandatory 15s pause + temp check)
            thermal_cooldown(threshold=gpu_temp_threshold, tracker=tracker)

            # Track progress
            if tracker:
                tracker.start_config(dataset_short, method_name + (f"_{importance_method}" if importance_method else ""), pool_factor)

            # Random method: run with each seed
            if method_name == "random":
                seed_results = []
                for seed in random_seeds_for_random:
                    pooled_docs = apply_pooling(
                        doc_embeddings, method_name, pool_factor,
                        random_seed=seed,
                    )
                    score_matrix = compute_maxsim_scores(query_embeddings, pooled_docs)
                    ndcg5 = compute_ndcg_at_k(score_matrix, relevance, k=5)
                    recall5 = compute_recall_at_k(score_matrix, relevance, k=5)
                    avg_pooled = np.mean([e.size(0) for e in pooled_docs])
                    seed_results.append({
                        "ndcg@5": ndcg5,
                        "recall@5": recall5,
                        "avg_tokens": avg_pooled,
                        "seed": seed,
                    })
                    if tracker:
                        tracker.finish_config(ndcg5, f"random_seed{seed}_PF{pool_factor}")

                # Report mean +/- std if multiple seeds
                if len(seed_results) > 1:
                    ndcg_vals = [r["ndcg@5"] for r in seed_results]
                    recall_vals = [r["recall@5"] for r in seed_results]
                    label = f"random (PF={pool_factor})"
                    msg = (
                        f"    {label}: NDCG@5={np.mean(ndcg_vals):.4f}+/-{np.std(ndcg_vals):.4f}  "
                        f"Recall@5={np.mean(recall_vals):.4f}+/-{np.std(recall_vals):.4f}  "
                        f"({len(seed_results)} seeds)"
                    )
                    if tracker:
                        tracker.log(msg)
                    else:
                        print(msg)

                for sr in seed_results:
                    compression = 1.0 - (sr["avg_tokens"] / avg_tokens_per_doc)
                    results.append({
                        "dataset": dataset_short,
                        "method": method_name,
                        "importance_method": "n/a",
                        "pool_factor": pool_factor,
                        "ndcg@5": sr["ndcg@5"],
                        "recall@5": sr["recall@5"],
                        "avg_tokens": sr["avg_tokens"],
                        "compression_ratio": compression,
                        "seed": sr["seed"],
                    })
                continue

            # Non-random methods
            label = method_name
            if importance_method:
                label = f"{method_name}_{importance_method}"

            scores = importance_cache.get(importance_method) if importance_method else None

            t0 = time.time()
            pooled_docs = apply_pooling(doc_embeddings, method_name, pool_factor, scores)
            pool_time = time.time() - t0

            avg_pooled_tokens = np.mean([e.size(0) for e in pooled_docs])
            compression_ratio = 1.0 - (avg_pooled_tokens / avg_tokens_per_doc)

            t0 = time.time()
            score_matrix = compute_maxsim_scores(query_embeddings, pooled_docs)
            score_time = time.time() - t0

            ndcg5 = compute_ndcg_at_k(score_matrix, relevance, k=5)
            recall5 = compute_recall_at_k(score_matrix, relevance, k=5)

            msg = (
                f"    [{label}] PF={pool_factor}: NDCG@5={ndcg5:.4f}  Recall@5={recall5:.4f}  "
                f"tokens={avg_pooled_tokens:.0f}  compression={compression_ratio:.1%}  "
                f"pool={pool_time:.1f}s  score={score_time:.1f}s"
            )
            if tracker:
                tracker.log(msg)
                tracker.finish_config(ndcg5, f"{label}_PF{pool_factor}")
            else:
                print(msg)

            # Track baseline and check early stopping
            if method_name == "none":
                baseline_ndcg = ndcg5
                if tracker:
                    tracker.set_baseline(dataset_short, ndcg5)
                if check_early_stop(ndcg5, ndcg5, "baseline", 1, early_stop_threshold, tracker):
                    raise EarlyStopError(f"Baseline NDCG@5={ndcg5:.4f} is near-zero.")
            elif baseline_ndcg > 0:
                check_early_stop(baseline_ndcg, ndcg5, label, pool_factor, early_stop_threshold, tracker)

            results.append({
                "dataset": dataset_short,
                "method": method_name,
                "importance_method": importance_method or "n/a",
                "pool_factor": pool_factor,
                "ndcg@5": ndcg5,
                "recall@5": recall5,
                "avg_tokens": avg_pooled_tokens,
                "compression_ratio": compression_ratio,
                "pool_time_s": pool_time,
                "score_time_s": score_time,
                "seed": None,
            })

    return results


def run_evaluation(
    model_name: str,
    dataset_names: List[str],
    pool_factors: List[int],
    output_dir: str,
    include_probe: bool = False,
    random_seeds: Optional[List[int]] = None,
    batch_size: int = 4,
    max_docs: Optional[int] = None,
    device: str = "cuda",
    early_stop_threshold: float = 0.3,
    allowed_methods: Optional[Set[str]] = None,
    resume: bool = False,
    gpu_temp_threshold: int = 78,
):
    """Main evaluation loop across all datasets."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Detect device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"
    dev = torch.device(device)

    # Load model
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Datasets: {dataset_names}")
    print(f"Pool factors: {pool_factors}")
    print(f"Include probe: {include_probe}")
    print(f"Random seeds: {random_seeds}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    print(f"\nLoading model: {model_name}")
    model, processor = load_model_and_processor(model_name, dev)

    # Derive a short model identifier for filenames
    model_short = model_name.split("/")[-1].lower().replace("-", "_")

    # Resume support: load already-completed configs
    completed_configs: Set[Tuple[str, str, str, int]] = set()
    if resume:
        completed_configs = load_completed_configs(output_path, model_short)
        all_results = load_existing_results(output_path, model_short)
        if completed_configs:
            print(f"[RESUME] Found {len(completed_configs)} completed configs, skipping them.")
    else:
        all_results = []

    if allowed_methods:
        print(f"[METHODS] Running only: {sorted(allowed_methods)}")

    # Estimate total configs across all datasets for progress tracking
    # (approximate — actual count depends on method/PF filtering)
    n_importance = 2 + (1 if include_probe else 0)  # self_sim + centroid + maybe probe
    n_methods_per_pf = 3 + (6 * n_importance)  # none/random/hier + 6 strategies * n_importance
    n_random_seeds = len(random_seeds) if random_seeds else 1
    # PF=1 only has "none"; other PFs have all methods except "none"
    configs_pf1 = 1  # just "none"
    configs_per_other_pf = (n_random_seeds) + 1 + (6 * n_importance)  # random(seeds) + hier + importance methods
    other_pfs = [pf for pf in pool_factors if pf > 1]
    total_configs_per_ds = configs_pf1 + len(other_pfs) * configs_per_other_pf
    total_configs = total_configs_per_ds * len(dataset_names)

    # Set up progress tracker
    status_file = output_path / "status.txt"
    log_file = output_path / "eval.log"
    tracker = ProgressTracker(
        total_configs=total_configs,
        total_datasets=len(dataset_names),
        status_file=status_file,
        log_file=log_file,
    )
    tracker.log(f"Model: {model_name}")
    tracker.log(f"Datasets: {dataset_names}")
    tracker.log(f"Pool factors: {pool_factors}")
    tracker.log(f"Estimated total configs: {total_configs}")
    tracker.log(f"Monitor progress: watch -n5 cat {status_file}")
    tracker.log(f"Full log: tail -f {log_file}")

    if not resume:
        all_results = []

    for ds_idx, ds_name in enumerate(dataset_names):
        tracker.log(f"\n{'~'*60}")
        tracker.log(f"Dataset {ds_idx + 1}/{len(dataset_names)}: {ds_name}")
        tracker.log(f"{'~'*60}")

        try:
            ds_results = evaluate_single_dataset(
                model=model,
                processor=processor,
                dataset_name=ds_name,
                pool_factors=pool_factors,
                include_probe=include_probe,
                random_seeds=random_seeds,
                batch_size=batch_size,
                max_docs=max_docs,
                device=dev,
                tracker=tracker,
                early_stop_threshold=early_stop_threshold,
                allowed_methods=allowed_methods,
                completed_configs=completed_configs,
                gpu_temp_threshold=gpu_temp_threshold,
            )
        except EarlyStopError as e:
            tracker.log(f"EARLY STOP on {ds_name}: {e}")
            tracker.log("Saving partial results and aborting.")
            break

        tracker.finish_dataset(ds_name)

        # Add model info to all results
        for r in ds_results:
            r["model"] = model_short

        all_results.extend(ds_results)

        # Save per-dataset results incrementally
        ds_short = ds_name.split("/")[-1]
        ds_csv = output_path / f"results_{model_short}_{ds_short}.csv"
        pd.DataFrame(ds_results).to_csv(ds_csv, index=False)
        tracker.log(f"  [SAVED] {ds_csv}")

    # Save combined results
    df = pd.DataFrame(all_results)
    combined_csv = output_path / f"results_{model_short}_all.csv"
    df.to_csv(combined_csv, index=False)
    tracker.log(f"[SAVED] Combined CSV: {combined_csv}")

    combined_json = output_path / f"results_{model_short}_all.json"
    with open(combined_json, "w") as f:
        json.dump(all_results, f, indent=2)
    tracker.log(f"[SAVED] Combined JSON: {combined_json}")

    # Generate plots
    try:
        _plot_results_per_dataset(df, output_path, model_short)
    except Exception as e:
        tracker.log(f"[WARN] Could not generate plots: {e}")

    # Print summary table
    _print_summary(df)

    total_time = time.time() - tracker.start_time
    tracker.log(f"\nDONE. Total time: {tracker._fmt_duration(total_time)}")

    # Final status update
    try:
        with open(status_file, "w") as f:
            f.write(f"=== COMPLETED ({tracker._now()}) ===\n")
            f.write(f"Total time: {tracker._fmt_duration(total_time)}\n")
            f.write(f"Results: {combined_csv}\n")
    except Exception:
        pass

    return df


def _print_summary(df: pd.DataFrame):
    """Print a concise summary of best methods per dataset and compression level."""
    print(f"\n{'='*60}")
    print("SUMMARY -- Best method per dataset x compression")
    print(f"{'='*60}")

    # For random with multiple seeds, average across seeds
    df_summary = df.copy()
    if "seed" in df_summary.columns:
        random_mask = df_summary["method"] == "random"
        if random_mask.any():
            random_avg = (
                df_summary[random_mask]
                .groupby(["dataset", "method", "importance_method", "pool_factor", "model"])
                .agg({
                    "ndcg@5": "mean",
                    "recall@5": "mean",
                    "avg_tokens": "mean",
                    "compression_ratio": "mean",
                })
                .reset_index()
            )
            # Replace random rows with averaged version
            df_nonrandom = df_summary[~random_mask].copy()
            shared_cols = ["dataset", "method", "importance_method", "pool_factor", "model",
                           "ndcg@5", "recall@5", "avg_tokens", "compression_ratio"]
            df_summary = pd.concat(
                [df_nonrandom[shared_cols], random_avg[shared_cols]],
                ignore_index=True,
            )

    for dataset in df_summary["dataset"].unique():
        print(f"\n  {dataset}:")
        ds_data = df_summary[df_summary["dataset"] == dataset]

        for pf in sorted(ds_data["pool_factor"].unique()):
            pf_data = ds_data[ds_data["pool_factor"] == pf]
            if pf_data.empty:
                continue
            best = pf_data.loc[pf_data["ndcg@5"].idxmax()]
            label = best["method"]
            if best["importance_method"] != "n/a":
                label += f"_{best['importance_method']}"
            print(
                f"    PF={pf:>2d}: {label:<40s} "
                f"NDCG@5={best['ndcg@5']:.4f}  Recall@5={best['recall@5']:.4f}"
            )


def _plot_results_per_dataset(df: pd.DataFrame, output_path: Path, model_short: str):
    """Generate NDCG@5 vs compression ratio plots, one per dataset."""
    import matplotlib.pyplot as plt

    datasets = df["dataset"].unique()

    for dataset in datasets:
        ds_data = df[df["dataset"] == dataset].copy()

        # For random, average across seeds
        if "seed" in ds_data.columns:
            random_mask = ds_data["method"] == "random"
            if random_mask.any():
                random_avg = (
                    ds_data[random_mask]
                    .groupby(["method", "importance_method", "pool_factor"])
                    .agg({"ndcg@5": "mean", "compression_ratio": "mean"})
                    .reset_index()
                )
                ds_data = pd.concat([ds_data[~random_mask], random_avg], ignore_index=True)

        ds_data["label"] = ds_data.apply(
            lambda r: r["method"] if r["importance_method"] == "n/a" else f"{r['method']}_{r['importance_method']}",
            axis=1,
        )

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "p", "<", ">", "8"]

        for idx, label in enumerate(ds_data["label"].unique()):
            subset = ds_data[ds_data["label"] == label]
            if label == "none":
                ax.axhline(
                    y=subset["ndcg@5"].iloc[0],
                    color="black", linestyle="--", alpha=0.5, label="no pooling",
                )
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
        ax.set_title(f"{dataset} -- Retrieval Quality vs Compression ({model_short})", fontsize=13)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        plot_path = output_path / f"ndcg_vs_compression_{model_short}_{dataset}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] Plot: {plot_path}")

    # Combined plot (all datasets, best methods only)
    if len(datasets) > 1:
        _plot_combined(df, output_path, model_short)


def _plot_combined(df: pd.DataFrame, output_path: Path, model_short: str):
    """Combined plot showing key methods across all datasets side by side."""
    import matplotlib.pyplot as plt

    datasets = sorted(df["dataset"].unique())
    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5), squeeze=False)

    # Methods to highlight in the combined plot
    highlight_labels = {
        "none", "hierarchical", "random",
        "weighted_hierarchical_centroid_distance",
        "weighted_hierarchical_self_similarity",
        "adaptive_centroid_distance",
    }
    # Check if probe results exist
    if "probe" in df["importance_method"].unique():
        highlight_labels.add("weighted_hierarchical_probe")
        highlight_labels.add("adaptive_probe")

    style_map = {
        "none": {"color": "black", "marker": "o", "linestyle": "--"},
        "hierarchical": {"color": "tab:blue", "marker": "s", "linestyle": "-"},
        "random": {"color": "tab:gray", "marker": "x", "linestyle": ":"},
        "weighted_hierarchical_centroid_distance": {"color": "tab:red", "marker": "D", "linestyle": "-"},
        "weighted_hierarchical_self_similarity": {"color": "tab:orange", "marker": "^", "linestyle": "-"},
        "weighted_hierarchical_probe": {"color": "tab:purple", "marker": "v", "linestyle": "-"},
        "adaptive_centroid_distance": {"color": "tab:green", "marker": "P", "linestyle": "--"},
        "adaptive_probe": {"color": "tab:pink", "marker": "*", "linestyle": "--"},
    }

    for col_idx, dataset in enumerate(datasets):
        ax = axes[0, col_idx]
        ds_data = df[df["dataset"] == dataset].copy()

        # Average random across seeds
        if "seed" in ds_data.columns:
            random_mask = ds_data["method"] == "random"
            if random_mask.any():
                random_avg = (
                    ds_data[random_mask]
                    .groupby(["method", "importance_method", "pool_factor"])
                    .agg({"ndcg@5": "mean", "compression_ratio": "mean"})
                    .reset_index()
                )
                ds_data = pd.concat([ds_data[~random_mask], random_avg], ignore_index=True)

        ds_data["label"] = ds_data.apply(
            lambda r: r["method"] if r["importance_method"] == "n/a" else f"{r['method']}_{r['importance_method']}",
            axis=1,
        )

        for label in ds_data["label"].unique():
            if label not in highlight_labels:
                continue

            subset = ds_data[ds_data["label"] == label]
            style = style_map.get(label, {"color": None, "marker": "o", "linestyle": "-"})

            if label == "none":
                ax.axhline(
                    y=subset["ndcg@5"].iloc[0],
                    color="black", linestyle="--", alpha=0.5, label="no pooling",
                )
                continue

            ax.plot(
                subset["compression_ratio"],
                subset["ndcg@5"],
                marker=style["marker"],
                label=label.replace("_", " "),
                linewidth=1.5,
                markersize=6,
                color=style["color"],
                linestyle=style["linestyle"],
            )

        ax.set_xlabel("Compression Ratio", fontsize=11)
        if col_idx == 0:
            ax.set_ylabel("NDCG@5", fontsize=11)
        ax.set_title(dataset.replace("_test_subsampled", ""), fontsize=12)
        ax.legend(fontsize=6, loc="lower left")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Retrieval Quality vs Compression -- {model_short}", fontsize=14, y=1.02)
    fig.tight_layout()
    plot_path = output_path / f"ndcg_combined_{model_short}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Combined plot: {plot_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate importance-guided token pooling for document retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (5 docs, single dataset)
  python experiments/evaluate_pooling.py --max-docs 5

  # Full run on all 10 ViDoRe v1 datasets (default)
  python experiments/evaluate_pooling.py

  # Specific subsets only
  python experiments/evaluate_pooling.py \\
      --datasets vidore/docvqa_test_subsampled vidore/infovqa_test_subsampled

  # With probe-based importance and random robustness
  python experiments/evaluate_pooling.py \\
      --include-probe --random-seeds 42 123 456

  # Second model
  python experiments/evaluate_pooling.py \\
      --model vidore/colSmolVLM-256M-base \\
      --datasets vidore/docvqa_test_subsampled
        """,
    )
    parser.add_argument(
        "--model", type=str, default="ModernVBERT/colmodernvbert",
        help="HuggingFace model name (auto-detects model class)",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+",
        default=VIDORE_V1_DATASETS,
        help="HuggingFace dataset name(s). Defaults to all 10 ViDoRe v1 benchmark datasets.",
    )
    parser.add_argument(
        "--pool-factors", type=int, nargs="+",
        default=[1, 2, 3, 4, 6, 8],
        help="Pool factors to evaluate",
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments/results",
        help="Output directory",
    )
    parser.add_argument(
        "--include-probe", action="store_true",
        help="Include probe-based importance estimation (requires encoding probes through model)",
    )
    parser.add_argument(
        "--random-seeds", type=int, nargs="*", default=None,
        help="Seeds for random baseline (multiple seeds = mean+/-std). "
             "Use '--random-seeds 42 123 456' for 3 seeds.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for embedding")
    parser.add_argument("--max-docs", type=int, default=None, help="Max unique documents per dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--early-stop-threshold", type=float, default=0.3,
        help="Abort if baseline NDCG@5 < 0.05. Warn if a method drops below "
             "this fraction of baseline at PF<=4 (default: 0.3 = 30%%).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from previous run: skip already-completed configs found in output CSV.",
    )
    parser.add_argument(
        "--methods", type=str, nargs="*", default=None,
        help="Only run these importance-guided methods (e.g., --methods split_allocate importance_weighted_distance weighted_hierarchical). "
             "Baselines (none, random, hierarchical) always run.",
    )
    parser.add_argument(
        "--gpu-temp-threshold", type=int, default=78,
        help="Pause evaluation if GPU temp exceeds this (default: 78°C). Set 0 to disable.",
    )
    args = parser.parse_args()

    run_evaluation(
        model_name=args.model,
        dataset_names=args.datasets,
        pool_factors=args.pool_factors,
        output_dir=args.output_dir,
        include_probe=args.include_probe,
        random_seeds=args.random_seeds,
        batch_size=args.batch_size,
        max_docs=args.max_docs,
        device=args.device,
        early_stop_threshold=args.early_stop_threshold,
        allowed_methods=set(args.methods) if args.methods else None,
        resume=args.resume,
        gpu_temp_threshold=args.gpu_temp_threshold if args.gpu_temp_threshold > 0 else 999,
    )


if __name__ == "__main__":
    main()
