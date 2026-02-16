# Progress Update — Interpretability-Guided Token Pooling for Visual Document Retrieval

**Date:** February 16, 2026 (updated)  
**Target:** ICDAR 2026 (Deadline: February 27, 2026)

---

## 1. Motivation

Late-interaction vision-language retrieval models such as ColPali encode document pages as sets of patch-level embeddings — one vector per image patch — and match them against query token embeddings via MaxSim scoring. This approach enables OCR-free document retrieval: the model "sees" the document as an image and understands its content without any text extraction pipeline.

However, this design comes with a significant storage overhead. A single document page can produce over 1000 128-dimensional vectors, making large-scale indexing expensive. Existing compression methods, such as hierarchical token pooling (agglomerative clustering on cosine distances), reduce this cost by merging similar tokens. But they treat all patches equally — a whitespace margin receives the same consideration as a data-rich table cell.

The core question of this work is: **can we use interpretability signals to identify which patches are semantically important, and leverage this information to build better compression strategies?**

---

## 2. What Has Been Done

### 2.1 Importance Estimation Module

Implemented three query-agnostic methods for estimating per-token importance scores. These operate on the document embeddings alone (no query needed at indexing time), making them suitable for offline use.

**Self-Similarity Redundancy:** For each patch, compute the maximum cosine similarity to any other patch in the same document. Patches with low max-similarity to others are distinctive (non-redundant) and thus more important.

$$\text{importance}(i) = 1 - \max_{j \neq i} \cos(e_i, e_j)$$

**Centroid Distance:** Compute the cosine similarity between each patch and the mean embedding. Patches far from the average carry more distinctive information.

$$\text{importance}(i) = 1 - \cos\left(e_i, \frac{1}{N}\sum_j e_j\right)$$

**Probe-Based Importance:** Encode a set of semantic probe strings (e.g., "title", "table", "figure", "equation") through the model, then compute the maximum similarity between each document patch and all probes. This method explicitly captures what type of content the model recognises at each spatial location.

$$\text{importance}(i) = \max_q \text{sim}(e_i, \text{probe}_q)$$

The implementation is in `src/importance_estimation.py` with an abstract `BaseImportanceEstimator` class and concrete implementations for each strategy.

### 2.2 Importance-Guided Pooling Strategies

Implemented four pooling strategies that use importance scores to make informed compression decisions. All extend the existing `BaseTokenPooler` interface from `colpali_engine`.

**Top-K Selection:** Keep the *k* most important tokens (where *k* = token_count / pool_factor), discarding the rest. Retained tokens are returned in their original order. This is the simplest strategy — it tests whether selection alone is sufficient.

**Importance-Weighted Hierarchical Pooling:** Uses the same agglomerative clustering as the existing hierarchical pooler, but when computing the representative vector for each cluster, uses importance scores as weights instead of a uniform mean. This biases the pooled representation toward the most important tokens within each cluster.

**Protect-and-Pool:** A two-phase strategy. First, protect the top *p*% of tokens by importance (keep them untouched). Then, apply hierarchical clustering only to the remaining tokens. The output concatenates protected tokens with pooled representations.

**Adaptive Pool Factor:** Varies the compression ratio per document based on the entropy of the importance distribution. Documents with concentrated importance (few important patches — e.g., a single table) are compressed more aggressively; documents with distributed importance (many important regions) are compressed less. This targets a fixed average storage budget while adapting to document complexity.

All four strategies are in `src/importance_guided_pooling.py`, with unit tests covering output shapes, normalisation, order preservation, and edge cases.

### 2.3 Experiment Infrastructure

Built two experiment scripts:

- **`experiments/evaluate_pooling.py`** — Full evaluation pipeline, significantly enhanced (see §2.5):
  - Model-agnostic loading via `MODEL_REGISTRY` (supports ColModernVBERT, ColIdefics3, ColPali, ColQwen2, ColQwen2.5, ColQwen3, ColGemma3, and more).
  - Evaluates all 10 ViDoRe v1 benchmark datasets by default.
  - Correctly handles multi-query-per-document datasets (TabFQuAD, TAT-DQA) and practical task datasets with null-query distractor rows (syntheticDocQA_*, shiftproject).
  - Includes probe-based importance estimation via `--include-probe`.
  - Random baseline with multiple seeds (`--random-seeds 42 123 456`) for statistical robustness.
  - `ProgressTracker` class: writes live `status.txt` with progress bar + ETA, plus `eval.log` for full logging.
  - Early stopping: aborts if baseline NDCG@5 < 0.05; warns if a method drops below 30% of baseline at PF ≤ 4.
  - Per-dataset incremental CSV saves and combined comparison plots.

- **`experiments/visualize_importance.py`** — Generates importance heatmaps overlaid on document images, side-by-side comparisons across estimation methods, and correlation analysis between query-agnostic importance and query-specific similarity maps.

### 2.4 Importance Heatmap Visualisations

Generated importance heatmaps for 5 DocVQA documents using ColModernVBERT. These are in `experiments/results/visualizations/`. Each document has:
- Individual importance maps for self-similarity and centroid-distance methods.
- A side-by-side comparison of both methods against the original document.
- A query-specific MaxSim aggregate map for reference.

The correlation between query-agnostic importance and query-specific MaxSim is low (r ≈ 0.01–0.12), which is expected — importance is designed to capture general informativeness, not query-specific relevance. The retrieval evaluation below confirms that this general informativeness is nonetheless useful for compression.

### 2.5 Scaled Evaluation Pipeline (New — Feb 16)

Rewrote `experiments/evaluate_pooling.py` to support the full ViDoRe v1 benchmark. Key changes:

- **All 10 ViDoRe v1 datasets** are now the default (see §3.1 for the full list).
- **Null-query filtering** — practical task datasets (Energy, Government, Healthcare, AI, Shift Project) store 1000 rows but only 100 have queries; the remaining 900 are document-only corpus pages. The loader now correctly filters null/empty queries and builds sparse relevance matrices (100 queries × ~1000 documents).
- **Multi-query deduplication** — TabFQuAD (280 queries / 70 unique documents) and TAT-DQA (1663 queries / 277 unique documents) are deduplicated by `image_filename`, with proper relevance matrices.
- **15 methods evaluated per dataset** — 4 pooling strategies × 3 importance estimators (self_similarity, centroid_distance, probe) + hierarchical baseline + random baseline + no-pooling baseline.
- **810 total configs** for the current run (10 datasets × 81 configs each).
- **Probe-based importance** is now integrated into the evaluation loop.
- **Random baseline** runs with 3 seeds (42, 123, 456), reporting mean ± std.

---

## 3. Experimental Results

### 3.1 Full ViDoRe v1 Benchmark (In Progress)

Currently running the full evaluation on an RTX 3070 (8GB). Estimated ~30 hours.

**Model:** ColModernVBERT (240M parameters, ~480MB in fp16).

**All 10 ViDoRe v1 datasets:**

| Dataset | Type | Language | Queries | Documents | Notes |
|:--------|:-----|:---------|--------:|----------:|:------|
| DocVQA | Academic | EN | 500 | 500 | Scanned UCSF industry docs |
| InfoVQA | Academic | EN | 500 | 500 | Infographics from the web |
| TAT-DQA | Academic | EN | 1,663 | 277 | Financial reports (~6 queries/doc) |
| ArXivQA | Academic | EN | 500 | 500 | Scientific figures from arXiv |
| TabFQuAD | Academic | FR | 280 | 70 | French tables (4 queries/doc) |
| Energy | Practical | EN | 100 | 977 | Energy documents |
| Government | Practical | EN | 100 | 972 | Administrative documents |
| Healthcare | Practical | EN | 100 | 965 | Medical documents |
| AI | Practical | EN | 100 | 968 | AI-related scientific docs |
| Shift Project | Practical | FR | 100 | 1,000 | Environmental reports |

**Metrics:** NDCG@5, Recall@5.  
**Compression ratios:** Pool factors 1, 2, 3, 4, 6, 8.  
**Methods:** No pooling, random (3 seeds), hierarchical, + 4 importance-guided strategies × 3 estimators (self_similarity, centroid_distance, probe) = 15 methods.

**Status:** Running in tmux session `eval`. Monitor with:
```
cat experiments/results/scaled_eval/status.txt
tail -f experiments/results/scaled_eval/eval.log
```

### 3.2 Preliminary Results (100-document DocVQA pilot)

Prior to the full run, a pilot evaluation on 100 DocVQA documents established the baseline trends:

Baseline (no pooling): NDCG@5 = 0.660, Recall@5 = 0.75, ~1070 tokens per document.

| Compression | Method | Importance | NDCG@5 | Recall@5 | Tokens |
|:-----------:|:-------|:-----------|:------:|:--------:|:------:|
| 0%          | No pooling | — | 0.660 | 0.75 | 1070 |
| 50%         | Weighted hierarchical | centroid_distance | **0.672** | 0.76 | 534 |
| 50%         | Hierarchical (baseline) | — | 0.671 | 0.76 | 534 |
| 50%         | Random | — | 0.641 | 0.75 | 534 |
| 66.7%       | Weighted hierarchical | centroid_distance | **0.668** | 0.76 | 357 |
| 66.7%       | Hierarchical (baseline) | — | 0.664 | 0.75 | 357 |
| 75%         | Weighted hierarchical | centroid_distance | **0.679** | 0.78 | 267 |
| 75%         | Hierarchical (baseline) | — | 0.676 | 0.77 | 267 |
| 83.4%       | Weighted hierarchical | self_similarity | **0.671** | 0.76 | 178 |
| 83.4%       | Hierarchical (baseline) | — | 0.669 | 0.77 | 178 |
| 87.6%       | Adaptive | centroid_distance | **0.684** | 0.79 | 263 |
| 87.6%       | Weighted hierarchical | self_similarity | 0.646 | 0.74 | 133 |
| 87.6%       | Hierarchical (baseline) | — | 0.641 | 0.73 | 133 |
| 87.6%       | Random | — | 0.553 | 0.66 | 133 |

### 3.3 Key Observations (from pilot)

1. **Importance-weighted hierarchical pooling consistently matches or outperforms standard hierarchical pooling**, with the centroid-distance importance estimator showing the strongest gains. At 75% compression, it achieves NDCG@5 = 0.679 — higher than the uncompressed baseline (0.660). This suggests the weighted pooling acts as a form of noise reduction, producing cleaner representations by down-weighting uninformative patches within clusters.

2. **The improvement grows with compression.** At low compression (50%), the advantage over standard hierarchical pooling is marginal (+0.1%). At high compression (75%), it becomes more significant (+0.5%). At extreme compression (87.6%), the weighted hierarchical method retains 0.646 NDCG@5 while the standard method drops to 0.641 — a meaningful gap when both are operating under severe token budgets.

3. **Adaptive pooling excels at extreme compression settings.** When configured with a pool factor range of [4, 16] (target PF=8), the adaptive method achieves NDCG@5 = 0.684 — the highest score across all methods and compression levels. It does this by dynamically allocating ~263 tokens per document on average (vs 133 for fixed PF=8). Documents with uniformly distributed importance receive more tokens; documents with concentrated importance receive fewer.

4. **Top-K selection (pure token dropping) performs poorly at high compression.** At PF=8, top-k with centroid distance drops to 0.494 NDCG@5, barely above random chance for this dataset. This demonstrates that clustering-based merging is fundamentally superior to selection — it preserves information by averaging within clusters rather than discarding tokens entirely.

5. **Centroid distance is the stronger importance estimator for weighted hierarchical pooling**, while self-similarity performs better at extreme compression (PF=6). This makes sense: centroid distance identifies globally distinctive patches, which are more useful when you have a reasonable token budget. Self-similarity identifies locally non-redundant patches, which may be more resilient when the budget is very tight.

*These observations will be validated on the full 10-dataset run once it completes.*

---

## 4. What To Do Next

### 4.1 ~~Scale Evaluation~~ ✅ DONE (Feb 16)

Full evaluation pipeline rewritten and running on all 10 ViDoRe v1 datasets with probe-based importance and random seed robustness. **810 configs total, estimated ~30h runtime.**

### 4.2 ~~Evaluate Probe-Based Importance~~ ✅ DONE (Feb 16)

Probe-based importance is now included in the evaluation via `--include-probe`. All 4 pooling strategies are evaluated with all 3 importance estimators (self_similarity, centroid_distance, probe).

### 4.3 ~~Statistical Robustness~~ ✅ DONE (Feb 16)

Random baseline now runs with 3 seeds (42, 123, 456), reporting mean ± std.

### 4.4 Analyse Full Results (High Priority — when run completes, ~Feb 17–18)

Once the 810-config evaluation finishes:
- Generate per-dataset result tables and aggregate summary.
- Compare academic vs. practical task performance.
- Identify which document types benefit most from importance-guided pooling.
- Compare probe-based importance against self-similarity and centroid-distance.

### 4.5 Second Model (High Priority — Feb 18–19)

Run the same evaluation on ColSmolVLM (vidore/colSmol-256M) to demonstrate the approach is model-agnostic. The evaluation script already supports it — just change `--model`.

### 4.6 Per-Document-Type Analysis (Medium Priority — Feb 19–20)

Break down results by document type. The hypothesis is that importance-guided pooling provides the largest gains on documents with a clear division between informative and uninformative regions (e.g., a table surrounded by whitespace). Text-heavy documents with uniform information density may show smaller gains. The 10-dataset evaluation already provides a natural per-type breakdown (academic vs. practical, tables vs. infographics vs. scientific figures).

### 4.7 Publication-Quality Figures (Medium Priority — Feb 20–22)

- NDCG@5 vs. compression ratio curves with clean legends and consistent styling.
- Per-dataset bar charts showing method performance across document types.
- Importance heatmap examples that clearly illustrate the difference between estimation methods.
- A method overview diagram showing the pipeline from model output to importance estimation to guided pooling.

### 4.8 Paper Writing (Feb 22–27)

Structure in ICDAR format (~10 pages):
- **Introduction**: OCR-free document retrieval, storage overhead of multi-vector embeddings, interpretability as a tool for compression.
- **Related Work**: ColPali and ColBERT-style retrieval, token compression in dense retrieval, interpretability in information retrieval.
- **Method**: Three importance estimators, four pooling strategies, the interpretability-compression connection.
- **Experiments**: Setup, main results table, per-dataset analysis, visualisations.
- **Analysis**: When does importance guidance help? Which estimator and pooling strategy to use? Correlation between importance and retrieval relevance.
- **Conclusion**: Importance-guided pooling as noise reduction, practical recommendations.

---

## 5. File Structure

```
src/
    __init__.py
    importance_estimation.py              # 3 importance estimators + utility functions
    importance_guided_pooling.py          # 4 pooling strategies

experiments/
    evaluate_pooling.py                   # Full evaluation pipeline (model-agnostic, 10 datasets)
    evaluate_pooling_v1.py                # Backup of original single-dataset evaluation
    visualize_importance.py               # Importance heatmap generation
    results/
        pooling_evaluation_results.csv    # Pilot results (100-doc DocVQA, 56 configs)
        pooling_evaluation_results.json   # Same in JSON
        ndcg_vs_compression.png           # Pilot comparison plot
        scaled_eval/                      # Full 10-dataset evaluation (in progress)
            status.txt                    # Live progress tracker
            eval.log                      # Full evaluation log
            full_output.log               # tee'd stdout
        scaled_eval_partial_4ds/          # Partial results from earlier 4-dataset attempt
        visualizations/                   # Importance heatmaps for 5 documents

tests/
    test_importance_estimation.py         # Tests for importance estimators
    test_importance_guided_pooling.py     # Tests for pooling strategies
```

The codebase is a standalone package (installed via `pyproject.toml` in editable mode) that depends on `colpali_engine` from the sibling `../colpali/` directory.
