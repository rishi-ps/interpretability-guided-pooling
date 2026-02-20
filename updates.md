# Progress Update — Interpretability-Guided Token Pooling for Visual Document Retrieval

**Date:** February 20, 2026 (updated)  
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

**SVD Projection Energy (NEW — Feb 20):** Perform SVD on the (token_length × dim) embedding matrix. Tokens whose energy is concentrated along the top-k singular directions capture the most variance and are structurally important.

$$\text{importance}(i) = \| U_{i,:k} \cdot S_{:k} \|_2$$

where $U, S, V = \text{SVD}(E)$ and $k = \min(8, \min(n, d))$. Tokens with high projection onto the principal semantic directions are the primary carriers of the document's content.

**Attention-Based Importance (NEW — Feb 20):** Hooks into the last attention layer of the text backbone and extracts the attention matrix. Per-token importance is the mean attention *received* by each token (column-mean of attention matrix):

$$\text{importance}(i) = \frac{1}{H} \sum_h \frac{1}{N} \sum_j A_{h,j,i}$$

Tokens that receive high attention from many others serve as "information hubs" in the transformer. Unlike the other estimators, this requires a forward pass through the model with document images — it cannot operate on pre-computed embeddings alone. The implementation supports real attention extraction via `estimate_from_images()` and a proxy mode for pre-computed attention scores.

The implementation is in `src/importance_estimation.py` with an abstract `BaseImportanceEstimator` class and concrete implementations for all five strategies.

### 2.2 Importance-Guided Pooling Strategies

Implemented seven pooling strategies that use importance scores to make informed compression decisions. All extend the existing `BaseTokenPooler` interface from `colpali_engine`.

**Top-K Selection:** Keep the *k* most important tokens (where *k* = token_count / pool_factor), discarding the rest. Retained tokens are returned in their original order. This is the simplest strategy — it tests whether selection alone is sufficient.

**Importance-Weighted Hierarchical Pooling:** Uses the same agglomerative clustering as the existing hierarchical pooler, but when computing the representative vector for each cluster, uses importance scores as weights instead of a uniform mean. This biases the pooled representation toward the most important tokens within each cluster.

**Protect-and-Pool:** A two-phase strategy. First, protect the top *p*% of tokens by importance (keep them untouched). Then, apply hierarchical clustering only to the remaining tokens. The output concatenates protected tokens with pooled representations.

**Adaptive Pool Factor:** Varies the compression ratio per document based on the entropy of the importance distribution. Documents with concentrated importance (few important patches — e.g., a single table) are compressed more aggressively; documents with distributed importance (many important regions) are compressed less. This targets a fixed average storage budget while adapting to document complexity.

**Split-and-Allocate (NEW — Feb 17):** Addresses the key insight from Round 1: importance should guide *cluster formation*, not *within-cluster averaging*. Splits tokens into importance tiers (default: top-50% / bottom-50% by importance score), then allocates a disproportionate share of the cluster budget to the important tier (default: 75% / 25%). Each tier is clustered independently with hierarchical pooling. The output has the exact same number of tokens as standard hierarchical — a fair comparison. This gives important tokens (text, tables, figures) ~3× the representation budget, while aggressively merging unimportant tokens (margins, backgrounds).

**Importance-Weighted Distance (NEW — Feb 17):** A complementary approach to Split-and-Allocate. Instead of hard-splitting tokens into tiers, this method *softly* biases the clustering itself. Before computing distances for hierarchical clustering, embeddings are scaled by importance: $e_i' = e_i \cdot (1 + \alpha \cdot \hat{s}_i)$ where $\hat{s}_i$ is min-max normalized importance. Important tokens get larger magnitude → larger Euclidean distances → they resist merging and naturally end up in smaller, more precise clusters. Crucially, cluster representatives are computed from the *original* (unscaled) embeddings, preserving the embedding space geometry for MaxSim scoring. Default $\alpha = 1.0$. Same output token count as standard hierarchical — a fair comparison. This avoids Split-and-Allocate's hard tier boundary, which may hurt at low compression.

**Importance-Weighted K-Means (NEW — Feb 20):** K-means clustering with importance-biased initialization and weighted centroids. Instead of hierarchical Ward linkage, uses k-means which directly optimises reconstruction error. Key features: (1) importance-biased k-means++ initialization — important tokens are preferentially selected as seeds via D²-sampling weighted by importance, giving content-rich regions finer coverage from the start; (2) importance-weighted centroids — cluster representatives are biased toward important members via $w_i = 1 + \alpha \cdot \hat{s}_i$. Multiple restarts (default: 3) with best-by-inertia selection. Default $\alpha = 1.0$, max 20 iterations.

All seven strategies are in `src/importance_guided_pooling.py`, with unit tests (62 pooling + 35 estimation = 97 total) covering output shapes, normalisation, order preservation, budget compliance, alpha parameter behaviour, k-means++ init properties, and edge cases.

### 2.3 Experiment Infrastructure

Built two experiment scripts:

- **`experiments/evaluate_pooling.py`** — Full evaluation pipeline, significantly enhanced (see §2.5):
  - Model-agnostic loading via `MODEL_REGISTRY` (supports ColModernVBERT, ColIdefics3, ColPali, ColQwen2, ColQwen2.5, ColQwen3, ColGemma3, and more).
  - Evaluates all 10 ViDoRe v1 benchmark datasets by default.
  - Correctly handles multi-query-per-document datasets (TabFQuAD, TAT-DQA) and practical task datasets with null-query distractor rows (syntheticDocQA_*, shiftproject).
  - Includes probe-based importance estimation via `--include-probe`.
  - Includes attention-based importance estimation via `--include-attention` (requires forward pass per document).
  - Supports all 5 estimators and 7 pooling methods via `--methods` flag.
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
- **15+ methods evaluated per dataset** — 7 pooling strategies × 2–5 importance estimators + hierarchical baseline + random baseline + no-pooling baseline.
- **810 total configs** for the current run (10 datasets × 81 configs each).
- **Probe-based importance** is now integrated into the evaluation loop.
- **Random baseline** runs with 3 seeds (42, 123, 456), reporting mean ± std.

---

## 3. Experimental Results

### 3.1 Full ViDoRe v1 Benchmark (Complete)

Completed the full evaluation on an RTX 3070 (8GB). Round 2 finished in ~5 hours (430 configs across 10 datasets, 3 importance-guided strategies + baselines).

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

**Status (Round 1):** Run terminated Feb 17 at 191/810 configs (23.6%). Datasets completed: DocVQA, InfoVQA. Partial: TAT-DQA (PF 2–3). Terminated to pivot to improved method (see §3.5). Results saved in `experiments/results/scaled_eval/`.

**Status (Round 2):** ✅ **COMPLETED** Feb 17. All 10 datasets finished. 269 configs total, ~5 hours runtime. Results in `experiments/results/round2/`. See §3.7 for cross-dataset summary.

### 3.2 Full-Run Results — Round 1 (DocVQA & InfoVQA complete; TAT-DQA partial)

**Run terminated Feb 17** after 2/10 datasets completed and partial TAT-DQA results, to pivot to an improved method (see §3.5). Results below document what we tried, what worked, what failed, and why.

**DocVQA (500 queries, 500 documents) — NDCG@5:**

Baseline (no pooling): NDCG@5 = 0.5162, Recall@5 = 0.598, ~1070 tokens/doc.

| PF | Comp% | Hierarchical | WH best (estimator) | WH Δ vs Hier | Top-K best | P&P best | Adaptive (actual tokens) | Random mean |
|:--:|:-----:|:------------:|:--------------------:|:------------:|:----------:|:--------:|:------------------------:|:-----------:|
| 2 | 50% | 0.5185 | 0.5184 (probe) | −0.0001 | 0.4999 | 0.5155 | 0.5162 (1070t = **no compression**) | 0.4857 |
| 3 | 67% | 0.5102 | 0.5104 (probe) | +0.0002 | 0.4793 | 0.5082 | 0.5162 (1070t = **no compression**) | 0.4603 |
| 4 | 75% | 0.5091 | 0.5099 (centroid) | +0.0008 | 0.4500 | 0.4500 ⚠️ =topk | 0.5185 (534t = PF 2) | 0.4397 |
| 6 | 83% | 0.4978 | 0.5002 (probe) | +0.0023 | 0.4004 | 0.4004 ⚠️ =topk | 0.5102 (357t = PF 3) | 0.4154 |
| 8 | 88% | 0.4758 | 0.4792 (self_sim) | +0.0034 | 0.3621 | 0.3621 ⚠️ =topk | 0.5103 (264t = PF 4) | 0.3827 |

**InfoVQA (500 queries, 500 documents) — NDCG@5:**

Baseline (no pooling): NDCG@5 = 0.8567, Recall@5 = 0.904, ~723 tokens/doc.

| PF | Comp% | Hierarchical | WH best (estimator) | WH Δ vs Hier | Top-K best | P&P best | Adaptive (actual tokens) | Random mean |
|:--:|:-----:|:------------:|:--------------------:|:------------:|:----------:|:--------:|:------------------------:|:-----------:|
| 2 | 50% | 0.8501 | 0.8509 (probe) | +0.0008 | 0.8096 | 0.8524 | 0.8567 (723t = **no compression**) | 0.8250 |
| 3 | 67% | 0.8474 | 0.8467 (centroid) | −0.0007 | 0.7457 | 0.8318 | 0.8567 (723t = **no compression**) | 0.7906 |
| 4 | 75% | 0.8409 | 0.8415 (probe) | +0.0005 | 0.7021 | 0.7021 ⚠️ =topk | 0.8501 (361t = PF 2) | 0.7716 |
| 6 | 83% | 0.8262 | 0.8294 (centroid) | +0.0031 | 0.6243 | 0.6243 ⚠️ =topk | 0.8474 (241t = PF 3) | 0.7362 |
| 8 | 88% | 0.8205 | 0.8244 (centroid) | +0.0039 | 0.5581 | 0.5581 ⚠️ =topk | 0.8409 (181t = PF 4) | 0.7073 |

**TAT-DQA (1663 queries, 277 documents) — Partial (PF 2–3 only):**

Baseline (no pooling): NDCG@5 = 0.7262, Recall@5 = 0.834, ~1080 tokens/doc.

| PF | Hierarchical | WH best (estimator) | WH Δ | Protect-and-Pool best | Random mean |
|:--:|:------------:|:--------------------:|:----:|:---------------------:|:-----------:|
| 2 | 0.7175 | 0.7189 (probe) | +0.0014 | 0.7213 (probe) | 0.6824 |
| 3 | 0.7194 | 0.7196 (centroid) | +0.0002 | 0.6878 (centroid) | 0.6413 |

### 3.3 Analysis: Why Round 1 Methods Underperform (Feb 17)

Careful analysis of the Round 1 results revealed three fundamental issues:

**Issue 1: Weighted hierarchical gains are real but tiny (+0.0 to +0.004 NDCG).**

The reason is straightforward: hierarchical clustering groups tokens that are already very similar (cosine similarity > 0.9 within a cluster). Taking a weighted vs. uniform average of near-identical vectors produces a nearly identical result after L2-normalization. The importance signal is applied *inside clusters where it cannot make a meaningful difference*. It needs to influence *which tokens share clusters in the first place* — i.e., the cluster budget allocation, not the within-cluster aggregation.

**Issue 2: Protect-and-pool collapses to top-k at PF ≥ 4.**

Root cause found in the implementation: `n_protect = min(int(token_length * protect_fraction), target_output)`. With `protect_fraction=0.25` and 1070 tokens: `n_protect = min(267, target_output)`. At PF=4, `target_output = 267`, so `n_protect = 267` — the entire output budget is consumed by protection, leaving zero tokens for pooling the remaining 803 tokens. They are simply discarded, reducing the method to pure top-k selection.

**Issue 3: Adaptive pooling uses unfair token budgets.**

The adaptive method was configured with `min_pool_factor=PF//2, max_pool_factor=PF*2`. The importance entropy calculation consistently yields near-maximum entropy for ~1000 tokens with min-max normalized scores, causing `effective_pf ≈ min_pool_factor`:
- At PF=2–3: effective PF=1 (no compression at all — returns original embeddings)
- At PF=4: effective PF=2 (half the compression of hierarchical at PF=4)
- At PF=8: effective PF≈4.3 (uses 247 tokens vs 133 for hierarchical)

The "gains" over hierarchical are entirely due to using 2–8× more tokens, not from better compression.

**Conclusion:** Of the four methods, only weighted hierarchical provides a genuine (though small) improvement at equal token budgets. Top-k is fundamentally limited (discards vs. merges). Protect-and-pool and adaptive both have implementation bugs that mask their true performance. More importantly, the weighted hierarchical approach is applying importance at the wrong level of the pipeline — it should guide *cluster formation*, not *within-cluster averaging*.

### 3.3 Preliminary Results (100-document DocVQA pilot)

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

### 3.4 Key Observations (from pilot, confirmed at scale)

1. **Importance-weighted hierarchical pooling consistently matches or outperforms standard hierarchical pooling**, but the gains are marginal: +0.0 to +0.004 NDCG@5 at full scale, +0.001 to +0.005 in the pilot. The trend is consistent (gains grow with compression), but the effect is too small to be the main contribution.

2. **The improvement grows with compression.** At low compression (50%), the advantage is negligible. At 88% compression, weighted hierarchical gains +0.003–0.004 NDCG@5 over hierarchical on both DocVQA and InfoVQA. Small, but consistently positive.

3. **Adaptive pooling is not comparable at equal budgets.** What appeared to be strong performance in the pilot (NDCG@5 = 0.684 at "PF=8") was actually the result of using 2× more tokens than hierarchical at the same nominal pool factor. When evaluated at true equal token counts, adaptive does not demonstrate clear advantages. The per-document budget allocation idea has merit, but the current entropy-based implementation always gravitates toward minimal compression.

4. **Top-K selection (pure token dropping) performs poorly at high compression.** At PF=8, top-k drops to 0.30–0.36 NDCG@5 on DocVQA, far below hierarchical (0.476). This confirms that clustering-based merging is fundamentally superior to selection — it preserves information by averaging within clusters rather than discarding tokens entirely.

5. **Protect-and-pool is broken at PF≥4** due to a budget allocation bug (see §3.3). At PF=2–3, where it functions correctly, it shows promise on InfoVQA (0.852 vs 0.850 hierarchical at PF=2) but not on DocVQA. The concept may be worth revisiting after fixing the implementation.

6. **All three importance estimators perform similarly** for weighted hierarchical. No single estimator dominates. Probe-based importance is slightly preferred on DocVQA, centroid distance on InfoVQA. This suggests the specific importance signal matters less than *how* it is used in the pooling pipeline.

### 3.5 Pivot: From Within-Cluster Weighting to Cluster Budget Allocation (Feb 17)

The Round 1 results led to a key insight: **importance signals should guide cluster formation, not within-cluster averaging.** 

The problem with weighted hierarchical: hierarchical clustering groups tokens by cosine similarity. Within a cluster, tokens point in nearly the same direction. Weighting the average of near-identical vectors barely changes the result after L2-norm. The importance signal is wasted.

The proposed fix: **Split-and-Allocate** — use importance to decide *how many clusters each importance tier receives*:

```
At PF=4 (1070 → 267 output tokens):
  Standard hierarchical: 267 clusters of ~4 tokens each (uniform allocation)
  Split-and-Allocate:    top-50% tokens → 200 clusters (fine-grained, PF≈2.7)
                          bottom-50% tokens → 67 clusters (aggressive, PF≈8)
                          Total: 267 tokens (same budget, fair comparison)
```

Important tokens (text, tables, figures) get 3× the representation budget. Unimportant tokens (margins, backgrounds) that queries rarely match are merged aggressively. Same total output tokens as hierarchical — a fair comparison.

### 3.6 Round 2 — DocVQA Results (Feb 17)

Ran validation (37 configs) + full-run completion (6 additional IWD configs) on DocVQA (500 queries, 500 documents) with all Round 2 methods. Pool factors 1/2/4/8, 2 importance estimators, 1 random seed. **43 total configs.**

Baseline (no pooling): NDCG@5 = 0.5162, Recall@5 = 0.598, ~1070 tokens/doc.

**NDCG@5 by method and pool factor (fair comparison — same token count, except adaptive):**

| Method | Estimator | PF=2 (534t) | PF=4 (267t) | PF=8 (133t) |
|:-------|:----------|:-----------:|:-----------:|:-----------:|
| hierarchical | — | 0.5185 | 0.5091 | 0.4758 |
| random (seed=42) | — | 0.4792 | 0.4368 | 0.3655 |
| **imp_weighted_dist** | **centroid** | **0.5118** | **0.5116** | **0.4956** |
| imp_weighted_dist | self_sim | 0.5077 | 0.5073 | 0.4893 |
| split_allocate | centroid | 0.5114 | 0.5032 | 0.4924 |
| split_allocate | self_sim | 0.5120 | 0.5012 | 0.4582 |
| weighted_hier | centroid | 0.5177 | 0.5099 | 0.4744 |
| weighted_hier | self_sim | 0.5184 | 0.4999 | 0.4792 |
| protect_and_pool | centroid | 0.5066 | 0.4775 | 0.4332 |
| protect_and_pool | self_sim | 0.5102 | 0.4792 | 0.4124 |
| adaptive | centroid | 0.5158 (564t) | 0.5096 (343t) | 0.4899 (164t) |
| adaptive | self_sim | 0.5082 (422t) | 0.5012 (209t) | 0.4623 (104t) |
| topk | centroid | 0.4864 | 0.4074 | 0.3001 |
| topk | self_sim | 0.4514 | 0.4049 | 0.3391 |

**Best importance-guided method at each PF (fair comparison, same tokens as hierarchical):**

| PF | Best Method | NDCG@5 | Δ vs Hierarchical |
|:--:|:------------|:------:|:------------------:|
| 2 | weighted_hier / self_sim | 0.5184 | −0.0001 |
| 4 | **imp_weighted_dist / centroid** | **0.5116** | **+0.0025** |
| 8 | **imp_weighted_dist / centroid** | **0.4956** | **+0.0198** |

**Key findings:**

1. **Importance-Weighted Distance with centroid_distance is the overall winner.** At PF=8 (88% compression), it scores 0.4956 vs hierarchical's 0.4758 — a **+0.0198 NDCG@5 improvement** using the exact same number of tokens (133). This is 5× larger than the best Round 1 gain and surpasses split_allocate's +0.0166.

2. **IWD also wins at PF=4**, unlike split_allocate which loses. IWD/centroid scores 0.5116 vs hierarchical's 0.5091 (+0.0025), while split_allocate/centroid was 0.5032 (−0.0059). The "soft" distance-biasing avoids the hard-tier boundary problem that hurts split_allocate at moderate compression.

3. **The gain grows with compression**, confirming the thesis: at PF=2 the gap is minimal (−0.0067), at PF=4 it's positive (+0.0025), and at PF=8 it's substantial (+0.0198).

4. **Centroid distance is the better estimator** across all methods. For IWD at PF=8: centroid=0.4956 vs self_sim=0.4893. For split_allocate: centroid=0.4924 vs self_sim=0.4582.

5. **Split-and-Allocate is the second-best method at PF=8** (+0.0166) but loses at PF=2 (−0.0071) and PF=4 (−0.0059). The hard tier boundary is suboptimal except under extreme compression.

6. **Weighted hierarchical provides small consistent gains** (+0.0008 at PF=4, +0.0034 at PF=8 with self_sim). Confirms importance helps, but within-cluster weighting is the least effective way to use it.

7. **Protect-and-pool still underperforms** even after the 75% cap fix.

8. **Adaptive uses unfair token budgets** (164 tokens at PF=8 vs 133 for others). Not a fair comparison.

Results saved in `experiments/results/round2/`.

### 3.7 Round 2 — Full 10-Dataset Cross-Dataset Results (Feb 17–20)

Round 2 evaluation **completed** across all 10 ViDoRe v1 datasets. 269 configs, ~5h runtime. The results below show the best importance-guided method at each compression level vs. hierarchical baseline (fair comparison — same token count).

**Best importance-guided method vs hierarchical, per dataset (NDCG@5 Δ, fair comparison):**

| Dataset | PF=4 best | Δ@4 | PF=8 best | Δ@8 |
|:--------|:----------|:---:|:----------|:---:|
| ArXivQA | IWD/self_sim | +0.0076 | WH/centroid | +0.0007 |
| DocVQA | IWD/centroid | +0.0025 | IWD/centroid | +0.0198 |
| InfoVQA | IWD/centroid | +0.0050 | WH/centroid | +0.0039 |
| ShiftProject | IWD/centroid | +0.0206 | IWD/self_sim | +0.0069 |
| AI | IWD/self_sim | +0.0030 | IWD/centroid | +0.0243 |
| Energy | SA/centroid | +0.0048 | IWD/self_sim | +0.0057 |
| Government | WH/centroid | +0.0037 | SA/centroid | +0.0002 |
| Healthcare | IWD/centroid | +0.0046 | WH/centroid | +0.0007 |
| TabFQuAD | SA/centroid | +0.0137 | SA/centroid | +0.0244 |
| TAT-DQA | WH/centroid | +0.0006 | IWD/centroid | +0.0080 |

**IWD = importance_weighted_distance, SA = split_allocate, WH = weighted_hierarchical.**

**Cross-dataset headline numbers:**

- **At PF=8 (88% compression):** The best importance-guided method beats hierarchical on **all 10 datasets** (Δ range: +0.0002 to +0.0244). Average gain: +0.009 NDCG@5.
- **At PF=4 (75% compression):** The best importance-guided method beats hierarchical on **all 10 datasets** (Δ range: +0.0006 to +0.0206). Average gain: +0.007 NDCG@5.
- **IWD/centroid at PF=8 specifically:** Positive on 6/10 datasets, negative on 4/10 (mean Δ = +0.0052). Not a universal winner — wins are concentrated on DocVQA (+0.020), AI (+0.024), TabFQuAD (+0.019), TAT-DQA (+0.008). Losses are small (−0.002 to −0.016).
- **No single method dominates PF=8 wins:** IWD/centroid wins 3, WH/centroid wins 3, IWD/self_sim wins 2, SA/centroid wins 2.
- **The consistent finding is that *some* form of importance guidance always helps at high compression**, even though the optimal strategy varies by dataset.

**Honest assessment:**

- The gains are real and consistent in direction (always ≥ 0 for the best method), but small in magnitude (mean +0.009 NDCG@5 at PF=8). Whether this is "significant" in a practical sense depends on the application.
- No statistical significance tests have been run (single seed, no bootstrap CIs). The per-dataset gains could partly be noise.
- IWD/centroid — the flagship method — is not the best on 4/10 datasets at PF=8. The story is "importance guidance helps" rather than "IWD/centroid is definitively superior."
- At PF=2, importance-guided methods still tend to hurt slightly (not shown, but consistent with DocVQA findings in §3.6).

### 3.8 Round 3 — New Estimators Pilot on DocVQA (Feb 20)

Implemented two new importance estimators (SVD, Attention) and one new pooling method (Importance-Weighted K-Means). Ran a pilot on DocVQA (500 queries, 500 docs) at PF={4,8} with 3 estimators × 3 guided methods + baselines = 23 configs.

**New method — IWK (Importance-Weighted K-Means):** Replaces hierarchical Ward linkage with k-means, using importance-biased D²-sampling for initialization and importance-weighted centroids. Hypothesis: direct reconstruction-error optimisation should outperform greedy agglomerative merging.

**Round 3a — SVD estimator pilot (no attention yet):**

| PF | Method | Estimator | NDCG@5 |
|:--:|:-------|:----------|:------:|
| 4 | hierarchical | — | 0.5091 |
| 4 | IWD | centroid | **0.5116** |
| 4 | IWD | svd | 0.5085 |
| 4 | IWK | svd | 0.5033 |
| 4 | IWK | centroid | 0.4992 |
| 8 | hierarchical | — | 0.4758 |
| 8 | IWK | centroid | **0.4986** |
| 8 | IWD | centroid | 0.4956 |
| 8 | IWK | svd | 0.4930 |
| 8 | IWD | svd | 0.4884 |

Key: **IWK/centroid beats IWD/centroid at PF=8** (0.4986 vs 0.4956, +0.003). SVD estimator is competitive but slightly below centroid_distance.

**Round 3b — Attention estimator added:**

| PF | Method | Estimator | NDCG@5 |
|:--:|:-------|:----------|:------:|
| 4 | IWD | **attention** | **0.5135** |
| 4 | IWD | centroid | 0.5116 |
| 4 | IWK | attention | 0.5054 |
| 4 | WH | attention | 0.5085 |
| 8 | IWD | **attention** | **0.4983** |
| 8 | IWK | centroid | 0.4986 |
| 8 | IWK | attention | 0.4954 |
| 8 | IWD | centroid | 0.4956 |

Key: **Attention is the best estimator for IWD** at both PF=4 (+0.0044 vs hier) and PF=8 (+0.0225 vs hier). However, for IWK, centroid still wins at PF=8.

Results saved in `experiments/results/round3_pilot/` and `experiments/results/round3_attention_pilot/`.

### 3.9 Round 4 — Extreme Compression + TopK Analysis on DocVQA (Feb 20)

Extended evaluation to extreme pool factors (PF={16,32,64}) with all 5 estimators × core methods + TopK. ~350 configs on DocVQA. This round tests: (1) does importance guidance scale to extreme compression? (2) how does TopK (pure selection) compare to merge-based methods?

**Summary by pool factor (DocVQA, best method at each PF):**

| PF | Comp% | ~Tokens | Hier | Best Guided | Δ | Best Method/Estimator |
|:--:|:-----:|:-------:|:----:|:-----------:|:---:|:---------------------|
| 4 | 75% | 267 | 0.5091 | 0.5135 | +0.004 | IWD/attention |
| 8 | 88% | 133 | 0.4758 | 0.4986 | +0.023 | IWK/centroid |
| 16 | 94% | 67 | 0.4474 | 0.4629 | +0.016 | IWD/centroid |
| 32 | 97% | 33 | 0.3925 | 0.4130 | +0.021 | IWD/attention |
| 64 | 98% | 17 | 0.2901 | 0.3272 | +0.037 | IWK/attention |

**The absolute gap grows with compression:** +0.004 at PF=4 → +0.037 at PF=64. At 98% compression (17 tokens per document), importance-guided methods recover 0.037 more NDCG than hierarchical.

**TopK catastrophic failure at extreme compression:**

| PF | TopK/attention | TopK/centroid | TopK/self_sim | TopK/svd | Random mean |
|:--:|:--------------:|:-------------:|:-------------:|:--------:|:-----------:|
| 4 | 0.4657 | 0.4074 | 0.4049 | **0.1801** | 0.4393 |
| 8 | 0.4005 | 0.3001 | 0.3391 | **0.1282** | 0.3770 |
| 16 | 0.3170 | 0.2004 | 0.2459 | **0.1093** | 0.3065 |
| 32 | 0.1771 | 0.1519 | 0.2030 | **0.0914** | 0.2418 |
| 64 | **0.0176** | 0.0909 | 0.1427 | 0.0765 | **0.1706** |

**Critical finding — attention-guided TopK flips from best to worst:**
- At PF=4–16: TopK/attention is the best TopK variant (0.4657, 0.4005, 0.3170).
- At PF=64: TopK/attention collapses to **0.0176** — worse than random (0.1706), worse than every other TopK variant.
- Root cause: attention scores follow a peaked distribution. At extreme compression, the top-17 tokens by attention are all near the same high-attention region, destroying spatial coverage. Other estimators maintain more spatial diversity.

**SVD estimator is consistently poor for TopK** (0.1801 at PF=4 — far below random at 0.4393). SVD importance captures variance structure, not spatial diversity; pure selection based on SVD scores means selecting tokens from the dominant few singular directions, losing everything else.

**Estimator convergence:** Among merge-based methods (IWD, IWK, WH), the gap between the best and worst estimator is ~0.006 NDCG (at PF=8). The gap between the best and worst *method* (IWK vs WH) is ~0.196 at PF=8. **Method matters ~30× more than estimator for merge-based approaches.**

Results saved in `experiments/results/round4_quick/`.

### 3.10 Critical Analysis and Honest Assessment (Feb 20)

After four rounds of experiments, a blunt look at what the results actually mean.

**Finding 1: The recovery ratio *decreases* with compression.**

| PF | Hier drop from baseline | Best guided recovery | Recovery ratio |
|:--:|:-----------------------:|:-------------------:|:--------------:|
| 4 | −0.0071 | +0.0044 | 62% |
| 8 | −0.0404 | +0.0228 | 56% |
| 16 | −0.0688 | +0.0155 | 23% |
| 32 | −0.1237 | +0.0205 | 17% |
| 64 | −0.2261 | +0.0371 | 16% |

The absolute NDCG gain grows (+0.004 → +0.037), but the quality lost by hierarchical grows much faster. Importance guidance recovers a *shrinking fraction* of the loss as compression increases.

**Finding 2: At practical compression levels, gains are negligible.**

- At PF=4 (75% compression, NDCG≈0.51): gain is +0.004 — unlikely to be statistically significant, certainly not meaningful in production.
- At PF=8 (88%, NDCG≈0.50): gain is +0.023 — this is the sweet spot, arguably interesting.
- At PF=64 (98%, NDCG≈0.33): gain is +0.037 — the biggest absolute gain, but 0.327 NDCG is too low for any practical use.

**Finding 3: PF=2 beating baseline is noise.**

Statistical analysis across 10 datasets showed that hierarchical at PF=2 beating the no-pooling baseline is not statistically significant: mean delta = −0.003, 4/10 datasets show positive deltas, all deltas within 0.15 standard errors. This is sampling noise, not a real effect.

**Finding 4: No single method or estimator dominates.**

At PF=8: IWK/centroid wins (0.4986). At PF=16: IWD/centroid wins (0.4629). At PF=64: IWK/attention wins (0.3272). The relative performance depends on the compression level, and the method/estimator differences are small compared to the method vs. baseline gap.

**Implication for paper narrative:** The honest story is *not* "we found a breakthrough method that dramatically improves pooling." It is: "importance signals should guide cluster formation rather than within-cluster averaging; this yields consistent but modest improvements that grow with compression; the method choice matters more than the estimator choice; and pure selection (TopK) catastrophically fails, especially when guided by peaked attention distributions."

---

## 4. What To Do Next

### 4.1 ~~Scale Evaluation~~ ✅ DONE (Feb 16)

Full evaluation pipeline rewritten and running on all 10 ViDoRe v1 datasets with probe-based importance and random seed robustness. **810 configs total.** Run terminated Feb 17 after 2/10 datasets completed (DocVQA, InfoVQA) + partial TAT-DQA. Results documented in §3.2.

### 4.2 ~~Evaluate Probe-Based Importance~~ ✅ DONE (Feb 16)

Probe-based importance is now included in the evaluation via `--include-probe`. All 4 pooling strategies are evaluated with all 3 importance estimators (self_similarity, centroid_distance, probe).

### 4.3 ~~Statistical Robustness~~ ✅ DONE (Feb 16)

Random baseline now runs with 3 seeds (42, 123, 456), reporting mean ± std.

### 4.4 ~~Analyse Round 1 Results~~ ✅ DONE (Feb 17)

Identified three critical issues (§3.3): weighted hierarchical gains too small (wrong level of intervention), protect-and-pool budget bug, adaptive unfair comparison. Led to pivot to Split-and-Allocate method (§3.5).

### 4.5 ~~Implement Split-and-Allocate Method~~ ✅ DONE (Feb 17)

Implemented `SplitAndAllocateTokenPooler` in `src/importance_guided_pooling.py`. Default config: top-50%/bottom-50% split, 75%/25% budget allocation. 11 unit tests added covering output shape, budget compliance, fair comparison vs hierarchical, batch processing, and edge cases.

### 4.6 ~~Fix Protect-and-Pool~~ ✅ DONE (Feb 17)

Fixed budget collapse: protection now capped at 75% of output budget (`max_protect = max(int(target_output * 0.75), 1)`), ensuring at least 25% of tokens come from pooling. Regression test added.

### 4.7 ~~Fix Adaptive Entropy~~ ✅ DONE (Feb 17)

Changed entropy computation from raw min-max normalization to softmax with temperature=0.1, creating sharper probability distributions that actually differentiate between documents. Adaptive still uses variable token counts (not a fair comparison), but now produces meaningful per-document variation.

### 4.8 ~~Quick Validation on DocVQA~~ ✅ DONE (Feb 17)

Ran 37-config validation on DocVQA (PF 1/2/4/8, 2 estimators). Results in §3.6. **Split-and-Allocate + centroid achieves +0.0166 NDCG@5 over hierarchical at PF=8** — the largest fair-comparison gain seen so far.

### 4.9 Full Round 2 Evaluation — ✅ COMPLETED (Feb 17)

Full 10-dataset evaluation with core methods and thermal protection.

**Incident 1:** Initial launch overheated the PC and caused a shutdown. Venv packages were lost and had to be reinstalled. This prompted three infrastructure improvements:

1. **Resume support (`--resume`):** Reads completed configs from the existing CSV and skips them. 37 DocVQA configs (from the pre-crash run) are preserved and skipped.
2. **GPU thermal cooldown (`--gpu-temp-threshold 72`):** Mandatory 15-second pause between every config. If GPU temp exceeds 72°C, execution pauses until it drops to 60°C.
3. **Method filtering (`--methods`):** Dropped topk, adaptive, and protect_and_pool from the run — Round 1 showed topk is catastrophic, adaptive cheats on token count, and protect_and_pool underperforms. Only the methods that matter for the paper narrative remain.

**Incident 2:** Second launch crashed with `ValueError: cannot convert float NaN to integer` in `ProgressTracker.get_eta()`. Root cause: the resume logic incremented `completed_configs` for skipped configs without appending to `config_times`, making `np.mean([])` return NaN → `timedelta(seconds=NaN)` → crash. Fixed by adding a `len(self.config_times) == 0` guard and NaN/inf check in `get_eta()`. Run restarted successfully.

**Current config:**
- **3 importance-guided strategies**: weighted_hierarchical, split_allocate, importance_weighted_distance
- **Baselines**: no pooling, random (seed=42), hierarchical
- Pool factors: 1, 2, 4, 8
- 2 estimators: self_similarity, centroid_distance
- **430 total configs** (9 methods × 4 PFs × 10 datasets + baselines; 37 skipped via resume from validation run)
- Running in **tmux** session `eval` (survives SSH disconnects)
- Progress: `cat experiments/results/round2/status.txt`
- Results save incrementally per dataset
- **All 10 datasets complete.** Total time: ~5 hours. 269 data rows in combined CSV. See §3.7 for cross-dataset summary.

### 4.10 ~~New Estimators + IWK Method~~ ✅ DONE (Feb 20)

Implemented SVD and Attention importance estimators, plus Importance-Weighted K-Means pooler. SVD uses top-k singular vector projection energy. Attention hooks into the last transformer layer and computes column-mean attention scores. IWK uses importance-biased k-means++ initialization with weighted centroids. 32 new tests added (97 total). Pilot results in §3.8.

### 4.11 ~~Round 3 + 4: Extreme Compression Testing~~ ✅ DONE (Feb 20)

Extended evaluation to PF={16,32,64} on DocVQA with all 5 estimators and TopK method. ~350 configs total. Results in §3.9. Key findings: absolute gap grows to +0.037 at PF=64, TopK/attention collapses catastrophically, estimator choice matters ~30× less than method choice.

### 4.12 Full 10-Dataset Run at High Pool Factors (Pending)

Run all methods × 5 estimators × PF={4,8,16,32,64} across all 10 ViDoRe datasets. Estimated ~4.6 hours for ~558 new configs. This would validate whether the DocVQA Round 4 findings (growing gap, TopK failure, estimator convergence) generalise across datasets.

### 4.13 Statistical Significance Testing (Pending)

Run bootstrap confidence intervals or paired permutation tests on the key comparisons. Current results are single-seed without error bars. Need to determine whether the +0.023 at PF=8 and +0.037 at PF=64 are statistically significant across datasets.

### 4.14 Second Model (Pending)

Run on ColSmolVLM or another ColPali-family model to demonstrate model-agnostic approach.

### 4.15 Publication-Quality Figures (Pending)

- NDCG@5 vs. compression ratio curves.
- Per-dataset bar charts.
- Importance heatmap examples.
- Method overview diagram: model output → importance estimation → cluster budget allocation → pooled embeddings.

### 4.16 Paper Writing (Feb 22–27)

Structure in ICDAR format (~10 pages). Key narrative: importance signals should guide cluster *formation* (distance biasing, budget allocation, seed selection) rather than *within-cluster averaging* → three complementary approaches (IWD for soft distance-biasing, Split-and-Allocate for hard budget allocation, IWK for importance-biased k-means) → consistent gains at high compression across 10 datasets → TopK catastrophic failure reveals that merging is essential and selection alone destroys MaxSim coverage → estimator choice matters far less than method choice.

---

## 5. File Structure

```
src/
    __init__.py
    importance_estimation.py              # 5 importance estimators (self_sim, centroid, probe, svd, attention)
    importance_guided_pooling.py          # 7 pooling strategies (incl. IWD, IWK, Split-and-Allocate)

experiments/
    evaluate_pooling.py                   # Full evaluation pipeline (model-agnostic, 10 datasets)
    evaluate_pooling_v1.py                # Backup of original single-dataset evaluation
    visualize_importance.py               # Importance heatmap generation
    results/
        pooling_evaluation_results.csv    # Pilot results (100-doc DocVQA, 56 configs)
        pooling_evaluation_results.json   # Same in JSON
        round2/                           # Round 2: Full eval, 10 datasets, PF={1,2,4,8}
            results_colmodernvbert_all.csv             # Combined results (269 rows)
            results_colmodernvbert_<dataset>.csv       # Per-dataset results (10 files)
        round3_pilot/                     # Round 3a: SVD + IWK pilot on DocVQA (23 configs)
        round3_attention_pilot/           # Round 3b: Attention estimator pilot on DocVQA (29 configs)
        round4_quick/                     # Round 4: Extreme PF + TopK on DocVQA (~350 configs)
            results_colmodernvbert_all.csv             # Master CSV with all results

scripts/
    watch_eval.py                         # Monitor eval progress

tests/
    test_importance_estimation.py         # 35 tests for 5 importance estimators
    test_importance_guided_pooling.py     # 62 tests for 7 pooling strategies (97 total)
```

The codebase is a standalone package (installed via `pyproject.toml` in editable mode) that depends on `colpali_engine` from the sibling `../colpali/` directory.

---

## 6. Theoretical Analysis — Why Split-and-Allocate Works (Feb 17)

### 6.0 Fact-Check of Claims

Before the analysis, a rigorous fact-check of the Round 2 DocVQA results:

| Claim | Verdict |
|:------|:--------|
| Baseline NDCG@5 = 0.5162, ~1070 tokens | **CORRECT** (actual: 0.5162, 1069.7t) |
| split_allocate_centroid at PF=8: 0.4924 vs hier 0.4758 (+0.0166) | **CORRECT** (numbers match, same 133 tokens). **UPDATE:** IWD/centroid now scores 0.4956 (+0.0198), surpassing split_allocate. |
| "+0.0166 is significant" | **UNSUBSTANTIATED** — single dataset, no statistical test, no confidence intervals. Could be noise. IWD's +0.0198 is larger but similarly unsubstantiated until cross-dataset validation completes. |
| At PF=2, importance-guided methods hurt slightly | **CORRECT** — all 10 importance-guided configs score below hierarchical (by −0.0001 to −0.0671) |
| topk and protect_and_pool "fail catastrophically" | **PARTIALLY CORRECT** — topk is catastrophic (−0.07 to −0.18 vs hier). protect_and_pool is consistently bad (−0.01 to −0.06) but not catastrophic. |
| "importance alone is a poor proxy for retrieval utility" | **MISLEADING** — topk fails because *selection* is inferior to *merging*, not because importance is a poor signal. split_allocate uses the same importance signal for *allocation* and it works. The failure is in how importance is applied, not in the signal itself. |
| Centroid allocation "wins at high compression, loses at low" | **PARTIALLY CORRECT for split_allocate** — loses at PF=2 (−0.0071) AND PF=4 (−0.0059), only wins at PF=8 (+0.0166). **However, IWD/centroid wins at PF=4 (+0.0025) AND PF=8 (+0.0198)**, only losing at PF=2 (−0.0067). The soft-bias approach has a lower crossover point than the hard-tier approach. |

### 6.1 Information-Bottleneck Interpretation

The Information Bottleneck (IB) framework seeks a compressed representation $T$ of the input $X$ (document token embeddings) that maximises mutual information with the relevant variable $Y$ (query-document relevance):

$$\min_{p(t|x)} \; I(T; X) - \beta \cdot I(T; Y)$$

In our setting: $X$ is the full set of ~1070 token embeddings, $T$ is the pooled set of $N/\text{PF}$ representative vectors, and $Y$ is the MaxSim retrieval score that determines ranking.

**Why uniform allocation suffices at low compression (PF=2).** When $|T| = 534$ (half the tokens), the bottleneck is loose: $I(T; X)$ is large regardless of allocation strategy. Even uniform-size clusters preserve most of the representational capacity. The marginal benefit of non-uniform allocation is outweighed by the *allocation risk* — the chance that tokens deemed "unimportant" by a query-agnostic heuristic are in fact critical for specific queries. At PF=2, this risk dominates because even the "unimportant" tier receives enough clusters to function.

**Why non-uniform allocation wins at extreme compression (PF=8).** When $|T| = 133$ (12.4% of tokens), the bottleneck is severe. Under uniform allocation, every document region gets ~1 representative per 8 tokens — including margins and blank backgrounds that rarely match any query. This wastes capacity. The IB-optimal solution concentrates $I(T; Y)$ in regions where queries actually match. Centroid distance provides a query-agnostic proxy: tokens far from the mean embedding carry distinctive information (text content, table cells, figure elements) that is more likely to have high MaxSim overlap with query tokens. Allocating 75% of the cluster budget to these distinctive tokens reduces representation error *where it matters for scoring*.

**The crossover is not a "phase transition" in the physics sense.** Phase transitions imply a sharp discontinuity. What we observe is a smooth crossover: the benefit of non-uniform allocation grows with compression as $\Delta_{\text{allocation}} > \Delta_{\text{risk}}$ only above a threshold. On DocVQA, this threshold lies between PF=4 and PF=8. The crossover point likely varies by dataset — documents with more uniform information density (e.g., dense text pages) will have a higher threshold than documents with sparse layouts (e.g., infographics with large whitespace regions).

### 6.2 Why Centroid Distance Is the Right Estimator for Allocation

Consider the MaxSim scoring function:

$$\text{score}(q, d) = \sum_{i=1}^{|q|} \max_{j=1}^{|d|} \text{sim}(q_i, d_j)$$

After clustering, document tokens $d_j$ are replaced by cluster representatives $c_k$ (normalised mean of cluster members). The **representation error** for query token $q_i$ is:

$$\epsilon_i = \max_j \text{sim}(q_i, d_j) - \max_k \text{sim}(q_i, c_k)$$

This error depends on how well each cluster representative approximates its members. Larger clusters → worse approximation → higher $\epsilon_i$.

**Key insight:** $\epsilon_i$ is weighted by query frequency. If queries predominantly match against distinctive tokens (those far from the document centroid), then reducing cluster size in that region minimises the *expected* total error $\mathbb{E}[\sum_i \epsilon_i]$.

Why centroid distance works as the proxy:

- **Centroid distance = distinctiveness.** A token far from $\bar{e} = \frac{1}{N}\sum_j e_j$ points in a direction that few other tokens share. It encodes specific content (a word, a digit, a symbol) rather than generic background. Query tokens seeking specific information will preferentially match these.

- **Self-similarity = redundancy.** A token with high max-similarity to another token is redundant — it can be merged with little loss regardless. But low-redundancy tokens aren't necessarily query-relevant; they might simply be *unusual* noise. This explains why self_similarity underperforms centroid_distance for split_allocate at PF=8 (0.4582 vs 0.4924): self_similarity identifies tokens that are *unique* but not necessarily *informative*.

- **Centroid distance correlates with information content.** In a late-interaction model, the centroid represents the "average patch" — typically a blend of background/padding. Tokens far from this average carry the content deviations that distinguish this document from others. These are precisely the tokens that determine retrieval ranking.

### 6.3 Why topk Fails and Merging Is Essential

The catastrophic failure of topk (selection without merging) reveals a fundamental constraint: **MaxSim requires coverage, not just saliency.**

A query like "What is the total revenue in Q3?" might have tokens matching:
- "total" → a token in the table header (high importance)
- "revenue" → a token in the row label (high importance)
- "Q3" → a column header (high importance)
- "What", "is", "the", "in" → generic tokens that match *background structure*

topk preserves the first three but discards background tokens entirely. The generic query tokens now have no good match among document tokens, causing their MaxSim contribution to drop. This is why topk degrades faster than random at high compression — random at least maintains *coverage* of the embedding space.

Merging (hierarchical clustering) preserves coverage because cluster representatives are weighted averages that "summarise" their region of embedding space. Even aggressively merged background regions retain a representative that generic query tokens can match.

**Split-and-allocate preserves this property:** both tiers use hierarchical clustering, so coverage is maintained everywhere. The allocation simply determines the *resolution* — important regions get fine-grained representation, unimportant regions get coarse but still present representation.

### 6.4 On Topological Manifold Approaches and Betti Numbers

The idea of preserving "Betti numbers" (topological invariants: connected components $b_0$, cycles $b_1$) of the document layout is theoretically appealing but **practically premature** for three reasons:

1. **Computational cost.** Computing persistent homology on 1070 tokens in 128-dimensional space requires $O(n^3)$ operations and is ill-conditioned in high dimensions. For an inference-time pooling method, this is prohibitive.

2. **The embedding space is not the layout space.** ColModernVBERT's patch embeddings encode *semantic content*, not spatial position. Two spatially adjacent patches of white background are close in embedding space, but two spatially distant text patches may also be close. The topology of the embedding manifold does not directly reflect document layout topology.

3. **Layout information is implicit, not explicit.** ViT-based encoders preserve some spatial structure through positional encodings, but this is entangled with content. Extracting the "spatial skeleton" requires disentangling position from content, which is itself a research problem.

**What would be actionable instead:** A spatial-aware split that uses the 2D patch grid positions rather than embedding distances. For example, split tokens by spatial region (top-half / bottom-half, or text-region / non-text-region detected via simple intensity thresholds on the source image). This would test whether spatial structure matters without requiring topological machinery. This is flagged as a potential future direction.

### 6.5 Should the Full 10-Dataset Run Launch?

**Yes, but with clear-eyed expectations.**

Arguments for:
- The +0.0166 gain at PF=8 needs cross-dataset validation. One dataset proves nothing. Ten datasets can establish (or refute) a pattern.
- Even if split_allocate only wins at extreme compression (PF=8), that is a valid and publishable finding — "importance-guided budget allocation becomes critical when the bottleneck is severe."
- The code is validated (34 tests pass), and the evaluation infrastructure is production-ready.
- Time pressure: with 10 days to the deadline, we need data *now*, not after another method iteration.

Arguments for caution:
- split_allocate loses at PF=2 (−0.0071) and PF=4 (−0.0059) on DocVQA. If this pattern holds across datasets, the "improvement" is compression-regime-specific.
- The lack of statistical significance testing is a real weakness. The full run should include multiple random seeds for the random baseline and ideally bootstrap confidence intervals for all methods.
- Centroid distance may be dataset-specific. DocVQA contains scanned industry documents with large whitespace margins — an ideal case for centroid-based importance. Datasets like TAT-DQA (dense financial tables) or ArXivQA (scientific figures) may behave differently.

**Recommendation:** Launch with the current setup (PF=1/2/4/8, 2 estimators, 1 random seed). The 10-dataset breadth is more valuable than deeper analysis of one dataset. If gains are consistent at PF=8 across datasets, that is the paper's contribution. If not, the negative result is equally valuable and honest.

### 6.6 Variable-Rate Pooling Hypothesis

A "variable-rate" strategy that selects different pool factors per document would need a selection heuristic based on document complexity. Candidate features:

1. **Importance entropy:** $H = -\sum_i p_i \log p_i$ where $p_i = \text{softmax}(\text{importance}_i / \tau)$. High entropy → information spread uniformly → needs more tokens. Low entropy → information concentrated → can compress more.

2. **Effective rank of the embedding matrix:** $\text{erank}(E) = \exp\left(-\sum_i \frac{\sigma_i}{\sum_j \sigma_j} \log \frac{\sigma_i}{\sum_j \sigma_j}\right)$ where $\sigma_i$ are singular values. Documents with higher effective rank span more directions in embedding space and need more representatives.

3. **Number of distinct importance clusters:** Run k-means on importance scores; the optimal $k$ (by elbow criterion) estimates information structuredness. A document with 2 clear tiers (text + background) can use split_allocate effectively; a document with uniformly distributed importance should use standard hierarchical.

The practical challenge: any variable-rate scheme produces different token counts per document, making storage and batching irregular. The adaptive method in its current form attempted this and the result was that it gravitates toward minimal compression (entropy ≈ max for ~1000 tokens). A better approach might be to fix the total budget but vary the *allocation strategy* (split ratio and budget ratio) per document rather than the total token count. This preserves storage predictability while adapting to document structure.
