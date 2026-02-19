# Explainer: What I've Done and Why

**For: PI meeting**
**Date: February 17, 2026**
**Deadline: ICDAR 2026, February 27**

---

## The Big Picture

The goal of this thesis is to make **visual document retrieval cheaper at scale** without sacrificing quality. Models like ColPali/ColModernVBERT encode each document page as ~1000 vectors (one per image patch) and match them against query tokens via MaxSim. This works well but means storing ~1000 × 128-dim vectors per page — expensive for millions of documents.

The existing compression method (hierarchical token pooling via agglomerative clustering) merges similar tokens into clusters. It **treats all patches equally** — a blank margin and a data-rich table cell get the same clustering budget. My thesis asks: **can query-agnostic interpretability signals guide this compression to be smarter?**

---

## Phase 0: The Reasoning Approach (Abandoned)

Before the current pooling work, I explored a different direction: **Reason-to-Retrieve (R2R)** — augmenting queries with chain-of-thought reasoning traces to help a small retriever (ColSmol, 256M params) match better.

### What I built (`/home/bs_thesis/Code/colsmol-reasoning/`)

- **Trace generator**: Used Qwen2.5-0.5B-Instruct to generate reasoning traces for queries (e.g., "I need to find a page with a financial table showing quarterly data...")
- **R2R training pipeline**: Fine-tuned ColSmol with LoRA on augmented queries (`query [SEP] reasoning_trace`)
- **Ablation controls**: Three modes — `use` (aligned trace), `none` (query only), `shuffle` (mismatched trace)
- **Full training**: Completed all three training runs (use/none/shuffle, seed=42)
- **Evaluation**: Ran on ViDoRe v1 and partial v2

### Why I abandoned it

**R2R made retrieval significantly worse.** The results on ViDoRe v1:

| Dataset | Baseline (none) | With Reasoning (use) | Delta |
|:--------|:---------------:|:--------------------:|:-----:|
| DocVQA | 55.7 | 43.1 | **-12.6** |
| InfoVQA | 82.5 | 71.9 | **-10.5** |
| ArXivQA | 72.0 | 57.2 | **-14.8** |
| TAT-DQA | 74.5 | 59.6 | **-14.9** |
| Healthcare | 95.1 | 76.4 | **-18.7** |
| TabFQuAD | 62.1 | 62.3 | +0.2 |
| **Average** | **77.7** | **~63** | **~-15** |

The reasoning traces added noise rather than useful grounding. The small reasoning model (0.5B) likely generated low-quality traces, and the extra tokens diluted the query signal in MaxSim scoring. Only TabFQuAD showed a negligible improvement.

On ViDoRe v2, results were mixed — a couple of datasets improved (Economics +8.3, ESG Reports +7.1) but these were partial results and the overall picture was negative.

### What I learned

1. **Query augmentation is risky for late interaction** — every extra token participates in MaxSim. Bad trace tokens actively hurt by creating spurious matches.
2. **A 0.5B reasoning model isn't good enough** — the traces weren't grounded in actual visual features.
3. **The query side isn't the bottleneck** — the retrieval problem is more about how documents are represented/compressed than about query understanding for these benchmarks.

This led me to pivot to the **document-side compression** problem, which is where the current work lives.

---

## Phase 1: Importance Estimation

### What I built

Three methods to estimate "how important is each patch?" without seeing any query:

1. **Centroid Distance**: How far is each patch from the document's mean embedding? Patches far from the average carry distinctive content (text, tables, figures). Formula: `importance(i) = 1 - cos(e_i, mean(e))`.

2. **Self-Similarity**: How redundant is each patch? If a patch has a near-twin elsewhere in the document, it's redundant. Formula: `importance(i) = 1 - max_{j≠i} cos(e_i, e_j)`.

3. **Probe-Based**: How well does each patch match semantic probe strings like "title", "table", "figure"? Uses the model's own text encoder to score alignment. Formula: `importance(i) = max_probe sim(e_i, probe)`.

### Why these three

- **Centroid distance** captures "distinctiveness" — the tokens that make this document different from a generic blank page. These are the tokens queries are most likely to match against.
- **Self-similarity** captures "redundancy" — if two tokens are nearly identical, you only need one.
- **Probes** capture "semantic content type" — explicitly asking "is this a table? a figure?"

All three are **query-agnostic** (computed at indexing time, not retrieval time) and **model-agnostic** (work on any ColPali-family model's embeddings).

### What I found

The correlation between query-agnostic importance and query-specific relevance (MaxSim) is low (r ≈ 0.01–0.12). This is expected — importance measures general informativeness, not "does this answer query X?" The key question is whether this general signal helps compression, even without knowing the query.

---

## Phase 2: First Pooling Strategies (Round 1)

### What I built

Four importance-guided pooling methods, all producing the same number of output tokens as standard hierarchical (fair comparison):

1. **Top-K Selection**: Keep the k most important tokens, discard the rest.
2. **Importance-Weighted Hierarchical**: Same clusters as standard hierarchical, but use importance-weighted mean instead of uniform mean within each cluster.
3. **Protect-and-Pool**: Keep top-25% tokens untouched, cluster the rest.
4. **Adaptive Pool Factor**: Vary compression per document based on importance entropy.

### What I found (Round 1 — DocVQA + InfoVQA, 500 queries each)

**Weighted hierarchical: gains are real but tiny (+0.001 to +0.004 NDCG@5).**

Why so small? Because hierarchical clustering already groups tokens that are very similar (cosine > 0.9 within a cluster). Taking a weighted vs. uniform average of near-identical vectors barely changes anything after L2 normalization. The importance signal is applied *inside clusters where it can't make a difference*. It needs to influence *which tokens share clusters*.

**Top-K: catastrophic at high compression (-0.07 to -0.18 vs hierarchical at PF=8).**

MaxSim needs *coverage*. If you drop all background tokens, generic query words ("what", "is", "the") have nothing to match against. Selection is fundamentally inferior to merging.

**Protect-and-Pool: collapsed to top-k at PF≥4.**

Bug: at pool factor 4 with 1070 tokens and 25% protection, the protected set (267 tokens) consumed the entire output budget (267 = 1070/4). Zero tokens left for pooling. Fixed later.

**Adaptive: looked good but cheated on token count.**

The entropy calculation always yielded near-maximum entropy for ~1000 tokens, causing effective pool factor ≈ 1. At nominal PF=8, adaptive was actually using 247 tokens while hierarchical used 133. Not a fair comparison.

### The key insight

**Importance should guide cluster _formation_, not within-cluster averaging.** The problem isn't how you summarize a cluster — it's how many clusters you allocate to important vs. unimportant regions.

---

## Phase 3: Split-and-Allocate + Bug Fixes (Round 2)

### What I built

**Split-and-Allocate**: Split tokens into two tiers by importance (top 50% / bottom 50%), then allocate 75% of the cluster budget to the important tier and 25% to the unimportant tier. Both tiers are clustered independently via hierarchical pooling.

```
Standard hierarchical at PF=4 (1070 → 267 clusters):
  All 1070 tokens → 267 clusters of ~4 tokens each

Split-and-Allocate at PF=4 (same 267 output tokens):
  Top 535 tokens → 200 clusters (effective PF ≈ 2.7)
  Bottom 535 tokens → 67 clusters (effective PF ≈ 8)
```

Important regions get 3× the representation fidelity. Unimportant regions get coarse but still present representation (coverage preserved).

Also fixed the bugs: protect-and-pool capped at 75% of output budget, adaptive uses softmax with temperature for sharper entropy.

### DocVQA validation results (500 queries, fair comparison)

| Pool Factor | Hierarchical | Split-and-Allocate (centroid) | Delta |
|:-----------:|:------------:|:-----------------------------:|:-----:|
| 2 (50% compression) | 0.5185 | 0.5114 | **-0.0071** |
| 4 (75% compression) | 0.5091 | 0.5032 | **-0.0059** |
| 8 (88% compression) | 0.4758 | 0.4924 | **+0.0166** |

**It works at high compression (PF=8) but hurts at low compression (PF=2, PF=4).**

### Why this pattern

At PF=2 (keeping 50% of tokens), the bottleneck is loose — there's enough capacity for everything. The hard tier boundary prevents natural cross-tier clustering and slightly hurts. At PF=8 (keeping 12.4% of tokens), the bottleneck is severe — uniform allocation wastes clusters on blank margins. Non-uniform allocation concentrates representation where queries actually match.

---

## Phase 4: Importance-Weighted Distance (Today)

### The problem with Split-and-Allocate

The hard tier boundary is artificial. A token at importance rank 535 (top of "unimportant" tier) is almost identical to one at rank 534 (bottom of "important" tier), but they end up in different tier clusters that can never merge. This hurts at low compression.

### What I built

**Importance-Weighted Distance Pooler**: Instead of hard splitting, *softly* bias the clustering. Before computing distances for hierarchical clustering, scale each embedding by importance:

```
e_i_scaled = e_i × (1 + α × normalized_importance_i)
```

Important tokens get larger magnitude → larger Euclidean distances → they resist merging and naturally end up in smaller clusters. No hard boundary. The scaling only affects the merge order — cluster representatives are computed from the **original** (unscaled) embeddings to preserve the embedding space for MaxSim scoring.

Default α = 1.0. Same output token count as standard hierarchical — fair comparison.

### Why this should work better

- No hard tier boundary → should avoid the PF=2/PF=4 regression
- Important tokens still get smaller clusters (more precision where it matters)
- Unimportant tokens still merge aggressively (efficient budget use)
- Smooth, not discrete — the effect scales continuously with importance

### Current status

**Running right now** on all 10 ViDoRe v1 datasets (430 configs, ~5h ETA). Results will tell us whether this fixes the low-compression problem that Split-and-Allocate had.

---

## The Full Method Lineup (Currently Evaluating)

| Method | How importance is used | Fair comparison? | Hypothesis |
|:-------|:----------------------|:----------------:|:-----------|
| hierarchical | Not used (uniform clustering) | Baseline | — |
| random | Not used (random subset) | Yes | Lower bound |
| topk | Keep top-k, discard rest | Yes | Selection alone insufficient |
| weighted_hier | Importance-weighted mean within clusters | Yes | Helps but small effect |
| protect_and_pool | Protect top tokens, cluster rest | Yes (fixed) | Hard boundary suboptimal |
| adaptive | Vary compression per document | **No** (variable tokens) | Per-document adaptation |
| **split_allocate** | **Hard split into tiers, unequal budget** | **Yes** | **Big gain at PF=8** |
| **importance_weighted_distance** | **Scale embeddings before clustering** | **Yes** | **Soft version, should work at all PFs** |

---

## What I Expect to Show in the Paper

### If importance_weighted_distance works across datasets:

**Main claim**: Query-agnostic importance signals can meaningfully improve token compression for late-interaction visual document retrieval, especially at high compression ratios (8×). The key insight is that importance must guide cluster *formation* (via distance scaling or budget allocation), not within-cluster averaging.

**Narrative arc**:
1. Tried the obvious approach (importance-weighted averaging) → marginal gains
2. Diagnosed why → importance is wasted inside already-similar clusters
3. Two principled fixes: hard allocation (Split-and-Allocate) and soft distance scaling (Importance-Weighted Distance)
4. Validated on 10 datasets with 2 importance estimators

### If results are mixed:

Still publishable as: "We systematically study where query-agnostic importance helps compression and where it doesn't. It helps at extreme compression (PF=8) but not at moderate compression (PF=2-4), consistent with an information bottleneck analysis."

---

## Key Numbers to Remember

| Metric | Value |
|:-------|:------|
| Model | ColModernVBERT (240M params) |
| GPU | RTX 3070 (8GB) |
| Tokens per document | ~1070 (128-dim each) |
| Benchmark | ViDoRe v1 (10 datasets, 3943 queries, 6729 documents) |
| Best gain so far | +0.0166 NDCG@5 at PF=8 (split_allocate + centroid, DocVQA) |
| Tests | 45 unit tests, all passing |
| Code | ~800 lines core (importance estimation + pooling), ~1200 lines eval pipeline |

---

## Potential PI Questions & Answers

**Q: Why not train a better model instead of post-hoc compression?**
A: This is a complementary approach. Training requires data, compute, and is model-specific. Our method is a drop-in post-processing step that works on any ColPali-family model's existing embeddings. No retraining needed.

**Q: Why is the gain only at PF=8?**
A: At low compression (PF=2), there's enough capacity for everything — importance doesn't matter much. At extreme compression (PF=8, keeping only 12.4% of tokens), every cluster matters, and wasting clusters on blank margins is costly. This is consistent with information bottleneck theory: non-uniform allocation only outweighs allocation risk when the bottleneck is severe.

**Q: How do you know +0.0166 NDCG@5 isn't noise?**
A: On a single dataset, I can't be sure. That's why we're running 10 datasets right now. If the gain is consistent across datasets, it's real. If it's only on DocVQA, we'll report that honestly.

**Q: Why did R2R fail?**
A: The 0.5B reasoning model generated low-quality traces that added noise to queries. In MaxSim, every query token participates in scoring — bad trace tokens create spurious matches that hurt ranking. The document-side compression problem turned out to be more tractable with our resources.

**Q: What about the reasoning module idea you suggested?**
A: A reasoning module inside the architecture would require modifying and retraining the model — a multi-month project beyond our 10-day deadline. Our importance estimation does something related (identifying which patches matter) but as a post-processing step that works on any existing model. We mention architectural reasoning as future work.

**Q: Why centroid distance over self-similarity?**
A: Centroid distance measures "distinctiveness" (how different from the average), while self-similarity measures "uniqueness" (how different from the nearest neighbor). For budget allocation, distinctiveness is the better proxy because distinctive tokens are more likely to be query-relevant (they carry content, not background). On DocVQA at PF=8: centroid = 0.4924 vs self_sim = 0.4582.

**Q: Is this contribution enough for ICDAR?**
A: If the 10-dataset results confirm the pattern: yes. The contribution is (1) a systematic study of where importance signals help compression, (2) the insight that importance must guide cluster formation not averaging, (3) two methods that operationalize this insight, (4) evaluation on the full ViDoRe v1 benchmark. ICDAR values applied document analysis work, and this is directly applicable to large-scale document retrieval systems.

**Q: What's the second model evaluation for?**
A: To demonstrate model-agnostic generalization. Plan is to run on ColSmolVLM after the ColModernVBERT results are in (Feb 19-20).

---

## Timeline to Deadline (Feb 27)

| Date | Task | Status |
|:-----|:-----|:------:|
| Feb 10-13 | R2R approach (reasoning) | ❌ Abandoned (negative results) |
| Feb 14-16 | Importance estimation + first pooling methods | ✅ Done |
| Feb 16-17 | Round 1 evaluation + analysis | ✅ Done |
| Feb 17 | Split-and-Allocate + Importance-Weighted Distance | ✅ Implemented |
| Feb 17 | Full 10-dataset Round 2 evaluation | ⏳ Running (~21:30 tonight) |
| Feb 18 | Analyze Round 2 results | |
| Feb 19-20 | Second model (ColSmolVLM) | |
| Feb 20-22 | Figures + visualizations | |
| Feb 22-27 | Paper writing | |
