#!/usr/bin/env python3
"""
Recover results from the crashed Round 2 evaluation run.

Merges:
1. The 6 IWD DocVQA configs from the per-dataset CSV
2. The 8 InfoVQA configs parsed from eval.log

into the results_colmodernvbert_all.csv so that --resume can skip them.
"""

import pandas as pd
from pathlib import Path

ROUND2 = Path("experiments/results/round2")
ALL_CSV = ROUND2 / "results_colmodernvbert_all.csv"

# 1. Load existing all.csv (37 rows from validation run)
df_all = pd.read_csv(ALL_CSV)
print(f"Existing _all.csv: {len(df_all)} rows")

# 2. Load IWD DocVQA results from per-dataset CSV
ds_csv = ROUND2 / "results_colmodernvbert_docvqa_test_subsampled.csv"
df_docvqa_iwd = pd.read_csv(ds_csv)
print(f"Per-dataset DocVQA CSV: {len(df_docvqa_iwd)} rows (all IWD)")

# 3. Parse the 8 InfoVQA configs from eval.log
# Format: [HH:MM:SS]     [method] PF=N: NDCG@5=X  Recall@5=Y  tokens=Z  compression=C%  pool=Ps  score=Ss
infovqa_rows = [
    # none PF=1
    {"dataset": "infovqa_test_subsampled", "method": "none", "importance_method": "n/a",
     "pool_factor": 1, "ndcg@5": 0.8567, "recall@5": 0.9040, "avg_tokens": 723,
     "compression_ratio": 0.0, "pool_time_s": 0.0, "score_time_s": 82.0, "seed": None, "model": "colmodernvbert"},
    # hierarchical PF=2
    {"dataset": "infovqa_test_subsampled", "method": "hierarchical", "importance_method": "n/a",
     "pool_factor": 2, "ndcg@5": 0.8501, "recall@5": 0.8960, "avg_tokens": 361,
     "compression_ratio": 0.501, "pool_time_s": 121.7, "score_time_s": 105.4, "seed": None, "model": "colmodernvbert"},
    # weighted_hierarchical self_similarity PF=2
    {"dataset": "infovqa_test_subsampled", "method": "weighted_hierarchical", "importance_method": "self_similarity",
     "pool_factor": 2, "ndcg@5": 0.8497, "recall@5": 0.9000, "avg_tokens": 361,
     "compression_ratio": 0.501, "pool_time_s": 124.6, "score_time_s": 112.9, "seed": None, "model": "colmodernvbert"},
    # split_allocate self_similarity PF=2
    {"dataset": "infovqa_test_subsampled", "method": "split_allocate", "importance_method": "self_similarity",
     "pool_factor": 2, "ndcg@5": 0.8466, "recall@5": 0.8920, "avg_tokens": 361,
     "compression_ratio": 0.501, "pool_time_s": 91.3, "score_time_s": 88.7, "seed": None, "model": "colmodernvbert"},
    # importance_weighted_distance self_similarity PF=2
    {"dataset": "infovqa_test_subsampled", "method": "importance_weighted_distance", "importance_method": "self_similarity",
     "pool_factor": 2, "ndcg@5": 0.8520, "recall@5": 0.9000, "avg_tokens": 361,
     "compression_ratio": 0.501, "pool_time_s": 55.9, "score_time_s": 124.6, "seed": None, "model": "colmodernvbert"},
    # weighted_hierarchical centroid_distance PF=2
    {"dataset": "infovqa_test_subsampled", "method": "weighted_hierarchical", "importance_method": "centroid_distance",
     "pool_factor": 2, "ndcg@5": 0.8488, "recall@5": 0.8960, "avg_tokens": 361,
     "compression_ratio": 0.501, "pool_time_s": 157.4, "score_time_s": 116.8, "seed": None, "model": "colmodernvbert"},
    # split_allocate centroid_distance PF=2
    {"dataset": "infovqa_test_subsampled", "method": "split_allocate", "importance_method": "centroid_distance",
     "pool_factor": 2, "ndcg@5": 0.8490, "recall@5": 0.8980, "avg_tokens": 361,
     "compression_ratio": 0.501, "pool_time_s": 93.3, "score_time_s": 95.0, "seed": None, "model": "colmodernvbert"},
    # importance_weighted_distance centroid_distance PF=2
    {"dataset": "infovqa_test_subsampled", "method": "importance_weighted_distance", "importance_method": "centroid_distance",
     "pool_factor": 2, "ndcg@5": 0.8501, "recall@5": 0.9000, "avg_tokens": 361,
     "compression_ratio": 0.501, "pool_time_s": 50.2, "score_time_s": 127.1, "seed": None, "model": "colmodernvbert"},
]

df_infovqa = pd.DataFrame(infovqa_rows)
print(f"Recovered InfoVQA configs: {len(df_infovqa)} rows")

# 4. Merge all three sources, deduplicating by (dataset, method, importance_method, pool_factor)
df_merged = pd.concat([df_all, df_docvqa_iwd, df_infovqa], ignore_index=True)
dedup_cols = ["dataset", "method", "importance_method", "pool_factor"]
df_merged = df_merged.drop_duplicates(subset=dedup_cols, keep="last")
print(f"Merged total: {len(df_merged)} rows (after dedup)")

# 5. Backup original and save
backup = ALL_CSV.with_suffix(".csv.bak")
df_all.to_csv(backup, index=False)
print(f"Backed up original to: {backup}")

df_merged.to_csv(ALL_CSV, index=False)
print(f"Saved merged CSV to: {ALL_CSV}")

# 6. Verify
print("\nVerification - rows per dataset:")
print(df_merged.groupby("dataset").size().to_string())
print(f"\nTotal unique configs: {len(df_merged)}")
print("Done!")
