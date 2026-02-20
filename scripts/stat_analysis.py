import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, bootstrap

# Load results
csv = '../experiments/results/round2/results_colmodernvbert_all.csv'
df = pd.read_csv(csv)

# Focus on DocVQA (main dataset in updates.md)
docvqa = df[df['dataset'] == 'docvqa_test_subsampled']

# Only compare methods with same token count (fair comparison)
def get_fair_methods(pool_factor):
    base = docvqa[(docvqa['method'] == 'hierarchical') & (docvqa['pool_factor'] == pool_factor)]
    if base.empty:
        return None
    base_tokens = base['avg_tokens'].iloc[0]
    fair = docvqa[(docvqa['pool_factor'] == pool_factor) & (np.abs(docvqa['avg_tokens'] - base_tokens) < 1e-2)]
    return fair

# For each pool factor, compare best importance-guided vs hierarchical
for pf in [2, 4, 8]:
    fair = get_fair_methods(pf)
    if fair is None:
        print(f"No hierarchical baseline for PF={pf}")
        continue
    hier = fair[fair['method'] == 'hierarchical']['ndcg@5'].values
    iwd = fair[(fair['method'] == 'importance_weighted_distance') & (fair['importance_method'] == 'centroid_distance')]['ndcg@5'].values
    sa = fair[(fair['method'] == 'split_allocate') & (fair['importance_method'] == 'centroid_distance')]['ndcg@5'].values
    wh = fair[(fair['method'] == 'weighted_hierarchical') & (fair['importance_method'] == 'centroid_distance')]['ndcg@5'].values
    print(f"\nPool factor {pf}:")
    print(f"  Hierarchical: {hier}")
    print(f"  IWD (centroid): {iwd}")
    print(f"  Split-Allocate (centroid): {sa}")
    print(f"  Weighted-Hier (centroid): {wh}")
    # Paired t-test (if multiple runs)
    if len(iwd) == len(hier) and len(iwd) > 1:
        t, p = ttest_rel(iwd, hier)
        print(f"  IWD vs Hierarchical: t={t:.3f}, p={p:.3g}")
    # Bootstrap CI
    if len(iwd) > 0 and len(hier) > 0:
        data = (iwd - hier) if len(iwd) == len(hier) else (iwd[0] - hier[0])
        if isinstance(data, np.ndarray):
            res = bootstrap((data,), np.mean, confidence_level=0.95, n_resamples=10000, method='basic')
            print(f"  IWD-Hier 95% CI: {res.confidence_interval}")
        else:
            print(f"  IWD-Hier diff: {data}")
