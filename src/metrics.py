# src/metrics.py
import numpy as np
from collections import Counter
from typing import List, Dict
import math

def hr_at_k(recommended: List[str], ground_truth: List[str], k=10):
    rec = recommended[:k]
    return 1.0 if any(r in ground_truth for r in rec) else 0.0

def ndcg_at_k(recommended: List[str], ground_truth: List[str], k=10):
    rec = recommended[:k]
    idcg = 1.0  # binary relevance: best is 1
    dcg = 0.0
    for i, r in enumerate(rec):
        rel = 1.0 if r in ground_truth else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 2)
    return dcg / idcg

def coverage(all_recommended: List[List[str]], catalog_size: int):
    unique = set([it for lst in all_recommended for it in lst])
    return len(unique) / float(catalog_size)

def catalog_coverage_list(all_recommended: List[List[str]]):
    unique = set([it for lst in all_recommended for it in lst])
    return unique

def mean_popularity_rank(novelty_inputs: List[List[str]], item_pop_rank: Dict[str,int]):
    # lower is more popular; novelty ~ average rank
    vals = []
    for lst in novelty_inputs:
        vals.extend([item_pop_rank.get(i, len(item_pop_rank)+1) for i in lst])
    return float(np.mean(vals)) if vals else None

def intra_list_diversity(all_recommended: List[List[str]], embeddings: Dict[str, np.ndarray]):
    # average pairwise distance per list, then mean
    import numpy as np
    out = []
    for lst in all_recommended:
        vecs = [embeddings.get(i) for i in lst if embeddings.get(i) is not None]
        if len(vecs) < 2:
            out.append(0.0)
            continue
        mat = np.stack(vecs)
        # cosine distances
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        matn = mat / (norms + 1e-9)
        sims = matn @ matn.T
        # upper triangle distances
        n = len(vecs)
        dists = []
        for i in range(n):
            for j in range(i+1, n):
                dists.append(1.0 - float(sims[i,j]))
        out.append(float(np.mean(dists)) if dists else 0.0)
    return float(np.mean(out))

# helper: compute item popularity from interactions (ratings list of (user,item))
def compute_item_popularity(interactions: List[tuple]):
    c = Counter([it for _, it in interactions])
    # rank: smaller number = more popular (1 = most popular)
    sorted_items = [it for it, _ in c.most_common()]
    rank = {it: i+1 for i, it in enumerate(sorted_items)}
    return rank
