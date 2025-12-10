# src/candidate_generation.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_topk_candidates(user_vec, news_emb, news_ids, k=200, exclude_set=None):
    sims = cosine_similarity([user_vec], news_emb)[0]
    order = sims.argsort()[::-1]
    res = []
    for idx in order:
        nid = news_ids[idx]
        if exclude_set and nid in exclude_set:
            continue
        res.append((nid, float(sims[idx]), idx))
        if len(res) >= k:
            break
    return res
