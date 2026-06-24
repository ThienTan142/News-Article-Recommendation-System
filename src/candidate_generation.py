import numpy as np

def get_topk_candidates(user_vec, news_emb, news_ids, k=200, exclude_set=None):
    if k <= 0:
        raise ValueError("k must be greater than 0")
    if news_emb.ndim != 2:
        raise ValueError("news_emb must be a 2D matrix")
    if len(news_ids) != news_emb.shape[0]:
        raise ValueError("news_ids length must match news_emb rows")

    user_vec = np.asarray(user_vec, dtype=float)
    if user_vec.ndim != 1 or user_vec.shape[0] != news_emb.shape[1]:
        raise ValueError("user_vec must be a 1D vector with the same dimension as news_emb")

    exclude_set = {str(news_id) for news_id in exclude_set} if exclude_set else set()
    denom = (np.linalg.norm(news_emb, axis=1) * np.linalg.norm(user_vec)) + 1e-12
    sims = news_emb.dot(user_vec) / denom
    order = sims.argsort()[::-1]
    res = []
    for idx in order:
        nid = str(news_ids[idx])
        if nid in exclude_set:
            continue
        res.append((nid, float(sims[idx]), idx))
        if len(res) >= k:
            break
    return res
