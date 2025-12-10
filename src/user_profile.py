# src/user_profile.py
import pickle
import numpy as np

def load_user_history(path="data/precompute/user_history.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def build_user_vector_from_history(user_id, user_history, news_emb, news_meta):
    # news_meta must have news_id column aligned with news_emb rows
    if user_id not in user_history:
        return None
    reads = [nid for nid in user_history[user_id] if nid in news_meta["news_id"].values]
    if len(reads) == 0:
        return None
    id2idx = {nid: i for i, nid in enumerate(news_meta["news_id"].astype(str).tolist())}
    idxs = [id2idx[n] for n in reads]
    vec = news_emb[idxs].mean(axis=0)
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec
