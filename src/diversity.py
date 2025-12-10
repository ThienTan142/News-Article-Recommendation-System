# src/diversity.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def mmr_rerank(query_emb, doc_embs, doc_ids, top_k=10, lambda_param=0.7):
    # normalize
    q = query_emb / (np.linalg.norm(query_emb) + 1e-12)
    docs = doc_embs.copy()
    docs = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-12)
    sim_q = docs.dot(q)
    selected = []
    selected_ids = []
    for _ in range(min(top_k, len(doc_ids))):
        scores = []
        for i in range(len(doc_ids)):
            if i in selected:
                scores.append(-1e9); continue
            if not selected:
                div = 0.0
            else:
                sims_selected = docs[i].dot(docs[selected].T)
                div = float(np.max(sims_selected))
            score = lambda_param * sim_q[i] - (1 - lambda_param) * div
            scores.append(score)
        best = int(np.argmax(scores))
        selected.append(best)
        selected_ids.append((doc_ids[best], float(sim_q[best])))
    return selected_ids
