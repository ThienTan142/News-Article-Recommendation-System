# run_recommend_cli.py
import argparse
import numpy as np
from src.embedding import load_news_embeddings
from src.user_profile import load_user_history, build_user_vector_from_history
from src.candidate_generation import get_topk_candidates
from src.ranking import rank_candidates
from src.diversity import mmr_rerank

def recommend(user_id, topk=10, candidate_k=200):
    news_emb, news_meta = load_news_embeddings()
    user_hist = load_user_history()
    user_vec = build_user_vector_from_history(user_id, user_hist, news_emb, news_meta)
    if user_vec is None:
        print("Cold-start: using mean news embedding")
        user_vec = news_emb.mean(axis=0)
        exclude = set()
    else:
        exclude = user_hist.get(user_id, set())
    cand = get_topk_candidates(user_vec, news_emb, news_meta["news_id"].astype(str).tolist(), k=candidate_k, exclude_set=exclude)
    # rank by ctr model (if exists)
    try:
        ranked = rank_candidates(user_vec, cand, news_emb, news_meta)
    except Exception as e:
        print("CTR ranking failed, fallback to similarity:", e)
        ranked = [(nid, sim) for nid, sim, idx in cand]
    # mmr diversity
    ids = [nid for nid,score in ranked]
    idxs = [news_meta[news_meta["news_id"]==nid].index[0] for nid in ids]
    doc_embs = news_emb[idxs]
    mmr_out = mmr_rerank(user_vec, doc_embs, ids, top_k=topk)
    return mmr_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user","-u", required=True)
    parser.add_argument("--topk","-k", type=int, default=10)
    args = parser.parse_args()
    out = recommend(args.user, topk=args.topk)
    print("Top recommendations:")
    for nid, score in out:
        print(f"{nid} | score={score:.4f}")
