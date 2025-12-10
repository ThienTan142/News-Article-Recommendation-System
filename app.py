# app.py
import streamlit as st
import numpy as np
import pandas as pd
from src.embedding import load_news_embeddings
from src.user_profile import load_user_history, build_user_vector_from_history
from src.candidate_generation import get_topk_candidates
from src.ranking import rank_candidates
from src.diversity import mmr_rerank

st.title("News Recommender — DL pipeline demo")

news_emb, news_meta = load_news_embeddings()
user_hist = load_user_history()

user_id = st.text_input("User ID (e.g. U8125):")
topk = st.slider("Top K", 5, 20, 10)

if st.button("Recommend"):
    if not user_id:
        st.error("Enter user_id")
    else:
        user_vec = build_user_vector_from_history(user_id, user_hist, news_emb, news_meta)
        if user_vec is None:
            st.info("Cold start — using mean news embedding")
            user_vec = news_emb.mean(axis=0)
            exclude = set()
        else:
            exclude = user_hist.get(user_id, set())

        cand = get_topk_candidates(user_vec, news_emb, news_meta["news_id"].astype(str).tolist(), k=200, exclude_set=exclude)
        try:
            ranked = rank_candidates(user_vec, cand, news_emb, news_meta)
        except Exception as e:
            st.warning(f"CTR rank not available: {e}. Using similarity.")
            ranked = [(nid, sim) for nid, sim, idx in cand]

        ids = [nid for nid,_ in ranked]
        idxs = [news_meta[news_meta["news_id"]==nid].index[0] for nid in ids]
        doc_embs = news_emb[idxs]
        mmr_out = mmr_rerank(user_vec, doc_embs, ids, top_k=topk)
        st.subheader(f"Top {topk}")
        for nid, score in mmr_out:
            row = news_meta[news_meta["news_id"]==nid].iloc[0]
            st.markdown(f"**{row['title'] if 'title' in row else nid}**")
            st.write(f"News ID: {nid} | score: {score:.4f}")
            if 'text' in row:
                st.write(row['text'][:300])
            st.write("---")
