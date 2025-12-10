# src/embeddings.py
import numpy as np
import pandas as pd

def load_news_embeddings(emb_path="data/precompute/news_embeddings.npy", meta_path="data/precompute/news_metadata.csv"):
    emb = np.load(emb_path)
    meta = pd.read_csv(meta_path)
    return emb, meta
