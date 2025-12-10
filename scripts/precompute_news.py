# scripts/precompute_news.py
import os
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from src.preprocess import load_news

NEWS_PATH = "data/MINDsmall/MINDsmall_train/news.tsv"
OUT_DIR = "data/precompute"
OUT_EMB = os.path.join(OUT_DIR, "news_embeddings.npy")
OUT_META = os.path.join(OUT_DIR, "news_metadata.csv")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading news...")
    df = load_news(NEWS_PATH)
    texts = df["text"].tolist()
    print("Loading SBERT model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    print("Encoding news...")
    emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # L2 normalize rows
    norms = (emb**2).sum(axis=1) ** 0.5
    emb = emb / norms[:, None]
    np.save(OUT_EMB, emb)
    df[["news_id","title","category","text"]].to_csv(OUT_META, index=False)
    print("Saved:", OUT_EMB, OUT_META)

if __name__ == "__main__":
    main()
