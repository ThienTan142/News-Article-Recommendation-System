# scripts/03_generate_embeddings.py
import pandas as pd
import numpy as np
import os
import json

from src.embedding import NewsEmbedder

DATA_DIR = "data/processed/"
OUTPUT_EMB = os.path.join(DATA_DIR, "news_embeddings.npy")
OUTPUT_MAP = os.path.join(DATA_DIR, "mappings.json")

def main():
    print("[INFO] Loading cleaned news...")
    news_path = os.path.join(DATA_DIR, "news_clean.csv")
    df = pd.read_csv(news_path)

    # news_clean.csv có cột: news_id, clean_text
    texts = df["clean_text"].tolist()
    news_ids = df["news_id"].tolist()

    # Load Sentence-BERT
    embedder = NewsEmbedder()

    # Generate embeddings
    embeddings = embedder.encode_news(texts)

    # Save embeddings as .npy
    np.save(OUTPUT_EMB, embeddings)
    print(f"[INFO] Saved embeddings → {OUTPUT_EMB}")

    # Save mapping news_id -> index
    mapping = {news_ids[i]: i for i in range(len(news_ids))}

    with open(OUTPUT_MAP, "w") as f:
        json.dump(mapping, f, indent=4)

    print(f"[INFO] Saved ID mapping → {OUTPUT_MAP}")
    print("[DONE] Embedding generation completed.")

if __name__ == "__main__":
    main()
