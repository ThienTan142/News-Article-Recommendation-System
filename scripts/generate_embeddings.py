import sys
import os

# Thêm thư mục Project vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import os

from src.embedding import NewsEmbedder


def main():
    input_path = "data/processed/news_clean.csv"
    output_emb_path = "data/interim/article_embeddings.npy"
    output_ids_path = "data/interim/article_ids.npy"

    print("[INFO] Loading cleaned news:", input_path)
    df = pd.read_csv(input_path)

    # Kiểm tra cột clean_text
    if "clean_text" not in df.columns:
        raise ValueError("[ERROR] Column 'clean_text' not found in news_clean.csv")

    texts = df["clean_text"].astype(str).tolist()
    article_ids = df["news_id"].tolist()

    print(f"[INFO] Total articles to embed: {len(texts)}")

    # Load model
    embedder = NewsEmbedder()

    # Generate embeddings
    embeddings = embedder.encode_news(texts)

    # Create output directory if missing
    os.makedirs("data/interim", exist_ok=True)

    # Save embeddings + IDs
    np.save(output_emb_path, embeddings)
    np.save(output_ids_path, np.array(article_ids))

    print("[INFO] Saved embeddings to:", output_emb_path)
    print("[INFO] Saved article IDs to:", output_ids_path)
    print("[DONE] Embedding generation completed!")


if __name__ == "__main__":
    main()

