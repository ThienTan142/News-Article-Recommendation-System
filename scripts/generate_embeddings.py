import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

def main():
    print("[INFO] Loading cleaned news...")
    df = pd.read_csv("data/processed/news_clean.csv")

    # Kiểm tra cột
    if "news_id" not in df.columns or "clean_text" not in df.columns:
        raise ValueError("File news_clean.csv phải có 2 cột: news_id và clean_text")

    # Convert clean_text về dạng list
    texts = df["clean_text"].astype(str).tolist()

    print(f"[INFO] Loaded {len(texts)} articles")

    # Load SBERT model
    print("[INFO] Loading model: all-MiniLM-L6-v2")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Encode toàn bộ bài báo (tạo embeddings)
    print("[INFO] Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print("[INFO] Embeddings shape:", embeddings.shape)

    # Tạo folder nếu chưa có
    os.makedirs("data/embeddings", exist_ok=True)

    # Lưu embeddings
    np.save("data/embeddings/news_embeddings.npy", embeddings)
    
    # Lưu lại news_id tương ứng
    df[["news_id"]].to_csv("data/embeddings/news_ids.csv", index=False)

    print("[DONE] Saved embeddings to data/embeddings/")


if __name__ == "__main__":
    main()
