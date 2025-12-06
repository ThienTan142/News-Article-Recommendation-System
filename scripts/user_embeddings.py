import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm

def main():
    start_time = time.time()
    print("[INFO] Loading data...")

    # Load train interactions
    train = pd.read_csv("data/processed/interactions_train.csv")

    # Load news embeddings
    news_emb = np.load("data/embeddings/news_embeddings.npy")
    news_ids = pd.read_csv("data/embeddings/news_ids.csv")["news_id"].tolist()

    # Create mapping news_id -> embedding index
    news_index = {nid: idx for idx, nid in enumerate(news_ids)}

    users = train["user_id"].unique()
    print(f"[INFO] Total users: {len(users)}")

    user_vectors = {}
    
    # Group interactions by user để tăng tốc
    user_groups = train.groupby("user_id")

    for user in tqdm(users, desc="Generating user embeddings"):
        user_data = user_groups.get_group(user)
        clicked_items = user_data[user_data["clicked"] == 1]["news_id"]

        valid_items = [nid for nid in clicked_items if nid in news_index]

        if not valid_items:
            # Không có bài báo nào → vector 0
            user_vectors[user] = np.zeros(news_emb.shape[1])
        else:
            # Stack embeddings nhanh hơn vòng for
            vectors = news_emb[[news_index[nid] for nid in valid_items]]
            user_vectors[user] = vectors.mean(axis=0)

    # Convert sang numpy để lưu
    user_ids = list(user_vectors.keys())
    user_matrix = np.vstack([user_vectors[uid] for uid in user_ids])

    # Save
    os.makedirs("data/embeddings", exist_ok=True)
    np.save("data/embeddings/user_embeddings.npy", user_matrix)

    pd.DataFrame({"user_id": user_ids}).to_csv(
        "data/embeddings/user_ids.csv", index=False
    )

    total_time = time.time() - start_time
    print(f"[DONE] User embeddings saved! Total time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
