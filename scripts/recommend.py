import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class NewsRecommender:
    def __init__(self, news_emb_path, user_emb_path, news_id_path, user_id_path):
        print("Loading news embeddings...")
        self.news_emb = np.load(news_emb_path)

        print("Loading user embeddings...")
        self.user_emb = np.load(user_emb_path)

        print("Loading news IDs...")
        df_ids = pd.read_csv(news_id_path)
        self.news_ids = df_ids["news_id"].tolist()

        print("Loading user IDs...")
        df_uids = pd.read_csv(user_id_path)
        self.user_ids = df_uids["user_id"].tolist()

        assert len(self.news_emb) == len(self.news_ids), \
            "Số lượng news_emb và news_id không giống nhau!"

        assert len(self.user_emb) == len(self.user_ids), \
            "Số lượng user_emb và user_id không giống nhau!"

        # Map user_id -> index trong user_emb
        self.user_index = {uid: idx for idx, uid in enumerate(self.user_ids)}

        # Map news_id -> index trong news_emb
        self.news_index = {nid: idx for idx, nid in enumerate(self.news_ids)}

        print("All embeddings loaded successfully!")

    def recommend_for_user(self, user_id, top_k=10, exclude_ids=None):

        if user_id not in self.user_index:
            raise ValueError(f"User ID {user_id} không tồn tại trong user embeddings!")

        user_vec = self.user_emb[self.user_index[user_id]]

        sims = cosine_similarity([user_vec], self.news_emb)[0]

        sorted_idx = sims.argsort()[::-1]

        recommendations = []
        for idx in sorted_idx:
            nid = self.news_ids[idx]

            if exclude_ids and nid in exclude_ids:
                continue

            recommendations.append((nid, sims[idx]))

            if len(recommendations) == top_k:
                break

        return recommendations


if __name__ == "__main__":
    recommender = NewsRecommender(
        news_emb_path="data/embeddings/news_embeddings.npy",
        user_emb_path="data/embeddings/user_embeddings.npy",
        news_id_path="data/embeddings/news_ids.csv",
        user_id_path="data/embeddings/user_ids.csv"
    )

    print("\n=== GỢI Ý BÀI BÁO ===")
    user_id = input("Nhập user_id muốn gợi ý: ").strip()   # KHÔNG ép kiểu int

    results = recommender.recommend_for_user(user_id=user_id, top_k=10)

    print(f"\nTop 10 bài báo gợi ý cho user {user_id}:")
    for news_id, score in results:
        print(f"News ID: {news_id} | Similarity: {score:.4f}")
