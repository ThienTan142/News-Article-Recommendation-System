import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from time import time
from tqdm import tqdm


class NewsRecommender:
    def __init__(self, news_emb_path, user_emb_path, news_id_path, user_id_path):
        print("[INFO] Loading embeddings...")

        self.news_emb = np.load(news_emb_path)
        self.user_emb = np.load(user_emb_path)

        self.news_ids = pd.read_csv(news_id_path)["news_id"].tolist()
        self.user_ids = pd.read_csv(user_id_path)["user_id"].tolist()

        print("[INFO] Loaded all files successfully!")

    def recommend_topk(self, user_id, top_k=10, exclude_ids=None):
        try:
            user_index = self.user_ids.index(user_id)
        except ValueError:
            return []

        user_vec = self.user_emb[user_index]

        sims = cosine_similarity([user_vec], self.news_emb)[0]
        sorted_idx = sims.argsort()[::-1]

        results = []
        for i in sorted_idx:
            nid = self.news_ids[i]

            if exclude_ids and nid in exclude_ids:
                continue

            results.append(nid)

            if len(results) == top_k:
                break

        return results


def evaluate(validation_path, top_k=10):
    print("[INFO] Loading validation file...")
    df = pd.read_csv(validation_path)

    recommender = NewsRecommender(
        news_emb_path="data/embeddings/news_embeddings.npy",
        user_emb_path="data/embeddings/user_embeddings.npy",
        news_id_path="data/embeddings/news_ids.csv",
        user_id_path="data/embeddings/user_ids.csv"
    )

    print("[INFO] Preparing ground truth...")
    gt = (
        df[df["clicked"] == 1]
        .groupby("user_id")["news_id"]
        .apply(set)
        .to_dict()
    )

    users = list(gt.keys())
    total_users = len(users)
    print(f"[INFO] Total users with clicks in validation: {total_users}")

    total_p = 0
    total_r = 0
    total_f1 = 0

    start = time()

    # ---------- PROGRESS BAR HERE ----------
    for uid in tqdm(users, desc="Evaluating users", ncols=90):

        true_items = gt[uid]
        recs = recommender.recommend_topk(uid, top_k)

        if not recs:
            continue

        recs_set = set(recs)
        hit = len(recs_set & true_items)

        precision = hit / top_k
        recall = hit / len(true_items)
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0
        )

        total_p += precision
        total_r += recall
        total_f1 += f1

    avg_p = total_p / total_users
    avg_r = total_r / total_users
    avg_f1 = total_f1 / total_users

    print("\n====== EVALUATION RESULT ======")
    print(f"Precision@{top_k}: {avg_p:.4f}")
    print(f"Recall@{top_k}:    {avg_r:.4f}")
    print(f"F1@{top_k}:        {avg_f1:.4f}")
    print("================================")

    print(f"Time used: {time() - start:.2f} sec")


if __name__ == "__main__":
    evaluate("data/processed/interactions_val.csv", top_k=10)
