# src/ctr_dataset.py
import pandas as pd
from src.preprocess import load_behaviors

def build_ctr_samples(news_meta, behaviors_path="data/MINDsmall/behaviors.tsv", neg_ratio=4, out_csv="data/precompute/ctr_dataset.csv"):
    df = load_behaviors(behaviors_path)
    rows = []
    for _, r in df.iterrows():
        uid = r["user_id"]
        impressions = r["impressions"]
        if not impressions: continue
        tokens = impressions.split()
        pos = [t.split("-")[0] for t in tokens if t.endswith("-1")]
        neg = [t.split("-")[0] for t in tokens if t.endswith("-0")]
        for p in pos:
            rows.append((uid, p, 1))
            # sample negatives
            sample_neg = neg[:neg_ratio] if len(neg) > 0 else []
            for n in sample_neg:
                rows.append((uid, n, 0))
    out = pd.DataFrame(rows, columns=["user_id","news_id","label"])
    valid_ids = set(news_meta["news_id"].astype(str).tolist())
    out = out[out["news_id"].isin(valid_ids)]
    out.to_csv(out_csv, index=False)
    print("Saved CTR samples ->", out_csv)
    return out
