import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PATHS, RANDOM_SEED
from src.preprocess import load_behaviors

def build_ctr_samples(
    news_meta,
    behaviors_path=PATHS.behaviors_path,
    neg_ratio=4,
    out_csv=PATHS.ctr_dataset_path,
    random_state=RANDOM_SEED,
):
    if neg_ratio < 0:
        raise ValueError("neg_ratio must be greater than or equal to 0")

    rng = np.random.default_rng(random_state)
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
            sample_size = min(neg_ratio, len(neg))
            sample_neg = rng.choice(neg, size=sample_size, replace=False).tolist() if sample_size else []
            for n in sample_neg:
                rows.append((uid, n, 0))
    out = pd.DataFrame(rows, columns=["user_id","news_id","label"])
    valid_ids = set(news_meta["news_id"].astype(str).tolist())
    out = out[out["news_id"].isin(valid_ids)]
    os.makedirs(Path(out_csv).parent, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(
        "Saved CTR samples ->",
        out_csv,
        "| positives:",
        int((out["label"] == 1).sum()),
        "| negatives:",
        int((out["label"] == 0).sum()),
    )
    return out
