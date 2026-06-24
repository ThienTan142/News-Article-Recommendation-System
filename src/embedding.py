import numpy as np
import pandas as pd

from src.artifacts import ensure_file, validate_news_artifacts
from src.config import PATHS


def load_news_embeddings(
    emb_path=PATHS.news_embeddings_path,
    meta_path=PATHS.news_metadata_path,
):
    emb_file = ensure_file(
        emb_path,
        "news embeddings",
        "python scripts/precompute_news.py",
    )
    meta_file = ensure_file(
        meta_path,
        "news metadata",
        "python scripts/precompute_news.py",
    )
    emb = np.load(emb_file)
    meta = pd.read_csv(meta_file)
    validate_news_artifacts(emb, meta)
    meta["news_id"] = meta["news_id"].astype(str)
    return emb, meta
