import json
import pickle
import warnings
from pathlib import Path

import numpy as np

from src.artifacts import InvalidArtifactError, MissingArtifactError, build_id_to_index
from src.config import PATHS


def _coerce_history(raw_history):
    if not isinstance(raw_history, dict):
        raise InvalidArtifactError("user history must be a mapping of user_id to news ids")

    history = {}
    for user_id, news_ids in raw_history.items():
        if news_ids is None:
            history[str(user_id)] = set()
            continue
        history[str(user_id)] = {str(news_id) for news_id in news_ids}
    return history


def load_user_history(
    path=PATHS.user_history_path,
    legacy_pickle_path=PATHS.legacy_user_history_pickle_path,
    allow_pickle=True,
):
    history_path = Path(path)
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            return _coerce_history(json.load(f))

    pickle_path = Path(legacy_pickle_path)
    if allow_pickle and pickle_path.exists():
        warnings.warn(
            "Loading legacy pickle user history. Only use pickle artifacts from trusted sources.",
            RuntimeWarning,
            stacklevel=2,
        )
        with open(pickle_path, "rb") as f:
            return _coerce_history(pickle.load(f))

    raise MissingArtifactError(
        "Missing user history artifact: "
        f"{history_path}. Run python scripts/precompute_user_history.py first."
    )


def build_user_vector_from_history(
    user_id,
    user_history,
    news_emb,
    news_meta,
    id_to_index=None,
):
    if user_id not in user_history:
        return None
    id_to_index = id_to_index or build_id_to_index(news_meta)
    reads = [str(nid) for nid in user_history[user_id] if str(nid) in id_to_index]
    if len(reads) == 0:
        return None
    idxs = [id_to_index[nid] for nid in reads]
    vec = news_emb[idxs].mean(axis=0)
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec
