from functools import lru_cache

import numpy as np
import torch

from src.artifacts import ensure_file
from src.config import DEVICE, PATHS
from src.dl_model import CTR_MLP


def _load_state_dict(model_path, device):
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location=device)


@lru_cache(maxsize=4)
def load_ranker(emb_dim, model_path=PATHS.ctr_model_path, device=DEVICE):
    model_file = ensure_file(
        model_path,
        "CTR model",
        "python -m src.train",
    )
    model = CTR_MLP(emb_dim)
    model.load_state_dict(_load_state_dict(model_file, device))
    model.to(device)
    model.eval()
    return model


def rank_candidates(
    user_vec,
    candidate_list,
    news_emb,
    news_meta,
    model_path=PATHS.ctr_model_path,
    device=DEVICE,
):
    """
    candidate_list: [(news_id, sim, idx), ...]
    returns: list of (news_id, score)
    """
    if not candidate_list:
        return []
    emb_dim = news_emb.shape[1]
    model = load_ranker(emb_dim, model_path=model_path, device=device)

    ids, item_vecs = [], []
    for nid, _sim, idx in candidate_list:
        ids.append(str(nid))
        item_vecs.append(news_emb[idx])

    users = np.repeat(np.asarray(user_vec, dtype="float32")[None, :], len(item_vecs), axis=0)
    items = np.stack(item_vecs)
    users_t = torch.tensor(users).float().to(device)
    items_t = torch.tensor(items).float().to(device)
    with torch.no_grad():
        probs = model(users_t, items_t).cpu().numpy().tolist()
    out = list(zip(ids, probs))
    out.sort(key=lambda x: x[1], reverse=True)
    return out
