# src/ranking.py
import torch
from sklearn.metrics.pairwise import cosine_similarity
from src.dl_model import CTR_MLP

def rank_candidates(user_vec, candidate_list, news_emb, news_meta, model_path="models/ctr_model.pt", device="cpu"):
    """
    candidate_list: [(news_id, sim, idx), ...]
    returns: list of (news_id, score)
    """
    # load model
    emb_dim = news_emb.shape[1]
    model = CTR_MLP(emb_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # prepare tensors
    user = torch.tensor(user_vec).float().unsqueeze(0)
    ids, item_vecs = [], []
    for nid, sim, idx in candidate_list:
        ids.append(nid)
        item_vecs.append(news_emb[idx])
    import numpy as np
    users = np.tile(user.numpy(), (len(item_vecs),1))
    items = np.stack(item_vecs)
    users_t = torch.tensor(users).float()
    items_t = torch.tensor(items).float()
    with torch.no_grad():
        probs = model(users_t, items_t).numpy().tolist()
    out = list(zip(ids, probs))
    out.sort(key=lambda x: x[1], reverse=True)
    return out
