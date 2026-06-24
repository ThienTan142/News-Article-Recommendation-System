from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.artifacts import ProjectArtifactError, build_id_to_index, validate_news_artifacts
from src.candidate_generation import get_topk_candidates
from src.config import DEFAULT_CANDIDATE_K, DEFAULT_TOP_K
from src.diversity import mmr_rerank
from src.embedding import load_news_embeddings
from src.ranking import rank_candidates
from src.user_profile import build_user_vector_from_history, load_user_history


@dataclass(frozen=True)
class RecommenderAssets:
    news_emb: np.ndarray
    news_meta: pd.DataFrame
    user_history: dict[str, set[str]]
    id_to_index: dict[str, int]


@dataclass(frozen=True)
class Recommendation:
    news_id: str
    score: float
    title: Optional[str] = None
    text: Optional[str] = None
    category: Optional[str] = None


@dataclass(frozen=True)
class RecommendationResult:
    recommendations: list[Recommendation]
    ranking_source: str
    cold_start: bool
    fallback_reason: Optional[str] = None


def load_recommender_assets() -> RecommenderAssets:
    news_emb, news_meta = load_news_embeddings()
    validate_news_artifacts(news_emb, news_meta)
    news_meta = news_meta.copy()
    news_meta["news_id"] = news_meta["news_id"].astype(str)
    return RecommenderAssets(
        news_emb=news_emb,
        news_meta=news_meta,
        user_history=load_user_history(),
        id_to_index=build_id_to_index(news_meta),
    )


def recommend(
    user_id: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    assets: Optional[RecommenderAssets] = None,
    fallback_to_similarity: bool = True,
) -> RecommendationResult:
    if not user_id:
        raise ValueError("user_id is required")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if candidate_k < top_k:
        raise ValueError("candidate_k must be greater than or equal to top_k")

    assets = assets or load_recommender_assets()
    user_id = str(user_id)
    user_vec = build_user_vector_from_history(
        user_id,
        assets.user_history,
        assets.news_emb,
        assets.news_meta,
        id_to_index=assets.id_to_index,
    )
    cold_start = user_vec is None
    exclude = set()
    if cold_start:
        user_vec = assets.news_emb.mean(axis=0)
    else:
        exclude = {str(news_id) for news_id in assets.user_history.get(user_id, set())}

    news_ids = assets.news_meta["news_id"].astype(str).tolist()
    candidates = get_topk_candidates(
        user_vec,
        assets.news_emb,
        news_ids,
        k=candidate_k,
        exclude_set=exclude,
    )

    ranking_source = "ctr"
    fallback_reason = None
    try:
        ranked = rank_candidates(user_vec, candidates, assets.news_emb, assets.news_meta)
    except (ProjectArtifactError, FileNotFoundError, RuntimeError, ValueError) as exc:
        if not fallback_to_similarity:
            raise
        ranking_source = "similarity"
        fallback_reason = str(exc)
        ranked = [(news_id, score) for news_id, score, _ in candidates]

    if not ranked:
        return RecommendationResult([], ranking_source, cold_start, fallback_reason)

    ranked_ids = [str(news_id) for news_id, _ in ranked]
    ranked_scores = {str(news_id): float(score) for news_id, score in ranked}
    doc_indices = [assets.id_to_index[news_id] for news_id in ranked_ids]
    doc_embs = assets.news_emb[doc_indices]
    diverse = mmr_rerank(user_vec, doc_embs, ranked_ids, top_k=top_k)

    recommendations = []
    for news_id, _ in diverse:
        row = assets.news_meta.iloc[assets.id_to_index[str(news_id)]]
        recommendations.append(
            Recommendation(
                news_id=str(news_id),
                score=ranked_scores.get(str(news_id), 0.0),
                title=str(row["title"]) if "title" in row and pd.notna(row["title"]) else None,
                text=str(row["text"]) if "text" in row and pd.notna(row["text"]) else None,
                category=str(row["category"]) if "category" in row and pd.notna(row["category"]) else None,
            )
        )

    return RecommendationResult(recommendations, ranking_source, cold_start, fallback_reason)
