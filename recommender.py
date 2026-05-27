from __future__ import annotations

from typing import Any, Dict, List, Optional

from recommenders import (
    get_config,
    get_maml_recommender,
    get_recommender,
    get_similarity_recommender,
    rerank_by_level,
)


def get_recommender_config() -> Dict[str, Any]:
    return get_config()


def recommend_by_similarity(
    query: str,
    topk: int = 6,
    user_level: Optional[str] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    results = get_similarity_recommender().recommend(query, topk=topk, **kwargs)
    return rerank_by_level(results, user_level)


def recommend_by_maml_meta(
    query: str,
    topk: int = 6,
    user_level: Optional[str] = None,
    user_id: Optional[str] = None,
    task_id: Optional[str] = None,
    support_set: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    results = get_maml_recommender().recommend(
        query,
        topk=topk,
        user_level=user_level,
        task_id=task_id,
        support_set=support_set,
        user_id=user_id,
        **kwargs,
    )
    return rerank_by_level(results, user_level)


def recommend_by_title_keywords(
    query: str,
    topk: int = 6,
    user_level: Optional[str] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Backward-compatible alias for the new MAML meta-ranker."""
    return recommend_by_maml_meta(query, topk=topk, user_level=user_level, **kwargs)


def recommend_selected(
    query: str,
    recommender_type: Optional[str] = None,
    topk: int = 6,
    user_level: Optional[str] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    selected = recommender_type or get_config()["recommender"].get("type", "maml_meta")
    results = get_recommender(selected).recommend(
        query, topk=topk, user_level=user_level, **kwargs
    )
    return rerank_by_level(results, user_level)


def compare_recommendations(
    query: str,
    topk: int = 6,
    user_level: Optional[str] = None,
    user_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "similarity": recommend_by_similarity(
            query, topk=topk, user_level=user_level, **kwargs
        ),
        "maml_meta": recommend_by_maml_meta(
            query,
            topk=topk,
            user_level=user_level,
            user_id=user_id,
            **kwargs,
        ),
    }
