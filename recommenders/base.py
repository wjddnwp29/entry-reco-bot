from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


Recommendation = Dict[str, Any]
SupportSample = Dict[str, Any]


class BaseRecommender(ABC):
    @abstractmethod
    def recommend(
        self,
        query: str,
        topk: int = 6,
        user_level: Optional[str] = None,
        task_id: Optional[str] = None,
        support_set: Optional[List[SupportSample]] = None,
        **kwargs: Any,
    ) -> List[Recommendation]:
        """Return recommendation cards compatible with the existing Flask UI."""


def normalize_level(level: str | int | None) -> str:
    raw = str(level or "").strip()
    mapping = {
        "1": "Beginner",
        "2": "Intermediate",
        "3": "Advanced",
        "beginner": "Beginner",
        "intermediate": "Intermediate",
        "advanced": "Advanced",
        "초급": "Beginner",
        "중급": "Intermediate",
        "고급": "Advanced",
    }
    return mapping.get(raw.lower(), mapping.get(raw, raw))


def rerank_by_level(
    results: List[Recommendation], user_level: str | None
) -> List[Recommendation]:
    """Small level-aware reranker shared by both recommenders."""
    if not results:
        return results

    level = normalize_level(user_level)
    if not level:
        return sorted(results, key=lambda item: -float(item.get("score", 0.0)))

    neighbors = {
        ("Beginner", "Intermediate"),
        ("Intermediate", "Beginner"),
        ("Intermediate", "Advanced"),
        ("Advanced", "Intermediate"),
    }

    def bonus(item: Recommendation) -> float:
        item_level = normalize_level(item.get("difficulty"))
        if item_level == level:
            return 0.10
        if (level, item_level) in neighbors:
            return 0.04
        return -0.06

    reranked = []
    for result in results:
        item = dict(result)
        item["score"] = float(item.get("score", 0.0)) + bonus(item)
        reranked.append(item)
    return sorted(reranked, key=lambda item: -float(item["score"]))
