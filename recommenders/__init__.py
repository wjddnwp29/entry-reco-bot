from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base import BaseRecommender, rerank_by_level
from .config import load_config
from .maml_recommender import MAMLRecommender
from .similarity_recommender import SimilarityRecommender


ROOT = Path(__file__).resolve().parent.parent
_CONFIG: Dict[str, Any] | None = None
_SIMILARITY: SimilarityRecommender | None = None
_MAML: MAMLRecommender | None = None


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def get_config() -> Dict[str, Any]:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config(ROOT / "configs" / "recommender_config.yaml")
    return _CONFIG


def get_similarity_recommender() -> SimilarityRecommender:
    global _SIMILARITY
    if _SIMILARITY is None:
        config = get_config()
        paths = config["paths"]
        _SIMILARITY = SimilarityRecommender(
            meta_path=_resolve(paths["meta_path"]),
            embedding_path=_resolve(paths["embedding_path"]),
            encoder_config=config.get("encoder", {}),
        )
    return _SIMILARITY


def get_maml_recommender() -> MAMLRecommender:
    global _MAML
    if _MAML is None:
        config = get_config()
        paths = config["paths"]
        _MAML = MAMLRecommender(
            meta_path=_resolve(paths["meta_path"]),
            embedding_path=_resolve(paths["embedding_path"]),
            checkpoint_path=_resolve(paths["checkpoint_path"]),
            logs_dir=_resolve(paths["logs_dir"]),
            encoder_config=config.get("encoder", {}),
            maml_config=config.get("maml", {}),
        )
    return _MAML


def get_recommender(recommender_type: str | None = None) -> BaseRecommender:
    config = get_config()
    selected = recommender_type or config["recommender"].get("type", "maml_meta")
    if selected == "similarity":
        return get_similarity_recommender()
    if selected == "maml_meta":
        return get_maml_recommender()
    raise ValueError(f"Unknown recommender type: {selected}")


__all__ = [
    "BaseRecommender",
    "MAMLRecommender",
    "SimilarityRecommender",
    "get_config",
    "get_recommender",
    "get_similarity_recommender",
    "get_maml_recommender",
    "rerank_by_level",
]
