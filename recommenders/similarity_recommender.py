from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseRecommender, Recommendation, SupportSample
from .content_store import ContentStore
from .feature_encoder import TextFeatureEncoder


_SPLIT_RE = re.compile(r"[^\w가-힣]+", re.UNICODE)


class SimilarityRecommender(BaseRecommender):
    """Baseline vector similarity recommender with keyword fallback."""

    def __init__(
        self,
        meta_path: str | Path,
        embedding_path: str | Path,
        encoder_config: Optional[Dict[str, Any]] = None,
    ):
        self.store = ContentStore(meta_path, embedding_path)
        encoder_config = encoder_config or {}
        self.encoder = TextFeatureEncoder(
            model_name=encoder_config.get("model_name", "skt/kobert-base-v1"),
            embedding_dim=int(
                encoder_config.get("embedding_dim") or self.store.embedding_dim or 768
            ),
            local_files_only=bool(encoder_config.get("local_files_only", False)),
            allow_hash_fallback=bool(encoder_config.get("allow_hash_fallback", True)),
            backend=encoder_config.get("backend"),
        )

    def recommend(
        self,
        query: str,
        topk: int = 6,
        user_level: Optional[str] = None,
        task_id: Optional[str] = None,
        support_set: Optional[List[SupportSample]] = None,
        **kwargs: Any,
    ) -> List[Recommendation]:
        if not self.store.meta:
            return []

        scores = self._semantic_scores(query)
        source = "similarity"
        if scores is None:
            scores = self._keyword_scores(query)
            source = "similarity-keyword"

        idxs = np.argsort(-scores)[:topk]
        return self.store.pack(idxs.tolist(), scores, source=source)

    def _semantic_scores(self, query: str) -> Optional[np.ndarray]:
        if self.store.embeddings is None or not self.encoder.is_semantic:
            return None
        q = self.encoder.encode_one(query)
        if q.shape[0] != self.store.embedding_dim:
            return None
        return (self.store.embeddings @ q).astype(np.float32)

    def _keyword_scores(self, query: str) -> np.ndarray:
        return np.asarray(
            [self._keyword_score(query, item) for item in self.store.meta],
            dtype=np.float32,
        )

    def _keyword_score(self, query: str, item: Dict[str, Any]) -> float:
        q_tokens = set(self._tokenize(query))
        if not q_tokens:
            return 0.0
        title_tokens = set(self._tokenize(str(item.get("title", ""))))
        desc_tokens = set(self._tokenize(str(item.get("description", ""))))
        goal_tokens = set(self._tokenize(str(item.get("goal", ""))))
        return (
            len(q_tokens & title_tokens) * 1.5
            + len(q_tokens & desc_tokens)
            + len(q_tokens & goal_tokens) * 0.8
        )

    def _tokenize(self, text: str) -> List[str]:
        return [token for token in _SPLIT_RE.split((text or "").lower()) if token]
