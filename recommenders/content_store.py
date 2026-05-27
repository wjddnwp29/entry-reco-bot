from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


class ContentStore:
    def __init__(self, meta_path: str | Path, embedding_path: str | Path):
        self.meta_path = Path(meta_path)
        self.embedding_path = Path(embedding_path)
        self.meta: List[Dict[str, Any]] = self._load_meta()
        self.embeddings: Optional[np.ndarray] = self._load_embeddings()
        self._by_id = {
            str(item.get("content_id") or item.get("id") or idx): item
            for idx, item in enumerate(self.meta)
        }

    def _load_meta(self) -> List[Dict[str, Any]]:
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Missing content meta file: {self.meta_path}")
        data = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Content meta must be a list: {self.meta_path}")
        return data

    def _load_embeddings(self) -> Optional[np.ndarray]:
        if not self.embedding_path.exists():
            return None

        with self.embedding_path.open("rb") as f:
            obj = pickle.load(f)

        embeddings = None
        meta = None

        if isinstance(obj, dict) and "embeddings" in obj:
            embeddings = np.asarray(obj["embeddings"], dtype=np.float32)
            if isinstance(obj.get("meta"), list):
                meta = list(obj["meta"])
        elif isinstance(obj, dict):
            rows = []
            metas = []
            for key, value in obj.items():
                if isinstance(value, dict) and "embedding" in value:
                    rows.append(np.asarray(value["embedding"], dtype=np.float32))
                    item_meta = dict(value.get("meta", {}))
                    item_meta.setdefault("content_id", key)
                    metas.append(item_meta)
            if rows:
                embeddings = np.vstack(rows)
                meta = metas
        elif isinstance(obj, list):
            rows = []
            metas = []
            for idx, value in enumerate(obj):
                if isinstance(value, dict) and "embedding" in value:
                    rows.append(np.asarray(value["embedding"], dtype=np.float32))
                    item_meta = dict(value.get("meta", {}))
                    item_meta.setdefault("content_id", item_meta.get("id", str(idx)))
                    metas.append(item_meta)
            if rows:
                embeddings = np.vstack(rows)
                meta = metas
        elif isinstance(obj, np.ndarray):
            embeddings = obj.astype(np.float32)

        if embeddings is None:
            return None

        if meta is not None:
            self.meta = meta

        if len(self.meta) != len(embeddings):
            n = min(len(self.meta), len(embeddings))
            self.meta = self.meta[:n]
            embeddings = embeddings[:n]

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        return (embeddings / norms).astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        if self.embeddings is not None and self.embeddings.ndim == 2:
            return int(self.embeddings.shape[1])
        return 0

    def get(self, content_id: str | int | None) -> Optional[Dict[str, Any]]:
        if content_id is None:
            return None
        return self._by_id.get(str(content_id))

    def candidate_text(self, item: Dict[str, Any]) -> str:
        return (
            item.get("search_text")
            or " ".join(
                [
                    str(item.get("title", "")),
                    str(item.get("description", "")),
                    str(item.get("goal", "")),
                ]
            )
        ).strip()

    def iter_candidates(self) -> Iterable[tuple[int, Dict[str, Any]]]:
        return enumerate(self.meta)

    def pack(
        self, idxs: Iterable[int], scores: np.ndarray | List[float], source: str
    ) -> List[Dict[str, Any]]:
        packed: List[Dict[str, Any]] = []
        score_arr = np.asarray(scores, dtype=np.float32)
        for idx in idxs:
            item = dict(self.meta[int(idx)])
            content_id = str(item.get("content_id") or item.get("id") or idx)
            score = float(score_arr[int(idx)]) if int(idx) < len(score_arr) else 0.0
            packed.append(
                {
                    "id": content_id,
                    "content_id": content_id,
                    "title": item.get("title", item.get("question", "")),
                    "description": item.get("description", ""),
                    "goal": item.get("goal", ""),
                    "difficulty": item.get("difficulty", item.get("level", "")),
                    "url": item.get("url", ""),
                    "source": source,
                    "selected": False,
                    "score": score,
                }
            )
        return packed
