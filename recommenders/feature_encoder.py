from __future__ import annotations

import hashlib
import re
from typing import Iterable, List, Optional

import numpy as np


_TOKEN_RE = re.compile(r"[\w가-힣]+", re.UNICODE)


class TextFeatureEncoder:
    """Frozen text encoder.

    Backends (선택 우선순위):
      1. ``kobert``              : transformers AutoModel ([CLS] + L2 정규화). 논문 기준.
      2. ``sentence-transformer``: sentence-transformers 모델.
      3. ``hash``                : 네트워크/모델 없이 동작하는 해시 임베딩 폴백.
    """

    def __init__(
        self,
        model_name: str,
        embedding_dim: int = 768,
        local_files_only: bool = False,
        allow_hash_fallback: bool = True,
        backend: Optional[str] = None,
        max_length: int = 256,
    ):
        self.model_name = model_name
        self.embedding_dim = int(embedding_dim)
        self.local_files_only = bool(local_files_only)
        self.allow_hash_fallback = bool(allow_hash_fallback)
        self.max_length = int(max_length)

        self._st_model = None      # sentence-transformers 모델
        self._hf_tok = None        # KoBERT 토크나이저
        self._hf_model = None      # KoBERT 모델
        self._torch = None
        self.backend = "hash"

        requested = (backend or "auto").lower()
        if requested in {"kobert", "bert", "transformers", "hf"}:
            self._try_load_kobert()
        elif requested in {"sentence-transformer", "sentence_transformers", "st"}:
            self._try_load_sentence_transformer()
        elif requested == "hash":
            pass
        else:  # auto: 모델 이름으로 추정
            if "kobert" in model_name.lower():
                self._try_load_kobert()
            else:
                self._try_load_sentence_transformer()

        if self.backend == "hash" and not self.allow_hash_fallback:
            raise RuntimeError(f"Could not load embedding model: {model_name}")

    @property
    def is_semantic(self) -> bool:
        return self.backend in {"kobert", "sentence-transformer"}

    # ----- 모델 로딩 -----
    def _try_load_kobert(self) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            kwargs = {"local_files_only": True} if self.local_files_only else {}
            self._hf_tok = AutoTokenizer.from_pretrained(
                self.model_name, use_fast=False, **kwargs
            )
            self._hf_model = AutoModel.from_pretrained(self.model_name, **kwargs).eval()
            self._torch = torch

            dim = int(getattr(self._hf_model.config, "hidden_size", 0) or 0)
            if dim > 0:
                self.embedding_dim = dim
            self.backend = "kobert"
        except Exception:
            self._hf_tok = None
            self._hf_model = None
            self._torch = None
            self.backend = "hash"

    def _try_load_sentence_transformer(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            kwargs = {"local_files_only": True} if self.local_files_only else {}
            self._st_model = SentenceTransformer(self.model_name, **kwargs)
            dim = int(self._st_model.get_sentence_embedding_dimension() or 0)
            if dim > 0:
                self.embedding_dim = dim
            self.backend = "sentence-transformer"
        except Exception:
            self._st_model = None
            self.backend = "hash"

    # ----- 인코딩 -----
    def encode(self, texts: Iterable[str]) -> np.ndarray:
        values = [text or "" for text in texts]
        if not values:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        if self.backend == "kobert":
            return self._encode_kobert(values)
        if self.backend == "sentence-transformer":
            embs = self._st_model.encode(
                values, convert_to_numpy=True, normalize_embeddings=True
            )
            return np.asarray(embs, dtype=np.float32)
        return np.vstack([self._hash_encode(text) for text in values]).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def _encode_kobert(self, values: List[str], batch_size: int = 32) -> np.ndarray:
        torch = self._torch
        chunks = []
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                batch_texts = values[start : start + batch_size]
                batch = self._hf_tok(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                # KoBERT 슬로우 토크나이저가 type_vocab_size(2)를 벗어난
                # token_type_ids를 반환하는 문제가 있어 제거(모두 0으로 처리).
                batch.pop("token_type_ids", None)
                output = self._hf_model(**batch)
                cls = output.last_hidden_state[:, 0]  # [CLS] 토큰
                cls = torch.nn.functional.normalize(cls, p=2, dim=1)
                chunks.append(cls.cpu().numpy().astype(np.float32))
        return np.vstack(chunks)

    # ----- 해시 폴백 -----
    def _hash_encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        for token in self._features(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.embedding_dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def _features(self, text: str) -> List[str]:
        lowered = (text or "").lower()
        tokens = _TOKEN_RE.findall(lowered)
        features = list(tokens)

        compact = "".join(tokens)
        for n in (2, 3):
            if len(compact) >= n:
                features.extend(compact[i : i + n] for i in range(len(compact) - n + 1))

        return features or ["<empty>"]


def lexical_features(query: str, candidate: str) -> np.ndarray:
    q_tokens = set(_TOKEN_RE.findall((query or "").lower()))
    c_tokens = set(_TOKEN_RE.findall((candidate or "").lower()))

    if q_tokens or c_tokens:
        overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens | c_tokens))
    else:
        overlap = 0.0

    q_len = max(1, len(query or ""))
    c_len = max(1, len(candidate or ""))
    length_ratio = min(q_len, c_len) / max(q_len, c_len)
    return np.asarray([overlap, length_ratio], dtype=np.float32)
