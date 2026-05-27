from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from .base import BaseRecommender, Recommendation, SupportSample
from .content_store import ContentStore
from .feature_encoder import TextFeatureEncoder, lexical_features

try:
    from torch.func import functional_call
except Exception:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call  # type: ignore


class RankerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MAMLRecommender(BaseRecommender):
    """MAML meta-ranker over frozen query/candidate text features."""

    def __init__(
        self,
        meta_path: str | Path,
        embedding_path: str | Path,
        checkpoint_path: str | Path,
        logs_dir: str | Path = "logs/user_logs",
        encoder_config: Optional[Dict[str, Any]] = None,
        maml_config: Optional[Dict[str, Any]] = None,
        load_checkpoint: bool = True,
    ):
        encoder_config = encoder_config or {}
        maml_config = maml_config or {}

        self.store = ContentStore(meta_path, embedding_path)
        self.logs_dir = Path(logs_dir)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(maml_config.get("device", "cpu"))
        self.inner_lr = float(maml_config.get("inner_lr", 0.05))
        self.inner_steps = int(maml_config.get("inner_steps", 1))
        self.support_size = int(maml_config.get("support_size", 6))
        self.adapt_on_inference = bool(maml_config.get("adapt_on_inference", True))
        self._feature_cache: Dict[Tuple[str, str], np.ndarray] = {}

        self.encoder = TextFeatureEncoder(
            model_name=encoder_config.get("model_name", "skt/kobert-base-v1"),
            embedding_dim=int(encoder_config.get("embedding_dim") or 768),
            local_files_only=bool(encoder_config.get("local_files_only", False)),
            allow_hash_fallback=bool(encoder_config.get("allow_hash_fallback", True)),
            backend=encoder_config.get("backend"),
        )

        self.input_dim = self.encoder.embedding_dim * 4 + 2
        self.model = RankerMLP(
            input_dim=self.input_dim,
            hidden_dim=int(maml_config.get("hidden_dim", 128)),
        ).to(self.device)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loaded_checkpoint = False

        if load_checkpoint:
            self.load_checkpoint(self.checkpoint_path)

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

        user_id = kwargs.get("user_id")
        support = support_set
        if support is None and user_id:
            support = self.support_from_user_logs(str(user_id), fallback_query=query)
        support = support or []

        adapted_params = None
        if self.adapt_on_inference and support:
            adapted_params, _ = self.adapt(support)

        scores = self.score_all_candidates(query, adapted_params=adapted_params)
        idxs = np.argsort(-scores)[:topk]
        results = self.store.pack(idxs.tolist(), scores, source="maml_meta")
        for result in results:
            result["adapted"] = bool(adapted_params)
            result["checkpoint_loaded"] = self.loaded_checkpoint
        return results

    def score_all_candidates(
        self,
        query: str,
        adapted_params: Optional[OrderedDict[str, torch.Tensor]] = None,
    ) -> np.ndarray:
        samples = [
            {
                "query": query,
                "candidate_id": str(item.get("content_id") or item.get("id") or idx),
                "candidate_text": self.store.candidate_text(item),
                "label": 0,
            }
            for idx, item in self.store.iter_candidates()
        ]
        x, _ = self._samples_to_batch(samples)
        with torch.no_grad():
            logits = self._forward(x, adapted_params)
            return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

    def inner_update(
        self,
        support_set: List[SupportSample],
        create_graph: bool = True,
        initial_params: Optional[OrderedDict[str, torch.Tensor]] = None,
    ) -> Tuple[OrderedDict[str, torch.Tensor], torch.Tensor]:
        params = initial_params or OrderedDict(self.model.named_parameters())
        last_loss = torch.tensor(0.0, device=self.device)

        if not support_set:
            return OrderedDict((k, v) for k, v in params.items()), last_loss

        for _ in range(self.inner_steps):
            loss = self._loss(support_set, params)
            grads = torch.autograd.grad(
                loss,
                tuple(params.values()),
                create_graph=create_graph,
                retain_graph=create_graph,
            )
            params = OrderedDict(
                (name, param - self.inner_lr * grad)
                for (name, param), grad in zip(params.items(), grads)
            )
            last_loss = loss

        return params, last_loss

    def outer_update(
        self,
        tasks: List[Dict[str, Any]],
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        optimizer.zero_grad()
        query_losses = []
        support_losses = []

        for task in tasks:
            support = list(task.get("support", []))
            query = list(task.get("query", []))
            if not support or not query:
                continue

            adapted_params, support_loss = self.inner_update(
                support, create_graph=True
            )
            query_loss = self._loss(query, adapted_params)
            support_losses.append(support_loss.detach())
            query_losses.append(query_loss)

        if not query_losses:
            return {"meta_loss": 0.0, "support_loss": 0.0, "tasks": 0.0}

        meta_loss = torch.stack(query_losses).mean()
        meta_loss.backward()
        optimizer.step()

        support_loss = torch.stack(support_losses).mean() if support_losses else meta_loss
        return {
            "meta_loss": float(meta_loss.detach().cpu()),
            "support_loss": float(support_loss.detach().cpu()),
            "tasks": float(len(query_losses)),
        }

    def adapt(
        self, support_set: List[SupportSample]
    ) -> Tuple[OrderedDict[str, torch.Tensor], float]:
        adapted_params, support_loss = self.inner_update(
            support_set, create_graph=False
        )
        detached = OrderedDict((name, value.detach()) for name, value in adapted_params.items())
        return detached, float(support_loss.detach().cpu())

    def save_checkpoint(self, path: str | Path, extra: Optional[Dict[str, Any]] = None) -> None:
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": self.model.state_dict(),
            "input_dim": self.input_dim,
            "encoder_dim": self.encoder.embedding_dim,
            "extra": extra or {},
        }
        # 파이썬 파일 핸들로 저장(비-ASCII 경로에서도 안전).
        with open(ckpt_path, "wb") as f:
            torch.save(payload, f)

    def load_checkpoint(self, path: str | Path) -> bool:
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            return False
        with open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location=self.device, weights_only=False)
        if int(checkpoint.get("input_dim", self.input_dim)) != self.input_dim:
            return False
        self.model.load_state_dict(checkpoint["model_state"])
        self.loaded_checkpoint = True
        return True

    def support_from_user_logs(
        self, user_id: str, fallback_query: str = ""
    ) -> List[SupportSample]:
        log_path = self.logs_dir / f"{user_id}.json"
        if not log_path.exists():
            return []

        try:
            data = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        samples: List[SupportSample] = []
        last_query = fallback_query
        for session in data.get("sessions", []):
            if session.get("query"):
                last_query = str(session["query"])

            action = session.get("action")
            if isinstance(action, dict):
                kind = str(action.get("kind", "")).lower()
                content_id = action.get("content_id")
                label = self._label_from_action(kind)
                sample = self._sample_from_content(content_id, last_query, label)
                if sample:
                    samples.append(sample)

            event = str(session.get("event", "")).lower()
            if event in {"select", "feedback"}:
                content = session.get("content") or {}
                content_id = content.get("id") or content.get("content_id")
                label = 1 if event == "select" else self._label_from_action(session.get("feedback"))
                sample = self._sample_from_content(content_id, last_query, label)
                if sample:
                    samples.append(sample)

        return samples[-self.support_size :]

    def _label_from_action(self, action: Any) -> int:
        value = str(action or "").lower()
        return 0 if value == "dislike" else 1

    def _sample_from_content(
        self, content_id: Any, query: str, label: int
    ) -> Optional[SupportSample]:
        item = self.store.get(content_id)
        if item is None:
            return None
        return {
            "user_id": "log",
            "task_id": "user_feedback",
            "query": query,
            "candidate_id": str(content_id),
            "candidate_text": self.store.candidate_text(item),
            "label": int(label),
        }

    def _loss(
        self,
        samples: List[SupportSample],
        params: Optional[OrderedDict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x, y = self._samples_to_batch(samples)
        logits = self._forward(x, params)
        return self.loss_fn(logits, y)

    def _forward(
        self,
        x: torch.Tensor,
        params: Optional[OrderedDict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if params is None:
            return self.model(x)
        return functional_call(self.model, params, (x,))

    def _samples_to_batch(
        self, samples: Iterable[SupportSample]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = []
        labels = []
        for sample in samples:
            query = str(sample.get("query") or "")
            candidate = str(sample.get("candidate_text") or "")
            features.append(self._pair_features(query, candidate))
            labels.append(float(sample.get("label", 0)))

        x = torch.as_tensor(np.vstack(features), dtype=torch.float32, device=self.device)
        y = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        return x, y

    def _pair_features(self, query: str, candidate: str) -> np.ndarray:
        key = (query, candidate)
        cached = self._feature_cache.get(key)
        if cached is not None:
            return cached

        q = self.encoder.encode_one(query)
        c = self.encoder.encode_one(candidate)
        feature = np.concatenate(
            [q, c, np.abs(q - c), q * c, lexical_features(query, candidate)]
        ).astype(np.float32)
        self._feature_cache[key] = feature
        return feature
