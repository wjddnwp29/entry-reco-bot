from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


MetaSample = Dict[str, Any]
MetaTask = Dict[str, Union[List[MetaSample], str]]


class MetaTaskDataset:
    """Builds support/query recommendation tasks for MAML."""

    def __init__(
        self,
        tasks_path: str | Path | None = None,
        contents_meta_path: str | Path = "dataset/built/contents_meta.json",
        support_size: int = 6,
        query_size: int = 6,
        negative_samples: int = 3,
        seed: int = 42,
    ):
        self.tasks_path = Path(tasks_path) if tasks_path else None
        self.contents_meta_path = Path(contents_meta_path)
        self.support_size = int(support_size)
        self.query_size = int(query_size)
        self.negative_samples = int(negative_samples)
        self.rng = random.Random(seed)

        self.tasks: List[MetaTask] = self._load_or_build_tasks()
        if not self.tasks:
            raise ValueError("No meta-learning tasks could be built.")

    def _load_or_build_tasks(self) -> List[MetaTask]:
        if self.tasks_path and self.tasks_path.exists():
            return self._load_tasks_file(self.tasks_path)
        return self._build_dummy_tasks()

    def _load_tasks_file(self, path: Path) -> List[MetaTask]:
        if path.suffix.lower() == ".jsonl":
            samples = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            return self._group_samples(samples)

        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "tasks" in data:
            return list(data["tasks"])
        if isinstance(data, dict) and "samples" in data:
            return self._group_samples(list(data["samples"]))
        if isinstance(data, list) and data and "support" in data[0]:
            return data
        if isinstance(data, list):
            return self._group_samples(data)
        raise ValueError(f"Unsupported meta task format: {path}")

    def _group_samples(self, samples: Iterable[MetaSample]) -> List[MetaTask]:
        grouped: Dict[str, List[MetaSample]] = defaultdict(list)
        for sample in samples:
            grouped[str(sample.get("task_id") or "default")].append(sample)

        tasks: List[MetaTask] = []
        for task_id, rows in grouped.items():
            if len(rows) < 2:
                continue
            shuffled = list(rows)
            self.rng.shuffle(shuffled)
            split = max(1, min(len(shuffled) - 1, self.support_size))
            support = self._fit_size(shuffled[:split], self.support_size)
            query = self._fit_size(shuffled[split:], self.query_size)
            tasks.append({"task_id": task_id, "support": support, "query": query})
        return tasks

    def _build_dummy_tasks(self) -> List[MetaTask]:
        if not self.contents_meta_path.exists():
            return []

        contents = json.loads(self.contents_meta_path.read_text(encoding="utf-8"))
        by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in contents:
            task_id = str(item.get("category") or item.get("difficulty") or "general")
            by_task[task_id].append(item)

        all_items = list(contents)
        tasks: List[MetaTask] = []
        for task_id, positives in by_task.items():
            samples: List[MetaSample] = []
            for item in positives:
                query = f"{item.get('title', '')} 만들고 싶어요"
                samples.append(self._make_sample(task_id, query, item, label=1))

                negatives = [
                    cand
                    for cand in self.rng.sample(all_items, min(len(all_items), 25))
                    if cand not in positives
                ][: self.negative_samples]
                for neg in negatives:
                    samples.append(self._make_sample(task_id, query, neg, label=0))

            self.rng.shuffle(samples)
            min_needed = self.support_size + self.query_size
            if len(samples) < min_needed:
                samples = self._fit_size(samples, min_needed)
            tasks.append(
                {
                    "task_id": task_id,
                    "support": self._fit_size(samples[: self.support_size], self.support_size),
                    "query": self._fit_size(
                        samples[self.support_size : min_needed], self.query_size
                    ),
                }
            )
        return tasks

    def _make_sample(
        self, task_id: str, query: str, item: Dict[str, Any], label: int
    ) -> MetaSample:
        candidate_id = str(item.get("content_id") or item.get("id") or item.get("title"))
        candidate_text = (
            item.get("search_text")
            or " ".join(
                [
                    str(item.get("title", "")),
                    str(item.get("description", "")),
                    str(item.get("goal", "")),
                ]
            )
        ).strip()
        return {
            "user_id": "dummy",
            "task_id": task_id,
            "query": query,
            "candidate_id": candidate_id,
            "candidate_text": candidate_text,
            "label": int(label),
        }

    def _fit_size(self, rows: List[MetaSample], size: int) -> List[MetaSample]:
        if not rows:
            return []
        fitted = list(rows)
        while len(fitted) < size:
            fitted.append(self.rng.choice(rows))
        return fitted[:size]

    def sample_tasks(self, count: int) -> List[MetaTask]:
        if count >= len(self.tasks):
            return [self._reshuffle_task(task) for task in self.rng.sample(self.tasks, len(self.tasks))]
        return [self._reshuffle_task(task) for task in self.rng.sample(self.tasks, count)]

    def _reshuffle_task(self, task: MetaTask) -> MetaTask:
        rows = list(task.get("support", [])) + list(task.get("query", []))
        self.rng.shuffle(rows)
        support = self._fit_size(rows[: self.support_size], self.support_size)
        query = self._fit_size(rows[self.support_size :], self.query_size)
        return {"task_id": str(task["task_id"]), "support": support, "query": query}

    def __len__(self) -> int:
        return len(self.tasks)
