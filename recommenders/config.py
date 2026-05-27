from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "recommender": {
        "type": "maml_meta",
        "debug_compare": True,
    },
    "paths": {
        "meta_path": "dataset/built/contents_meta.json",
        "embedding_path": "dataset/built/contents_embeddings.pkl",
        "checkpoint_path": "models/maml_ranker.pt",
        "logs_dir": "logs/user_logs",
        "meta_tasks_path": "dataset/built/meta_tasks.json",
    },
    "encoder": {
        "backend": "kobert",
        "model_name": "skt/kobert-base-v1",
        "embedding_dim": 768,
        "local_files_only": False,
        "allow_hash_fallback": True,
    },
    "maml": {
        "hidden_dim": 128,
        "inner_lr": 0.05,
        "outer_lr": 0.001,
        "inner_steps": 1,
        "support_size": 6,
        "query_size": 6,
        "tasks_per_batch": 4,
        "epochs": 5,
        "negative_samples": 3,
        "device": "cpu",
        "adapt_on_inference": True,
    },
}


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def _minimal_yaml_load(path: Path) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(-1, root)]

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if value == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)

    return root


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path = "configs/recommender_config.yaml") -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return deepcopy(DEFAULT_CONFIG)

    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        loaded = _minimal_yaml_load(config_path)

    return _deep_merge(DEFAULT_CONFIG, loaded)
