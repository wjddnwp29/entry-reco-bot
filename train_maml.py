from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.meta_dataset import MetaTaskDataset
from recommenders.config import load_config
from recommenders.maml_recommender import MAMLRecommender


ROOT = Path(__file__).resolve().parent


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MAML meta-ranker.")
    parser.add_argument("--config", default="configs/recommender_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--tasks-per-batch", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tasks", default=None, help="Optional JSON/JSONL meta task file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(resolve(args.config))
    paths = config["paths"]
    maml_cfg = dict(config["maml"])

    if args.epochs is not None:
        maml_cfg["epochs"] = args.epochs
    if args.tasks_per_batch is not None:
        maml_cfg["tasks_per_batch"] = args.tasks_per_batch
    if args.checkpoint:
        paths["checkpoint_path"] = args.checkpoint

    tasks_path = args.tasks or paths.get("meta_tasks_path")
    resolved_tasks = resolve(tasks_path) if tasks_path else None
    if resolved_tasks and not resolved_tasks.exists():
        resolved_tasks = None

    dataset = MetaTaskDataset(
        tasks_path=resolved_tasks,
        contents_meta_path=resolve(paths["meta_path"]),
        support_size=int(maml_cfg["support_size"]),
        query_size=int(maml_cfg["query_size"]),
        negative_samples=int(maml_cfg["negative_samples"]),
    )

    recommender = MAMLRecommender(
        meta_path=resolve(paths["meta_path"]),
        embedding_path=resolve(paths["embedding_path"]),
        checkpoint_path=resolve(paths["checkpoint_path"]),
        logs_dir=resolve(paths["logs_dir"]),
        encoder_config=config.get("encoder", {}),
        maml_config=maml_cfg,
        load_checkpoint=False,
    )

    optimizer = torch.optim.Adam(
        recommender.model.parameters(), lr=float(maml_cfg["outer_lr"])
    )

    epochs = int(maml_cfg["epochs"])
    tasks_per_batch = int(maml_cfg["tasks_per_batch"])
    print(
        f"Training MAML ranker: tasks={len(dataset)}, epochs={epochs}, "
        f"tasks_per_batch={tasks_per_batch}, encoder={recommender.encoder.backend}"
    )

    last_metrics = {}
    for epoch in range(1, epochs + 1):
        tasks = dataset.sample_tasks(tasks_per_batch)
        last_metrics = recommender.outer_update(tasks, optimizer)
        print(
            f"epoch={epoch} meta_loss={last_metrics['meta_loss']:.4f} "
            f"support_loss={last_metrics['support_loss']:.4f} "
            f"tasks={int(last_metrics['tasks'])}"
        )

    checkpoint_path = resolve(paths["checkpoint_path"])
    recommender.save_checkpoint(
        checkpoint_path,
        extra={
            "epochs": epochs,
            "tasks_per_batch": tasks_per_batch,
            "last_metrics": last_metrics,
            "task_source": str(resolved_tasks or "dummy_from_contents_meta"),
        },
    )
    print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
