"""KoBERT로 콘텐츠 임베딩을 재생성한다.

contents_meta.json의 각 콘텐츠 텍스트를 KoBERT([CLS] + L2 정규화)로 임베딩하여
ContentStore가 읽는 형식({"embeddings", "meta"})으로 contents_embeddings.pkl에 저장한다.

사용법:
    python scripts/embed_kobert.py
    python scripts/embed_kobert.py --config configs/recommender_config.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recommenders.config import load_config  # noqa: E402
from recommenders.feature_encoder import TextFeatureEncoder  # noqa: E402


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def candidate_text(item: dict) -> str:
    """ContentStore.candidate_text와 동일한 규칙."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate content embeddings with KoBERT.")
    parser.add_argument("--config", default="configs/recommender_config.yaml")
    args = parser.parse_args()

    config = load_config(resolve(args.config))
    paths = config["paths"]
    encoder_cfg = dict(config.get("encoder", {}))
    # 임베딩 재생성은 모델 다운로드가 필요하므로 로컬 전용 해제
    encoder_cfg["local_files_only"] = False

    meta_path = resolve(paths["meta_path"])
    embedding_path = resolve(paths["embedding_path"])

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, list):
        raise ValueError(f"Content meta must be a list: {meta_path}")

    # ContentStore.candidate_text와 동일한 규칙으로 임베딩 대상 텍스트 구성
    texts = [candidate_text(item) for item in meta]

    encoder = TextFeatureEncoder(
        model_name=encoder_cfg.get("model_name", "skt/kobert-base-v1"),
        embedding_dim=int(encoder_cfg.get("embedding_dim") or 768),
        local_files_only=False,
        allow_hash_fallback=bool(encoder_cfg.get("allow_hash_fallback", True)),
        backend=encoder_cfg.get("backend", "kobert"),
    )
    print(f"인코더 backend={encoder.backend}, dim={encoder.embedding_dim}, 콘텐츠={len(texts)}개")
    if encoder.backend != "kobert":
        print("경고: KoBERT 로드 실패. 폴백 백엔드로 임베딩합니다. 결과 차원을 확인하세요.")

    embeddings = encoder.encode(texts).astype(np.float32)
    print(f"임베딩 shape={embeddings.shape}")

    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    with embedding_path.open("wb") as f:
        pickle.dump({"embeddings": embeddings, "meta": meta}, f)
    print(f"저장 완료: {embedding_path}")


if __name__ == "__main__":
    main()
