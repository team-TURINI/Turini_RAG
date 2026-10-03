"""서빙 기동 준비 — 검색 데이터가 없으면 HF 비공개 Dataset 에서 내려받는다.

코퍼스·FAISS 인덱스·국가 라벨은 git 에 없다(데이터는 공개 저장소에 안 올리는 팀 원칙).
HF Space 는 공개 저장소라 코드만 두고, 데이터는 **비공개 Dataset** 에 두고 기동 때 토큰으로 받는다.

환경변수
  HF_DATASET   예: "team-turini/rag-data"  (비어 있으면 내려받지 않고 로컬 파일만 확인)
  HF_TOKEN     비공개 Dataset 읽기 토큰 (Space 의 Secrets 에 등록)

Dataset 안의 경로는 이 저장소의 상대 경로와 같아야 한다:
  data/chunking_data/fixed_450_70/clean_chunks_450_70_v2.jsonl
  data/jurisdiction.json
  vectorstores/fixed_450_70_v2/index.faiss
  vectorstores/fixed_450_70_v2/index.pkl
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import config as cfg

log = logging.getLogger(__name__)


def required_files() -> list[Path]:
    return [cfg.FINAL_CORPUS, cfg.JURISDICTION_PATH,
            cfg.FINAL_INDEX_DIR / "index.faiss", cfg.FINAL_INDEX_DIR / "index.pkl"]


def missing_files() -> list[Path]:
    return [p for p in required_files() if not p.exists()]


def ensure_data() -> None:
    missing = missing_files()
    if not missing:
        return
    repo = os.getenv("HF_DATASET", "").strip()
    if not repo:
        lines = "\n".join(f"  - {p.relative_to(cfg.PROJECT_ROOT)}" for p in missing)
        raise RuntimeError(f"검색 데이터가 없고 HF_DATASET 도 설정되지 않았습니다:\n{lines}")
    from huggingface_hub import snapshot_download
    log.info("검색 데이터 내려받기: %s (%d개 없음)", repo, len(missing))
    snapshot_download(repo_id=repo, repo_type="dataset", token=os.getenv("HF_TOKEN") or None,
                      local_dir=str(cfg.PROJECT_ROOT),
                      allow_patterns=[str(p.relative_to(cfg.PROJECT_ROOT)).replace("\\", "/")
                                      for p in required_files()])
    still = missing_files()
    if still:
        lines = "\n".join(f"  - {p.relative_to(cfg.PROJECT_ROOT)}" for p in still)
        raise RuntimeError(f"Dataset 에 다음 파일이 없습니다:\n{lines}")
