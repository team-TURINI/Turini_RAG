"""RAG 베이스라인 설정 — 실험 노브의 단일 진입점.

이 파일 하나만 바꾸면 파이프라인 동작이 바뀐다.
실험(k값·모델 스윕)은 여기 상수만 조정하고 스크립트를 재실행하면 된다.

환경변수 (.env 또는 OS env):
  - OPENAI_API_KEY (필수) — 임베딩 + 생성 LLM
  - INDEX_DIR      (선택) — 인덱스 디렉토리. 기본: ./vectorstores
  - CHUNK_SIZE / CHUNK_OVERLAP / SPLITTER (선택) — 청킹 스윕용 override

================================================================
실험 가능한 축:
  - CHUNK_SIZE / CHUNK_OVERLAP / SPLITTER : 청킹 (바꾸면 인덱스 재빌드 필수)
  - EMBEDDING_MODEL : 임베딩 모델 (바꾸면 인덱스 재빌드 필수)
  - RETRIEVE_K      : FAISS 에서 가져올 후보 수
  - TOP_K_GEN       : LLM 에 넘길 최종 문서 수 (RETRIEVE_K 이하)
  - MODEL           : 생성 LLM
  - TEMPERATURE / MAX_TOKENS / TOP_P : 생성 파라미터
================================================================
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 프로젝트 루트 = 이 파일이 있는 디렉토리
PROJECT_ROOT = Path(__file__).resolve().parent

# .env 로드 (OS 환경변수가 항상 우선)
load_dotenv(PROJECT_ROOT / ".env")


# =============================================================================
# 필수 키
# =============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# =============================================================================
# 청킹 (청크 사이즈 실험 축)
#   data/docs.jsonl 을 이 설정으로 잘라 인덱싱한다.
#   설정마다 인덱스가 다른 디렉토리에 저장되므로 스윕해도 서로 덮어쓰지 않는다.
#
#   스윕은 환경변수로도 가능 (config.py 를 안 고쳐도 됨):
#     CHUNK_SIZE=300 CHUNK_OVERLAP=30 python scripts/build_index.py
#
#   주의: docs.jsonl 은 scripts/prepare_docs.py 가 만든다. 규칙은
#         data/PREPARE_RULES.md 참조.
# =============================================================================
SPLITTER = os.getenv("SPLITTER", "recursive")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


# =============================================================================
# 인덱스 디렉토리
#   <PROJECT_ROOT>/vectorstores/portfolio/chunk_recursive_500_50/
#                                        └ 청킹 설정이 이름에 들어간다
# =============================================================================
INDEX_DIR = Path(os.getenv("INDEX_DIR", str(PROJECT_ROOT / "vectorstores")))
DOMAIN = "portfolio"
INDEX_NAME = os.getenv("INDEX_NAME", f"chunk_{SPLITTER}_{CHUNK_SIZE}_{CHUNK_OVERLAP}")


# =============================================================================
# 임베딩 모델
#   주의: FAISS 인덱스 빌드 시점과 런타임이 동일해야 함. 바꾸면 재빌드.
# =============================================================================
EMBEDDING_MODEL = "text-embedding-3-large"


# =============================================================================
# Retriever (FAISS Dense)
# =============================================================================
RETRIEVE_K = 5        # FAISS 에서 가져올 후보 수
TOP_K_GEN = 5         # LLM 에 넘기는 최종 문서 수 (RETRIEVE_K 이하)


# =============================================================================
# Generator (LLM)
# =============================================================================
MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.1
MAX_TOKENS = 768
TOP_P = 0.9
