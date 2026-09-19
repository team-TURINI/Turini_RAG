"""RAG 베이스라인 설정 — 실험 노브의 단일 진입점.

이 파일 하나만 바꾸면 파이프라인 동작이 바뀐다.
실험(k값·모델 스윕)은 여기 상수만 조정하고 스크립트를 재실행하면 된다.

환경변수 (.env 또는 OS env):
  - OPENAI_API_KEY (필수) — 임베딩 + 생성 LLM
  - COHERE_API_KEY (최종 파이프라인 필수) — Cohere 리랭커
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
COHERE_API_KEY = os.getenv("COHERE_API_KEY") or os.getenv("CO_API_KEY")   # 리랭커


# =============================================================================
# Multi-turn Query Understanding / Conversation Summary
# =============================================================================
QUERY_REWRITE_MODEL = os.getenv("QUERY_REWRITE_MODEL", "gpt-4.1-mini")
QUERY_REWRITE_REASONING_EFFORT = os.getenv("QUERY_REWRITE_REASONING_EFFORT") or None
QUERY_SUMMARY_MODEL = os.getenv("QUERY_SUMMARY_MODEL", "gpt-4.1-mini")


def _env_bool(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_TRACING = _env_bool("LANGSMITH_TRACING")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "turini-query-rewriter")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT") or None


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


# =============================================================================
# ★ 최종 파이프라인 확정 사양 (2026-09-19, 팀 회의 확정)
#
#   위의 베이스라인 상수(RETRIEVE_K=5, TOP_K_GEN=5, chunk_recursive_500_50 …)는
#   초기 Dense 단독 베이스라인용이며 그대로 둔다. 최종 파이프라인(core/pipeline.py
#   FinalPipeline)은 **이 섹션만** 읽는다. 실험 근거는 docs/final_pipeline_spec.md.
# =============================================================================

# ── ① 청킹 — 긴 일반 문서만 450/70, 짧은 문서는 통째로 (whole/faq/section/atomic)
#     이미 잘라둔 코퍼스를 그대로 쓴다. 재청킹하지 않는다.
FINAL_CORPUS = PROJECT_ROOT / "data" / "chunking_data" / "fixed_450_70" / "clean_chunks_450_70_v2.jsonl"
FINAL_INDEX_DIR = PROJECT_ROOT / "vectorstores" / "fixed_450_70_v2"      # FAISS (text-embedding-3-large)
FINAL_TESTSET = PROJECT_ROOT / "data" / "testset" / "rag_testset_retriever_v1v2_fixed450_team_eval_v2.json"
FINAL_SPLIT = PROJECT_ROOT / "data" / "gen_contexts" / "split.json"      # dev 107 / holdout 50

# ── ② Retriever — Dense + BM25, RRF 로 결합 (가중치 5:5 = unweighted)
FINAL_DENSE_K = 50                       # Dense 후보
FINAL_BM25_K = 50                        # BM25 후보
FINAL_BM25_TOKENIZER = "noun"            # Kiwi 명사·고유명사·영문·숫자·한자 (NNG/NNP/SL/SN/SH)
FINAL_BM25_K1 = 1.2
FINAL_BM25_B = 0.75
FINAL_RRF_K = 20                         # score = 1/(20+rank_dense) + 1/(20+rank_bm25)
FINAL_RRF_W_DENSE = 1.0                  # 5:5 → RRF 는 크기 불변이라 1:1 과 동일
FINAL_RRF_W_BM25 = 1.0
FINAL_HYBRID_POOL = 50                   # 각 리트리버에서 가져올 수

# ── ③ Reranker — Cohere
FINAL_RERANK_PROVIDER = "cohere"
FINAL_RERANK_MODEL = "rerank-v4.0-pro"
FINAL_CANDIDATE_K = 20                   # Hybrid 상위 20개만 리랭커에 넘김 (c20 ≈ c30, 후보 33% 절감)

# ── ④ Generator 에 넘길 컨텍스트
FINAL_TOP_K_GEN = 3                      # 리랭커 재정렬 Top3
FINAL_CTX_FIELD = "embedding_text"       # 제목 + 본문

# ── ⑤ 프롬프트 — V2_4 (v2 시스템 프롬프트 + 질문 뒤 "출력 전 확인" 블록)
FINAL_PROMPT_VERSION = "v2"
FINAL_INSTRUCTION_PLACEMENT = "check"    # core.generator.USER_TEMPLATES["check"]

# ── 생성 파라미터 (위 MODEL/TEMPERATURE/MAX_TOKENS/TOP_P 와 동일값, 명시적으로 고정)
FINAL_MODEL = "gpt-4.1-mini"
FINAL_TEMPERATURE = 0.1
FINAL_MAX_TOKENS = 768
FINAL_TOP_P = 0.9

# ── 판정
FINAL_JUDGE_MODEL = "gpt-4.1-2025-04-14"  # 생성 후보와 분리 (자기선호 편향 방지)
