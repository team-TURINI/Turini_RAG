"""RAG 파이프라인 — 한 질문을 처리하는 단일 진입점.

베이스라인 흐름:
  ① retrieve  — FAISS dense 검색 (top RETRIEVE_K)
  ② generate  — 상위 TOP_K_GEN 문서를 근거로 LLM 답변

리랭커·하이브리드 검색 등 확장은 ① 과 ② 사이에 단계를 끼워 넣으면 된다.
리트리버는 한 번만 로드해 인스턴스 수명 동안 재사용한다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from config import TOP_K_GEN
from core.generator import GenerationResult, generate
from core.retriever import DenseRetriever


class RAGPipeline:
    """리트리버 + 제너레이터를 묶은 최소 RAG 파이프라인."""

    def __init__(self) -> None:
        self.retriever = DenseRetriever()

    def run(self, question: str) -> Tuple[str, List[str], Dict[str, Any]]:
        """한 질문 처리.

        Returns:
            (answer, doc_ids, meta)
              - answer  : 생성된 답변
              - doc_ids : LLM 에 넘긴 문서들의 doc_id 리스트
              - meta    : 디버그/평가용 메타데이터
        """
        # ① retrieve
        candidates = self.retriever.search(question)
        top_k = candidates[:TOP_K_GEN]

        contexts = [c["page_content"] for c in top_k]
        doc_ids = [c["doc_id"] for c in top_k]

        # ② generate
        result: GenerationResult = generate(question, contexts)

        meta = {
            "n_candidates": len(candidates),
            "top_k_doc_ids": doc_ids,
            "latency_s": round(result.latency, 3),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }
        return result.answer, doc_ids, meta
