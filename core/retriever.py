"""Dense 리트리버 — FAISS similarity search 단일 래퍼.

RAG 베이스라인의 검색 구성요소. 쿼리를 임베딩해 FAISS 에서 top-k 유사 문서를
가져온다. Hybrid(BM25)·rerank 같은 확장은 이 클래스 위에 얹으면 된다.

벡터스토어를 한 번만 로드해 인스턴스 수명 동안 재사용한다.
"""

from __future__ import annotations

from typing import Any, Dict, List

from config import RETRIEVE_K
from core.vectorstore import load_faiss_vectorstore


class DenseRetriever:
    """FAISS 기반 dense 리트리버 (single instance)."""

    def __init__(self) -> None:
        self._vs = load_faiss_vectorstore("chunk_record")

    def search(self, query: str, k: int = RETRIEVE_K) -> List[Dict[str, Any]]:
        """query 로 top-k 문서 검색.

        Returns:
            각 항목 = {
                "doc_id":       str,   # 표시·평가용 식별자
                "page_content": str,   # 문서 본문
                "metadata":     dict,  # 원본 메타데이터
                "score":        float, # L2 거리 (낮을수록 가까움)
                "rank":         int,   # 1-based 순위
            }
        """
        pairs = self._vs.similarity_search_with_score(query, k=k)
        results: List[Dict[str, Any]] = []
        for rank, (doc, dist) in enumerate(pairs, start=1):
            meta = dict(doc.metadata or {})
            doc_id = meta.get("chunk_id") or meta.get("doc_id") or doc.page_content[:80]
            results.append(
                {
                    "doc_id": doc_id,
                    "page_content": doc.page_content,
                    "metadata": meta,
                    "score": float(dist),
                    "rank": rank,
                }
            )
        return results
