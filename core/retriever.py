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
        self._vs = load_faiss_vectorstore()  # config.INDEX_NAME = 현재 청킹 설정

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


# =============================================================================
# 최종 파이프라인용 리트리버 3종
#
#   아래 세 클래스는 scripts/run_dense_baseline.py · run_bm25.py · run_hybrid.py 의
#   검색 로직을 **그대로 옮긴 것**이다. 동점 처리 규칙까지 같아야 오프라인 run 파일과
#   순위 단위로 대조할 수 있다(docs/final_pipeline_spec.md 2단계 회귀 검증).
#   새로 짜지 않는 이유: BM25 토크나이저 태그 집합이 구현마다 달라 성능이
#   0.757 vs 0.846 으로 갈렸던 전례가 있다.
#
#   공통 반환: 순위 순서의 chunk_id 리스트. 본문·메타는 corpus dict 에서 꺼낸다.
# =============================================================================
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union


def load_corpus(path: Union[str, Path]) -> Dict[str, dict]:
    """chunk_id → 청크 dict. 검색·리랭킹·생성이 같은 객체를 공유한다."""
    with open(path, "r", encoding="utf-8") as f:
        return {c["chunk_id"]: c for c in (json.loads(l) for l in f if l.strip())}


class DenseIndexRetriever:
    """FAISS 인덱스를 **경로로 직접** 열어 검색한다 (run_dense_baseline.py 와 동일).

    동점 처리: L2 거리 오름차순, 같으면 chunk_id 오름차순.
    """

    def __init__(self, index_dir: Union[str, Path]) -> None:
        from core.vectorstore import load_faiss_from_dir
        self._vs = load_faiss_from_dir(index_dir)

    def search(self, query: str, k: int) -> List[str]:
        pairs = self._vs.similarity_search_with_score(query, k=k)
        ranked = sorted(((d.metadata.get("chunk_id"), float(s)) for d, s in pairs),
                        key=lambda x: (x[1], x[0]))
        return [cid for cid, _ in ranked]


class BM25Retriever:
    """Kiwi 형태소 분석 + BM25Okapi (run_bm25.py --tokenizer noun 과 동일).

    토크나이저 `noun` = {NNG, NNP, SL, SN, SH}
      명사·고유명사에 **영문(SL)·숫자(SN)·한자(SH)** 를 포함한다. ETF·IPO·PER 같은
      영문 약어가 이 도메인의 최고 IDF 토큰이라 빼면 −9%p 다. 의존명사(NNB)·수사(NR)·
      대명사(NP) 는 IDF 가 0 에 수렴해 뺀다.
    동점 처리: 점수 내림차순, 같으면 chunk_id 오름차순.
    """

    NOUN_TAGS = frozenset({"NNG", "NNP", "SL", "SN", "SH"})

    def __init__(self, corpus: Dict[str, dict], *, k1: float, b: float,
                 text_field: str = "embedding_text") -> None:
        from kiwipiepy import Kiwi
        from rank_bm25 import BM25Okapi
        self._kiwi = Kiwi()
        self._ids = list(corpus.keys())
        tokens = [self.tokenize(corpus[c][text_field]) for c in self._ids]
        self._bm25 = BM25Okapi(tokens, k1=k1, b=b)

    def tokenize(self, text: str) -> List[str]:
        return [t.form for t in self._kiwi.tokenize(text) if t.tag in self.NOUN_TAGS]

    def search(self, query: str, k: int) -> List[str]:
        scores = self._bm25.get_scores(self.tokenize(query))
        ranked = sorted(zip(self._ids, scores), key=lambda x: (-x[1], x[0]))[:k]
        return [cid for cid, _ in ranked]


def rrf(dense: Sequence[str], bm25: Sequence[str], *, k: int, wd: float, wb: float,
        pool: int, out_k: int) -> List[str]:
    """Reciprocal Rank Fusion — run_hybrid.py 의 rrf() 를 그대로 옮김.

    score(c) = wd / (k + rank_dense) + wb / (k + rank_bm25)
    순위만 쓰므로 Dense(L2)와 BM25(양수 점수)의 스케일을 맞출 필요가 없다.
    가중치는 크기 불변이라 5:5 와 1:1 은 같은 결과를 낸다.
    동점 처리: 점수 내림차순, 같으면 chunk_id 오름차순.
    """
    score: Dict[str, float] = {}
    for rank, cid in enumerate(dense[:pool], start=1):
        score[cid] = score.get(cid, 0.0) + wd / (k + rank)
    for rank, cid in enumerate(bm25[:pool], start=1):
        score[cid] = score.get(cid, 0.0) + wb / (k + rank)
    return [c for c, _ in sorted(score.items(), key=lambda x: (-x[1], x[0]))][:out_k]


class HybridRetriever:
    """Dense + BM25 → RRF. 각 단계 결과를 함께 돌려줘 회귀 대조에 쓴다."""

    def __init__(self, dense: DenseIndexRetriever, bm25: BM25Retriever, *,
                 dense_k: int, bm25_k: int, rrf_k: int, w_dense: float, w_bm25: float,
                 pool: int, out_k: int) -> None:
        self.dense, self.bm25 = dense, bm25
        self.dense_k, self.bm25_k = dense_k, bm25_k
        self.rrf_k, self.w_dense, self.w_bm25 = rrf_k, w_dense, w_bm25
        self.pool, self.out_k = pool, out_k

    def search(self, query: str) -> Dict[str, List[str]]:
        d = self.dense.search(query, self.dense_k)
        b = self.bm25.search(query, self.bm25_k)
        h = rrf(d, b, k=self.rrf_k, wd=self.w_dense, wb=self.w_bm25,
                pool=self.pool, out_k=self.out_k)
        return {"dense": d, "bm25": b, "hybrid": h}
