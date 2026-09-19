from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from multiturn.query_rewriter import QueryRewriteResult


class FinalPipelineProtocol(Protocol):
    """실제 core.pipeline.FinalPipeline 중 adapter가 사용하는 표면."""

    def retrieve(self, question: str) -> dict[str, list[str]]:
        ...

    def build_context(self, ranked_ids: list[str]) -> list[str]:
        ...


@dataclass(frozen=True)
class RAGContextResult:
    """향후 Generator에 넘길 검색 결과. 개인 portfolio는 포함하지 않는다."""

    original_question: str
    retrieval_query: str
    needs_portfolio: bool
    reranked_ids: tuple[str, ...]
    retrieved_context: tuple[str, ...]


class MultiTurnRAGAdapter:
    """Query Rewrite 결과와 FinalPipeline의 검색·컨텍스트 조립만 연결한다."""

    def __init__(self, pipeline: FinalPipelineProtocol) -> None:
        self.pipeline = pipeline

    def retrieve_context(
        self,
        rewrite_result: QueryRewriteResult,
    ) -> RAGContextResult | None:
        if rewrite_result.route != "rag":
            return None

        retrieval_result = self.pipeline.retrieve(rewrite_result.retrieval_query)
        reranked = retrieval_result["reranked"]
        if not isinstance(reranked, list) or not all(
            isinstance(chunk_id, str) for chunk_id in reranked
        ):
            raise TypeError("FinalPipeline.retrieve()['reranked']는 list[str]여야 합니다.")

        reranked_ids = list(reranked)
        contexts = self.pipeline.build_context(reranked_ids)
        return RAGContextResult(
            original_question=rewrite_result.original_question,
            retrieval_query=rewrite_result.retrieval_query,
            needs_portfolio=rewrite_result.needs_portfolio,
            reranked_ids=tuple(reranked_ids),
            retrieved_context=tuple(contexts),
        )
