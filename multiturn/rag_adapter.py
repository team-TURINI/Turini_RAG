from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from typing import Protocol

from multiturn.query_rewriter import QueryRewriteResult


class FinalPipelineProtocol(Protocol):
    """실제 core.pipeline.FinalPipeline 중 adapter가 사용하는 표면."""

    def retrieve(self, question: str) -> dict[str, list[str]]:
        ...

    def build_context(self, ranked_ids: list[str]) -> list[str]:
        ...

    def retrieve_quiz(self, question: str) -> dict[str, list[str]]:
        ...

    def build_quiz_context(self, ranked_ids: list[str]) -> list[str]:
        ...


@dataclass(frozen=True)
class RAGContextResult:
    """향후 Generator에 넘길 검색 결과. 개인 portfolio는 포함하지 않는다."""

    original_question: str
    retrieval_query: str
    needs_portfolio: bool
    reranked_ids: tuple[str, ...]
    retrieved_context: tuple[str, ...]
    source: Literal["financial", "quiz"] = "financial"


class MultiTurnRAGAdapter:
    """Query Rewrite 결과와 FinalPipeline의 검색·컨텍스트 조립만 연결한다."""

    def __init__(self, pipeline: FinalPipelineProtocol) -> None:
        self.pipeline = pipeline

    def retrieve_context(
        self,
        rewrite_result: QueryRewriteResult,
    ) -> RAGContextResult | None:
        if rewrite_result.route not in {"rag", "quiz"}:
            return None

        is_quiz = rewrite_result.route == "quiz"
        retrieval_result = (
            self.pipeline.retrieve_quiz(rewrite_result.retrieval_query)
            if is_quiz
            else self.pipeline.retrieve(rewrite_result.retrieval_query)
        )
        reranked = retrieval_result["reranked"]
        if not isinstance(reranked, list) or not all(
            isinstance(chunk_id, str) for chunk_id in reranked
        ):
            method = "retrieve_quiz" if is_quiz else "retrieve"
            raise TypeError(f"FinalPipeline.{method}()['reranked']는 list[str]여야 합니다.")

        reranked_ids = list(reranked)
        contexts = (
            self.pipeline.build_quiz_context(reranked_ids)
            if is_quiz
            else self.pipeline.build_context(reranked_ids)
        )
        return RAGContextResult(
            original_question=rewrite_result.original_question,
            retrieval_query=rewrite_result.retrieval_query,
            needs_portfolio=rewrite_result.needs_portfolio,
            reranked_ids=tuple(reranked_ids),
            retrieved_context=tuple(contexts),
            source="quiz" if is_quiz else "financial",
        )
