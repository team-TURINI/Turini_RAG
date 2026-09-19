from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from multiturn.query_rewriter import QueryRewriteResult
from multiturn.rag_adapter import MultiTurnRAGAdapter, RAGContextResult
from multiturn.state import ConversationState


class QueryRewriterProtocol(Protocol):
    def rewrite(
        self,
        current_question: str,
        state: ConversationState,
        *,
        scenario_name: str | None = None,
    ) -> QueryRewriteResult:
        ...


@dataclass(frozen=True)
class MultiTurnTurnResult:
    rewrite_result: QueryRewriteResult
    rag_result: RAGContextResult | None


class MultiTurnOrchestrator:
    """한 질문을 rewrite하고 route가 rag일 때만 retrieval을 준비한다.

    Generator 및 state message 추가는 다음 통합 단계의 책임이다. 정상 lifecycle에서는
    assistant 응답 생성 후 user/assistant 메시지를 state에 추가하고 summary maintenance를
    호출한다.
    """

    def __init__(
        self,
        rewriter: QueryRewriterProtocol,
        rag_adapter: MultiTurnRAGAdapter,
    ) -> None:
        self.rewriter = rewriter
        self.rag_adapter = rag_adapter

    def prepare_turn(
        self,
        current_question: str,
        state: ConversationState,
        *,
        scenario_name: str | None = None,
    ) -> MultiTurnTurnResult:
        rewrite_result = self.rewriter.rewrite(
            current_question,
            state,
            scenario_name=scenario_name,
        )
        rag_result = self.rag_adapter.retrieve_context(rewrite_result)
        return MultiTurnTurnResult(
            rewrite_result=rewrite_result,
            rag_result=rag_result,
        )
