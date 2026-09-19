from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from core.generator import GenerationResult
from multiturn.generation_adapter import MultiTurnGeneratorAdapter
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


GenerationStatus = Literal["generated", "portfolio_required", "direct", "clarify"]


@dataclass(frozen=True)
class MultiTurnGenerationResult:
    rewrite_result: QueryRewriteResult
    rag_result: RAGContextResult | None
    generation_status: GenerationStatus
    generation_result: GenerationResult | None
    generation_profile: str | None


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
        generator_adapter: MultiTurnGeneratorAdapter | None = None,
    ) -> None:
        self.rewriter = rewriter
        self.rag_adapter = rag_adapter
        self.generator_adapter = generator_adapter

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

    def generate_turn(
        self,
        current_question: str,
        state: ConversationState,
        *,
        scenario_name: str | None = None,
    ) -> MultiTurnGenerationResult:
        """RAG turn을 생성까지 연결하되 state의 message/summary는 변경하지 않는다."""

        prepared = self.prepare_turn(
            current_question,
            state,
            scenario_name=scenario_name,
        )
        rewrite = prepared.rewrite_result

        if rewrite.route in {"direct", "clarify"}:
            return MultiTurnGenerationResult(
                rewrite_result=rewrite,
                rag_result=None,
                generation_status=rewrite.route,
                generation_result=None,
                generation_profile=None,
            )

        if prepared.rag_result is None:
            raise RuntimeError("route='rag'인데 retrieval 결과가 없습니다.")
        if rewrite.needs_portfolio and state.portfolio_context is None:
            return MultiTurnGenerationResult(
                rewrite_result=rewrite,
                rag_result=prepared.rag_result,
                generation_status="portfolio_required",
                generation_result=None,
                generation_profile=None,
            )
        if self.generator_adapter is None:
            raise RuntimeError("generation을 실행하려면 generator_adapter가 필요합니다.")

        if rewrite.needs_portfolio:
            assert state.portfolio_context is not None
            generation = self.generator_adapter.generate_portfolio(
                prepared.rag_result,
                state.portfolio_context,
            )
            profile = self.generator_adapter.portfolio_profile
        else:
            generation = self.generator_adapter.generate_general(prepared.rag_result)
            profile = self.generator_adapter.general_prompt_preset

        return MultiTurnGenerationResult(
            rewrite_result=rewrite,
            rag_result=prepared.rag_result,
            generation_status="generated",
            generation_result=generation,
            generation_profile=profile,
        )
