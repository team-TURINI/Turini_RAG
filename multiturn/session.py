"""멀티턴 세션 — 한 대화를 끝까지 잇는 층.

orchestrator.generate_turn() 은 한 턴의 재작성·검색·생성까지만 하고 state 를 건드리지 않는다
(팀원 설계: "state message 추가는 다음 통합 단계의 책임"). 이 모듈이 그 다음 단계다.

  ask(question)
    1. orchestrator.generate_turn(question, state)          — 재작성 → 라우팅 → 검색 → 생성
    2. 사용자에게 보낼 문장 확정
         generated          → 생성 답변
         direct             → 인사 / 감사·작별 / 범위 밖 정형 문구 (질문 키워드로 셋 중 하나)
         clarify            → 되묻기 정형 문구
         portfolio_required → 포트폴리오 등록 안내 정형 문구
    3. state 에 user/assistant 메시지 추가                    — 이제부터 다음 턴의 문맥
    4. maintain_conversation_summary()                      — 메시지가 recent_message_limit 를 넘으면 앞부분을 요약으로 압축

요약 실패는 턴을 실패시키지 않는다 — 답변은 이미 나갔고, 메시지는 남아 있어 다음 턴에 다시 시도된다.

사용:
  from multiturn.session import build_session
  s = build_session(portfolio=None)          # FinalPipeline·QueryRewriter·Summarizer 실제 로드
  r = s.ask("채권이 뭐야?");  print(r.answer)
  r = s.ask("그럼 종류는?");  print(r.rewrite.retrieval_query, r.answer)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, cast

from core.generator import GenerationResult
from multiturn.conversation_summarizer import (
    ConversationSummarizer,
    ConversationSummaryError,
    maintain_conversation_summary,
)
from multiturn.orchestrator import GenerationStatus, MultiTurnOrchestrator
from multiturn.query_rewriter import QueryRewriteResult
from multiturn.rag_adapter import RAGContextResult
from multiturn.state import ConversationState

log = logging.getLogger(__name__)

# 생성기를 안 거치는 상태의 답변. 서비스 톤에 맞게 바꿔도 되고 build_session(replies=...) 로 덮어쓸 수 있다.
# direct 는 재작성기가 인사·감사·잡담·범위 밖 질문을 한 라우트로 묶으므로 여기서 문구만 갈라 쓴다.
DEFAULT_REPLIES: dict[str, str] = {
    "direct": "금융상품이나 투자에 대해 궁금한 점을 물어보시면 자료를 바탕으로 안내해 드립니다.",
    "direct_greeting": "안녕하세요. 금융상품이나 투자에 대해 궁금한 점을 물어보시면 자료를 바탕으로 안내해 드립니다.",
    "direct_thanks": "도움이 되었다니 다행입니다. 더 궁금한 점이 있으면 언제든 물어보세요.",
    "direct_offtopic": "죄송하지만 금융상품·투자 관련 질문에만 답변드릴 수 있습니다. 관련해서 궁금한 점이 있으면 알려주세요.",
    "clarify": "어떤 상품이나 개념을 말씀하시는지 조금 더 구체적으로 알려주시면 정확히 안내해 드리겠습니다.",
    "portfolio_required": "이 질문은 보유 자산 정보가 있어야 답할 수 있습니다. 포트폴리오를 등록하시면 그에 맞춰 안내해 드립니다.",
}

_GREETING = re.compile(r"안녕|하이|헬로|hello|\bhi\b|반가|처음", re.IGNORECASE)
_THANKS = re.compile(r"고마|감사|땡큐|thank|잘 ?알겠|알겠(어|습니다)|수고|잘가|안녕히|바이|bye", re.IGNORECASE)


def direct_reply_key(question: str) -> str:
    """direct 라우트를 인사/감사·작별/범위 밖으로 나눈다. 짧은 인사·감사만 골라내고 나머지는 범위 밖."""
    q = question.strip()
    if _THANKS.search(q):
        return "direct_thanks"
    if _GREETING.search(q) and len(q) <= 20:
        return "direct_greeting"
    return "direct_offtopic"


@dataclass(frozen=True)
class TurnResponse:
    """한 턴의 결과. answer 가 사용자에게 보낼 문장이고 나머지는 기록·평가용."""

    question: str
    answer: str
    status: GenerationStatus
    rewrite: QueryRewriteResult
    rag: Optional[RAGContextResult]
    generation: Optional[GenerationResult]
    profile: Optional[str]
    summary_updated: bool
    turn_index: int


class MultiTurnSession:
    def __init__(
        self,
        orchestrator: MultiTurnOrchestrator,
        *,
        summarizer: Optional[ConversationSummarizer] = None,
        state: Optional[ConversationState] = None,
        recent_message_limit: int = 6,
        replies: Optional[dict[str, str]] = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.summarizer = summarizer
        self.state = state if state is not None else ConversationState()
        self.recent_message_limit = recent_message_limit
        self.replies = {**DEFAULT_REPLIES, **(replies or {})}
        self.history: list[TurnResponse] = []

    def ask(self, question: str, *, scenario_name: Optional[str] = None) -> TurnResponse:
        if not question.strip():
            raise ValueError("question 은 비어 있을 수 없습니다.")
        result = self.orchestrator.generate_turn(question, self.state, scenario_name=scenario_name)

        if result.generation_status == "generated":
            assert result.generation_result is not None
            answer = result.generation_result.answer
        elif result.generation_status == "direct":
            answer = self.replies[direct_reply_key(question)]
        else:
            answer = self.replies[result.generation_status]

        # 여기서부터 이번 턴이 문맥이 된다 — rewriter 는 "완료된 이전 turn 만" state 에 있길 기대하므로 생성 뒤에 넣는다
        self.state.add_user_message(question)
        self.state.add_assistant_message(answer)

        summary_updated = False
        if self.summarizer is not None:
            try:
                summary_updated = maintain_conversation_summary(
                    self.state, self.summarizer, recent_message_limit=self.recent_message_limit)
            except ConversationSummaryError as e:          # 답변은 이미 나갔다 — 다음 턴에 다시 시도
                log.warning("대화 요약 실패 (다음 턴에 재시도): %s", e)

        resp = TurnResponse(
            question=question, answer=answer, status=result.generation_status,
            rewrite=result.rewrite_result, rag=result.rag_result,
            generation=result.generation_result, profile=result.generation_profile,
            summary_updated=summary_updated, turn_index=len(self.history) + 1,
        )
        self.history.append(resp)
        return resp

    def set_portfolio(self, portfolio: Optional[dict[str, Any]]) -> None:
        self.state.set_portfolio_context(portfolio)

    def reset_conversation(self) -> None:
        """대화와 턴 history만 초기화하고 portfolio는 유지한다."""

        self.state.messages.clear()
        self.state.set_summary("")
        self.history.clear()

    def reset_all(self) -> None:
        """대화, 턴 history, portfolio를 모두 초기화한다."""

        self.reset_conversation()
        self.state.set_portfolio_context(None)

    def transcript(self) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.state.messages]


_PORTFOLIO_UNSET = object()


def build_session(
    *,
    pipeline=None,
    portfolio: Optional[dict[str, Any]] | object = _PORTFOLIO_UNSET,
    state: Optional[ConversationState] = None,
    general_prompt_preset: Optional[str] = None,
    summarize: bool = True,
    recent_message_limit: int = 6,
    replies: Optional[dict[str, str]] = None,
    verbose: bool = False,
) -> MultiTurnSession:
    """실제 구성요소로 세션을 조립한다.

    pipeline을 주면 여러 세션이 인덱스를 공유할 수 있다. 저장된 state를 복원할 때는
    state만 전달해야 하며, portfolio와 state를 함께 명시하면 모호성을 막기 위해 거부한다.
    """
    from core.pipeline import FinalPipeline
    from multiturn.generation_adapter import MultiTurnGeneratorAdapter
    from multiturn.query_rewriter import QueryRewriter
    from multiturn.rag_adapter import MultiTurnRAGAdapter

    if state is not None and portfolio is not _PORTFOLIO_UNSET:
        raise ValueError("portfolio와 state는 동시에 전달할 수 없습니다.")
    if state is not None and not isinstance(state, ConversationState):
        raise TypeError("state는 ConversationState여야 합니다.")

    if pipeline is None:
        pipeline = FinalPipeline(verbose=verbose)
    gen_kwargs = {"general_prompt_preset": general_prompt_preset} if general_prompt_preset else {}
    orchestrator = MultiTurnOrchestrator(
        QueryRewriter(recent_message_limit=recent_message_limit),
        MultiTurnRAGAdapter(pipeline),
        MultiTurnGeneratorAdapter(**gen_kwargs),
    )
    if state is None:
        initial_portfolio = (
            None
            if portfolio is _PORTFOLIO_UNSET
            else cast(Optional[dict[str, Any]], portfolio)
        )
        state = ConversationState(portfolio_context=initial_portfolio)
    return MultiTurnSession(
        orchestrator,
        summarizer=ConversationSummarizer() if summarize else None,
        state=state, recent_message_limit=recent_message_limit, replies=replies,
    )
