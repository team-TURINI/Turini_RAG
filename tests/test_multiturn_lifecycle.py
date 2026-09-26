from __future__ import annotations

import unittest
from unittest.mock import patch

from core.generator import GenerationResult
from multiturn.conversation_summarizer import ConversationSummaryError
from multiturn.generation_adapter import MultiTurnGeneratorAdapter
from multiturn.orchestrator import MultiTurnOrchestrator
from multiturn.query_rewriter import QueryRewriteResult
from multiturn.rag_adapter import MultiTurnRAGAdapter
from multiturn.session import MultiTurnSession, build_session
from multiturn.state import ConversationState


class FixedRewriter:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error

    def rewrite(self, current_question, state, *, scenario_name=None):
        if self.error is not None:
            raise self.error
        return QueryRewriteResult(
            original_question=current_question,
            retrieval_query=f"복원 검색: {current_question}",
            is_followup=bool(state.messages),
            needs_portfolio=False,
            route="rag",
        )


class FakePipeline:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.retrieve_calls: list[str] = []

    def retrieve(self, question: str) -> dict[str, list[str]]:
        self.retrieve_calls.append(question)
        if self.error is not None:
            raise self.error
        return {"reranked": ["chunk-1"]}

    def build_context(self, ranked_ids: list[str]) -> list[str]:
        return ["금융 context"]


def generate_answer(*args, **kwargs) -> GenerationResult:
    return GenerationResult("생성 답변", 0.01, 10, 5)


def fail_generation(*args, **kwargs) -> GenerationResult:
    raise RuntimeError("generation failed")


def make_session(
    *,
    state: ConversationState | None = None,
    rewriter: FixedRewriter | None = None,
    pipeline: FakePipeline | None = None,
    generate=generate_answer,
    summarizer=None,
    limit: int = 6,
) -> MultiTurnSession:
    adapter = MultiTurnGeneratorAdapter(
        general_generate=generate,
        portfolio_generate=generate,
    )
    orchestrator = MultiTurnOrchestrator(
        rewriter or FixedRewriter(),
        MultiTurnRAGAdapter(pipeline or FakePipeline()),
        adapter,
    )
    return MultiTurnSession(
        orchestrator,
        state=state,
        summarizer=summarizer,
        recent_message_limit=limit,
    )


class FailingSummarizer:
    def summarize(self, previous_summary, messages_to_summarize):
        raise ConversationSummaryError("summary failed")


class MultiTurnLifecycleTest(unittest.TestCase):
    def test_build_session_restores_existing_state_and_accepts_arbitrary_question(self) -> None:
        saved = {
            "messages": [
                {"role": "user", "content": "ETF가 뭐야?"},
                {"role": "assistant", "content": "상장지수펀드입니다."},
            ],
            "conversation_summary": "ETF 기본 개념을 설명함",
            "portfolio_context": {"allocation": {"채권": 0.0}},
        }
        state = ConversationState.from_dict(saved)
        generator = MultiTurnGeneratorAdapter(
            general_generate=generate_answer,
            portfolio_generate=generate_answer,
        )

        with (
            patch(
                "multiturn.query_rewriter.QueryRewriter",
                return_value=FixedRewriter(),
            ),
            patch(
                "multiturn.generation_adapter.MultiTurnGeneratorAdapter",
                return_value=generator,
            ),
        ):
            session = build_session(
                state=state,
                pipeline=FakePipeline(),
                summarize=False,
            )
            response = session.ask("사용자가 그때그때 입력한 임의의 후속 질문")

        self.assertIs(session.state, state)
        self.assertEqual(response.answer, "생성 답변")
        self.assertTrue(response.rewrite.is_followup)
        self.assertEqual(state.messages[-2].content, response.question)
        self.assertEqual(state.messages[-1].content, response.answer)

    def test_build_session_rejects_state_and_explicit_portfolio_together(self) -> None:
        with self.assertRaises(ValueError):
            build_session(
                state=ConversationState(),
                portfolio=None,
                pipeline=FakePipeline(),
                summarize=False,
            )

    def test_reset_conversation_keeps_portfolio(self) -> None:
        portfolio = {"allocation": {"채권": 0.0}}
        session = make_session(
            state=ConversationState(
                conversation_summary="기존 요약",
                portfolio_context=portfolio,
            )
        )
        session.ask("임의 질문")

        session.reset_conversation()

        self.assertEqual(session.state.messages, [])
        self.assertEqual(session.state.conversation_summary, "")
        self.assertEqual(session.history, [])
        self.assertIs(session.state.portfolio_context, portfolio)

    def test_reset_all_removes_portfolio(self) -> None:
        session = make_session(
            state=ConversationState(
                conversation_summary="기존 요약",
                portfolio_context={"allocation": {"채권": 0.0}},
            )
        )
        session.ask("임의 질문")

        session.reset_all()

        self.assertEqual(session.state.messages, [])
        self.assertEqual(session.state.conversation_summary, "")
        self.assertEqual(session.history, [])
        self.assertIsNone(session.state.portfolio_context)

    def test_failed_rewrite_retrieval_or_generation_does_not_append_turn(self) -> None:
        sessions = [
            make_session(rewriter=FixedRewriter(error=RuntimeError("rewrite failed"))),
            make_session(pipeline=FakePipeline(error=RuntimeError("retrieve failed"))),
            make_session(generate=fail_generation),
        ]

        for session in sessions:
            with self.subTest(component=type(session.orchestrator.rewriter).__name__):
                session.state.conversation_summary = "기존 요약"
                session.state.add_user_message("완료된 이전 질문")
                session.state.add_assistant_message("완료된 이전 답변")
                before = session.state.to_dict()

                with self.assertRaises(RuntimeError):
                    session.ask("실패할 현재 질문")

                self.assertEqual(session.state.to_dict(), before)
                self.assertEqual(session.history, [])

    def test_summary_failure_preserves_completed_turn_and_previous_state(self) -> None:
        state = ConversationState(conversation_summary="기존 요약")
        for index in range(2):
            state.add_user_message(f"이전 질문 {index}")
            state.add_assistant_message(f"이전 답변 {index}")
        before_messages = list(state.messages)
        session = make_session(
            state=state,
            summarizer=FailingSummarizer(),
            limit=4,
        )

        response = session.ask("새 질문")

        self.assertEqual(response.answer, "생성 답변")
        self.assertFalse(response.summary_updated)
        self.assertEqual(state.conversation_summary, "기존 요약")
        self.assertEqual(state.messages[:4], before_messages)
        self.assertEqual(state.messages[-2].content, "새 질문")
        self.assertEqual(state.messages[-1].content, "생성 답변")
        self.assertEqual(len(session.history), 1)


if __name__ == "__main__":
    unittest.main()
