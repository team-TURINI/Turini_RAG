from __future__ import annotations

import unittest

from core.generator import GenerationResult
from multiturn.generation_adapter import MultiTurnGeneratorAdapter
from multiturn.orchestrator import MultiTurnOrchestrator
from multiturn.query_rewriter import QueryRewriteResult
from multiturn.rag_adapter import MultiTurnRAGAdapter
from multiturn.state import ConversationState


class FakeFinalPipeline:
    def __init__(self) -> None:
        self.retrieve_calls: list[str] = []
        self.build_context_calls: list[list[str]] = []

    def retrieve(self, question: str) -> dict[str, list[str]]:
        self.retrieve_calls.append(question)
        return {"reranked": ["chunk-2", "chunk-1"]}

    def build_context(self, ranked_ids: list[str]) -> list[str]:
        self.build_context_calls.append(list(ranked_ids))
        return ["금융 context 2", "금융 context 1"]


class FakeQueryRewriter:
    def __init__(self, result: QueryRewriteResult) -> None:
        self.result = result
        self.states: list[ConversationState] = []

    def rewrite(
        self,
        current_question: str,
        state: ConversationState,
        *,
        scenario_name: str | None = None,
    ) -> QueryRewriteResult:
        self.states.append(state)
        return self.result


class CapturingGenerationFunctions:
    def __init__(self) -> None:
        self.general_calls: list[tuple[tuple, dict]] = []
        self.portfolio_calls: list[tuple[tuple, dict]] = []

    @staticmethod
    def _result() -> GenerationResult:
        return GenerationResult("answer", 0.01, 10, 5)

    def general(self, *args, **kwargs) -> GenerationResult:
        self.general_calls.append((args, kwargs))
        return self._result()

    def portfolio(self, *args, **kwargs) -> GenerationResult:
        self.portfolio_calls.append((args, kwargs))
        return self._result()


def rewrite(
    *,
    route: str = "rag",
    needs_portfolio: bool = False,
) -> QueryRewriteResult:
    return QueryRewriteResult(
        original_question="그럼 내 포트폴리오에서는 어떤 의미야?",
        retrieval_query="금리 상승과 채권 가격 변화가 포트폴리오에 미치는 일반적인 영향",
        is_followup=True,
        needs_portfolio=needs_portfolio,
        route=route,  # type: ignore[arg-type]
    )


def build_orchestrator(result: QueryRewriteResult):
    pipeline = FakeFinalPipeline()
    calls = CapturingGenerationFunctions()
    generator_adapter = MultiTurnGeneratorAdapter(
        general_generate=calls.general,
        portfolio_generate=calls.portfolio,
    )
    orchestrator = MultiTurnOrchestrator(
        FakeQueryRewriter(result),
        MultiTurnRAGAdapter(pipeline),
        generator_adapter,
    )
    return orchestrator, pipeline, calls


class MultiTurnGenerationTest(unittest.TestCase):
    def test_general_question_uses_v2_4j_without_portfolio(self) -> None:
        orchestrator, pipeline, calls = build_orchestrator(rewrite())
        state = ConversationState(portfolio_context={"secret": "must-not-pass"})

        result = orchestrator.generate_turn("현재 질문", state)

        self.assertEqual(result.generation_status, "generated")
        self.assertEqual(result.generation_profile, "v2_4j")
        self.assertEqual(pipeline.retrieve_calls, [result.rewrite_result.retrieval_query])
        self.assertEqual(len(calls.general_calls), 1)
        self.assertEqual(calls.portfolio_calls, [])
        args, kwargs = calls.general_calls[0]
        self.assertEqual(args[0], result.rewrite_result.original_question)
        self.assertNotIn("portfolio_context", kwargs)
        self.assertNotIn("must-not-pass", repr((args, kwargs)))

    def test_portfolio_question_uses_v2_p_once_with_trusted_state_value(self) -> None:
        orchestrator, pipeline, calls = build_orchestrator(
            rewrite(needs_portfolio=True)
        )
        portfolio = {"portfolio": {"allocation": {"채권": 0.0}}}

        result = orchestrator.generate_turn(
            "현재 질문",
            ConversationState(portfolio_context=portfolio),
        )

        self.assertEqual(result.generation_status, "generated")
        self.assertEqual(result.generation_profile, "v2_p")
        self.assertEqual(len(calls.portfolio_calls), 1)
        self.assertEqual(calls.general_calls, [])
        args, kwargs = calls.portfolio_calls[0]
        self.assertEqual(args[0], result.rewrite_result.original_question)
        self.assertIs(args[2], portfolio)
        self.assertEqual(kwargs["profile"], "v2_p")
        self.assertEqual(pipeline.retrieve_calls, [result.rewrite_result.retrieval_query])

    def test_missing_portfolio_keeps_retrieval_and_skips_generation(self) -> None:
        orchestrator, pipeline, calls = build_orchestrator(
            rewrite(needs_portfolio=True)
        )

        result = orchestrator.generate_turn("현재 질문", ConversationState())

        self.assertEqual(result.generation_status, "portfolio_required")
        self.assertIsNotNone(result.rag_result)
        self.assertIsNone(result.generation_result)
        self.assertEqual(pipeline.retrieve_calls, [result.rewrite_result.retrieval_query])
        self.assertEqual(calls.general_calls, [])
        self.assertEqual(calls.portfolio_calls, [])

    def test_direct_skips_retrieval_and_generation(self) -> None:
        orchestrator, pipeline, calls = build_orchestrator(
            rewrite(route="direct")
        )

        result = orchestrator.generate_turn("안녕", ConversationState())

        self.assertEqual(result.generation_status, "direct")
        self.assertIsNone(result.rag_result)
        self.assertEqual(pipeline.retrieve_calls, [])
        self.assertEqual(calls.general_calls, [])
        self.assertEqual(calls.portfolio_calls, [])

    def test_clarify_skips_retrieval_and_generation(self) -> None:
        orchestrator, pipeline, calls = build_orchestrator(
            rewrite(route="clarify")
        )

        result = orchestrator.generate_turn("그건?", ConversationState())

        self.assertEqual(result.generation_status, "clarify")
        self.assertIsNone(result.rag_result)
        self.assertEqual(pipeline.retrieve_calls, [])
        self.assertEqual(calls.general_calls, [])
        self.assertEqual(calls.portfolio_calls, [])

    def test_retrieval_and_generation_questions_remain_distinct(self) -> None:
        orchestrator, pipeline, calls = build_orchestrator(rewrite())

        result = orchestrator.generate_turn("현재 질문", ConversationState())

        self.assertEqual(pipeline.retrieve_calls, [result.rewrite_result.retrieval_query])
        self.assertEqual(
            calls.general_calls[0][0][0],
            result.rewrite_result.original_question,
        )
        self.assertNotEqual(pipeline.retrieve_calls[0], calls.general_calls[0][0][0])

    def test_generate_turn_does_not_mutate_state(self) -> None:
        orchestrator, _, _ = build_orchestrator(rewrite())
        state = ConversationState(conversation_summary="요약")
        state.add_user_message("이전 질문")
        original_messages = list(state.messages)

        orchestrator.generate_turn("현재 질문", state)

        self.assertEqual(state.conversation_summary, "요약")
        self.assertEqual(state.messages, original_messages)


if __name__ == "__main__":
    unittest.main()
