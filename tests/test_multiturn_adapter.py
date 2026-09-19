from __future__ import annotations

import unittest

from multiturn.orchestrator import MultiTurnOrchestrator
from multiturn.query_rewriter import QueryRewriteResult
from multiturn.rag_adapter import MultiTurnRAGAdapter
from multiturn.state import ConversationState


class FakeFinalPipeline:
    def __init__(self) -> None:
        self.retrieve_calls: list[str] = []
        self.build_context_calls: list[list[str]] = []
        self.run_calls = 0
        self.retrieval_result = {
            "dense": ["dense-1"],
            "bm25": ["bm25-1"],
            "hybrid": ["hybrid-1", "hybrid-2"],
            "reranked": ["chunk-3", "chunk-1", "chunk-2", "chunk-4"],
        }

    def retrieve(self, question: str) -> dict[str, list[str]]:
        self.retrieve_calls.append(question)
        return self.retrieval_result

    def build_context(self, ranked_ids: list[str]) -> list[str]:
        self.build_context_calls.append(list(ranked_ids))
        return [f"context:{chunk_id}" for chunk_id in ranked_ids[:3]]

    def run(self, question: str) -> None:
        self.run_calls += 1
        raise AssertionError("Generator 경로인 FinalPipeline.run()은 호출하면 안 됩니다.")


class FakeQueryRewriter:
    def __init__(self, result: QueryRewriteResult) -> None:
        self.result = result
        self.calls: list[tuple[str, ConversationState, str | None]] = []

    def rewrite(
        self,
        current_question: str,
        state: ConversationState,
        *,
        scenario_name: str | None = None,
    ) -> QueryRewriteResult:
        self.calls.append((current_question, state, scenario_name))
        return self.result


def rewrite_result(
    *,
    original_question: str = "그럼 둘의 추적 오차는 왜 생겨?",
    retrieval_query: str = "인덱스 ETF와 액티브 ETF의 추적 오차 발생 원인",
    needs_portfolio: bool = False,
    route: str = "rag",
) -> QueryRewriteResult:
    return QueryRewriteResult(
        original_question=original_question,
        retrieval_query=retrieval_query,
        is_followup=True,
        needs_portfolio=needs_portfolio,
        route=route,  # type: ignore[arg-type]
    )


class MultiTurnRAGAdapterTest(unittest.TestCase):
    def test_rag_route_calls_retrieve_once_with_exact_retrieval_query(self) -> None:
        pipeline = FakeFinalPipeline()
        rewrite = rewrite_result()

        MultiTurnRAGAdapter(pipeline).retrieve_context(rewrite)

        self.assertEqual(pipeline.retrieve_calls, [rewrite.retrieval_query])

    def test_original_and_retrieval_questions_remain_separate(self) -> None:
        pipeline = FakeFinalPipeline()
        rewrite = rewrite_result(
            original_question="그럼 둘은 왜 달라?",
            retrieval_query="인덱스 ETF와 액티브 ETF의 차이",
        )

        result = MultiTurnRAGAdapter(pipeline).retrieve_context(rewrite)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.original_question, "그럼 둘은 왜 달라?")
        self.assertEqual(result.retrieval_query, "인덱스 ETF와 액티브 ETF의 차이")
        self.assertNotEqual(result.original_question, result.retrieval_query)

    def test_direct_route_does_not_call_retrieval(self) -> None:
        pipeline = FakeFinalPipeline()

        result = MultiTurnRAGAdapter(pipeline).retrieve_context(
            rewrite_result(retrieval_query="", route="direct")
        )

        self.assertIsNone(result)
        self.assertEqual(pipeline.retrieve_calls, [])

    def test_clarify_route_does_not_call_retrieval(self) -> None:
        pipeline = FakeFinalPipeline()

        result = MultiTurnRAGAdapter(pipeline).retrieve_context(
            rewrite_result(retrieval_query="", route="clarify")
        )

        self.assertIsNone(result)
        self.assertEqual(pipeline.retrieve_calls, [])

    def test_reranked_ids_are_passed_to_build_context(self) -> None:
        pipeline = FakeFinalPipeline()

        result = MultiTurnRAGAdapter(pipeline).retrieve_context(rewrite_result())

        expected = pipeline.retrieval_result["reranked"]
        self.assertEqual(pipeline.build_context_calls, [expected])
        assert result is not None
        self.assertEqual(result.reranked_ids, tuple(expected))
        self.assertEqual(
            result.retrieved_context,
            ("context:chunk-3", "context:chunk-1", "context:chunk-2"),
        )

    def test_adapter_has_no_generator_call_path(self) -> None:
        pipeline = FakeFinalPipeline()

        MultiTurnRAGAdapter(pipeline).retrieve_context(rewrite_result())

        self.assertEqual(pipeline.run_calls, 0)

    def test_portfolio_context_is_not_mixed_into_retrieval(self) -> None:
        pipeline = FakeFinalPipeline()
        rewrite = rewrite_result()
        rewriter = FakeQueryRewriter(rewrite)
        state = ConversationState(
            portfolio_context={
                "sample_id": "SECRET-P-01",
                "allocation": {"채권": 0.55},
                "total_amount_krw": 12000000,
            }
        )

        MultiTurnOrchestrator(
            rewriter,
            MultiTurnRAGAdapter(pipeline),
        ).prepare_turn(rewrite.original_question, state)

        self.assertEqual(pipeline.retrieve_calls, [rewrite.retrieval_query])
        self.assertNotIn("SECRET-P-01", pipeline.retrieve_calls[0])
        self.assertNotIn("0.55", pipeline.retrieve_calls[0])
        self.assertNotIn("12000000", pipeline.retrieve_calls[0])

    def test_needs_portfolio_true_and_rag_still_retrieves(self) -> None:
        pipeline = FakeFinalPipeline()
        rewrite = rewrite_result(
            original_question="그럼 내 포트폴리오에는 어떤 의미야?",
            retrieval_query="금리 상승과 채권 가격 변화가 포트폴리오에 미치는 영향",
            needs_portfolio=True,
        )

        result = MultiTurnRAGAdapter(pipeline).retrieve_context(rewrite)

        self.assertEqual(pipeline.retrieve_calls, [rewrite.retrieval_query])
        assert result is not None
        self.assertTrue(result.needs_portfolio)

    def test_orchestrator_does_not_mutate_conversation_state(self) -> None:
        pipeline = FakeFinalPipeline()
        rewrite = rewrite_result()
        rewriter = FakeQueryRewriter(rewrite)
        state = ConversationState(conversation_summary="기존 요약")
        state.add_user_message("인덱스 ETF가 뭐야?")
        original_messages = list(state.messages)

        result = MultiTurnOrchestrator(
            rewriter,
            MultiTurnRAGAdapter(pipeline),
        ).prepare_turn(rewrite.original_question, state)

        self.assertEqual(state.conversation_summary, "기존 요약")
        self.assertEqual(state.messages, original_messages)
        self.assertEqual(result.rewrite_result, rewrite)


if __name__ == "__main__":
    unittest.main()
