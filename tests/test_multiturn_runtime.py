from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import Any

from multiturn.conversation_summarizer import (
    ConversationSummarizer,
    ConversationSummaryError,
    _sanitize_summary_traceable_inputs,
    build_conversation_summary_trace_inputs,
    maintain_conversation_summary,
)
from multiturn.prompts.query_rewrite import build_query_rewrite_input
from multiturn.query_rewriter import QueryRewriteError, QueryRewriter
from multiturn.state import ConversationState


class FakeResponsesAPI:
    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            output_text=json.dumps(self.payload, ensure_ascii=False)
            if self.payload is not None
            else "",
        )


class FakeOpenAIClient:
    def __init__(self, responses: FakeResponsesAPI) -> None:
        self.responses = responses


def query_payload(
    question: str,
    retrieval_query: str,
    *,
    is_followup: bool,
    needs_portfolio: bool,
    route: str = "rag",
) -> dict[str, Any]:
    return {
        "original_question": question,
        "retrieval_query": retrieval_query,
        "is_followup": is_followup,
        "needs_portfolio": needs_portfolio,
        "route": route,
    }


def state_with_messages(count: int) -> ConversationState:
    state = ConversationState()
    for index in range(1, count + 1):
        if index % 2:
            state.add_user_message(f"메시지 {index}")
        else:
            state.add_assistant_message(f"메시지 {index}")
    return state


class MultiTurnRuntimeTest(unittest.TestCase):
    def test_query_rewriter_keeps_followup_semantics_and_strict_schema(self) -> None:
        current = "그럼 둘의 추적 오차는 왜 생겨?"
        state = ConversationState()
        state.add_user_message("인덱스 ETF와 액티브 ETF의 차이가 뭐야?")
        state.add_assistant_message("운용 재량과 지수 추종 방식이 다릅니다.")
        responses = FakeResponsesAPI(
            query_payload(
                "모델이 바꾼 원 질문",
                "인덱스 ETF와 액티브 ETF의 추적 오차 발생 원인",
                is_followup=True,
                needs_portfolio=False,
            )
        )

        result = QueryRewriter(client=FakeOpenAIClient(responses)).rewrite(
            current,
            state,
        )

        self.assertEqual(result.original_question, current)
        self.assertTrue(result.is_followup)
        self.assertEqual(result.route, "rag")
        response_format = responses.calls[0]["text"]["format"]
        self.assertTrue(response_format["strict"])
        self.assertEqual(
            response_format["schema"]["properties"]["route"]["enum"],
            ["rag", "direct", "clarify"],
        )

    def test_portfolio_is_hidden_but_needs_portfolio_rag_is_allowed(self) -> None:
        current = "그럼 내 포트폴리오에서는 어떤 의미야?"
        state = ConversationState(
            portfolio_context={
                "sample_id": "SECRET-P-01",
                "allocation": {"채권": 0.55},
                "total_amount_krw": 12000000,
            }
        )
        state.add_user_message("금리가 오르면 채권 가격은 어떻게 돼?")
        state.add_assistant_message("기존 채권 가격은 하락 압력을 받습니다.")
        responses = FakeResponsesAPI(
            query_payload(
                current,
                "금리 상승과 채권 가격 변화가 포트폴리오에 미치는 일반적인 영향",
                is_followup=True,
                needs_portfolio=True,
            )
        )

        result = QueryRewriter(client=FakeOpenAIClient(responses)).rewrite(current, state)

        self.assertTrue(result.needs_portfolio)
        self.assertEqual(result.route, "rag")
        serialized = json.dumps(responses.calls[0]["input"], ensure_ascii=False)
        for forbidden in ("SECRET-P-01", "0.55", "12000000", "portfolio_context"):
            self.assertNotIn(forbidden, serialized)

    def test_non_rag_routes_allow_empty_retrieval_query(self) -> None:
        for route, question, followup in (
            ("direct", "안녕!", False),
            ("clarify", "그건?", True),
        ):
            with self.subTest(route=route):
                responses = FakeResponsesAPI(
                    query_payload(
                        question,
                        "",
                        is_followup=followup,
                        needs_portfolio=False,
                        route=route,
                    )
                )
                result = QueryRewriter(client=FakeOpenAIClient(responses)).rewrite(
                    question,
                    ConversationState(),
                )
                self.assertEqual(result.route, route)
                self.assertEqual(result.retrieval_query, "")

    def test_rag_route_rejects_empty_retrieval_query(self) -> None:
        responses = FakeResponsesAPI(
            query_payload(
                "ETF가 뭐야?",
                "",
                is_followup=False,
                needs_portfolio=False,
            )
        )
        with self.assertRaises(QueryRewriteError):
            QueryRewriter(client=FakeOpenAIClient(responses)).rewrite(
                "ETF가 뭐야?",
                ConversationState(),
            )

    def test_summary_skips_when_six_or_fewer_messages(self) -> None:
        state = state_with_messages(6)
        responses = FakeResponsesAPI({"summary": "호출되면 안 됨"})
        summarizer = ConversationSummarizer(client=FakeOpenAIClient(responses))

        self.assertFalse(maintain_conversation_summary(state, summarizer))
        self.assertEqual(responses.calls, [])

    def test_summary_compacts_first_two_and_keeps_recent_six(self) -> None:
        state = state_with_messages(8)
        state.set_summary("기존 요약")
        responses = FakeResponsesAPI({"summary": "기존 요약과 앞 두 메시지의 누적 요약"})
        summarizer = ConversationSummarizer(client=FakeOpenAIClient(responses))

        self.assertTrue(maintain_conversation_summary(state, summarizer))

        request_input = responses.calls[0]["input"]
        self.assertIn("기존 요약", request_input)
        self.assertIn("메시지 1", request_input)
        self.assertIn("메시지 2", request_input)
        self.assertNotIn("메시지 3", request_input)
        self.assertEqual(len(state.messages), 6)
        self.assertEqual(state.messages[0].content, "메시지 3")

    def test_summary_failure_preserves_all_state(self) -> None:
        state = state_with_messages(8)
        state.set_summary("변경되면 안 되는 요약")
        original_messages = list(state.messages)
        summarizer = ConversationSummarizer(
            client=FakeOpenAIClient(FakeResponsesAPI(error=RuntimeError("failure")))
        )

        with self.assertRaises(ConversationSummaryError):
            maintain_conversation_summary(state, summarizer)

        self.assertEqual(state.conversation_summary, "변경되면 안 되는 요약")
        self.assertEqual(state.messages, original_messages)

    def test_summary_request_and_trace_exclude_hidden_portfolio(self) -> None:
        state = state_with_messages(8)
        state.portfolio_context = {
            "sample_id": "SECRET-P-01",
            "allocation": {"채권": 0.55},
        }
        responses = FakeResponsesAPI({"summary": "안전한 요약"})
        summarizer = ConversationSummarizer(client=FakeOpenAIClient(responses))

        maintain_conversation_summary(state, summarizer)
        trace_inputs = build_conversation_summary_trace_inputs("", state.messages[:2])
        processed = _sanitize_summary_traceable_inputs(
            {"trace_inputs": trace_inputs, "operation": object()}
        )
        serialized = json.dumps(
            {"request": responses.calls[0], "trace": processed},
            ensure_ascii=False,
        )
        for forbidden in ("portfolio_context", "SECRET-P-01", "0.55"):
            self.assertNotIn(forbidden, serialized)

    def test_summary_and_recent_six_feed_query_rewriter_input(self) -> None:
        state = state_with_messages(8)
        responses = FakeResponsesAPI({"summary": "오래된 채권 대화 요약"})
        summarizer = ConversationSummarizer(client=FakeOpenAIClient(responses))

        maintain_conversation_summary(state, summarizer)
        query_input = build_query_rewrite_input("그럼 둘은?", state)

        self.assertIn("[대화 요약]\n오래된 채권 대화 요약", query_input)
        self.assertNotIn("메시지 1", query_input)
        self.assertNotIn("메시지 2", query_input)
        for index in range(3, 9):
            self.assertIn(f"메시지 {index}", query_input)


if __name__ == "__main__":
    unittest.main()
