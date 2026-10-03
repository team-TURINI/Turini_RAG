from __future__ import annotations

import unittest
from contextvars import copy_context
from types import SimpleNamespace
from unittest.mock import patch

from multiturn.query_rewriter import (
    QueryRewriteResult,
    QueryRewriter,
    _run_traced_query_rewrite,
    _serialize_traceable_outputs,
)
from multiturn.state import ConversationState


class OriginalAPIError(RuntimeError):
    pass


class FailingResponsesAPI:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def create(self, **kwargs):
        raise self.error


def trace_container_without_network() -> dict:
    return {
        "new_run": None,
        "context": copy_context(),
    }


class QueryRewriterTracingTest(unittest.TestCase):
    def test_output_serializer_preserves_success_structure(self) -> None:
        result = QueryRewriteResult(
            original_question="ETF가 뭐야?",
            retrieval_query="ETF 정의",
            is_followup=False,
            needs_portfolio=False,
            route="rag",
        )

        self.assertEqual(
            _serialize_traceable_outputs(result),
            {
                "original_question": "ETF가 뭐야?",
                "retrieval_query": "ETF 정의",
                "is_followup": False,
                "needs_portfolio": False,
                "route": "rag",
            },
        )

    def test_output_serializer_accepts_exception_path_none(self) -> None:
        self.assertEqual(_serialize_traceable_outputs(None), {})

    def test_traced_operation_preserves_original_exception(self) -> None:
        original = OriginalAPIError("credit_balance_exhausted")

        def fail() -> QueryRewriteResult:
            raise original

        with (
            patch(
                "langsmith.run_helpers._setup_run",
                return_value=trace_container_without_network(),
            ),
            patch("langsmith.run_helpers.LOGGER.warning") as warning,
        ):
            with self.assertRaises(OriginalAPIError) as raised:
                _run_traced_query_rewrite(
                    {
                        "current_question": "ETF가 뭐야?",
                        "recent_messages": [],
                        "conversation_summary": "",
                    },
                    fail,
                )

        self.assertIs(raised.exception, original)
        warning.assert_not_called()

    def test_tracing_enabled_failure_preserves_state_and_api_exception(self) -> None:
        original = OriginalAPIError("credit_balance_exhausted")
        client = SimpleNamespace(responses=FailingResponsesAPI(original))
        state = ConversationState(
            conversation_summary="기존 요약",
            portfolio_context={"allocation": {"채권": 0.0}},
        )
        state.add_user_message("이전 질문")
        state.add_assistant_message("이전 답변")
        before = state.to_dict()
        rewriter = QueryRewriter(
            client=client,
            enable_tracing=True,
        )

        with patch(
            "langsmith.run_helpers._setup_run",
            return_value=trace_container_without_network(),
        ):
            with self.assertRaises(OriginalAPIError) as raised:
                rewriter.rewrite("ETF가 뭐야?", state)

        self.assertIs(raised.exception, original)
        self.assertEqual(state.to_dict(), before)


if __name__ == "__main__":
    unittest.main()
