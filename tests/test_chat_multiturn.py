from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from multiturn.state import ConversationState
from scripts import chat_multiturn
from scripts.chat_multiturn import chat_loop, format_state


class FakeSession:
    def __init__(self) -> None:
        self.state = ConversationState(
            conversation_summary="공개 가능한 대화 요약",
            portfolio_context={"api_key": "never-print-this"},
        )
        self.state.add_user_message("이전 질문")
        self.state.add_assistant_message("이전 답변")
        self.ask_calls: list[str] = []
        self.reset_calls = 0

    def ask(self, question: str):
        self.ask_calls.append(question)
        if len(self.ask_calls) == 1:
            raise RuntimeError("temporary failure")
        return SimpleNamespace(
            answer="정상 답변",
            rewrite=SimpleNamespace(
                retrieval_query="복원된 검색 질문",
                route="rag",
                is_followup=True,
                needs_portfolio=False,
            ),
            profile="v2_4j",
            summary_updated=False,
        )

    def reset_conversation(self) -> None:
        self.reset_calls += 1
        self.state.messages.clear()
        self.state.conversation_summary = ""


class InterruptingSession(FakeSession):
    def ask(self, question: str):
        raise KeyboardInterrupt


class ChatMultiturnTest(unittest.TestCase):
    def test_main_initializes_pipeline_once_and_reuses_it_for_session(self) -> None:
        pipeline = object()
        session = object()
        with (
            patch.object(chat_multiturn.cfg, "OPENAI_API_KEY", "test"),
            patch.object(chat_multiturn.cfg, "COHERE_API_KEY", "test"),
            patch.object(
                chat_multiturn.FinalPipeline,
                "__new__",
                return_value=pipeline,
            ) as pipeline_new,
            patch.object(
                chat_multiturn,
                "build_session",
                return_value=session,
            ) as build_session,
            patch.object(chat_multiturn, "chat_loop") as chat_loop_mock,
            patch("builtins.print"),
        ):
            chat_multiturn.main([])

        pipeline_new.assert_called_once()
        build_session.assert_called_once_with(
            pipeline=pipeline,
            portfolio=None,
            summarize=True,
        )
        chat_loop_mock.assert_called_once_with(session, debug=False)

    def test_state_output_hides_raw_portfolio(self) -> None:
        rendered = format_state(FakeSession().state)

        self.assertIn("공개 가능한 대화 요약", rendered)
        self.assertIn("recent_messages: 2", rendered)
        self.assertIn("user: 이전 질문", rendered)
        self.assertIn("portfolio_present: yes", rendered)
        self.assertNotIn("never-print-this", rendered)
        self.assertNotIn("api_key", rendered)

    def test_loop_skips_empty_recovers_from_error_and_passes_question_unchanged(self) -> None:
        session = FakeSession()
        inputs = iter(
            [
                "   ",
                "  첫 번째 임의 질문  ",
                "/debug on",
                "두 번째 임의 질문",
                "/reset",
                "/exit",
            ]
        )
        outputs: list[str] = []

        chat_loop(
            session,  # type: ignore[arg-type]
            input_fn=lambda prompt: next(inputs),
            output_fn=outputs.append,
        )

        self.assertEqual(
            session.ask_calls,
            ["  첫 번째 임의 질문  ", "두 번째 임의 질문"],
        )
        self.assertTrue(any(line.startswith("[error] RuntimeError:") for line in outputs))
        self.assertIn("정상 답변", outputs)
        debug_output = "\n".join(outputs)
        self.assertIn("retrieval_query: 복원된 검색 질문", debug_output)
        self.assertIn("generation_profile: v2_4j", debug_output)
        self.assertEqual(session.reset_calls, 1)
        self.assertIsNotNone(session.state.portfolio_context)

    def test_keyboard_interrupt_from_turn_is_not_swallowed(self) -> None:
        with self.assertRaises(KeyboardInterrupt):
            chat_loop(
                InterruptingSession(),  # type: ignore[arg-type]
                input_fn=lambda prompt: "질문",
                output_fn=lambda message: None,
            )


if __name__ == "__main__":
    unittest.main()
