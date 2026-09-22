from __future__ import annotations

import unittest

from core.generator import GenerationResult
from multiturn.conversation_summarizer import ConversationSummaryError
from multiturn.generation_adapter import MultiTurnGeneratorAdapter
from multiturn.orchestrator import MultiTurnOrchestrator
from multiturn.query_rewriter import QueryRewriteResult
from multiturn.rag_adapter import MultiTurnRAGAdapter
from multiturn.session import DEFAULT_REPLIES, MultiTurnSession
from multiturn.state import ConversationState, Message


class FakeFinalPipeline:
    def retrieve(self, question: str) -> dict[str, list[str]]:
        return {"reranked": ["chunk-1"]}

    def build_context(self, ranked_ids: list[str]) -> list[str]:
        return ["금융 context"]


class ScriptedRewriter:
    """턴마다 정해진 결과를 순서대로 돌려주고, 호출 시점의 state 스냅샷을 남긴다."""

    def __init__(self, results: list[QueryRewriteResult]) -> None:
        self.results = list(results)
        self.seen_message_counts: list[int] = []

    def rewrite(self, current_question, state, *, scenario_name=None):
        self.seen_message_counts.append(len(state.messages))
        return self.results.pop(0)


class FakeSummarizer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def summarize(self, previous_summary, messages_to_summarize):
        self.calls += 1
        if self.fail:
            raise ConversationSummaryError("boom")
        from multiturn.conversation_summarizer import ConversationSummaryResult
        return ConversationSummaryResult(summary=f"요약({len(messages_to_summarize)}개)")


def rw(route="rag", needs_portfolio=False, q="현재 질문", rq="검색 질문"):
    return QueryRewriteResult(original_question=q, retrieval_query=rq if route == "rag" else "",
                              is_followup=False, needs_portfolio=needs_portfolio, route=route)  # type: ignore[arg-type]


def gen(*args, **kwargs) -> GenerationResult:
    return GenerationResult("생성 답변", 0.01, 10, 5)


def session(results, *, summarizer=None, portfolio=None, limit=6):
    orch = MultiTurnOrchestrator(
        ScriptedRewriter(results), MultiTurnRAGAdapter(FakeFinalPipeline()),
        MultiTurnGeneratorAdapter(general_generate=gen, portfolio_generate=gen))
    return MultiTurnSession(orch, summarizer=summarizer,
                            state=ConversationState(portfolio_context=portfolio), recent_message_limit=limit)


class MultiTurnSessionTest(unittest.TestCase):
    def test_generated_turn_appends_both_messages_after_rewrite(self) -> None:
        s = session([rw(), rw()])
        r1 = s.ask("채권이 뭐야?")
        r2 = s.ask("그럼 종류는?")
        self.assertEqual(r1.status, "generated"); self.assertEqual(r1.answer, "생성 답변")
        # rewriter 는 '완료된 이전 turn' 만 봐야 한다 — 1턴째 0개, 2턴째 2개
        self.assertEqual(s.orchestrator.rewriter.seen_message_counts, [0, 2])
        self.assertEqual([m.role for m in s.state.messages], ["user", "assistant", "user", "assistant"])
        self.assertEqual(s.state.messages[2].content, "그럼 종류는?")
        self.assertEqual(r2.turn_index, 2); self.assertEqual(len(s.history), 2)

    def test_direct_clarify_portfolio_required_use_fixed_replies(self) -> None:
        s = session([rw(route="direct"), rw(route="clarify"), rw(needs_portfolio=True)])
        self.assertEqual(s.ask("안녕").answer, DEFAULT_REPLIES["direct_greeting"])
        self.assertEqual(s.ask("그건?").answer, DEFAULT_REPLIES["clarify"])
        r = s.ask("내 포트폴리오엔 어때?")
        self.assertEqual(r.status, "portfolio_required")
        self.assertEqual(r.answer, DEFAULT_REPLIES["portfolio_required"])
        self.assertIsNone(r.generation)
        self.assertEqual(len(s.state.messages), 6)          # 정형 답변도 문맥에 남는다

    def test_direct_splits_greeting_thanks_offtopic(self) -> None:
        s = session([rw(route="direct")] * 4)
        self.assertEqual(s.ask("안녕!").answer, DEFAULT_REPLIES["direct_greeting"])
        self.assertEqual(s.ask("고마워").answer, DEFAULT_REPLIES["direct_thanks"])
        self.assertEqual(s.ask("오늘 날씨 어때?").answer, DEFAULT_REPLIES["direct_offtopic"])
        self.assertEqual(s.ask("네 잘 알겠습니다 감사합니다").answer, DEFAULT_REPLIES["direct_thanks"])

    def test_custom_replies_override_defaults(self) -> None:
        orch = session([rw(route="direct")]).orchestrator
        s = MultiTurnSession(orch, replies={"direct_greeting": "반가워요"})
        self.assertEqual(s.ask("안녕").answer, "반가워요")
        self.assertEqual(s.replies["clarify"], DEFAULT_REPLIES["clarify"])

    def test_summary_compacts_when_messages_exceed_limit(self) -> None:
        fs = FakeSummarizer()
        s = session([rw()] * 4, summarizer=fs, limit=4)
        flags = [s.ask(f"질문 {i}").summary_updated for i in range(4)]
        # 1·2턴: 2·4개 → 압축 없음. 3턴: 6개 > 4 → 앞 2개 요약. 4턴: 6개 → 다시 앞 2개
        self.assertEqual(flags, [False, False, True, True])
        self.assertEqual(fs.calls, 2)
        self.assertEqual(len(s.state.messages), 4)
        self.assertTrue(s.state.conversation_summary.startswith("요약("))

    def test_summary_failure_does_not_fail_turn(self) -> None:
        s = session([rw()] * 3, summarizer=FakeSummarizer(fail=True), limit=4)
        for i in range(3):
            r = s.ask(f"질문 {i}")
        self.assertEqual(r.answer, "생성 답변"); self.assertFalse(r.summary_updated)
        self.assertEqual(len(s.state.messages), 6)           # 압축 안 됐지만 메시지는 보존
        self.assertEqual(s.state.conversation_summary, "")

    def test_empty_question_rejected(self) -> None:
        with self.assertRaises(ValueError):
            session([rw()]).ask("   ")


if __name__ == "__main__":
    unittest.main()
