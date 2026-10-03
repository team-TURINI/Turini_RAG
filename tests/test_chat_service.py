from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.generator import GenerationResult
from multiturn.chat_service import ChatService
from multiturn.generation_adapter import MultiTurnGeneratorAdapter
from multiturn.orchestrator import MultiTurnOrchestrator
from multiturn.query_rewriter import QueryRewriteResult
from multiturn.rag_adapter import MultiTurnRAGAdapter
from multiturn.store import SqliteConversationStore


class FakePipeline:
    def retrieve(self, question: str) -> dict[str, list[str]]:
        return {"reranked": ["chunk-1", "chunk-2", "chunk-3", "chunk-4"]}

    def build_context(self, ranked_ids: list[str]) -> list[str]:
        return ["금융 context"]


class EchoRewriter:
    """state 를 그대로 받아 기록하고, 메시지가 있으면 후속질문으로 표시한다."""

    def __init__(self) -> None:
        self.seen_summaries: list[str] = []
        self.seen_message_counts: list[int] = []
        self.seen_portfolio: list[bool] = []

    def rewrite(self, current_question, state, *, scenario_name=None):
        self.seen_summaries.append(state.conversation_summary)
        self.seen_message_counts.append(len(state.messages))
        self.seen_portfolio.append(state.portfolio_context is not None)
        return QueryRewriteResult(original_question=current_question,
                                  retrieval_query=f"검색:{current_question}", is_followup=bool(state.messages),
                                  needs_portfolio=False, route="rag")


def fake_generate(question, contexts, **kwargs) -> GenerationResult:
    return GenerationResult(f"답변:{question}", 0.01, 10, 5)


class FakeSummarizer:
    """메시지가 limit 를 넘으면 state 를 압축 — 실제 요약기와 같은 효과."""

    def __init__(self) -> None:
        self.calls = 0

    def summarize(self, previous_summary, messages_to_summarize):
        from multiturn.conversation_summarizer import ConversationSummaryResult
        self.calls += 1
        return ConversationSummaryResult(summary=f"요약{self.calls}")


class ChatServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.rewriter = EchoRewriter()
        self.summarizer = FakeSummarizer()
        orch = MultiTurnOrchestrator(
            self.rewriter, MultiTurnRAGAdapter(FakePipeline()),
            MultiTurnGeneratorAdapter(general_generate=fake_generate, portfolio_generate=fake_generate))
        self.store = SqliteConversationStore(Path(self.tmp.name) / "c.db")
        self.svc = ChatService(store=self.store, orchestrator=orch, summarizer=self.summarizer,
                               recent_message_limit=4)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_first_ask_creates_conversation_and_history(self) -> None:
        t = self.svc.ask("u1", "채권이 뭐야?")
        self.assertTrue(t.is_new_conversation)
        self.assertEqual(t.answer, "답변:채권이 뭐야?")
        self.assertEqual(t.seq, 2)
        msgs = self.svc.history("u1", t.conversation_id)
        self.assertEqual([(m.role, m.content) for m in msgs],
                         [("user", "채권이 뭐야?"), ("assistant", "답변:채권이 뭐야?")])
        self.assertEqual(msgs[1].meta["route"], "rag")
        self.assertEqual(msgs[1].meta["context_chunk_ids"], ["chunk-1", "chunk-2", "chunk-3"])
        self.assertEqual(self.svc.list_conversations("u1")[0].title, "채권이 뭐야?")

    def test_second_ask_resumes_context_from_store(self) -> None:
        """새 프로세스처럼 매 요청 state 를 저장소에서 복원 — 재작성기가 이전 턴을 봐야 한다."""
        t1 = self.svc.ask("u1", "채권이 뭐야?")
        t2 = self.svc.ask("u1", "그럼 종류는?", conversation_id=t1.conversation_id)
        self.assertFalse(t2.is_new_conversation)
        self.assertTrue(t2.is_followup)
        self.assertEqual(self.rewriter.seen_message_counts, [0, 2])      # 2턴째엔 이전 턴 2개가 보인다
        self.assertEqual(len(self.svc.history("u1", t1.conversation_id)), 4)

    def test_login_flow_lists_and_restores_after_restart(self) -> None:
        t = self.svc.ask("u1", "펀드가 뭐야?")
        self.svc.ask("u1", "수수료는?", conversation_id=t.conversation_id)

        fresh_store = SqliteConversationStore(Path(self.tmp.name) / "c.db")   # 서버 재시작
        fresh = ChatService(store=fresh_store, orchestrator=self.svc.orchestrator,
                            summarizer=self.summarizer, recent_message_limit=4)
        convs = fresh.list_conversations("u1")
        self.assertEqual(len(convs), 1)
        self.assertEqual(fresh.latest_conversation_id("u1"), t.conversation_id)
        self.assertEqual([m.content for m in fresh.history("u1", t.conversation_id)],
                         ["펀드가 뭐야?", "답변:펀드가 뭐야?", "수수료는?", "답변:수수료는?"])

    def test_history_kept_after_summary_compaction(self) -> None:
        cid = None
        for i in range(4):
            t = self.svc.ask("u1", f"질문{i}", conversation_id=cid)
            cid = t.conversation_id
        self.assertGreater(self.summarizer.calls, 0)                     # 압축이 일어났고
        rec = self.store.load("u1", cid)
        self.assertEqual(len(rec.state["messages"]), 4)                  # state 는 최근 4개만 남았지만
        self.assertEqual(len(self.svc.history("u1", cid)), 8)            # 표시용 내역은 전부 남는다
        self.assertTrue(rec.state["conversation_summary"].startswith("요약"))

    def test_portfolio_not_persisted_by_default(self) -> None:
        pf = {"sample_id": "P-01", "allocation": {"국내주식": 0.55}}
        t = self.svc.ask("u1", "내 자산은?", portfolio=pf)
        self.assertTrue(self.rewriter.seen_portfolio[-1])                # 이번 턴엔 쓰였지만
        self.assertIsNone(self.store.load("u1", t.conversation_id).state["portfolio_context"])  # 저장은 안 됨

        self.svc.ask("u1", "다시 물어볼게", conversation_id=t.conversation_id)
        self.assertFalse(self.rewriter.seen_portfolio[-1])               # 안 넣으면 없는 상태로 진행
        self.svc.ask("u1", "또 물어볼게", conversation_id=t.conversation_id, portfolio=pf)
        self.assertTrue(self.rewriter.seen_portfolio[-1])                # 앱이 넣어주면 다시 쓰인다

    def test_portfolio_persisted_when_opted_in(self) -> None:
        svc = ChatService(store=self.store, orchestrator=self.svc.orchestrator,
                          summarizer=self.summarizer, persist_portfolio=True)
        t = svc.ask("u1", "내 자산은?", portfolio={"sample_id": "P-01"})
        self.assertEqual(self.store.load("u1", t.conversation_id).state["portfolio_context"],
                         {"sample_id": "P-01"})

    def test_conversations_are_isolated_per_user(self) -> None:
        t = self.svc.ask("u1", "내 대화")
        self.svc.ask("u2", "남의 대화")
        self.assertEqual(len(self.svc.list_conversations("u1")), 1)
        self.assertEqual(self.svc.history("u2", t.conversation_id), [])
        with self.assertRaises(KeyError):
            self.svc.ask("u2", "끼어들기", conversation_id=t.conversation_id)

    def test_unknown_conversation_id_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.svc.ask("u1", "질문", conversation_id="없는id")

    def test_rename_and_delete(self) -> None:
        t = self.svc.ask("u1", "채권이 뭐야?")
        self.assertTrue(self.svc.rename("u1", t.conversation_id, "채권 공부"))
        self.assertEqual(self.svc.list_conversations("u1")[0].title, "채권 공부")
        self.assertFalse(self.svc.rename("u2", t.conversation_id, "뺏기"))
        self.assertTrue(self.svc.delete("u1", t.conversation_id))
        self.assertEqual(self.svc.list_conversations("u1"), [])

    def test_blank_question_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.svc.ask("u1", "   ")


class StatelessTurnTest(unittest.TestCase):
    """무상태 경로 — state 왕복이 저장소 경로와 같은 문맥을 만든다."""

    def setUp(self) -> None:
        self.rewriter = EchoRewriter()
        self.summarizer = FakeSummarizer()
        orch = MultiTurnOrchestrator(
            self.rewriter, MultiTurnRAGAdapter(FakePipeline()),
            MultiTurnGeneratorAdapter(general_generate=fake_generate, portfolio_generate=fake_generate))
        self.svc = ChatService(store=None, orchestrator=orch, summarizer=self.summarizer, recent_message_limit=4)

    def test_first_turn_without_state(self) -> None:
        t = self.svc.ask_stateless("u1", "채권이 뭐야?")
        self.assertEqual(t.answer, "답변:채권이 뭐야?")
        self.assertEqual(t.suggested_title, "채권이 뭐야?")
        self.assertFalse(t.is_followup)
        self.assertEqual(t.state["version"], 1)
        self.assertEqual([m["role"] for m in t.state["messages"]], ["user", "assistant"])
        self.assertNotIn("portfolio_context", t.state)
        self.assertEqual([m["role"] for m in t.messages], ["user", "assistant"])
        self.assertEqual(t.messages[1]["meta"]["route"], "rag")
        self.assertEqual(t.messages[1]["meta"]["context_chunk_ids"], ["chunk-1", "chunk-2", "chunk-3"])

    def test_state_round_trip_gives_rewriter_previous_turns(self) -> None:
        t1 = self.svc.ask_stateless("u1", "채권이 뭐야?")
        t2 = self.svc.ask_stateless("u1", "그럼 종류는?", state=t1.state, conversation_id="c1")
        self.assertTrue(t2.is_followup)
        self.assertIsNone(t2.suggested_title)
        self.assertEqual(self.rewriter.seen_message_counts, [0, 2])     # 2턴째엔 이전 턴 2개가 보인다
        self.assertEqual(len(t2.state["messages"]), 4)

    def test_state_compaction_keeps_summary_in_state(self) -> None:
        state = None
        for i in range(4):
            t = self.svc.ask_stateless("u1", f"질문{i}", state=state)
            state = t.state
        self.assertGreater(self.summarizer.calls, 0)
        self.assertEqual(len(state["messages"]), 4)                     # 최근 4개만 원문
        self.assertTrue(state["conversation_summary"].startswith("요약"))
        self.assertEqual(self.rewriter.seen_summaries[-1], state and self.rewriter.seen_summaries[-1])

    def test_portfolio_used_but_not_in_state(self) -> None:
        pf = {"diagnosis": {"risk_type": "안정형"}}
        t = self.svc.ask_stateless("u1", "내 자산은?", portfolio=pf)
        self.assertTrue(self.rewriter.seen_portfolio[-1])
        self.assertNotIn("portfolio_context", t.state)
        t2 = self.svc.ask_stateless("u1", "다시", state=t.state)
        self.assertFalse(self.rewriter.seen_portfolio[-1])                 # 다음 요청에 안 넣으면 없음

    def test_matches_store_backed_path(self) -> None:
        """같은 질문 순서를 저장소 경로로 돌렸을 때와 state 내용이 같아야 한다."""
        with tempfile.TemporaryDirectory() as d:
            stored = ChatService(store=SqliteConversationStore(Path(d) / "c.db"), orchestrator=self.svc.orchestrator,
                                 summarizer=FakeSummarizer(), recent_message_limit=4)
            cid = None
            for q in ["a", "b", "c"]:
                cid = stored.ask("u1", q, conversation_id=cid).conversation_id
            rec = stored.store.load("u1", cid).state
        state = None
        for q in ["a", "b", "c"]:
            state = self.svc.ask_stateless("u1", q, state=state).state
        self.assertEqual(state["messages"], rec["messages"])
        self.assertEqual(state["conversation_summary"], rec["conversation_summary"])

    def test_bad_state_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.svc.ask_stateless("u1", "q", state={"version": 2})
        with self.assertRaises(ValueError):
            self.svc.ask_stateless("u1", "q", state={"messages": [{"role": "system", "content": "x"}]})
        with self.assertRaises(ValueError):
            self.svc.ask_stateless("u1", "q", state="문자열")  # type: ignore[arg-type]

    def test_store_methods_fail_clearly_without_store(self) -> None:
        with self.assertRaises(RuntimeError):
            self.svc.list_conversations("u1")
        with self.assertRaises(RuntimeError):
            self.svc.ask("u1", "q")


if __name__ == "__main__":
    unittest.main()
