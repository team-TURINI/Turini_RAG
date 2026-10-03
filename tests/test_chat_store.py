from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from multiturn.store import (
    ConversationMeta,
    ConversationRecord,
    JsonConversationStore,
    SqliteConversationStore,
    make_title,
    new_conversation_id,
    now_iso,
)


def record(user_id="u1", cid=None, title="첫 질문", n=0, summary="", messages=None):
    cid = cid or new_conversation_id()
    ts = now_iso()
    return ConversationRecord(
        meta=ConversationMeta(conversation_id=cid, user_id=user_id, title=title,
                              created_at=ts, updated_at=ts, n_messages=n),
        state={"messages": messages or [], "conversation_summary": summary, "portfolio_context": None})


class StoreContractMixin:
    """두 구현이 같은 규약을 지키는지 한 벌의 테스트로 검사한다."""

    def make_store(self):
        raise NotImplementedError

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = self.make_store()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_save_load_roundtrip(self) -> None:
        r = record(summary="지난 요약", messages=[{"role": "user", "content": "채권이 뭐야?"}])
        self.store.save(r)
        got = self.store.load("u1", r.meta.conversation_id)
        self.assertIsNotNone(got)
        self.assertEqual(got.meta.title, "첫 질문")
        self.assertEqual(got.state["conversation_summary"], "지난 요약")
        self.assertEqual(got.state["messages"][0]["content"], "채권이 뭐야?")

    def test_other_user_cannot_read_or_delete(self) -> None:
        r = record(user_id="u1")
        self.store.save(r)
        self.assertIsNone(self.store.load("u2", r.meta.conversation_id))
        self.assertEqual(self.store.messages("u2", r.meta.conversation_id), [])
        self.assertFalse(self.store.delete("u2", r.meta.conversation_id))
        self.assertIsNotNone(self.store.load("u1", r.meta.conversation_id))

    def test_messages_append_and_order(self) -> None:
        r = record()
        cid = r.meta.conversation_id
        self.store.save(r)
        self.store.append_messages("u1", cid, [("user", "q1", {}), ("assistant", "a1", {"route": "rag"})])
        last = self.store.append_messages("u1", cid, [("user", "q2", {}), ("assistant", "a2", {})])
        self.assertEqual(last, 4)
        msgs = self.store.messages("u1", cid)
        self.assertEqual([m.seq for m in msgs], [1, 2, 3, 4])
        self.assertEqual([m.content for m in msgs], ["q1", "a1", "q2", "a2"])
        self.assertEqual(msgs[1].meta["route"], "rag")

    def test_messages_survive_state_compaction(self) -> None:
        """요약 압축으로 state.messages 가 비어도 표시용 내역은 남아야 한다 — 이 기능의 핵심."""
        r = record()
        cid = r.meta.conversation_id
        self.store.save(r)
        self.store.append_messages("u1", cid, [("user", "q1", {}), ("assistant", "a1", {})])
        compacted = ConversationRecord(
            meta=ConversationMeta(conversation_id=cid, user_id="u1", title="첫 질문",
                                  created_at=r.meta.created_at, updated_at=now_iso(), n_messages=2),
            state={"messages": [], "conversation_summary": "압축된 요약", "portfolio_context": None})
        self.store.save(compacted)
        self.assertEqual(self.store.load("u1", cid).state["messages"], [])
        self.assertEqual([m.content for m in self.store.messages("u1", cid)], ["q1", "a1"])

    def test_messages_limit_and_paging(self) -> None:
        r = record(); cid = r.meta.conversation_id
        self.store.save(r)
        self.store.append_messages("u1", cid, [("user", f"q{i}", {}) for i in range(1, 7)])
        recent = self.store.messages("u1", cid, limit=2)
        self.assertEqual([m.content for m in recent], ["q5", "q6"])
        older = self.store.messages("u1", cid, limit=2, before_seq=recent[0].seq)
        self.assertEqual([m.content for m in older], ["q3", "q4"])

    def test_list_sorted_by_updated_at_and_scoped_to_user(self) -> None:
        a, b, other = record(title="A"), record(title="B"), record(user_id="u2", title="남의 것")
        self.store.save(a); self.store.save(other); self.store.save(b)
        self.store.append_messages("u1", b.meta.conversation_id, [("user", "q", {})])
        titles = [c.title for c in self.store.list_conversations("u1")]
        self.assertEqual(titles[0], "B")
        self.assertEqual(sorted(titles), ["A", "B"])
        self.assertEqual([c.title for c in self.store.list_conversations("u2")], ["남의 것"])

    def test_delete_removes_messages(self) -> None:
        r = record(); cid = r.meta.conversation_id
        self.store.save(r)
        self.store.append_messages("u1", cid, [("user", "q", {})])
        self.assertTrue(self.store.delete("u1", cid))
        self.assertIsNone(self.store.load("u1", cid))
        self.assertEqual(self.store.messages("u1", cid), [])

    def test_append_by_other_user_is_rejected(self) -> None:
        r = record(); cid = r.meta.conversation_id
        self.store.save(r)
        with self.assertRaises(PermissionError):
            self.store.append_messages("u2", cid, [("user", "q", {})])

    def test_blank_user_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.store.list_conversations("  ")


class SqliteStoreTest(StoreContractMixin, unittest.TestCase):
    def make_store(self):
        return SqliteConversationStore(Path(self.tmp.name) / "c.db")

    def test_reopening_file_keeps_data(self) -> None:
        r = record(); cid = r.meta.conversation_id
        self.store.save(r)
        self.store.append_messages("u1", cid, [("user", "q", {})])
        again = SqliteConversationStore(Path(self.tmp.name) / "c.db")       # 서버 재시작 상황
        self.assertEqual([m.content for m in again.messages("u1", cid)], ["q"])


class JsonStoreTest(StoreContractMixin, unittest.TestCase):
    def make_store(self):
        return JsonConversationStore(Path(self.tmp.name) / "conv")

    def test_rejects_unsafe_conversation_id(self) -> None:
        with self.assertRaises(ValueError):
            self.store.load("u1", "../../etc/passwd")


class TitleTest(unittest.TestCase):
    def test_title_trimmed(self) -> None:
        self.assertEqual(make_title("  채권이   뭐야?  "), "채권이 뭐야?")
        self.assertTrue(make_title("가" * 80).endswith("…"))
        self.assertEqual(len(make_title("가" * 80)), 41)


if __name__ == "__main__":
    unittest.main()
