from __future__ import annotations

import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient

import serve.api as api
from multiturn.chat_service import StatelessTurn


class FakeService:
    """파이프라인·LLM 없이 API 층만 검사한다. state 를 받으면 메시지 수를 늘려 돌려준다."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.reranker = SimpleNamespace(last_failed=False)
        self.orchestrator = SimpleNamespace(
            rag_adapter=SimpleNamespace(pipeline=SimpleNamespace(corpus={"c": 1}, reranker=self.reranker)))

    def ask_stateless(self, user_id, question, *, state=None, portfolio=None, conversation_id=None):
        self.calls.append({"user_id": user_id, "question": question, "state": state,
                           "portfolio": portfolio, "conversation_id": conversation_id})
        if state is not None and state.get("version") not in (None, 1):
            raise ValueError("state.version 지원 안 함")
        prev = (state or {}).get("messages", [])
        msgs = prev + [{"role": "user", "content": question}, {"role": "assistant", "content": f"답변:{question}"}]
        return StatelessTurn(
            answer=f"답변:{question}", status="generated",
            state={"version": 1, "messages": msgs[-6:], "conversation_summary": ""},
            messages=[{"role": "user", "content": question},
                      {"role": "assistant", "content": f"답변:{question}", "meta": {"status": "generated"}}],
            suggested_title=question[:40] if not prev else None,
            retrieval_query=f"검색:{question}", route="rag", is_followup=bool(prev),
            needs_portfolio=False, profile="v2_4j", summary_updated=False)


def _reset():
    api._state.update({"service": None, "ready": False, "error": None, "boot_s": None})


class ServeApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.svc = FakeService()
        api.app.state.service_factory = lambda: self.svc
        self._key = api.API_KEY
        api.API_KEY = ""
        self.client = TestClient(api.app)
        self.client.__enter__()

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)
        api.API_KEY = self._key
        _reset()

    def test_health_ready(self) -> None:
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ready"]); self.assertEqual(r.json()["corpus_chunks"], 1)

    def test_new_conversation_returns_state_messages_and_title(self) -> None:
        r = self.client.post("/chat", json={"user_id": "u1", "question": "채권이 뭐야?"})
        self.assertEqual(r.status_code, 200)
        b = r.json()
        self.assertEqual(b["answer"], "답변:채권이 뭐야?")
        self.assertEqual(b["suggested_title"], "채권이 뭐야?")
        self.assertEqual([m["role"] for m in b["messages"]], ["user", "assistant"])
        self.assertEqual(b["messages"][1]["meta"]["status"], "generated")
        self.assertEqual(b["state"]["version"], 1)
        self.assertEqual(len(b["state"]["messages"]), 2)
        self.assertFalse(b["degraded"])
        self.assertIsNone(self.svc.calls[-1]["state"])

    def test_state_round_trip_continues_conversation(self) -> None:
        """앱이 받은 state 를 그대로 돌려보내면 후속 질문으로 이어져야 한다 — 이 API 의 핵심 계약."""
        first = self.client.post("/chat", json={"user_id": "u1", "question": "채권이 뭐야?"}).json()
        second = self.client.post("/chat", json={"user_id": "u1", "question": "그럼 종류는?",
                                                 "state": first["state"], "conversation_id": "c1"}).json()
        self.assertTrue(second["is_followup"])
        self.assertIsNone(second["suggested_title"])
        self.assertEqual(len(second["state"]["messages"]), 4)
        self.assertEqual(self.svc.calls[-1]["state"], first["state"])
        self.assertEqual(self.svc.calls[-1]["conversation_id"], "c1")

    def test_portfolio_passed_through_not_in_state(self) -> None:
        pf = {"diagnosis": {"risk_type": "안정형"}, "portfolio": {"allocation": {"국내주식": 0.55}}}
        r = self.client.post("/chat", json={"user_id": "u1", "question": "내 자산은?", "portfolio": pf}).json()
        self.assertEqual(self.svc.calls[-1]["portfolio"], pf)
        self.assertNotIn("portfolio_context", r["state"])

    def test_bad_state_version_is_422(self) -> None:
        r = self.client.post("/chat", json={"user_id": "u1", "question": "q",
                                            "state": {"version": 99, "messages": []}})
        self.assertEqual(r.status_code, 422)

    def test_reports_degraded_when_rerank_failed(self) -> None:
        self.svc.reranker.last_failed = True
        self.assertTrue(self.client.post("/chat", json={"user_id": "u1", "question": "질문"}).json()["degraded"])

    def test_schema_validation(self) -> None:
        self.assertEqual(self.client.post("/chat", json={"user_id": "u1", "question": ""}).status_code, 422)
        self.assertEqual(self.client.post("/chat", json={"question": "q"}).status_code, 422)
        self.assertEqual(self.client.post("/chat", json={"user_id": "u", "question": "x" * 2001}).status_code, 422)

    def test_conversation_endpoints_removed(self) -> None:
        self.assertEqual(self.client.get("/conversations", params={"user_id": "u1"}).status_code, 404)


class ServeApiAuthTest(unittest.TestCase):
    def setUp(self) -> None:
        api.app.state.service_factory = FakeService
        self._key = api.API_KEY
        api.API_KEY = "secret"
        self.client = TestClient(api.app)
        self.client.__enter__()

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)
        api.API_KEY = self._key
        _reset()

    def test_requires_api_key(self) -> None:
        body = {"user_id": "u1", "question": "질문"}
        self.assertEqual(self.client.post("/chat", json=body).status_code, 401)
        self.assertEqual(self.client.post("/chat", json=body, headers={"X-API-Key": "wrong-key"}).status_code, 401)
        self.assertEqual(self.client.post("/chat", json=body, headers={"X-API-Key": "secret"}).status_code, 200)

    def test_health_is_open(self) -> None:
        self.assertEqual(self.client.get("/health").status_code, 200)


class ServeApiBootTest(unittest.TestCase):
    """기동 중·실패 시 /health 와 /chat 은 503 + Retry-After 를 준다 — 앱 문서의 기대."""

    def test_boot_failure_reports_503(self) -> None:
        def boom():
            raise RuntimeError("인덱스 없음")
        api.app.state.service_factory = boom
        key, api.API_KEY = api.API_KEY, ""
        try:
            with TestClient(app=api.app, raise_server_exceptions=False) as client:
                h = client.get("/health")
                self.assertEqual(h.status_code, 503)
                self.assertIn("인덱스 없음", h.json()["detail"]["error"])
                self.assertEqual(h.headers.get("retry-after"), "10")
                c = client.post("/chat", json={"user_id": "u", "question": "q"})
                self.assertEqual(c.status_code, 503)
                self.assertEqual(c.headers.get("retry-after"), "10")
        finally:
            api.API_KEY = key
            _reset()


if __name__ == "__main__":
    unittest.main()
