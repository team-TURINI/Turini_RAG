from __future__ import annotations

import json
import sys
import types
import unittest
from unittest import mock

from core import trace


class MaskTest(unittest.TestCase):
    PF = {"sample_id": "P-01",
          "diagnosis": {"risk_type": "안정형", "level_type": "초급", "level_score": 33.3,
                        "weak_tags": ["분산투자 기본 원리"]},
          "portfolio": {"input_mode": "실제", "total_amount_krw": 12_000_000,
                        "allocation": {"국내주식": 0.55, "현금성자산": 0.30}}}

    def test_values_masked_but_structure_kept(self) -> None:
        m = trace.mask_portfolio(self.PF)
        self.assertEqual(m["portfolio"]["allocation"]["국내주식"], trace.MASK)
        self.assertEqual(m["portfolio"]["total_amount_krw"], trace.MASK)
        self.assertEqual(m["diagnosis"]["level_score"], trace.MASK)
        self.assertEqual(m["diagnosis"]["weak_tags"], [trace.MASK])
        self.assertEqual(set(m["portfolio"]["allocation"]), {"국내주식", "현금성자산"})   # 키는 남는다
        self.assertEqual(m["diagnosis"]["risk_type"], "안정형")                      # 식별 정보 아님
        self.assertEqual(m["portfolio"]["input_mode"], "실제")

    def test_raw_mode_passes_through(self) -> None:
        with mock.patch.dict("os.environ", {"TRACE_PORTFOLIO": "raw"}):
            self.assertEqual(trace.mask_portfolio(self.PF), self.PF)

    def test_prompt_block_masked_and_rest_intact(self) -> None:
        from core.generator import build_portfolio_messages
        msgs = build_portfolio_messages("내 포트폴리오 괜찮아?", ["금융상품 자료 본문"], self.PF)
        masked = trace.mask_messages(msgs)
        user = masked[1]["content"]
        self.assertNotIn("12000000", user)
        self.assertNotIn("0.55", user)
        self.assertIn("국내주식", user)                 # 어떤 필드가 왔는지는 보인다
        self.assertIn("금융상품 자료 본문", user)        # 자료·질문·지시는 그대로
        self.assertIn("내 포트폴리오 괜찮아?", user)
        self.assertEqual(masked[0]["content"], msgs[0]["content"])   # system 프롬프트 불변
        self.assertEqual(msgs[1]["content"].count("12000000"), 1)    # 원본은 안 건드린다

    def test_malformed_block_is_fully_masked(self) -> None:
        text = "# 사용자 포트폴리오\n{깨진 JSON\n\n# 사용자 질문\n질문"
        out = trace._mask_portfolio_block(text)
        self.assertIn(trace.MASK, out)
        self.assertNotIn("깨진 JSON", out)

    def test_messages_without_portfolio_unchanged(self) -> None:
        msgs = [{"role": "user", "content": "# 금융상품 자료\n자료\n\n# 사용자 질문\n질문"}]
        self.assertEqual(trace.mask_messages(msgs), msgs)


class CollectTest(unittest.TestCase):
    def test_annotate_outside_turn_is_noop(self) -> None:
        trace.annotate({"x": 1})                      # 예외가 나면 안 된다

    def test_collect_gathers_and_drops_none(self) -> None:
        with trace.collect() as diag:
            trace.annotate({"a": 1, "b": None})
            trace.annotate({"c": "x"})
        self.assertEqual(diag, {"a": 1, "c": "x"})

    def test_nested_collect_shares_one_bucket(self) -> None:
        with trace.collect() as outer:
            with trace.collect() as inner:
                trace.annotate({"a": 1})
            self.assertIs(outer, inner)
        self.assertEqual(outer, {"a": 1})

    def test_bucket_is_released_after_turn(self) -> None:
        with trace.collect():
            pass
        self.assertIsNone(trace._turn_diag.get())


class SpanTest(unittest.TestCase):
    def test_disabled_span_is_noop(self) -> None:
        with mock.patch.object(trace, "enabled", return_value=False):
            with trace.span("x", inputs={"a": 1}) as sp:
                sp.end(outputs={"b": 2}); sp.add_metadata({"c": 3}); sp.add_tags(["d"])

    def test_enabled_span_sends_to_langsmith(self) -> None:
        calls: dict = {}

        class FakeRunTree:
            def add_outputs(self, o): calls.setdefault("outputs", []).append(o)
            def add_metadata(self, m): calls.setdefault("metadata", []).append(m)
            def add_tags(self, t): calls.setdefault("tags", []).append(t)

        class FakeCM:
            def __enter__(self): return FakeRunTree()
            def __exit__(self, *a): return False

        def fake_trace(**kw):
            calls["start"] = kw
            return FakeCM()

        fake_mod = types.ModuleType("langsmith")
        fake_mod.trace = fake_trace
        with mock.patch.object(trace, "enabled", return_value=True), \
             mock.patch.dict(sys.modules, {"langsmith": fake_mod}):
            with trace.span("retrieval", run_type="retriever", inputs={"q": "채권"},
                            metadata={"thread_id": "c1"}, tags=["t"]) as sp:
                sp.end(outputs={"n": 3}, latency_s=1.2)
        self.assertEqual(calls["start"]["name"], "retrieval")
        self.assertEqual(calls["start"]["run_type"], "retriever")
        self.assertEqual(calls["start"]["inputs"], {"q": "채권"})
        self.assertEqual(calls["start"]["metadata"], {"thread_id": "c1"})
        self.assertEqual(calls["outputs"], [{"n": 3}])
        self.assertEqual(calls["metadata"], [{"latency_s": 1.2}])

    def test_langsmith_failure_does_not_break_caller(self) -> None:
        fake_mod = types.ModuleType("langsmith")

        def boom(**kw):
            raise RuntimeError("langsmith 다운")

        fake_mod.trace = boom
        with mock.patch.object(trace, "enabled", return_value=True), \
             mock.patch.dict(sys.modules, {"langsmith": fake_mod}):
            with trace.span("x") as sp:
                sp.end(outputs={"ok": True})          # no-op 으로 떨어져야 한다

    def test_body_exception_propagates(self) -> None:
        with mock.patch.object(trace, "enabled", return_value=False):
            with self.assertRaises(ValueError):
                with trace.span("x"):
                    raise ValueError("본문 오류")


if __name__ == "__main__":
    unittest.main()
