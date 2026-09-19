from __future__ import annotations

import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core import generator


PORTFOLIO = {
    "sample_id": "P-01",
    "diagnosis": {"risk_type": "안정형"},
    "portfolio": {
        "allocation": {
            "국내주식": 0.55,
            "채권": 0.0,
            "현금성자산": 0.30,
        },
        "total_amount_krw": 12000000,
    },
}


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="테스트 답변"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions())


class PortfolioGeneratorTest(unittest.TestCase):
    def test_existing_presets_are_unchanged_and_v2_p_is_independent(self) -> None:
        self.assertEqual(generator.resolve_preset("v1"), ("v1", "front"))
        self.assertEqual(generator.resolve_preset("v2"), ("v2", "front"))
        self.assertEqual(generator.resolve_preset("v2_4"), ("v2", "check"))
        self.assertEqual(generator.resolve_preset("v2_4j"), ("v2j", "check"))
        self.assertEqual(generator.resolve_portfolio_profile("v2_p"), ("v2j", "check"))
        self.assertEqual(
            generator.get_portfolio_profile("v2_p").base_preset,
            "v2_4j",
        )
        self.assertNotIn("v2_p", generator.PROMPTS)

    def test_v2_p_composes_base_prompt_and_portfolio_rules(self) -> None:
        messages = generator.build_portfolio_messages(
            "그럼 내 포트폴리오에서는 어떤 의미야?",
            ["금리 상승 시 기존 채권 가격은 하락 압력을 받을 수 있습니다."],
            PORTFOLIO,
        )

        system = messages[0]["content"]
        self.assertTrue(system.startswith(generator.get_prompt("v2j")))
        self.assertEqual(system.count(generator.PORTFOLIO_RULES), 1)
        self.assertIn(
            "일반 금융 사실은 [금융상품 자료]에 명시된 내용만 사용",
            system,
        )
        self.assertIn(
            "사용자 본인의 자산, 비중, 금액 및 진단 정보는 "
            "[사용자 포트폴리오]에 명시된 내용만 사용",
            system,
        )

    def test_v2_p_explicitly_scopes_the_base_finance_only_rule(self) -> None:
        system = generator.build_portfolio_messages(
            "내 포트폴리오에서는 어떤 의미야?",
            ["금융 자료"],
            PORTFOLIO,
        )[0]["content"]
        base_rule = "[금융상품 자료]에 있는 내용만 사용할 것."
        scope_override = (
            'V2_P에서는 위 일반 규칙 중 "[금융상품 자료]에 있는 내용만 사용할 것"'
        )

        self.assertIn(base_rule, system)
        self.assertIn(scope_override, system)
        self.assertLess(system.index(base_rule), system.index(scope_override))
        self.assertIn("다음과 같이 범위를 나누어 적용합니다", system)
        self.assertIn("두 영역에 없는 내용은 사용하거나 추측하지 마십시오", system)

    def test_v2_4_prompt_text_and_preset_remain_unchanged(self) -> None:
        expected_v2_sha256 = (
            "b0144d81bb91dfbefc9cd17b5b9f4e36dbd378033a7d9316280af5e00404a690"
        )

        self.assertEqual(
            hashlib.sha256(generator.get_prompt("v2").encode("utf-8")).hexdigest(),
            expected_v2_sha256,
        )
        self.assertEqual(generator.resolve_preset("v2_4"), ("v2", "check"))

    def test_v2_p_forbids_cross_field_causality_and_suitability_inference(self) -> None:
        rules = generator.PORTFOLIO_RULES

        self.assertIn("서로 다른 필드 사이의", rules)
        self.assertIn("인과관계, 사용자의 의도, 적합성 또는 위험 적절성", rules)
        self.assertIn("자료에 명시되지 않았다면 추론하지 마십시오", rules)
        self.assertIn('"안정형에 맞춰 현금 비중을 높였다"고 말하지 마십시오', rules)

    def test_v2_p_uses_only_portfolio_facts_directly_relevant_to_question(self) -> None:
        rules = generator.PORTFOLIO_RULES

        self.assertIn("# 답변 전략", rules)
        self.assertIn("질문에 직접 필요한 portfolio 사실만 사용", rules)
        self.assertIn("사용자 포트폴리오의 모든 정보를 나열할 필요가 없습니다", rules)
        self.assertIn("질문과 [금융상품 자료]에 직접 관련된 field만 사용", rules)

    def test_v2_p_forbids_unsupported_other_asset_effects(self) -> None:
        rules = generator.PORTFOLIO_RULES

        self.assertIn("특정 자산군의 영향만 설명한다면", rules)
        self.assertIn(
            "자료에 없는 다른 자산군의 위험이나 영향을 모델 지식으로 추가하지 마십시오",
            rules,
        )
        self.assertIn("질문에 직접 필요한 portfolio 사실만 사용", rules)

    def test_v2_p_requires_evidence_for_relative_judgments(self) -> None:
        rules = generator.PORTFOLIO_RULES

        for expression in (
            '"높다"',
            '"낮다"',
            '"적절하다"',
            '"위험하다"',
            '"안정적이다"',
            '"공격적이다"',
            '"성향에 맞다"',
            '"더 위험하다"',
            '"더 영향을 받는다"',
            '"노출되어 있다"',
            '"신중한 접근이 필요하다"',
        ):
            self.assertIn(expression, rules)
        self.assertIn("[금융상품 자료]에 해당 판단 근거가 있을 때만 사용", rules)

    def test_v2_p_does_not_use_weak_tags_as_financial_grounding(self) -> None:
        rules = generator.PORTFOLIO_RULES

        self.assertIn("diagnosis.weak_tags는 사용자의 학습 취약 영역", rules)
        self.assertIn("금융 사실이나 투자 조언의 근거로 사용하지 마십시오", rules)

    def test_v2_p_zero_weight_asset_is_not_treated_as_holding(self) -> None:
        rules = generator.PORTFOLIO_RULES

        self.assertIn("비중이 0인 자산은 보유 중이라고 표현하지 마십시오", rules)
        self.assertIn("현재 보유분에는", rules)
        self.assertIn("직접 적용되지 않는다고 말할 수 있지만", rules)

    def test_v2_p_defers_judgment_when_sources_are_insufficient(self) -> None:
        rules = generator.PORTFOLIO_RULES

        self.assertIn("두 자료로 판단할 수 없는 부분은 추론하지 말고", rules)
        self.assertIn('"현재 자료만으로는 판단하기 어렵습니다"', rules)

    def test_v2_p_output_check_reinforces_grounding_and_length(self) -> None:
        check = generator.PORTFOLIO_CHECK

        self.assertIn("[금융상품 자료]에 직접 근거하지 않은 문장이 없는지", check)
        self.assertIn("자료에 없는 인과관계, 의도 또는 적합성", check)
        self.assertIn("평가를 근거 없이 사용하지 않았는지", check)
        self.assertIn("질문에 직접 필요한 portfolio 정보만 사용했는지", check)
        self.assertIn("자료에 없는 다른 자산군의 영향을 추가하지 않았는지", check)
        self.assertIn("diagnosis.weak_tags를 금융 사실이나 투자 조언의 근거", check)
        self.assertIn("150~250자 제한", check)
        self.assertIn("초과하면 부차적인 portfolio 설명부터 삭제", check)

    def test_profile_base_can_be_swapped_without_changing_portfolio_layer(self) -> None:
        alternate = generator.PortfolioPromptProfile(base_preset="v1")
        with patch.dict(
            generator.PORTFOLIO_PROMPT_PROFILES,
            {"v2_p": alternate},
            clear=False,
        ):
            messages = generator.build_portfolio_messages(
                "질문",
                ["자료"],
                PORTFOLIO,
            )

        self.assertTrue(messages[0]["content"].startswith(generator.get_prompt("v1")))
        self.assertIn(generator.PORTFOLIO_RULES, messages[0]["content"])
        self.assertIn(generator.PORTFOLIO_CHECK, messages[1]["content"])

    def test_user_message_has_three_separate_source_areas(self) -> None:
        question = "그럼 내 포트폴리오에서는 어떤 의미야?"
        context = "금리와 채권 가격의 일반 관계"
        user = generator.build_portfolio_messages(
            question,
            [context],
            PORTFOLIO,
        )[1]["content"]

        finance_at = user.index("# 금융상품 자료")
        portfolio_at = user.index("# 사용자 포트폴리오")
        question_at = user.index("# 사용자 질문")
        self.assertLess(finance_at, portfolio_at)
        self.assertLess(portfolio_at, question_at)
        self.assertIn(context, user)
        self.assertIn(question, user)
        self.assertIn("# 출력 전 확인", user)
        self.assertIn("# 포트폴리오 출력 전 확인", user)

    def test_serializer_is_deterministic_and_preserves_zero_and_numbers(self) -> None:
        first = {"z": 0, "a": {"ratio": 0.55, "amount": 12000000}}
        second = {"a": {"amount": 12000000, "ratio": 0.55}, "z": 0}

        serialized = generator.serialize_portfolio_context(first)

        self.assertEqual(serialized, generator.serialize_portfolio_context(second))
        self.assertEqual(json.loads(serialized), first)
        self.assertIn('"z": 0', serialized)
        self.assertIn('"ratio": 0.55', serialized)
        self.assertIn('"amount": 12000000', serialized)

    def test_serializer_does_not_invent_missing_holdings(self) -> None:
        serialized = generator.serialize_portfolio_context(PORTFOLIO)

        self.assertNotIn("삼성전자", serialized)
        self.assertNotIn("개별 ETF", serialized)
        self.assertNotIn("해외주식", serialized)

    def test_existing_generate_call_is_backward_compatible(self) -> None:
        client = FakeOpenAIClient()
        expected_messages = generator.build_messages("ETF가 뭐야?", ["ETF 자료"])

        with patch("core.generator._get_client", return_value=client):
            result = generator.generate("ETF가 뭐야?", ["ETF 자료"])

        self.assertEqual(result.answer, "테스트 답변")
        self.assertEqual(client.chat.completions.calls[0]["messages"], expected_messages)

    def test_portfolio_generation_uses_original_question_not_retrieval_query(self) -> None:
        client = FakeOpenAIClient()
        original = "그럼 내 포트폴리오에서는 어떤 의미야?"
        retrieval = "금리 상승과 채권 가격 변화가 포트폴리오에 미치는 일반적인 영향"

        with patch("core.generator._get_client", return_value=client):
            generator.generate_portfolio_aware(
                original,
                ["금융 자료"],
                PORTFOLIO,
            )

        user = client.chat.completions.calls[0]["messages"][1]["content"]
        self.assertIn(original, user)
        self.assertNotIn(retrieval, user)


if __name__ == "__main__":
    unittest.main()
