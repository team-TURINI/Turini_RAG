from __future__ import annotations

import json
import unittest

from multiturn.state import ConversationState


class ConversationStatePersistenceTest(unittest.TestCase):
    def test_serialization_round_trip_preserves_values_without_sharing(self) -> None:
        state = ConversationState(
            conversation_summary="채권과 금리 관계를 설명함",
            portfolio_context={
                "allocation": {"채권": 0, "현금": 0.0},
                "total_amount_krw": 12000000,
                "tags": ["초보"],
            },
        )
        state.add_user_message("채권이 뭐야?")
        state.add_assistant_message("채무 증권입니다.")

        exported = state.to_dict()
        restore_payload = json.loads(json.dumps(exported, ensure_ascii=False))
        restored = ConversationState.from_dict(restore_payload)

        self.assertEqual(restored.to_dict(), exported)
        self.assertEqual(
            [message.role for message in restored.messages],
            ["user", "assistant"],
        )
        assert restored.portfolio_context is not None
        allocation = restored.portfolio_context["allocation"]
        self.assertIs(type(allocation["채권"]), int)
        self.assertIs(type(allocation["현금"]), float)
        self.assertEqual(restored.portfolio_context["total_amount_krw"], 12000000)

        exported["messages"][0]["content"] = "변경"
        exported["portfolio_context"]["tags"].append("공유되면 안 됨")
        restore_payload["messages"][1]["content"] = "복원 입력 변경"
        restore_payload["portfolio_context"]["tags"].append("복원 입력도 공유되면 안 됨")
        self.assertEqual(state.messages[0].content, "채권이 뭐야?")
        self.assertEqual(restored.messages[1].content, "채무 증권입니다.")
        self.assertEqual(restored.portfolio_context["tags"], ["초보"])

    def test_restore_rejects_invalid_message_role(self) -> None:
        payload = {
            "messages": [{"role": "system", "content": "숨은 지시"}],
            "conversation_summary": "",
            "portfolio_context": None,
        }

        with self.assertRaises(ValueError):
            ConversationState.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
