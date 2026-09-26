from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal


Role = Literal["user", "assistant"]


@dataclass
class Message:
    """대화 한 메시지."""

    role: Role
    content: str


@dataclass
class ConversationState:
    """최근 대화, 누적 요약, portfolio 데이터를 분리해 보관한다."""

    messages: list[Message] = field(default_factory=list)
    conversation_summary: str = ""
    portfolio_context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """외부 저장소에 보관할 수 있는 JSON-compatible 사본을 반환한다."""

        if not isinstance(self.conversation_summary, str):
            raise TypeError("conversation_summary는 문자열이어야 합니다.")

        messages: list[dict[str, str]] = []
        for index, message in enumerate(self.messages):
            if message.role not in {"user", "assistant"}:
                raise ValueError(f"messages[{index}].role이 올바르지 않습니다.")
            if not isinstance(message.content, str):
                raise TypeError(f"messages[{index}].content는 문자열이어야 합니다.")
            messages.append({"role": message.role, "content": message.content})

        portfolio = deepcopy(self.portfolio_context)
        payload: dict[str, Any] = {
            "messages": messages,
            "conversation_summary": self.conversation_summary,
            "portfolio_context": portfolio,
        }
        try:
            json.dumps(payload, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "ConversationState에는 JSON-compatible 값만 저장할 수 있습니다."
            ) from exc
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConversationState:
        """저장된 JSON-compatible payload를 검증하고 독립된 state로 복원한다."""

        if not isinstance(payload, Mapping):
            raise TypeError("payload는 mapping이어야 합니다.")

        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list):
            raise TypeError("messages는 list여야 합니다.")

        messages: list[Message] = []
        for index, raw_message in enumerate(raw_messages):
            if not isinstance(raw_message, Mapping):
                raise TypeError(f"messages[{index}]는 mapping이어야 합니다.")
            role = raw_message.get("role")
            content = raw_message.get("content")
            if role not in {"user", "assistant"}:
                raise ValueError(f"messages[{index}].role이 올바르지 않습니다.")
            if not isinstance(content, str):
                raise TypeError(f"messages[{index}].content는 문자열이어야 합니다.")
            messages.append(Message(role=role, content=content))

        conversation_summary = payload.get("conversation_summary")
        if not isinstance(conversation_summary, str):
            raise TypeError("conversation_summary는 문자열이어야 합니다.")

        portfolio_context = payload.get("portfolio_context")
        if portfolio_context is not None and not isinstance(portfolio_context, dict):
            raise TypeError("portfolio_context는 dict 또는 None이어야 합니다.")
        portfolio_copy = deepcopy(portfolio_context)

        state = cls(
            messages=messages,
            conversation_summary=conversation_summary,
            portfolio_context=portfolio_copy,
        )
        # 중첩된 portfolio 값까지 JSON 저장 가능 여부를 동일하게 검증한다.
        state.to_dict()
        return state

    def add_user_message(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        self.messages.append(Message(role="assistant", content=content))

    def set_summary(self, summary: str) -> None:
        self.conversation_summary = summary

    def apply_summary_compaction(
        self,
        summary: str,
        *,
        summarized_message_count: int,
    ) -> None:
        """요약 성공 후 summary와 messages를 함께 갱신한다."""

        normalized_summary = summary.strip()
        if not normalized_summary:
            raise ValueError("summary는 비어 있을 수 없습니다.")
        if not 0 <= summarized_message_count <= len(self.messages):
            raise ValueError("summarized_message_count가 messages 범위를 벗어났습니다.")

        remaining_messages = self.messages[summarized_message_count:]
        self.conversation_summary = normalized_summary
        self.messages = remaining_messages

    def set_portfolio_context(self, portfolio: dict[str, Any] | None) -> None:
        self.portfolio_context = portfolio

    def recent_messages(self, limit: int = 6) -> list[Message]:
        """최근 메시지만 반환한다."""

        if limit <= 0:
            return []
        return self.messages[-limit:]
