from __future__ import annotations

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
