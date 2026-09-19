from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol, Sequence, TypedDict, cast

from langsmith import traceable

from config import (
    LANGSMITH_API_KEY,
    LANGSMITH_PROJECT,
    LANGSMITH_TRACING,
    OPENAI_API_KEY,
    QUERY_SUMMARY_MODEL,
)
from multiturn.state import ConversationState, Message


CONVERSATION_SUMMARY_SYSTEM_PROMPT = """당신은 멀티턴 금융 대화에서 후속 질문을
해석하는 데 필요한 상태를 압축하는 Conversation Summarizer다. 일반적인 대화 요약이
아니라 다음 턴의 대명사, 지시어, 생략된 대상을 복원할 수 있는 간결한 한국어 상태를
지정된 JSON Schema로만 반환하라.

반드시 지킬 규칙:
1. 오직 previous_summary와 messages_to_summarize에 명시된 내용만 사용한다.
2. 제공되지 않은 정보나 숨겨진 사용자 정보가 있다고 추측하지 않는다.
3. 숫자, 자산 비중, 투자금액, 사용자 진단값을 임의로 생성하지 않는다.
4. 이전 summary를 폐기하지 말고 새 메시지와 합쳐 누적 summary를 만든다.
5. 주요 금융 주제, 사용자가 질문한 핵심 대상, 비교 중인 상품/개념을 보존한다.
6. 첫 번째/두 번째 등 이후 참조될 수 있는 항목과 사용자가 명시한 조건을 보존한다.
7. topic switch가 있었다면 최신 활성 주제를 분명히 하되, 이후 다시 참조될 가능성이
   있는 오래된 주제는 지나치게 삭제하지 않는다.
8. 인사, 감사, 욕설/감탄사 자체, 반복 표현, 장황한 부연, 종료된 불필요한 세부는
   제거한다. 이런 메시지만 추가되었고 previous_summary가 있다면 그대로 반환한다.
9. previous_summary도 없고 보존할 실질적 내용도 없다면 사실을 만들지 말고
   '없음'이라고만 반환한다.
10. 같은 내용을 반복하지 않고, 정보량이 충분할 때는 대체로 300~600자 이내를
    목표로 한다. 정보가 적으면 더 짧아도 된다.
"""


CONVERSATION_SUMMARY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ConversationSummaryResult:
    summary: str


class ConversationSummaryError(RuntimeError):
    """대화 요약 API 호출 또는 구조화 응답 변환에 실패했을 때 발생한다."""


class ResponsesAPI(Protocol):
    def create(self, **kwargs: Any) -> Any:
        ...


class OpenAIClient(Protocol):
    responses: ResponsesAPI


class SummaryTraceMessage(TypedDict):
    role: str
    content: str


class ConversationSummaryTraceInputs(TypedDict):
    previous_summary: str
    messages_to_summarize: list[SummaryTraceMessage]


def build_conversation_summary_input(
    previous_summary: str,
    messages_to_summarize: Sequence[Message],
) -> str:
    summary_text = previous_summary.strip() or "(없음)"
    messages_text = "\n".join(
        f"{message.role}: {message.content}" for message in messages_to_summarize
    ) or "(없음)"
    return (
        "[기존 대화 요약]\n"
        f"{summary_text}\n\n"
        "[새로 요약할 오래된 대화]\n"
        f"{messages_text}"
    )


def build_conversation_summary_trace_inputs(
    previous_summary: str,
    messages_to_summarize: Sequence[Message],
) -> ConversationSummaryTraceInputs:
    return {
        "previous_summary": previous_summary,
        "messages_to_summarize": [
            {"role": message.role, "content": message.content}
            for message in messages_to_summarize
        ],
    }


def _sanitize_summary_traceable_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    trace_inputs = cast(ConversationSummaryTraceInputs, inputs["trace_inputs"])
    return {
        "previous_summary": trace_inputs["previous_summary"],
        "messages_to_summarize": trace_inputs["messages_to_summarize"],
    }


def _serialize_summary_traceable_outputs(
    output: ConversationSummaryResult,
) -> dict[str, Any]:
    return asdict(output)


@traceable(
    name="conversation_summary_update",
    run_type="chain",
    tags=["component:conversation_summarizer"],
    process_inputs=_sanitize_summary_traceable_inputs,
    process_outputs=_serialize_summary_traceable_outputs,
    enabled=True,
)
def _run_traced_conversation_summary(
    trace_inputs: ConversationSummaryTraceInputs,
    operation: Callable[[], ConversationSummaryResult],
) -> ConversationSummaryResult:
    return operation()


class ConversationSummarizer:
    """오래된 메시지를 후속 질문 해석용 누적 summary로 압축한다."""

    def __init__(
        self,
        *,
        client: OpenAIClient | None = None,
        model: str = QUERY_SUMMARY_MODEL,
        enable_tracing: bool | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model은 비어 있을 수 없습니다.")
        using_default_client = client is None
        if enable_tracing is None:
            enable_tracing = bool(
                using_default_client and LANGSMITH_TRACING and LANGSMITH_API_KEY
            )
        self.tracing_enabled = enable_tracing
        self.client = client if client is not None else self._create_default_client(
            enable_tracing=enable_tracing
        )
        self.model = model

    def summarize(
        self,
        previous_summary: str,
        messages_to_summarize: Sequence[Message],
    ) -> ConversationSummaryResult:
        if not messages_to_summarize:
            raise ValueError("messages_to_summarize는 비어 있을 수 없습니다.")

        request: dict[str, Any] = {
            "model": self.model,
            "instructions": CONVERSATION_SUMMARY_SYSTEM_PROMPT,
            "input": build_conversation_summary_input(
                previous_summary,
                messages_to_summarize,
            ),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "conversation_summary_result",
                    "description": "후속 질문 해석용 누적 대화 요약",
                    "strict": True,
                    "schema": CONVERSATION_SUMMARY_JSON_SCHEMA,
                }
            },
            "store": False,
        }

        def invoke() -> ConversationSummaryResult:
            try:
                response = self.client.responses.create(**request)
            except Exception as exc:
                raise ConversationSummaryError(
                    "Conversation Summary API 호출에 실패했습니다."
                ) from exc
            return self._parse_response(response)

        if not self.tracing_enabled:
            return invoke()

        trace_inputs = build_conversation_summary_trace_inputs(
            previous_summary,
            messages_to_summarize,
        )
        return _run_traced_conversation_summary(
            trace_inputs,
            invoke,
            langsmith_extra={
                "project_name": LANGSMITH_PROJECT,
                "metadata": {
                    "component": "conversation_summarizer",
                    "model": self.model,
                    "message_count": len(messages_to_summarize),
                },
                "tags": ["component:conversation_summarizer"],
            },
        )

    @staticmethod
    def _create_default_client(*, enable_tracing: bool) -> OpenAIClient:
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY가 없습니다. .env에 키를 설정하거나 client를 주입하세요."
            )
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        if not enable_tracing:
            return client
        from langsmith.wrappers import wrap_openai

        return wrap_openai(
            client,
            chat_name="conversation_summary_openai_responses",
            tracing_extra={
                "metadata": {"component": "conversation_summarizer"},
                "tags": ["component:conversation_summarizer"],
            },
        )

    @staticmethod
    def _parse_response(response: Any) -> ConversationSummaryResult:
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise ConversationSummaryError(
                "Conversation Summarizer 응답에 output_text가 없습니다."
            )
        try:
            payload = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise ConversationSummaryError(
                "Conversation Summarizer 응답이 유효한 JSON이 아닙니다."
            ) from exc
        if not isinstance(payload, dict):
            raise ConversationSummaryError(
                "Conversation Summarizer 응답은 JSON object여야 합니다."
            )
        summary = payload.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ConversationSummaryError("summary는 비어 있지 않은 문자열이어야 합니다.")
        return ConversationSummaryResult(summary=summary.strip())


def maintain_conversation_summary(
    state: ConversationState,
    summarizer: ConversationSummarizer,
    *,
    recent_message_limit: int = 6,
) -> bool:
    """오래된 messages를 성공한 요약과 교환하고 갱신 여부를 반환한다."""

    if recent_message_limit <= 0:
        raise ValueError("recent_message_limit은 1 이상이어야 합니다.")
    if len(state.messages) <= recent_message_limit:
        return False

    summarized_message_count = len(state.messages) - recent_message_limit
    messages_to_summarize = tuple(state.messages[:summarized_message_count])
    result = summarizer.summarize(
        state.conversation_summary,
        messages_to_summarize,
    )

    # 외부 호출과 parsing이 모두 성공한 후에만 상태를 변경한다.
    state.apply_summary_compaction(
        result.summary,
        summarized_message_count=summarized_message_count,
    )
    return True
