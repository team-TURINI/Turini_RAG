from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal, Protocol, TypedDict, cast

from langsmith import traceable

from config import (
    LANGSMITH_API_KEY,
    LANGSMITH_PROJECT,
    LANGSMITH_TRACING,
    OPENAI_API_KEY,
    QUERY_REWRITE_MODEL,
    QUERY_REWRITE_REASONING_EFFORT,
)
from multiturn.prompts.query_rewrite import (
    QUERY_REWRITE_SYSTEM_PROMPT,
    build_query_rewrite_input,
    query_rewrite_recent_messages,
)
from multiturn.state import ConversationState


ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
QueryRoute = Literal["rag", "direct", "clarify"]
ALLOWED_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


@dataclass(frozen=True)
class QueryRewriteResult:
    original_question: str
    retrieval_query: str
    is_followup: bool
    needs_portfolio: bool
    route: QueryRoute


class QueryRewriteError(RuntimeError):
    """LLM 응답을 QueryRewriteResult로 변환할 수 없을 때 발생한다."""


class ResponsesAPI(Protocol):
    def create(self, **kwargs: Any) -> Any:
        ...


class OpenAIClient(Protocol):
    responses: ResponsesAPI


class TraceMessage(TypedDict):
    role: str
    content: str


class QueryRewriteTraceInputs(TypedDict):
    current_question: str
    recent_messages: list[TraceMessage]
    conversation_summary: str


QUERY_REWRITE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "original_question": {"type": "string"},
        "retrieval_query": {"type": "string"},
        "is_followup": {"type": "boolean"},
        "needs_portfolio": {"type": "boolean"},
        "route": {"type": "string", "enum": ["rag", "direct", "clarify"]},
    },
    "required": [
        "original_question",
        "retrieval_query",
        "is_followup",
        "needs_portfolio",
        "route",
    ],
    "additionalProperties": False,
}


def build_query_rewrite_trace_inputs(
    current_question: str,
    state: ConversationState,
    *,
    recent_message_limit: int = 6,
) -> QueryRewriteTraceInputs:
    messages = query_rewrite_recent_messages(
        current_question,
        state,
        recent_message_limit=recent_message_limit,
    )
    return {
        "current_question": current_question,
        "recent_messages": [
            {"role": message.role, "content": message.content}
            for message in messages
        ],
        "conversation_summary": state.conversation_summary,
    }


def build_query_rewrite_trace_metadata(
    *,
    model: str,
    reasoning_effort: ReasoningEffort | None,
    state: ConversationState,
    scenario_name: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "component": "query_rewriter",
        "query_rewrite_model": model,
        "reasoning_effort": reasoning_effort,
        "portfolio_available": state.portfolio_context is not None,
    }
    if scenario_name:
        metadata["scenario_name"] = scenario_name
    return metadata


def _sanitize_traceable_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    trace_inputs = cast(QueryRewriteTraceInputs, inputs["trace_inputs"])
    return {
        "current_question": trace_inputs["current_question"],
        "recent_messages": trace_inputs["recent_messages"],
        "conversation_summary": trace_inputs["conversation_summary"],
    }


def _serialize_traceable_outputs(output: QueryRewriteResult) -> dict[str, Any]:
    return asdict(output)


@traceable(
    name="query_rewrite",
    run_type="chain",
    tags=["component:query_rewriter"],
    process_inputs=_sanitize_traceable_inputs,
    process_outputs=_serialize_traceable_outputs,
    enabled=True,
)
def _run_traced_query_rewrite(
    trace_inputs: QueryRewriteTraceInputs,
    operation: Callable[[], QueryRewriteResult],
) -> QueryRewriteResult:
    return operation()


class QueryRewriter:
    """대화 문맥을 Query Understanding/Rewrite/Route 결과로 변환한다.

    호출 시 state에는 완료된 이전 turn만 둔다. 현재 질문은 rewrite와 이후 retrieval,
    generation을 마친 다음 state에 추가한다. Raw portfolio는 LLM 입력에 포함하지 않는다.
    """

    def __init__(
        self,
        *,
        client: OpenAIClient | None = None,
        model: str = QUERY_REWRITE_MODEL,
        reasoning_effort: ReasoningEffort | None = cast(
            ReasoningEffort | None, QUERY_REWRITE_REASONING_EFFORT
        ),
        recent_message_limit: int = 6,
        enable_tracing: bool | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model은 비어 있을 수 없습니다.")
        if recent_message_limit < 0:
            raise ValueError("recent_message_limit은 0 이상이어야 합니다.")
        if reasoning_effort is not None and reasoning_effort not in ALLOWED_REASONING_EFFORTS:
            allowed = ", ".join(sorted(ALLOWED_REASONING_EFFORTS))
            raise ValueError(f"지원하지 않는 reasoning_effort입니다: {allowed}")

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
        self.reasoning_effort = reasoning_effort
        self.recent_message_limit = recent_message_limit

    def rewrite(
        self,
        current_question: str,
        state: ConversationState,
        *,
        scenario_name: str | None = None,
    ) -> QueryRewriteResult:
        if not current_question.strip():
            raise ValueError("current_question은 비어 있을 수 없습니다.")

        request: dict[str, Any] = {
            "model": self.model,
            "instructions": QUERY_REWRITE_SYSTEM_PROMPT,
            "input": build_query_rewrite_input(
                current_question,
                state,
                recent_message_limit=self.recent_message_limit,
            ),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "query_rewrite_result",
                    "description": "검색 질문, 문맥/포트폴리오 판단, routing 결과",
                    "strict": True,
                    "schema": QUERY_REWRITE_JSON_SCHEMA,
                }
            },
            "store": False,
        }
        if self.reasoning_effort is not None:
            request["reasoning"] = {"effort": self.reasoning_effort}

        def invoke() -> QueryRewriteResult:
            response = self.client.responses.create(**request)
            return self._parse_response(response, original_question=current_question)

        if not self.tracing_enabled:
            return invoke()

        trace_inputs = build_query_rewrite_trace_inputs(
            current_question,
            state,
            recent_message_limit=self.recent_message_limit,
        )
        metadata = build_query_rewrite_trace_metadata(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            state=state,
            scenario_name=scenario_name,
        )
        tags = ["component:query_rewriter"]
        if scenario_name:
            tags.append(f"scenario:{scenario_name}")
        return _run_traced_query_rewrite(
            trace_inputs,
            invoke,
            langsmith_extra={
                "project_name": LANGSMITH_PROJECT,
                "metadata": metadata,
                "tags": tags,
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
            chat_name="query_rewrite_openai_responses",
            tracing_extra={
                "metadata": {"component": "query_rewriter"},
                "tags": ["component:query_rewriter"],
            },
        )

    @staticmethod
    def _parse_response(response: Any, *, original_question: str) -> QueryRewriteResult:
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise QueryRewriteError("Query Rewriter 응답에 output_text가 없습니다.")
        try:
            payload = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise QueryRewriteError("Query Rewriter 응답이 유효한 JSON이 아닙니다.") from exc
        if not isinstance(payload, dict):
            raise QueryRewriteError("Query Rewriter 응답은 JSON object여야 합니다.")

        generated_original_question = payload.get("original_question")
        retrieval_query = payload.get("retrieval_query")
        is_followup = payload.get("is_followup")
        needs_portfolio = payload.get("needs_portfolio")
        route = payload.get("route")
        if not isinstance(generated_original_question, str):
            raise QueryRewriteError("original_question은 문자열이어야 합니다.")
        if not isinstance(retrieval_query, str):
            raise QueryRewriteError("retrieval_query는 문자열이어야 합니다.")
        if type(is_followup) is not bool:
            raise QueryRewriteError("is_followup은 bool이어야 합니다.")
        if type(needs_portfolio) is not bool:
            raise QueryRewriteError("needs_portfolio는 bool이어야 합니다.")
        if route not in {"rag", "direct", "clarify"}:
            raise QueryRewriteError("route는 rag, direct, clarify 중 하나여야 합니다.")
        if route == "rag" and not retrieval_query.strip():
            raise QueryRewriteError("route가 rag이면 retrieval_query는 비어 있을 수 없습니다.")

        return QueryRewriteResult(
            original_question=original_question,
            retrieval_query=retrieval_query.strip(),
            is_followup=is_followup,
            needs_portfolio=needs_portfolio,
            route=cast(QueryRoute, route),
        )
