"""사람이 임의 질문을 이어서 입력하는 멀티턴 RAG CLI."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg  # noqa: E402
from core.pipeline import FinalPipeline  # noqa: E402
from multiturn.session import MultiTurnSession, TurnResponse, build_session  # noqa: E402
from multiturn.state import ConversationState  # noqa: E402


def load_portfolio(path: Path | None) -> dict | None:
    """선택한 JSON 파일을 기존 portfolio_context 구조 그대로 읽는다."""

    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("portfolio JSON의 최상위 값은 object여야 합니다.")
    return payload


def format_state(state: ConversationState) -> str:
    """민감한 portfolio 원문 없이 현재 대화 state를 표시한다."""

    lines = [
        "conversation_summary:",
        state.conversation_summary or "(없음)",
        f"recent_messages: {len(state.messages)}",
    ]
    lines.extend(f"- {message.role}: {message.content}" for message in state.messages)
    lines.append(
        "portfolio_present: "
        + ("yes" if state.portfolio_context is not None else "no")
    )
    return "\n".join(lines)


def format_debug(response: TurnResponse) -> str:
    return "\n".join(
        [
            f"retrieval_query: {response.rewrite.retrieval_query}",
            f"route: {response.rewrite.route}",
            f"is_followup: {response.rewrite.is_followup}",
            f"needs_portfolio: {response.rewrite.needs_portfolio}",
            f"generation_profile: {response.profile or '-'}",
            f"summary_updated: {response.summary_updated}",
        ]
    )


def chat_loop(
    session: MultiTurnSession,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    debug: bool = False,
) -> None:
    """단일 session을 재사용하는 입력 루프."""

    debug_enabled = debug
    while True:
        try:
            question = input_fn("you> ")
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            output_fn("대화를 종료합니다.")
            return

        command = question.strip().lower()
        if command == "/exit":
            output_fn("대화를 종료합니다.")
            return
        if command == "/reset":
            session.reset_conversation()
            output_fn("대화를 초기화했습니다. 포트폴리오는 유지됩니다.")
            continue
        if command == "/state":
            output_fn(format_state(session.state))
            continue
        if command == "/debug on":
            debug_enabled = True
            output_fn("debug를 켰습니다.")
            continue
        if command == "/debug off":
            debug_enabled = False
            output_fn("debug를 껐습니다.")
            continue
        if not question.strip():
            continue

        try:
            response = session.ask(question)
        except Exception as exc:
            output_fn(f"[error] {type(exc).__name__}: {exc}")
            continue

        output_fn(response.answer)
        if debug_enabled:
            output_fn(format_debug(response))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive multi-turn RAG chat")
    parser.add_argument(
        "--portfolio",
        type=Path,
        help="기존 portfolio_context 구조의 JSON 파일",
    )
    parser.add_argument("--debug", action="store_true", help="턴 진단 정보 표시")
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="conversation summary compaction 비활성화",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not cfg.OPENAI_API_KEY or not cfg.COHERE_API_KEY:
        raise SystemExit("[fatal] OPENAI_API_KEY와 COHERE_API_KEY가 필요합니다.")

    try:
        portfolio = load_portfolio(args.portfolio)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise SystemExit(f"[fatal] portfolio 파일을 읽을 수 없습니다: {exc}") from exc

    # 무거운 corpus/index/reranker는 프로세스 시작 시 한 번만 초기화한다.
    pipeline = FinalPipeline(verbose=True)
    session = build_session(
        pipeline=pipeline,
        portfolio=portfolio,
        summarize=not args.no_summary,
    )
    print("명령: /exit, /reset, /state, /debug on, /debug off")
    chat_loop(session, debug=args.debug)


if __name__ == "__main__":
    main()
