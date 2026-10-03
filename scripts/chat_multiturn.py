"""사람이 임의 질문을 이어서 입력하는 멀티턴 RAG CLI.

  python scripts/chat_multiturn.py                     # 일회성 — 끝나면 대화가 사라짐
  python scripts/chat_multiturn.py --user yejin        # 로그인 모드 — 지난 대화를 이어서 함
  python scripts/chat_multiturn.py --user yejin --new  # 로그인하되 새 대화로 시작

로그인 모드는 multiturn/chat_service.ChatService 를 거치므로 매 턴이 저장되고, 다시 실행하면
가장 최근 대화가 복원된다 (명령 /list · /open · /new · /history · /title · /delete).
"""

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
from multiturn.chat_service import ChatService  # noqa: E402
from multiturn.session import MultiTurnSession, TurnResponse, build_session  # noqa: E402
from multiturn.state import ConversationState  # noqa: E402
from multiturn.store import JsonConversationStore, SqliteConversationStore  # noqa: E402


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


HELP_PERSISTENT = ("명령: /exit, /new, /list, /open <번호|id>, /history [개수], /title <제목>, "
                   "/delete, /state, /debug on, /debug off")


def format_history(messages, *, width: int = 100) -> str:
    """저장된 대화 내역을 한 줄씩 — 로그인 직후와 /history 에서 쓴다."""

    if not messages:
        return "(지난 대화 없음)"
    lines = []
    for message in messages:
        who = "you" if message.role == "user" else "bot"
        text = " ".join(message.content.split())
        clipped = text[:width] + ("..." if len(text) > width else "")
        lines.append(f"  {message.seq:>3} {who}> {clipped}")
    return "\n".join(lines)


def format_conversations(conversations) -> str:
    if not conversations:
        return "(저장된 대화 없음)"
    return "\n".join(
        f"  {i:>2}. {c.title}  ({c.n_messages}개 · {c.updated_at[:16]} · {c.conversation_id[:8]})"
        for i, c in enumerate(conversations, start=1)
    )


def format_turn_debug(turn) -> str:
    return "\n".join(
        [
            f"conversation_id: {turn.conversation_id}",
            f"retrieval_query: {turn.retrieval_query}",
            f"route: {turn.route}",
            f"is_followup: {turn.is_followup}",
            f"needs_portfolio: {turn.needs_portfolio}",
            f"generation_profile: {turn.profile or '-'}",
            f"summary_updated: {turn.summary_updated}",
        ]
    )


def chat_loop_persistent(
    service: ChatService,
    user_id: str,
    *,
    conversation_id: str | None = None,
    portfolio: dict | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    debug: bool = False,
) -> None:
    """로그인한 사용자의 대화를 저장소에 이어 쓰는 입력 루프."""

    debug_enabled = debug
    cid = conversation_id

    def show_recent(limit: int = 6) -> None:
        if cid:
            output_fn(format_history(service.history(user_id, cid, limit=limit)))

    show_recent()
    while True:
        try:
            question = input_fn("you> ")
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            output_fn("대화를 종료합니다.")
            return

        raw = question.strip()
        command = raw.lower()
        if command == "/exit":
            output_fn("대화를 종료합니다.")
            return
        if command == "/new":
            cid = None
            output_fn("새 대화를 시작합니다. 다음 질문부터 새 대화로 저장됩니다.")
            continue
        if command == "/list":
            output_fn(format_conversations(service.list_conversations(user_id)))
            continue
        if command.startswith("/open"):
            arg = raw[len("/open"):].strip()
            conversations = service.list_conversations(user_id)
            picked = None
            if arg.isdigit() and 1 <= int(arg) <= len(conversations):
                picked = conversations[int(arg) - 1].conversation_id
            elif arg:
                picked = next(
                    (c.conversation_id for c in conversations if c.conversation_id.startswith(arg)),
                    None,
                )
            if picked is None:
                output_fn("그런 대화가 없습니다. /list 로 확인하세요.")
                continue
            cid = picked
            output_fn(f"대화를 열었습니다: {cid[:8]}")
            show_recent()
            continue
        if command.startswith("/history"):
            if not cid:
                output_fn("(아직 대화가 없습니다)")
                continue
            arg = raw[len("/history"):].strip()
            output_fn(format_history(service.history(user_id, cid, limit=int(arg) if arg.isdigit() else None)))
            continue
        if command.startswith("/title"):
            title = raw[len("/title"):].strip()
            if not cid or not title:
                output_fn("사용법: /title <새 제목> — 대화를 연 뒤에 쓸 수 있습니다.")
                continue
            service.rename(user_id, cid, title)
            output_fn("제목을 바꿨습니다.")
            continue
        if command == "/delete":
            if not cid:
                output_fn("(열린 대화가 없습니다)")
                continue
            service.delete(user_id, cid)
            cid = None
            output_fn("대화를 삭제했습니다.")
            continue
        if command == "/state":
            record = service.store.load(user_id, cid) if cid else None
            if record is None:
                output_fn("(아직 대화가 없습니다)")
                continue
            output_fn(format_state(ConversationState.from_dict(record.state)))
            continue
        if command == "/debug on":
            debug_enabled = True
            output_fn("debug를 켰습니다.")
            continue
        if command == "/debug off":
            debug_enabled = False
            output_fn("debug를 껐습니다.")
            continue
        if not raw:
            continue

        try:
            turn = service.ask(user_id, question, conversation_id=cid, portfolio=portfolio)
        except Exception as exc:
            output_fn(f"[error] {type(exc).__name__}: {exc}")
            continue

        cid = turn.conversation_id
        output_fn(turn.answer)
        if debug_enabled:
            output_fn(format_turn_debug(turn))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive multi-turn RAG chat")
    parser.add_argument(
        "--portfolio",
        type=Path,
        help="기존 portfolio_context 구조의 JSON 파일",
    )
    parser.add_argument("--debug", action="store_true", help="턴 진단 정보 표시")
    parser.add_argument("--user", help="로그인 사용자 id — 주면 대화가 저장되고 다음 실행 때 이어진다")
    parser.add_argument("--new", action="store_true", help="로그인 모드에서 지난 대화를 잇지 않고 새로 시작")
    parser.add_argument("--conversation", help="이어서 할 대화 id (기본: 가장 최근 대화)")
    parser.add_argument("--store", choices=["sqlite", "json"], default="sqlite", help="대화 저장 방식")
    parser.add_argument("--store-path", type=Path, help="대화 저장 위치 (기본 data/chat/)")
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

    if args.user:
        store = (
            JsonConversationStore(args.store_path or "data/chat/conversations")
            if args.store == "json"
            else SqliteConversationStore(args.store_path or "data/chat/conversations.db")
        )
        service = ChatService(store=store, pipeline=pipeline, summarize=not args.no_summary)
        conversations = service.list_conversations(args.user)
        print(f"\n[{args.user}] 저장된 대화 {len(conversations)}개")
        print(format_conversations(conversations))
        cid = args.conversation or (None if args.new else service.latest_conversation_id(args.user))
        if cid:
            print(f"\n이어서 합니다: {cid[:8]}   (/new 새 대화, /list 목록, /history 전체 보기)")
        print(HELP_PERSISTENT)
        chat_loop_persistent(service, args.user, conversation_id=cid, portfolio=portfolio, debug=args.debug)
        return

    session = build_session(
        pipeline=pipeline,
        portfolio=portfolio,
        summarize=not args.no_summary,
    )
    print("명령: /exit, /reset, /state, /debug on, /debug off")
    chat_loop(session, debug=args.debug)


if __name__ == "__main__":
    main()
